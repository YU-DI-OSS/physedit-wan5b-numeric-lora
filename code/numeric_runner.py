import os, json, torch, importlib, random
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def save_training_args(args):
    output_path = getattr(args, "output_path", None) if args is not None else None
    if output_path is None:
        return
    try:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, "training_args.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False, default=str)
        print(f"Training arguments saved to `{save_path}`.")
    except Exception as e:
        print(f"Warning: failed to save training arguments: {e}")


def exclude_quantized_params_from_ddp_sync(accelerator: Accelerator, model: DiffusionTrainingModule):
    """DDP broadcasts every parameter when it is constructed, but a quantized weight backed by a
    tensor subclass cannot be flattened into a broadcast bucket. Such weights are frozen and every
    rank loads them from the same checkpoint, so let DDP skip them."""
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass
        quant_configs = [module.quantize_config for module in model.modules() if getattr(module, "quantize_config", None) is not None]
        ignored = [
            f"{name}.weight" for name, module in model.named_modules()
            if any(quantize.is_quantized_linear(module) for quantize in quant_configs)
            and not module.weight.requires_grad and is_traceable_wrapper_subclass(module.weight)
        ]
        if len(ignored) > 0:
            model._ddp_params_and_buffers_to_ignore = ignored
            if accelerator.is_main_process:
                print(f"{len(ignored)} quantized weights are excluded from DDP state synchronization.")
    except Exception as e:
        print(f"Warning: failed to exclude quantized weights from DDP state synchronization: {e}")


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer

    if accelerator.is_main_process:
        save_training_args(args)

    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    state_root = getattr(args, "phydataset_full_state_dir", None) if args is not None else None
    resume_path = getattr(args, "phydataset_resume_state", None) if args is not None else None
    transition_path = getattr(args, "phydataset_stage_transition_state", None) if args is not None else None
    fingerprint_path = getattr(args, "phydataset_fingerprint_json", None) if args is not None else None
    checkpoint_steps_arg = getattr(args, "phydataset_checkpoint_steps", None) if args is not None else None
    checkpoint_steps = ({int(step) for step in checkpoint_steps_arg.split(",") if step.strip()}
                        if checkpoint_steps_arg else set())
    sampler = None
    if state_root or resume_path or transition_path:
        import numpy as np
        if num_workers != 0:
            raise ValueError("exact PhyDataset resume requires dataset_num_workers=0 (no prefetch cursor drift)")
        from phydataset_training import ResumableRandomSampler, FullStateCheckpoint, load_fingerprint
        sampler = ResumableRandomSampler(dataset, torch, seed=getattr(args, "phydataset_sampler_seed", 0))
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, shuffle=False, collate_fn=lambda x: x[0], num_workers=0)
        fingerprint = load_fingerprint(fingerprint_path)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        exclude_quantized_params_from_ddp_sync(accelerator, model)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    global_step, start_epoch = 0, 0
    if resume_path:
        # Adapter was loaded before optimizer construction in train.py. Restore
        # optimizer/scheduler/RNG/cursor only after accelerator.prepare.
        _, progress, rng = FullStateCheckpoint.load(
            resume_path, expected_fingerprint=fingerprint, optimizer=optimizer,
            scheduler=scheduler, scaler=getattr(accelerator, "scaler", None), torch_module=torch)
        sampler.load_state_dict(progress["sampler"])
        FullStateCheckpoint.restore_rng(rng, torch, np)
        global_step, start_epoch = int(progress["global_step"]), int(progress["epoch"])
        if sampler.advance_if_exhausted():
            start_epoch = sampler.epoch
        model_logger.num_steps = global_step
        resume_trace_path = os.environ.get("PHYDATASET_RESUME_TRACE")
        if resume_trace_path:
            optimizer_state = optimizer.state_dict()
            scheduler_state = scheduler.state_dict()
            cuda_matches = True
            if "torch_cuda" in rng and torch.cuda.is_available():
                cuda_matches = all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), rng["torch_cuda"]))
            numpy_matches = True
            if "numpy" in rng:
                current_numpy = np.random.get_state()
                numpy_matches = (current_numpy[0] == rng["numpy"][0] and
                                 np.array_equal(current_numpy[1], rng["numpy"][1]) and
                                 current_numpy[2:] == rng["numpy"][2:])
            evidence = {
                "global_step": global_step, "start_epoch": start_epoch,
                "sampler": sampler.state_dict(),
                "optimizer_state_entries": len(optimizer_state["state"]),
                "optimizer_param_groups": len(optimizer_state["param_groups"]),
                "scheduler_last_epoch": scheduler_state.get("last_epoch"),
                "python_rng_matches": random.getstate() == rng["python"],
                "torch_cpu_rng_matches": torch.equal(torch.get_rng_state(), rng["torch_cpu"]),
                "torch_cuda_rng_matches": cuda_matches, "numpy_rng_matches": numpy_matches,
            }
            with open(resume_trace_path, "w", encoding="utf-8") as trace:
                json.dump(evidence, trace, sort_keys=True)
    elif transition_path:
        # Preserve model, optimizer, scheduler, scaler and RNG from the completed
        # source stage. A new disjoint dataset gets its own sampler at position 0.
        from pathlib import Path
        source_meta = json.loads((Path(transition_path) / "metadata.json").read_text())
        _, source_progress, source_rng = FullStateCheckpoint.load(
            transition_path, expected_fingerprint=source_meta["fingerprint"],
            optimizer=optimizer, scheduler=scheduler,
            scaler=getattr(accelerator, "scaler", None), torch_module=torch)
        source_sampler = source_progress["sampler"]
        if (source_progress["global_step"] != fingerprint["transition_source_step"] or
                source_sampler["dataset_length"] != fingerprint["transition_source_records"] or
                source_sampler["position"] != fingerprint["transition_source_records"] or
                source_meta["fingerprint"]["stage_id"] != fingerprint["transition_source_stage_id"]):
            raise RuntimeError("source stage is not at its verified terminal cursor")
        FullStateCheckpoint.restore_rng(source_rng, torch, np)
        if sampler.epoch != 0 or sampler.position != 0 or len(dataset) != fingerprint["records"]:
            raise RuntimeError("new stage sampler did not start at position zero")
        global_step, start_epoch = 0, 0
        model_logger.num_steps = 0
        transition_trace = os.environ.get("PHYDATASET_TRANSITION_TRACE")
        if transition_trace:
            with open(transition_trace, "w", encoding="utf-8") as handle:
                json.dump({"source_checkpoint": transition_path,
                           "source_step": source_progress["global_step"],
                           "source_cursor": source_sampler["position"],
                           "target_records": len(dataset), "target_cursor": sampler.position,
                           "optimizer_entries": len(optimizer.state_dict()["state"]),
                           "scheduler_last_epoch": scheduler.state_dict().get("last_epoch"),
                           "model_initialized_from_source": True,
                           "rng_restored": True}, handle, sort_keys=True, indent=2)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    ga_trace_path = os.environ.get("PHYDATASET_GA_TRACE")
    micro_since_update = 0
    last_full_state_step = None
    stop_requested = False
    stop_request_file = os.environ.get("PHYDATASET_STOP_REQUEST_FILE")

    def save_full_state(step, epoch_id):
        """Save an atomic state with the logical (non-prefetched) cursor."""
        nonlocal last_full_state_step
        if not state_root or last_full_state_step == step:
            return
        world_size = int(getattr(accelerator, "num_processes", 1))
        ga_steps = int(getattr(args, "gradient_accumulation_steps", 1))
        # Accelerator can advance the sampler iterator before an optimizer
        # update. Derive the durable cursor from completed effective batches
        # within this epoch instead of the iterator's look-ahead position.
        epoch_steps = (len(dataset) + world_size * ga_steps - 1) // (world_size * ga_steps)
        logical_position = min(len(dataset),
                               (step - epoch_id * epoch_steps) * world_size * ga_steps)
        if not 0 <= logical_position <= len(dataset):
            raise RuntimeError("derived checkpoint cursor is outside the dataset")
        full_state = accelerator.get_state_dict(model)
        adapter_state = accelerator.unwrap_model(model).export_trainable_state_dict(
            full_state, remove_prefix=model_logger.remove_prefix_in_ckpt)
        adapter_state = model_logger.state_dict_converter(adapter_state)
        FullStateCheckpoint.save(
            state_root, adapter_state=adapter_state, optimizer=optimizer,
            scheduler=scheduler, scaler=getattr(accelerator, "scaler", None),
            progress={"global_step": step, "epoch": epoch_id,
                      "sampler": {"seed": getattr(args, "phydataset_sampler_seed", 0),
                                  "epoch": epoch_id, "position": logical_position,
                                  "dataset_length": len(dataset)}},
            rng_state=FullStateCheckpoint.capture_rng(torch, np),
            fingerprint=fingerprint, torch_module=torch)
        last_full_state_step = step

    for epoch_id in range(start_epoch, num_epochs):
        for data in tqdm(dataloader):
            micro_since_update += 1
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                before_update = None
                ga_evidence = None
                if ga_trace_path and accelerator.sync_gradients:
                    unwrapped = accelerator.unwrap_model(model)
                    trainable = [(n, p) for n, p in unwrapped.named_parameters() if p.requires_grad]
                    frozen = [(n, p) for n, p in unwrapped.named_parameters() if not p.requires_grad]
                    grads = [p.grad for _, p in trainable if p.grad is not None]
                    before_update = [(n, p.detach().clone()) for n, p in trainable]
                    ga_evidence = {
                        "global_step_before": global_step,
                        "microsteps_since_update": micro_since_update,
                        "trainable_parameter_count": sum(p.numel() for _, p in trainable),
                        "frozen_parameter_count": sum(p.numel() for _, p in frozen),
                        "frozen_with_grad": sum(1 for _, p in frozen if p.grad is not None),
                        "gradient_tensor_count": len(grads),
                        "gradient_finite": all(bool(torch.isfinite(g).all()) for g in grads),
                        "gradient_norm": float(torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads)).cpu()),
                        "loss": float(loss.detach().float().cpu()),
                    }
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # Accelerate suppresses optimizer updates on non-sync GA
                # microsteps. Count and checkpoint only real optimizer updates.
                if accelerator.sync_gradients:
                    global_step += 1
                    if ga_trace_path:
                        unwrapped = accelerator.unwrap_model(model)
                        current = dict(unwrapped.named_parameters())
                        ga_evidence["global_step_after"] = global_step
                        ga_evidence["max_trainable_delta"] = max(
                            float((current[n].detach() - old).abs().max().cpu()) for n, old in before_update)
                        ga_evidence["optimizer_step_observed"] = True
                        os.makedirs(os.path.dirname(ga_trace_path), exist_ok=True)
                        with open(ga_trace_path, "a", encoding="utf-8") as trace:
                            trace.write(json.dumps(ga_evidence, sort_keys=True) + "\n")
                        micro_since_update = 0
                    stop_requested = bool(stop_request_file and os.path.isfile(stop_request_file))
                    should_checkpoint = (stop_requested or global_step in checkpoint_steps or
                                         (save_steps is not None and global_step % save_steps == 0))
                    model_logger.on_step_end(accelerator, model, 1 if should_checkpoint else None, loss=loss)
                    if state_root and should_checkpoint:
                        save_full_state(global_step, epoch_id)
            if stop_requested:
                break
        if sampler is not None:
            if not stop_requested and (sampler.epoch != epoch_id or sampler.position != len(dataset)):
                raise RuntimeError("sampler epoch ended with an invalid cursor")
            if not stop_requested:
                sampler.next_epoch()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        if stop_requested:
            break

    # A normal terminal state is mandatory even when the last step is not a
    # periodic boundary.  This occurs before the inference-only export below.
    if state_root and global_step > 0:
        save_full_state(global_step, max(start_epoch, num_epochs - 1))

    model_logger.on_training_end(accelerator, model, None if checkpoint_steps else save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        exclude_quantized_params_from_ddp_sync(accelerator, model)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
