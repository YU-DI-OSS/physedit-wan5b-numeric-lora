"""Isolated runtime patch for a one-sample Wan5B numeric-conditioning smoke.

Loaded only when its directory is explicitly placed on PYTHONPATH. It does not
modify the shared DiffSynth source tree or production manifests.
"""
from __future__ import annotations

import json
import os
import hashlib
import shutil
import tempfile
import importlib.util
import sys
from pathlib import Path

_training_dir = "/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora/repos/DiffSynth-Studio-phydataset/examples/wanvideo/model_training"
if _training_dir not in sys.path:
    sys.path.insert(0, _training_dir)

import torch
from torch import nn

import diffsynth.diffusion as diffusion
import diffsynth.models.wan_video_dit as wan_dit

from wan_numeric_conditioning import NumericBatch
from wan_numeric_production_bridge import (
    WanNumericConditioner,
    cross_attention_with_numeric_regularizer,
)
from wan_numeric_runtime import get_current

# The fixed training entrypoint imports launch_training_task from the installed
# package. Load the audited PhyDataset runner as an isolated sibling module and
# replace only this process' exported entrypoint.
_runner_path = Path(__file__).with_name("numeric_runner.py")
if _runner_path.is_file():
    _spec = importlib.util.spec_from_file_location(
        "diffsynth.diffusion.numeric_runner", _runner_path)
    _numeric_runner = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_numeric_runner)
    diffusion.launch_training_task = _numeric_runner.launch_training_task


REGULARIZER_WEIGHT = 0.01


def _safe_checkpoint_load(path, *, numpy_rng=False):
    """Read verified legacy tensor/RNG payloads without unrestricted pickle."""
    if not numpy_rng:
        return torch.load(path, map_location="cpu", weights_only=True)
    import numpy as np
    from torch.serialization import safe_globals
    allowed = [np.ndarray, np.dtype, np.core.multiarray._reconstruct,
               type(np.dtype("uint32"))]
    with safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)
SAMPLED_BLOCK = 15
SAMPLED_QUERIES = 64


class NumericTextProjection(nn.Module):
    def __init__(self, base: nn.Module, target_cross_attention: nn.Module):
        super().__init__()
        self.base = base
        self.conditioner = WanNumericConditioner(
            model_dim=3072, max_components=1, num_types=8,
            harmonics=8, hidden_dim=512,
        )
        # Observation target is an existing model submodule. Keep only a plain
        # reference so state_dict names remain under blocks.15.cross_attn and
        # are not aliased through text_embedding.
        object.__setattr__(self, "_target_cross_attention", target_cross_attention)

    def forward(self, context):
        projected = self.base(context)
        bsz = projected.shape[0]
        if bsz != 1:
            raise RuntimeError("Wave01 numeric runtime requires per-rank batch size one")
        current = get_current()
        values = current["values"]
        type_ids = current["type_ids"]
        token_indices = current["token_indices"]
        count = len(token_indices)
        batch = NumericBatch(
            values=torch.tensor(values, device=projected.device, dtype=torch.float32).view(1, count, 1),
            type_ids=torch.tensor(type_ids, device=projected.device, dtype=torch.long).view(1, count),
            token_indices=torch.tensor(
                token_indices, device=projected.device, dtype=torch.long,
            ).view(1, count),
            component_mask=torch.ones(
                (bsz, count, 1), device=projected.device, dtype=torch.float32,
            ),
        )
        conditioned = self.conditioner(projected, batch)
        self._target_cross_attention._numeric_mask = conditioned.text_numeric_mask
        return conditioned.context


_original_cross_attention_forward = wan_dit.CrossAttention.forward


def _numeric_cross_attention_forward(self, x, y):
    if not getattr(self, "_numeric_enabled", False):
        return _original_cross_attention_forward(self, x, y)
    mask = getattr(self, "_numeric_mask", None)
    if mask is None:
        raise RuntimeError("numeric token mask was not supplied by text projection")
    q_count = min(SAMPLED_QUERIES, x.shape[1])
    indices = torch.linspace(
        0, x.shape[1] - 1, q_count, device=x.device,
    ).round().to(torch.long).unsqueeze(0).expand(x.shape[0], -1)
    output, regularizer = cross_attention_with_numeric_regularizer(
        self, x, y, mask, indices,
        threshold=float(0.75 * mask.sum(dim=1).float().mean().item() / mask.shape[1]),
    )
    self._last_numeric_regularizer = regularizer
    return output


wan_dit.CrossAttention.forward = _numeric_cross_attention_forward


_original_switch = diffusion.DiffusionTrainingModule.switch_pipe_to_training_mode


def _install_rank_aware_fullstate():
    """Patch the isolated smoke only: one shared state plus per-rank RNG/cursor."""
    import phydataset_training as pt
    if getattr(pt.FullStateCheckpoint, "_numeric_rank_aware", False):
        return
    cls = pt.FullStateCheckpoint

    def _sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_and_rotate(root, final):
        """Verify the new atomic state, then retain only three states in this run.

        Rotation is disabled unless WAN_NUMERIC_ROTATE_ROOT resolves exactly to
        the full-state root.  This prevents this isolated patch from ever
        touching historical runs through a parent path or a symlink alias.
        """
        configured = os.environ.get("WAN_NUMERIC_ROTATE_ROOT")
        if not configured:
            return
        root = Path(root).resolve(strict=True)
        allowed = Path(configured).resolve(strict=True)
        if root != allowed or final.parent.resolve(strict=True) != allowed:
            raise RuntimeError("checkpoint rotation root identity mismatch")
        keep = int(os.environ.get("WAN_NUMERIC_KEEP_LAST", "3"))
        if keep != 3:
            raise RuntimeError("formal numeric run requires keep_last=3")
        required = ["adapter.pt", "training_state.pt", "metadata.json", "COMPLETE"]
        required.extend(f"rank-{rank:05d}.pt" for rank in range(
            int(json.loads((final / "metadata.json").read_text(encoding="utf-8"))["world_size"])))
        verification = {}
        for name in required:
            path = final / name
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"new checkpoint failed verification: {path}")
            verification[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        (final / "verification.json").write_text(
            json.dumps(verification, sort_keys=True, indent=2), encoding="utf-8")
        # Re-read the verification record before any removal.
        recorded = json.loads((final / "verification.json").read_text(encoding="utf-8"))
        if recorded != verification:
            raise RuntimeError("checkpoint verification record mismatch")
        states = []
        for path in root.glob("step-????????"):
            if not path.is_dir() or not (path / "COMPLETE").is_file():
                continue
            try:
                step = int(path.name.removeprefix("step-"))
            except ValueError:
                continue
            states.append((step, path))
        states.sort()
        for _, path in states[:-keep]:
            if path.resolve().parent != allowed or path == final:
                raise RuntimeError("refusing unsafe checkpoint rotation target")
            shutil.rmtree(path)

    @classmethod
    def save(rank_cls, root, *, adapter_state, optimizer, scheduler, scaler, progress,
             rng_state, fingerprint, torch_module):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        final = root / f"step-{int(progress['global_step']):08d}"
        staging = root / (final.name + ".staging")
        if rank == 0:
            staging.mkdir()
        if world > 1:
            dist.barrier()
        rank_tmp = staging / f"rank-{rank:05d}.tmp"
        rank_final = staging / f"rank-{rank:05d}.pt"
        torch_module.save({"progress": progress, "rng": rng_state}, rank_tmp)
        os.replace(rank_tmp, rank_final)
        if world > 1:
            dist.barrier()
        if rank == 0:
            torch_module.save(adapter_state, staging / "adapter.pt")
            torch_module.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
            }, staging / "training_state.pt")
            meta = {"format": rank_cls.FORMAT, "fingerprint": fingerprint,
                    "world_size": world, "rank_aware": True}
            (staging / "metadata.json").write_text(
                json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8")
            (staging / "COMPLETE").write_text("ok\n", encoding="ascii")
            if final.exists():
                raise FileExistsError(final)
            os.replace(staging, final)
            _verify_and_rotate(root, final)
        if world > 1:
            dist.barrier()
        return final

    @classmethod
    def load(rank_cls, path, *, expected_fingerprint, optimizer, scheduler, scaler,
             torch_module):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        path = Path(path)
        if not (path / "COMPLETE").is_file():
            raise ValueError(f"incomplete checkpoint: {path}")
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if (meta.get("format") != rank_cls.FORMAT or
                meta.get("fingerprint") != expected_fingerprint or
                not meta.get("rank_aware") or int(meta.get("world_size", -1)) != world):
            raise ValueError("rank-aware checkpoint format/fingerprint/world mismatch")
        common = _safe_checkpoint_load(path / "training_state.pt")
        optimizer.load_state_dict(common["optimizer"])
        scheduler.load_state_dict(common["scheduler"])
        if scaler and common["scaler"] is not None:
            scaler.load_state_dict(common["scaler"])
        local = _safe_checkpoint_load(path / f"rank-{rank:05d}.pt", numpy_rng=True)
        return (_safe_checkpoint_load(path / "adapter.pt"),
                local["progress"], local["rng"])

    @classmethod
    def inspect(rank_cls, path, expected_fingerprint):
        path = Path(path)
        if not (path / "COMPLETE").is_file():
            raise ValueError(f"incomplete checkpoint: {path}")
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("format") != rank_cls.FORMAT or meta.get("fingerprint") != expected_fingerprint:
            raise ValueError("checkpoint format/fingerprint mismatch")
        return meta

    cls.save = save
    cls.load = load
    cls.inspect = inspect
    cls._numeric_rank_aware = True


_install_rank_aware_fullstate()


def _numeric_switch(self, pipe, *args, **kwargs):
    _install_rank_aware_fullstate()
    args = list(args)
    lora_only = os.environ.get("WAN_NUMERIC_LORA_ONLY_CKPT")
    if lora_only:
        if len(args) < 5:
            raise RuntimeError("unexpected switch_pipe_to_training_mode signature")
        args[4] = lora_only
    result = _original_switch(self, pipe, *args, **kwargs)
    if getattr(pipe, "dit", None) is None:
        return result
    target = pipe.dit.blocks[SAMPLED_BLOCK].cross_attn
    target._numeric_enabled = True
    pipe.dit.text_embedding = NumericTextProjection(pipe.dit.text_embedding, target)
    for parameter in pipe.dit.text_embedding.conditioner.parameters():
        parameter.requires_grad_(True)
    restore_path = os.environ.get("WAN_NUMERIC_RESTORE_CKPT")
    restored = False
    if restore_path:
        if restore_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(restore_path, device="cpu")
        else:
            state = _safe_checkpoint_load(restore_path)
        prefix = "text_embedding.conditioner."
        numeric_state = {
            key[len(prefix):]: value
            for key, value in state.items() if key.startswith(prefix)
        }
        pipe.dit.text_embedding.conditioner.load_state_dict(numeric_state, strict=True)
        restored = True
    trainable = sum(p.numel() for p in pipe.dit.parameters() if p.requires_grad)
    numeric = sum(p.numel() for p in pipe.dit.text_embedding.conditioner.parameters())
    print("WAN_NUMERIC_ATTACH " + json.dumps({
        "block": SAMPLED_BLOCK,
        "numeric_parameters": numeric,
        "total_trainable_parameters": trainable,
        "conditioning": "dynamic_per_sample_sidecar",
        "initial_speed_supervised": False,
        "regularizer_weight": REGULARIZER_WEIGHT,
        "threshold_policy": "0.75_times_numeric_key_fraction",
        "numeric_restore_strict": restored,
    }, sort_keys=True), flush=True)
    return result


diffusion.DiffusionTrainingModule.switch_pipe_to_training_mode = _numeric_switch


_original_loss = diffusion.FlowMatchSFTLoss


def _state_sha(state):
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def _tensor_summary(value):
    x = value.detach().float()
    flat = x.reshape(-1)
    return {
        "shape": list(x.shape),
        "sum": float(x.sum().cpu()),
        "sq_sum": float((x * x).sum().cpu()),
        "head": [float(v) for v in flat[:4].cpu()],
    }


def _numeric_loss(pipe, **inputs):
    # Instrumented literal equivalent of FlowMatchSFTLoss. This is used only
    # by the bounded exact-resume diagnostic overlay.
    cpu_rng_before = _state_sha(torch.get_rng_state())
    cuda_rng_before = _state_sha(torch.cuda.get_rng_state())
    input_summary = _tensor_summary(inputs["input_latents"])
    max_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_boundary, max_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    noise_summary = _tensor_summary(noise)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    target_value = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
    if "first_frame_latents" in inputs:
        prediction = prediction[:, :, 1:]
        target_value = target_value[:, :, 1:]
    base_loss = torch.nn.functional.mse_loss(prediction.float(), target_value.float())
    base_loss = base_loss * pipe.scheduler.training_weight(timestep)
    target = pipe.dit.blocks[SAMPLED_BLOCK].cross_attn
    regularizer = getattr(target, "_last_numeric_regularizer", None)
    if regularizer is None:
        raise RuntimeError("numeric attention regularizer was not produced")
    total = base_loss + REGULARIZER_WEIGHT * regularizer
    rank = int(os.environ.get("RANK", "0"))
    evidence = {
        "rank": rank,
        "cpu_rng_before": cpu_rng_before,
        "cuda_rng_before": cuda_rng_before,
        "timestep_id": int(timestep_id.item()),
        "timestep": float(timestep.detach().float().cpu().item()),
        "input_latents": input_summary,
        "noise": noise_summary,
        "base_loss": float(base_loss.detach().float().cpu()),
        "regularizer": float(regularizer.detach().float().cpu()),
        "total_loss": float(total.detach().float().cpu()),
        "finite": bool(torch.isfinite(total.detach()).all()),
    }
    print("WAN_NUMERIC_LOSS " + json.dumps(evidence, sort_keys=True), flush=True)
    trace_dir = os.environ.get("WAN_NUMERIC_STOCHASTIC_TRACE_DIR")
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        with open(os.path.join(trace_dir, f"rank-{rank}.jsonl"), "a", encoding="utf-8") as trace:
            trace.write(json.dumps(evidence, sort_keys=True) + "\n")
    return total


diffusion.FlowMatchSFTLoss = _numeric_loss
