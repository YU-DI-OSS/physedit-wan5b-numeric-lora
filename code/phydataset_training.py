"""PhyDataset reliability helpers for the fixed DiffSynth training entrypoint.

This module deliberately stores only trainable/adapter state plus optimizer,
scheduler, RNG and deterministic data cursor metadata. Frozen base weights are
identified by hashes in ``fingerprint`` and never copied into every checkpoint.
"""
from __future__ import annotations
import json, os, random, shutil, tempfile
from pathlib import Path


class ResumableRandomSampler:
    """Deterministic single-process sampler with an explicit next-item cursor."""
    def __init__(self, data_source, torch_module, seed=0, epoch=0, position=0):
        self.data_source, self.torch = data_source, torch_module
        self.seed, self.epoch, self.position = int(seed), int(epoch), int(position)

    def __iter__(self):
        generator = self.torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = self.torch.randperm(len(self.data_source), generator=generator).tolist()
        while self.position < len(order):
            item = order[self.position]
            self.position += 1
            yield item

    def __len__(self):
        return max(0, len(self.data_source) - self.position)

    def state_dict(self):
        return {"seed": self.seed, "epoch": self.epoch, "position": self.position,
                "dataset_length": len(self.data_source)}

    def load_state_dict(self, state):
        if int(state["dataset_length"]) != len(self.data_source):
            raise ValueError("sampler dataset length mismatch")
        self.seed, self.epoch, self.position = int(state["seed"]), int(state["epoch"]), int(state["position"])

    def next_epoch(self):
        self.epoch += 1
        self.position = 0

    def advance_if_exhausted(self):
        length = len(self.data_source)
        if self.position < length:
            return False
        if self.position > length:
            raise ValueError("sampler cursor exceeds dataset length")
        self.next_epoch()
        return True


def enforce_expert_boundary(inputs_shared, minimum, maximum, expert_name=None):
    if not (0.0 <= minimum < maximum <= 1.0):
        raise ValueError(f"invalid timestep interval [{minimum}, {maximum})")
    expected = {"high": (0.0, 0.358), "low": (0.358, 1.0)}
    if expert_name and (minimum, maximum) != expected[expert_name]:
        raise ValueError(f"{expert_name} expert requires {expected[expert_name]}, got {(minimum, maximum)}")
    result = dict(inputs_shared)
    result["min_timestep_boundary"] = minimum
    result["max_timestep_boundary"] = maximum
    assert result["min_timestep_boundary"] == minimum
    assert result["max_timestep_boundary"] == maximum
    return result


class FullStateCheckpoint:
    FORMAT = 1

    @staticmethod
    def capture_rng(torch_module, numpy_module=None):
        state = {"python": random.getstate(), "torch_cpu": torch_module.get_rng_state()}
        if torch_module.cuda.is_available():
            state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
        if numpy_module is not None:
            state["numpy"] = numpy_module.random.get_state()
        return state

    @staticmethod
    def restore_rng(state, torch_module, numpy_module=None):
        random.setstate(state["python"])
        torch_module.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch_module.cuda.is_available():
            torch_module.cuda.set_rng_state_all(state["torch_cuda"])
        if numpy_module is not None and "numpy" in state:
            numpy_module.random.set_state(state["numpy"])

    @classmethod
    def save(cls, root, *, adapter_state, optimizer, scheduler, scaler, progress,
             rng_state, fingerprint, torch_module):
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        final = root / f"step-{int(progress['global_step']):08d}"
        tmp = Path(tempfile.mkdtemp(prefix=final.name + ".tmp-", dir=root))
        try:
            torch_module.save(adapter_state, tmp / "adapter.pt")
            torch_module.save({"optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(),
                               "scaler": scaler.state_dict() if scaler else None,
                               "rng": rng_state}, tmp / "training_state.pt")
            meta = {"format": cls.FORMAT, "progress": progress, "fingerprint": fingerprint}
            (tmp / "metadata.json").write_text(json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8")
            (tmp / "COMPLETE").write_text("ok\n", encoding="ascii")
            if final.exists():
                raise FileExistsError(final)
            os.replace(tmp, final)
            return final
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True); raise

    @classmethod
    def load(cls, path, *, expected_fingerprint, optimizer, scheduler, scaler,
             torch_module):
        path = Path(path)
        if not (path / "COMPLETE").is_file():
            raise ValueError(f"incomplete checkpoint: {path}")
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("format") != cls.FORMAT or meta.get("fingerprint") != expected_fingerprint:
            raise ValueError("checkpoint format/fingerprint mismatch")
        import numpy as np
        allowed = [np.ndarray, np.dtype, np.core.multiarray._reconstruct,
                   type(np.dtype("uint32"))]
        with torch_module.serialization.safe_globals(allowed):
            state = torch_module.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        if scaler and state["scaler"] is not None: scaler.load_state_dict(state["scaler"])
        return torch_module.load(path / "adapter.pt", map_location="cpu", weights_only=True), meta["progress"], state["rng"]

    @classmethod
    def inspect(cls, path, expected_fingerprint):
        path = Path(path)
        if not (path / "COMPLETE").is_file():
            raise ValueError(f"incomplete checkpoint: {path}")
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("format") != cls.FORMAT or meta.get("fingerprint") != expected_fingerprint:
            raise ValueError("checkpoint format/fingerprint mismatch")
        return meta


def load_fingerprint(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"stage_id", "dataset_manifest_sha256", "model_manifest_sha256",
                "source_commit", "sampling_policy", "expert"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"fingerprint missing fields: {sorted(missing)}")
    return value
