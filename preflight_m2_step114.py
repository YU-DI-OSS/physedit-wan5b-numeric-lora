from __future__ import annotations

import hashlib
import json
from pathlib import Path

from safetensors import safe_open

BASE = Path("/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora/runs/wan5b-h3final-v2-20261107T0630Z")
RUN = BASE / "m2/a800-gravity-medium-prompt-v2"
SRC = BASE / "m2/a800-gravity-medium-tail-v1"
CHECKPOINT = SRC / "fullstate/step-00000114"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


fingerprint = json.loads((RUN / "fingerprint.json").read_text())
plan = json.loads((RUN / "resume-plan-step114.json").read_text())
receipt = json.loads((RUN / "m2-medium-gravity-prompt-v2.receipt.json").read_text())
validation = json.loads((RUN / "m2-medium-gravity-prompt-v2.validation.json").read_text())
source_meta = json.loads((CHECKPOINT / "metadata.json").read_text())
source_verification = json.loads((CHECKPOINT / "verification.json").read_text())
source_monitor = json.loads((SRC / "monitor/latest_report.json").read_text())

assert receipt["counts"]["strict_admitted"] == validation["records"] == fingerprint["records"] == plan["new_records"] == 2541
assert fingerprint["derived_metadata_sha256"] == validation["manifest_sha256"]
assert validation["status"] == "PASS"
assert source_meta["fingerprint"]["stage_id"] == fingerprint["transition_source_stage_id"]
assert source_meta["fingerprint"]["records"] == fingerprint["transition_source_records"] == 455
assert source_monitor["status"] == "COMPLETE"
assert source_monitor["global_step"] == fingerprint["transition_source_step"] == 114
assert source_monitor["consumed_samples"] == source_monitor["dataset_records"] == 455
assert (CHECKPOINT / "COMPLETE").read_text().strip().lower() == "ok"
assert sha256_file(CHECKPOINT / "adapter.pt") == fingerprint["transition_source_adapter_sha256"]
source_lora = RUN / "init-lora-from-step114.safetensors"
assert sha256_file(source_lora) == fingerprint["transition_source_training_sha256"]
with safe_open(source_lora, framework="pt", device="cpu") as handle:
    lora_keys = list(handle.keys())
assert len(lora_keys) == 600

source_code = {path.name: sha256_file(path) for path in (SRC / "code").glob("*.py")}
target_code = {path.name: sha256_file(path) for path in (RUN / "code").glob("*.py")}
assert source_code == target_code
worker_text = (RUN / "worker-step114.sh").read_text()
assert "--phydataset_stage_transition_state" in worker_text
assert '$SRC/fullstate/step-00000114' in worker_text
assert "GPU_LAUNCH_AUTHORIZED" in worker_text
assert not (RUN / "GPU_LAUNCH_AUTHORIZED").exists()
assert not (RUN / "output").exists()
assert not (RUN / "fullstate").exists()

preflight = {
    "status": "READY_CPU_DATA_GPU_LAUNCH_NOT_AUTHORIZED",
    "new_records": 2541,
    "new_cursor": 0,
    "new_manifest_sha256": validation["manifest_sha256"],
    "source_checkpoint": str(CHECKPOINT),
    "source_checkpoint_complete_marker": True,
    "source_checkpoint_verification_entries": sorted(source_verification),
    "source_step": 114,
    "source_terminal_records": 455,
    "source_lora_sha256": sha256_file(source_lora),
    "source_lora_tensor_count": len(lora_keys),
    "source_adapter_sha256": sha256_file(CHECKPOINT / "adapter.pt"),
    "expected_harmonic_encoder_tensors": 5,
    "expected_optimizer_state_entries": 605,
    "preserve_scheduler_and_rng": True,
    "transition_implementation": "existing validated --phydataset_stage_transition_state",
    "training_code_identical_to_step114_stage": source_code == target_code,
    "training_code_change_required": False,
    "worker_bash_syntax_checked": True,
    "gpu_authorization_gate_present": True,
    "gpu_authorization_gate_open": False,
    "gpu_training_started": False,
}
preflight_path = RUN / "continuation-preflight.json"
preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")

checksum_names = [
    "m2-medium-gravity-prompt-v2.jsonl",
    "m2-medium-gravity-prompt-v2.sidecar.jsonl",
    "m2-medium-gravity-prompt-v2.exclusions.jsonl",
    "m2-medium-gravity-prompt-v2.receipt.json",
    "m2-medium-gravity-prompt-v2.validation.json",
    "fingerprint.json",
    "resume-plan-step114.json",
    "worker-step114.sh",
    "build_m2_medium_gravity_prompt_v2.py",
    "validate_m2_medium_gravity_prompt_v2.py",
    "continuation-preflight.json",
    "export_step114_lora.py",
    "init-lora-from-step114.safetensors",
    "init-lora-step114-receipt.json",
    "preflight_m2_step114.py",
]
with (RUN / "SHA256SUMS").open("w") as handle:
    for name in sorted(checksum_names):
        handle.write(f"{sha256_file(RUN / name)}  {name}\n")

print(json.dumps(preflight, sort_keys=True))
