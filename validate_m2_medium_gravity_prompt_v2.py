from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

WAN = Path("/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora")
BASE = WAN / "runs/wan5b-h3final-v2-20261107T0630Z"
OUT = BASE / "m2/a800-gravity-medium-prompt-v2"
H3 = Path("/HOME/sysu_xdliang/sysu_xdliang_3/Storage01/h3_numeric_attention_20261106")
TOKENIZER_PATH = WAN / "models/Wan2.2-TI2V-5B/google/umt5-xxl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


receipt_path = OUT / "m2-medium-gravity-prompt-v2.receipt.json"
manifest_path = OUT / "m2-medium-gravity-prompt-v2.jsonl"
sidecar_path = OUT / "m2-medium-gravity-prompt-v2.sidecar.jsonl"
exclusions_path = OUT / "m2-medium-gravity-prompt-v2.exclusions.jsonl"
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True, use_fast=True)
assert tokenizer.is_fast

consumed = set()
for path in [
    BASE / "data/m2-t15-strict-v1.jsonl",
    BASE / "data/m2-gravity-next-v1.jsonl",
    BASE / "m2/a800-gpu1-migration-v1/m2-gravity-medium-tail-v1.jsonl",
]:
    consumed.update(row["numeric_video_id"] for row in read_jsonl(path))
assert len(consumed) == 7678
historical = set()
for path in [
    WAN / "runs/wan5b-wave01-m1-medium-v2-20261105T1625Z/metadata-main404.jsonl",
    WAN / "runs/wan5b-wave01-m1-recovery-tail2-20261106T0140Z/metadata-tail2.jsonl",
]:
    historical.update(Path(row["video"]).stem for row in read_jsonl(path))
assert len(historical) == 406
with gzip.open(H3 / "final_trajectory_only/CORE100_AND_SAME_TRAJECTORY_IDS.json.gz", "rt") as source:
    held = set(json.load(source))

manifest_rows = list(read_jsonl(manifest_path))
sidecar_rows = list(read_jsonl(sidecar_path))
assert len(manifest_rows) == len(sidecar_rows) == receipt["counts"]["strict_admitted"]
ids = []
gravity_values = Counter()
for manifest_row, sidecar in zip(manifest_rows, sidecar_rows):
    video_id = manifest_row["numeric_video_id"]
    assert video_id == sidecar["video_id"]
    ids.append(video_id)
    assert video_id not in consumed and video_id not in historical and video_id not in held
    assert manifest_row["video"] == str(H3 / "media" / f"{video_id}.mp4")
    assert hashlib.sha256(manifest_row["prompt"].encode()).hexdigest() == manifest_row["numeric_prompt_sha256"] == sidecar["prompt_sha256"]
    assert sidecar["parameter"] == "gravity" and sidecar["unit"] == "m/s^2" and sidecar["type_id"] == 0
    assert math.isclose(sidecar["normalized_value"], sidecar["raw_value"] / 25.0, abs_tol=1e-12)
    assert manifest_row["numeric_values"] == [sidecar["normalized_value"]] * len(sidecar["token_indices"])
    assert manifest_row["numeric_type_ids"] == [0] * len(sidecar["token_indices"])
    assert manifest_row["numeric_parameter_names"] == ["gravity"] * len(sidecar["token_indices"])
    assert manifest_row["numeric_token_indices"] == sidecar["token_indices"]
    encoded = tokenizer(manifest_row["prompt"], add_special_tokens=True, return_offsets_mapping=True, truncation=False)
    assert len(encoded["input_ids"]) == sidecar["token_count_with_special_tokens"] <= 512
    start, end = sidecar["char_span"]
    rebound = [i for i, (left, right) in enumerate(encoded["offset_mapping"]) if right > start and left < end]
    assert rebound == sidecar["token_indices"]
    rebound_ids = [encoded["input_ids"][i] for i in rebound]
    assert rebound_ids == sidecar["token_ids"]
    assert tokenizer.convert_ids_to_tokens(rebound_ids) == sidecar["token_text"]
    video_path = Path(manifest_row["video"])
    with video_path.open("rb") as source:
        assert source.read(12)[4:8] == b"ftyp"
    actual_video_sha256 = sha256_file(video_path)
    assert actual_video_sha256 == manifest_row["numeric_video_sha256"] == sidecar["video_sha256"]
    gravity_values[f"{sidecar['raw_value']:g}"] += 1

assert len(ids) == len(set(ids))
assert not (set(ids) & consumed)
assert not (set(ids) & historical)
assert not (set(ids) & held)

exclusion_counts = Counter(row["reason"] for row in read_jsonl(exclusions_path))
for reason in [
    "training_split_excluded",
    "protection_excluded",
    "already_consumed_excluded",
    "historical_t15_excluded",
    "no_clear_physical_value",
    "medium_over_512",
    "token_binding_failed",
    "final_media_not_present",
    "video_unreadable",
]:
    assert exclusion_counts[reason] == receipt["counts"][reason]
assert sum(exclusion_counts.values()) + len(ids) == receipt["counts"]["scanned"]

for name, expected in receipt["outputs_sha256"].items():
    if name == receipt_path.name:
        continue
    assert sha256_file(OUT / name) == expected

validation = {
    "status": "PASS",
    "records": len(ids),
    "unique_ids": len(set(ids)),
    "duplicate_ids": len(ids) - len(set(ids)),
    "consumed_overlap": len(set(ids) & consumed),
    "historical_t15_overlap": len(set(ids) & historical),
    "held_protection_overlap": len(set(ids) & held),
    "all_real_umt5_bindings_recomputed": True,
    "all_medium_token_counts_lte_512": True,
    "all_video_sha256_recomputed": True,
    "all_video_mp4_signatures_valid": True,
    "gravity_values": dict(gravity_values),
    "manifest_sha256": sha256_file(manifest_path),
    "sidecar_sha256": sha256_file(sidecar_path),
    "exclusions_sha256": sha256_file(exclusions_path),
}
validation_path = OUT / "m2-medium-gravity-prompt-v2.validation.json"
validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(validation, sort_keys=True))
