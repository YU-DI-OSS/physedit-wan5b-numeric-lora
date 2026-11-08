import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

WAN = Path("/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora")
BASE = WAN / "runs/wan5b-h3final-v2-20261107T0630Z"
OUT = BASE / "m2/a800-gravity-medium-m1paths-v1-prep"
H3 = Path("/HOME/sysu_xdliang/sysu_xdliang_3/Storage01/h3_numeric_attention_20261106")
TOKENIZER_PATH = WAN / "models/Wan2.2-TI2V-5B/google/umt5-xxl"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


manifest_path = OUT / "m2-medium-gravity-m1paths-v1.jsonl"
sidecar_path = OUT / "m2-medium-gravity-m1paths-v1.sidecar.jsonl"
exclusions_path = OUT / "m2-medium-gravity-m1paths-v1.exclusions.jsonl"
receipt_path = OUT / "m2-medium-gravity-m1paths-v1.receipt.json"
receipt = json.loads(receipt_path.read_text())
manifest = list(read_jsonl(manifest_path))
sidecar = list(read_jsonl(sidecar_path))
assert len(manifest) == len(sidecar) == receipt["counts"]["strict_admitted"]

consumed = set()
for path in [
    BASE / "data/m2-t15-strict-v1.jsonl",
    BASE / "data/m2-gravity-next-v1.jsonl",
    BASE / "m2/a800-gpu1-migration-v1/m2-gravity-medium-tail-v1.jsonl",
    BASE / "m2/a800-gravity-medium-prompt-v2/m2-medium-gravity-prompt-v2.jsonl",
]:
    consumed.update(row["numeric_video_id"] for row in read_jsonl(path))
historical = set()
for path in [
    WAN / "runs/wan5b-wave01-m1-medium-v2-20261105T1625Z/metadata-main404.jsonl",
    WAN / "runs/wan5b-wave01-m1-recovery-tail2-20261106T0140Z/metadata-tail2.jsonl",
]:
    historical.update(Path(row["video"]).stem for row in read_jsonl(path))
with gzip.open(H3 / "final_trajectory_only/CORE100_AND_SAME_TRAJECTORY_IDS.json.gz", "rt") as source:
    held = set(json.load(source))

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True, use_fast=True)
ids = []
gravity_values = Counter()
sources = Counter()
for train, side in zip(manifest, sidecar):
    video_id = train["numeric_video_id"]
    assert side["video_id"] == video_id
    ids.append(video_id)
    assert video_id not in consumed and video_id not in historical and video_id not in held
    assert train["numeric_prompt_sha256"] == side["prompt_sha256"] == hashlib.sha256(train["prompt"].encode()).hexdigest()
    assert side["parameter"] == "gravity" and side["unit"] == "m/s^2" and side["type_id"] == 0
    assert math.isclose(side["normalized_value"], side["raw_value"] / 25.0, abs_tol=1e-12)
    encoded = tokenizer(train["prompt"], add_special_tokens=True, return_offsets_mapping=True, truncation=False)
    assert len(encoded["input_ids"]) == side["token_count_with_special_tokens"] <= 512
    start, end = side["char_span"]
    indices = [i for i, (left, right) in enumerate(encoded["offset_mapping"]) if right > start and left < end]
    assert indices == side["token_indices"] == train["numeric_token_indices"]
    assert [encoded["input_ids"][i] for i in indices] == side["token_ids"]
    video = Path(train["video"])
    with video.open("rb") as handle:
        assert handle.read(12)[4:8] == b"ftyp"
    assert sha256_file(video) == train["numeric_video_sha256"] == side["video_sha256"]
    gravity_values[f"{side['raw_value']:g}"] += 1
    sources[side["m1_index_source"]] += 1

id_set = set(ids)
assert len(ids) == len(id_set)
exclusions = Counter(row["reason"] for row in read_jsonl(exclusions_path))
for reason, count in exclusions.items():
    assert receipt["counts"][reason] == count
assert sum(exclusions.values()) + len(ids) == receipt["counts"]["scanned"]

validation = {
    "status": "PASS",
    "records": len(ids),
    "unique_ids": len(id_set),
    "duplicate_ids": len(ids) - len(id_set),
    "prior_consumed_overlap": len(id_set & consumed),
    "historical_t15_overlap": len(id_set & historical),
    "protection_overlap": len(id_set & held),
    "all_real_umt5_bindings_recomputed": True,
    "all_medium_tokens_lte_512": True,
    "all_actual_video_sha256_recomputed": True,
    "gravity_values": dict(gravity_values),
    "sources": dict(sources),
    "manifest_sha256": sha256_file(manifest_path),
    "sidecar_sha256": sha256_file(sidecar_path),
    "exclusions_sha256": sha256_file(exclusions_path),
}
validation_path = OUT / "m2-medium-gravity-m1paths-v1.validation.json"
validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
files = [manifest_path, sidecar_path, exclusions_path, receipt_path, validation_path, OUT / "build_m2_m1paths_v1.py", OUT / "validate_m2_m1paths_v1.py"]
with (OUT / "SHA256SUMS").open("w") as handle:
    for path in sorted(files):
        handle.write(f"{sha256_file(path)}  {path.name}\n")
print(json.dumps(validation, sort_keys=True))
