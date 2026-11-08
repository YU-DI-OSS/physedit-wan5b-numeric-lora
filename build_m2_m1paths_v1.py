"""Build the next M2 tranche from the bounded M1 verified-media index."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

WAN = Path("/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora")
H3 = Path("/HOME/sysu_xdliang/sysu_xdliang_3/Storage01/h3_numeric_attention_20261106")
FROZEN = H3 / "final_trajectory_only"
BASE = WAN / "runs/wan5b-h3final-v2-20261107T0630Z"
INDEX = BASE / "data/resolved-media-v1.jsonl"
OUT = BASE / "m2/a800-gravity-medium-m1paths-v1-prep"
TOKENIZER_PATH = WAN / "models/Wan2.2-TI2V-5B/google/umt5-xxl"
MANIFEST = FROZEN / "TRAIN_MEDIUM_DETAILED.jsonl.gz"
TRAIN_IDS = FROZEN / "TRAIN_IDS.json.gz"
HELD_IDS = FROZEN / "CORE100_AND_SAME_TRAJECTORY_IDS.json.gz"
EXPECTED_MANIFEST_SHA256 = "1ba18226f991fd1f9cd12c19c5a49fc9dfc2ad980712bb929e881d4f9abfeba6"
ALLOWED_GRAVITY = (1.62, 3.71, 9.81, 15.0, 24.0)
MAX_TOKENS = 512
GRAVITY_RE = re.compile(
    r"(?i)(?<![A-Za-z])(?P<label>(?:the\s+)?gravity(?:\s+strength)?)"
    r"\s+is\s+(?P<value>(?:\d+(?:\.\d*)?|\.\d+))"
    r"\s*(?P<unit>m\s*/\s*s\s*(?:\^\s*2|²))"
)
sys.path.insert(0, str(WAN / "runs/wan5b-wave01-fourgroup-prep-20261105T1615Z/scripts"))
from wave01_numeric_schema import bind_parameter  # noqa: E402


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


def write_jsonl(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    assert sha256_file(MANIFEST) == EXPECTED_MANIFEST_SHA256

    with gzip.open(TRAIN_IDS, "rt") as source:
        train_ids = set(json.load(source))
    with gzip.open(HELD_IDS, "rt") as source:
        held_ids = set(json.load(source))
    assert len(train_ids) == 57279 and len(held_ids) == 321 and not (train_ids & held_ids)

    media_index = {}
    index_sources = Counter()
    for row in read_jsonl(INDEX):
        video_id = row["video_id"]
        if video_id in media_index:
            raise RuntimeError(f"duplicate M1 index ID: {video_id}")
        media_index[video_id] = row
        index_sources[row["source"]] += 1
    assert len(media_index) == 36208

    consumed = set()
    consumed_paths = [
        BASE / "data/m2-t15-strict-v1.jsonl",
        BASE / "data/m2-gravity-next-v1.jsonl",
        BASE / "m2/a800-gpu1-migration-v1/m2-gravity-medium-tail-v1.jsonl",
        BASE / "m2/a800-gravity-medium-prompt-v2/m2-medium-gravity-prompt-v2.jsonl",
    ]
    consumed_source_counts = {}
    for path in consumed_paths:
        before = len(consumed)
        for row in read_jsonl(path):
            video_id = row["numeric_video_id"]
            if video_id in consumed:
                raise RuntimeError(f"duplicate consumed ID: {video_id}")
            consumed.add(video_id)
        consumed_source_counts[str(path)] = len(consumed) - before
    assert len(consumed) == 10219

    historical = set()
    for path in [
        WAN / "runs/wan5b-wave01-m1-medium-v2-20261105T1625Z/metadata-main404.jsonl",
        WAN / "runs/wan5b-wave01-m1-recovery-tail2-20261106T0140Z/metadata-tail2.jsonl",
    ]:
        historical.update(Path(row["video"]).stem for row in read_jsonl(path))
    assert len(historical) == 406 and not (historical & consumed)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True, use_fast=True)
    assert tokenizer.is_fast

    manifest_path = OUT / "m2-medium-gravity-m1paths-v1.jsonl"
    sidecar_path = OUT / "m2-medium-gravity-m1paths-v1.sidecar.jsonl"
    exclusions_path = OUT / "m2-medium-gravity-m1paths-v1.exclusions.jsonl"
    counts = Counter({
        "scanned": 0,
        "not_in_final_train": 0,
        "protection_excluded": 0,
        "not_in_m1_verified_index": 0,
        "m1_index_id_sha_mismatch": 0,
        "already_consumed_excluded": 0,
        "historical_t15_excluded": 0,
        "no_clear_physical_value": 0,
        "medium_over_512": 0,
        "token_binding_failed": 0,
        "m1_path_missing": 0,
        "video_unreadable_or_sha_mismatch": 0,
        "strict_admitted": 0,
    })
    scenes = Counter()
    gravity_values = Counter()
    admitted_sources = Counter()
    admitted_ids = set()
    manifest_ids = set()
    examples = []

    def exclude(handle, video_id, reason, **details):
        counts[reason] += 1
        write_jsonl(handle, {"video_id": video_id, "reason": reason, **details})

    with (
        gzip.open(MANIFEST, "rt", encoding="utf-8") as source,
        manifest_path.with_suffix(".jsonl.partial").open("w", encoding="utf-8") as train_out,
        sidecar_path.with_suffix(".jsonl.partial").open("w", encoding="utf-8") as sidecar_out,
        exclusions_path.with_suffix(".jsonl.partial").open("w", encoding="utf-8") as exclusions_out,
    ):
        for manifest_index, line in enumerate(source):
            counts["scanned"] += 1
            row = json.loads(line)
            video_id = row["video_id"]
            if video_id in manifest_ids:
                raise RuntimeError(f"duplicate final manifest ID: {video_id}")
            manifest_ids.add(video_id)
            if video_id not in train_ids:
                exclude(exclusions_out, video_id, "not_in_final_train")
                continue
            if video_id in held_ids or row.get("needs_review") is not False:
                exclude(exclusions_out, video_id, "protection_excluded")
                continue
            indexed = media_index.get(video_id)
            if indexed is None:
                exclude(exclusions_out, video_id, "not_in_m1_verified_index")
                continue
            if indexed["source_video_sha256"] != row["source_video_sha256"]:
                exclude(exclusions_out, video_id, "m1_index_id_sha_mismatch")
                continue
            counts["m1_index_exact_id_sha_match"] += 1
            if video_id in consumed:
                exclude(exclusions_out, video_id, "already_consumed_excluded")
                continue
            if video_id in historical:
                exclude(exclusions_out, video_id, "historical_t15_excluded")
                continue

            prompt = row["medium"]
            prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
            if prompt_sha256 != row["medium_sha256"]:
                exclude(exclusions_out, video_id, "token_binding_failed", detail="medium_sha256_mismatch")
                continue
            matches = list(GRAVITY_RE.finditer(prompt))
            if len(matches) != 1:
                exclude(exclusions_out, video_id, "no_clear_physical_value", gravity_phrase_matches=len(matches))
                continue
            match = matches[0]
            value = float(match.group("value"))
            if not math.isfinite(value) or value <= 0 or not any(math.isclose(value, allowed, abs_tol=1e-9) for allowed in ALLOWED_GRAVITY):
                exclude(exclusions_out, video_id, "no_clear_physical_value", parsed_value=value)
                continue
            try:
                bound = bind_parameter(prompt, "gravity", value, "m/s^2")
            except ValueError as exc:
                exclude(exclusions_out, video_id, "token_binding_failed", detail=str(exc))
                continue
            if bound is None or (bound.char_start, bound.char_end) != (match.start("value"), match.end("value")):
                exclude(exclusions_out, video_id, "token_binding_failed", detail="semantic_span_disagreement")
                continue
            encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True, truncation=False)
            token_count = len(encoded["input_ids"])
            if token_count > MAX_TOKENS:
                exclude(exclusions_out, video_id, "medium_over_512", token_count=token_count)
                continue
            indices = [i for i, (start, end) in enumerate(encoded["offset_mapping"]) if end > bound.char_start and start < bound.char_end]
            if not indices or max(indices) >= MAX_TOKENS or any(encoded["offset_mapping"][i][0] == encoded["offset_mapping"][i][1] for i in indices):
                exclude(exclusions_out, video_id, "token_binding_failed", detail="invalid_token_overlap")
                continue

            video_path = Path(indexed["video"])
            if not video_path.is_file():
                exclude(exclusions_out, video_id, "m1_path_missing", video=str(video_path), source=indexed["source"])
                continue
            try:
                size = video_path.stat().st_size
                with video_path.open("rb") as handle:
                    header = handle.read(12)
                actual_sha256 = sha256_file(video_path)
            except OSError as exc:
                exclude(exclusions_out, video_id, "video_unreadable_or_sha_mismatch", detail=repr(exc))
                continue
            if size < 12 or header[4:8] != b"ftyp" or actual_sha256 != row["source_video_sha256"]:
                exclude(exclusions_out, video_id, "video_unreadable_or_sha_mismatch", actual_sha256=actual_sha256, expected_sha256=row["source_video_sha256"], video=str(video_path))
                continue

            if video_id in admitted_ids:
                raise RuntimeError(f"duplicate admitted ID: {video_id}")
            admitted_ids.add(video_id)
            counts["strict_admitted"] += 1
            scenes[row["scene"]] += 1
            gravity_values[f"{value:g}"] += 1
            admitted_sources[indexed["source"]] += 1
            token_ids = [encoded["input_ids"][i] for i in indices]
            train_row = {
                "prompt": prompt,
                "video": str(video_path),
                "numeric_values": [bound.normalized_value] * len(indices),
                "numeric_type_ids": [0] * len(indices),
                "numeric_token_indices": indices,
                "numeric_parameter_names": ["gravity"] * len(indices),
                "numeric_prompt_sha256": prompt_sha256,
                "numeric_video_id": video_id,
                "numeric_video_sha256": actual_sha256,
            }
            sidecar_row = {
                "video_id": video_id,
                "scene": row["scene"],
                "source_manifest_index": manifest_index,
                "m1_index_source": indexed["source"],
                "video_path": str(video_path),
                "video_sha256": actual_sha256,
                "prompt_sha256": prompt_sha256,
                "parameter": "gravity",
                "raw_value": value,
                "unit": "m/s^2",
                "normalized_value": bound.normalized_value,
                "type_id": 0,
                "char_span": [bound.char_start, bound.char_end],
                "token_count_with_special_tokens": token_count,
                "token_indices": indices,
                "token_ids": token_ids,
                "token_text": tokenizer.convert_ids_to_tokens(token_ids),
                "token_offsets": [encoded["offset_mapping"][i] for i in indices],
            }
            write_jsonl(train_out, train_row)
            write_jsonl(sidecar_out, sidecar_row)
            if len(examples) < 10:
                examples.append({"video_id": video_id, "video": str(video_path), "source": indexed["source"], "gravity": value, "tokens": indices})

    assert manifest_ids == train_ids
    assert not (admitted_ids & consumed) and not (admitted_ids & historical) and not (admitted_ids & held_ids)
    classified = sum(counts[key] for key in [
        "not_in_final_train", "protection_excluded", "not_in_m1_verified_index", "m1_index_id_sha_mismatch",
        "already_consumed_excluded", "historical_t15_excluded", "no_clear_physical_value", "medium_over_512",
        "token_binding_failed", "m1_path_missing", "video_unreadable_or_sha_mismatch", "strict_admitted",
    ])
    assert classified == counts["scanned"], (classified, counts["scanned"])

    manifest_partial = manifest_path.with_suffix(".jsonl.partial")
    sidecar_partial = sidecar_path.with_suffix(".jsonl.partial")
    exclusions_partial = exclusions_path.with_suffix(".jsonl.partial")
    manifest_partial.replace(manifest_path)
    sidecar_partial.replace(sidecar_path)
    exclusions_partial.replace(exclusions_path)
    outputs = {path.name: sha256_file(path) for path in [manifest_path, sidecar_path, exclusions_path]}
    receipt = {
        "status": "STRICT_ADMITTED_M2_FROM_M1_VERIFIED_PATHS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "m1_index": str(INDEX),
        "m1_index_records": len(media_index),
        "m1_index_sources": dict(index_sources),
        "counts": dict(counts),
        "consumed_unique": len(consumed),
        "consumed_source_counts": consumed_source_counts,
        "historical_t15_unique": len(historical),
        "duplicates": counts["strict_admitted"] - len(admitted_ids),
        "protection_overlap": len(admitted_ids & held_ids),
        "scenes_admitted": dict(scenes),
        "gravity_values_admitted": dict(gravity_values),
        "admitted_sources": dict(admitted_sources),
        "rules": {
            "config_sha256": "not used",
            "media": "bounded M1 resolved-media-v1 path; stored and recomputed SHA256 must equal H3 final manifest",
            "numeric": "unique explicit gravity, m/s^2, value/25, type0",
            "tokenizer": "real Wan2.2 UMT5 fast tokenizer offsets, Medium <=512",
        },
        "outputs_sha256": outputs,
        "examples": examples,
    }
    receipt_path = OUT / "m2-medium-gravity-m1paths-v1.receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    with (OUT / "SHA256SUMS").open("w") as handle:
        for path in sorted([manifest_path, sidecar_path, exclusions_path, receipt_path]):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
