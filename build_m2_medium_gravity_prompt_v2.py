"""Build a fail-closed Medium-prompt gravity tranche for M2.

This builder intentionally ignores the obsolete manifest ``split`` field.
Admission is based on the frozen TRAIN_IDS/protection set, exact prompt/video
hashes, one explicit gravity phrase, and real UMT5 offset bindings.
"""
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
OUT = BASE / "m2/a800-gravity-medium-prompt-v2"
TOKENIZER_PATH = WAN / "models/Wan2.2-TI2V-5B/google/umt5-xxl"
MANIFEST = FROZEN / "TRAIN_MEDIUM_DETAILED.jsonl.gz"
TRAIN_IDS = FROZEN / "TRAIN_IDS.json.gz"
HELD_IDS = FROZEN / "CORE100_AND_SAME_TRAJECTORY_IDS.json.gz"

EXPECTED_MANIFEST_SHA256 = "1ba18226f991fd1f9cd12c19c5a49fc9dfc2ad980712bb929e881d4f9abfeba6"
EXPECTED_TRAIN_IDS_SHA256 = "030adeff322d9cbee2096c8b1f0acc4679e3537a69597b6302e6be12396d6917"
EXPECTED_HELD_IDS_SHA256 = "9b0a0bd95315a965d53a9901f5a3a25ff247c969f369451549df5d94cf356154"
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


def load_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def write_jsonl(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def file_set_digest(names: set[str]) -> str:
    payload = "".join(name + "\n" for name in sorted(names)).encode()
    return hashlib.sha256(payload).hexdigest()


def tokenizer_fingerprint() -> dict[str, str]:
    result = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model"):
        path = TOKENIZER_PATH / name
        if path.is_file():
            result[name] = sha256_file(path)
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite stage: {OUT}")
    OUT.mkdir(parents=True)

    assert sha256_file(MANIFEST) == EXPECTED_MANIFEST_SHA256
    assert sha256_file(TRAIN_IDS) == EXPECTED_TRAIN_IDS_SHA256
    assert sha256_file(HELD_IDS) == EXPECTED_HELD_IDS_SHA256

    train_ids_list = load_gzip_json(TRAIN_IDS)
    held_ids_list = load_gzip_json(HELD_IDS)
    train_ids = set(train_ids_list)
    held_ids = set(held_ids_list)
    assert len(train_ids_list) == len(train_ids) == 57279
    assert len(held_ids_list) == len(held_ids) == 321
    assert not (train_ids & held_ids)

    consumed: set[str] = set()
    consumed_sources = [
        BASE / "data/m2-t15-strict-v1.jsonl",
        BASE / "data/m2-gravity-next-v1.jsonl",
        BASE / "m2/a800-gpu1-migration-v1/m2-gravity-medium-tail-v1.jsonl",
    ]
    consumed_source_counts = {}
    for path in consumed_sources:
        before = len(consumed)
        with path.open(encoding="utf-8") as source:
            for line in source:
                video_id = json.loads(line)["numeric_video_id"]
                if video_id in consumed:
                    raise RuntimeError(f"duplicate consumed ID: {video_id}")
                consumed.add(video_id)
        consumed_source_counts[str(path)] = len(consumed) - before
    assert len(consumed) == 7678, len(consumed)

    historical: set[str] = set()
    historical_sources = [
        WAN / "runs/wan5b-wave01-m1-medium-v2-20261105T1625Z/metadata-main404.jsonl",
        WAN / "runs/wan5b-wave01-m1-recovery-tail2-20261106T0140Z/metadata-tail2.jsonl",
    ]
    for path in historical_sources:
        with path.open(encoding="utf-8") as source:
            historical.update(Path(json.loads(line)["video"]).stem for line in source)
    assert len(historical) == 406, len(historical)
    assert not (consumed & historical)

    # Freeze the available final-media snapshot before scanning. New files that
    # arrive concurrently are intentionally deferred to the next tranche.
    media_dir = H3 / "media"
    available_media = {path.stem for path in media_dir.glob("*.mp4") if path.is_file()}
    available_snapshot_count = len(available_media)
    available_snapshot_sha256 = file_set_digest(available_media)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, local_files_only=True, use_fast=True
    )
    if not tokenizer.is_fast:
        raise RuntimeError("UMT5 fast tokenizer with offset mapping is required")

    manifest_path = OUT / "m2-medium-gravity-prompt-v2.jsonl"
    sidecar_path = OUT / "m2-medium-gravity-prompt-v2.sidecar.jsonl"
    exclusions_path = OUT / "m2-medium-gravity-prompt-v2.exclusions.jsonl"
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    sidecar_tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".partial")
    exclusions_tmp = exclusions_path.with_suffix(exclusions_path.suffix + ".partial")

    counts = Counter(
        {
            "scanned": 0,
            "training_split_excluded": 0,
            "protection_excluded": 0,
            "already_consumed_excluded": 0,
            "historical_t15_excluded": 0,
            "no_clear_physical_value": 0,
            "medium_over_512": 0,
            "token_binding_failed": 0,
            "final_media_not_present": 0,
            "video_unreadable": 0,
            "strict_admitted": 0,
        }
    )
    scenes = Counter()
    gravity_values = Counter()
    legacy_splits_scanned = Counter()
    legacy_splits_admitted = Counter()
    token_lengths_admitted = Counter()
    manifest_ids: set[str] = set()
    admitted_ids: set[str] = set()
    examples = []

    def exclude(handle, video_id: str, reason: str, **details) -> None:
        counts[reason] += 1
        write_jsonl(handle, {"video_id": video_id, "reason": reason, **details})

    with (
        gzip.open(MANIFEST, "rt", encoding="utf-8") as source,
        manifest_tmp.open("w", encoding="utf-8") as manifest_out,
        sidecar_tmp.open("w", encoding="utf-8") as sidecar_out,
        exclusions_tmp.open("w", encoding="utf-8") as exclusions_out,
    ):
        for manifest_index, line in enumerate(source):
            counts["scanned"] += 1
            row = json.loads(line)
            video_id = row["video_id"]
            if video_id in manifest_ids:
                raise RuntimeError(f"duplicate manifest ID: {video_id}")
            manifest_ids.add(video_id)
            legacy_splits_scanned[str(row.get("split"))] += 1

            if video_id not in train_ids:
                exclude(exclusions_out, video_id, "training_split_excluded")
                continue
            if video_id in held_ids or row.get("needs_review") is not False:
                exclude(exclusions_out, video_id, "protection_excluded")
                continue
            if video_id in consumed:
                exclude(exclusions_out, video_id, "already_consumed_excluded")
                continue
            if video_id in historical:
                exclude(exclusions_out, video_id, "historical_t15_excluded")
                continue

            prompt = row["medium"]
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if prompt_sha256 != row["medium_sha256"]:
                exclude(exclusions_out, video_id, "token_binding_failed", detail="medium_sha256_mismatch")
                continue

            gravity_matches = list(GRAVITY_RE.finditer(prompt))
            if len(gravity_matches) != 1:
                exclude(
                    exclusions_out,
                    video_id,
                    "no_clear_physical_value",
                    gravity_phrase_matches=len(gravity_matches),
                )
                continue
            match = gravity_matches[0]
            value = float(match.group("value"))
            if not math.isfinite(value) or value <= 0 or not any(
                math.isclose(value, allowed, rel_tol=0.0, abs_tol=1e-9)
                for allowed in ALLOWED_GRAVITY
            ):
                exclude(
                    exclusions_out,
                    video_id,
                    "no_clear_physical_value",
                    detail="gravity_outside_validated_regimes",
                    parsed_value=value,
                )
                continue

            try:
                bound = bind_parameter(prompt, "gravity", value, "m/s^2")
            except ValueError as exc:
                exclude(exclusions_out, video_id, "token_binding_failed", detail=str(exc))
                continue
            if bound is None or bound.char_start != match.start("value") or bound.char_end != match.end("value"):
                exclude(
                    exclusions_out,
                    video_id,
                    "token_binding_failed",
                    detail="semantic_regex_and_validated_binding_span_disagree",
                )
                continue
            if bound.type_id != 0 or not math.isclose(bound.normalized_value, value / 25.0, abs_tol=1e-12):
                raise RuntimeError("validated gravity schema changed unexpectedly")

            encoded = tokenizer(
                prompt,
                add_special_tokens=True,
                return_offsets_mapping=True,
                truncation=False,
            )
            token_count = len(encoded["input_ids"])
            if token_count > MAX_TOKENS:
                exclude(exclusions_out, video_id, "medium_over_512", token_count=token_count)
                continue
            token_indices = [
                index
                for index, (start, end) in enumerate(encoded["offset_mapping"])
                if end > bound.char_start and start < bound.char_end
            ]
            if not token_indices or max(token_indices) >= MAX_TOKENS:
                exclude(exclusions_out, video_id, "token_binding_failed", detail="no_in_range_token_overlap")
                continue
            token_offsets = [encoded["offset_mapping"][index] for index in token_indices]
            if any(start == end for start, end in token_offsets):
                exclude(exclusions_out, video_id, "token_binding_failed", detail="special_token_overlap")
                continue

            if video_id not in available_media:
                exclude(exclusions_out, video_id, "final_media_not_present")
                continue
            video_path = media_dir / f"{video_id}.mp4"
            try:
                size = video_path.stat().st_size
                with video_path.open("rb") as video_source:
                    header = video_source.read(12)
                actual_video_sha256 = sha256_file(video_path)
            except OSError as exc:
                exclude(exclusions_out, video_id, "video_unreadable", detail=repr(exc))
                continue
            if size < 12 or header[4:8] != b"ftyp" or actual_video_sha256 != row["source_video_sha256"]:
                exclude(
                    exclusions_out,
                    video_id,
                    "video_unreadable",
                    detail="invalid_mp4_signature_or_source_sha256_mismatch",
                    bytes=size,
                    actual_sha256=actual_video_sha256,
                    expected_sha256=row["source_video_sha256"],
                )
                continue

            if video_id in admitted_ids:
                raise RuntimeError(f"duplicate admitted ID: {video_id}")
            admitted_ids.add(video_id)
            counts["strict_admitted"] += 1
            scenes[row["scene"]] += 1
            gravity_values[f"{value:g}"] += 1
            legacy_splits_admitted[str(row.get("split"))] += 1
            token_lengths_admitted[str(token_count)] += 1

            token_ids = [encoded["input_ids"][index] for index in token_indices]
            token_text = tokenizer.convert_ids_to_tokens(token_ids)
            training_row = {
                "prompt": prompt,
                "video": str(video_path),
                "numeric_values": [bound.normalized_value] * len(token_indices),
                "numeric_type_ids": [bound.type_id] * len(token_indices),
                "numeric_token_indices": token_indices,
                "numeric_parameter_names": ["gravity"] * len(token_indices),
                "numeric_prompt_sha256": prompt_sha256,
                "numeric_video_id": video_id,
                "numeric_video_sha256": actual_video_sha256,
            }
            sidecar_row = {
                "video_id": video_id,
                "scene": row["scene"],
                "source_manifest_index": manifest_index,
                "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "prompt_tier": "medium",
                "prompt_sha256": prompt_sha256,
                "parameter": "gravity",
                "semantic_label": match.group("label"),
                "direction_convention": "positive downward gravity magnitude, matching prior vector-norm rule",
                "raw_text": match.group("value"),
                "raw_value": value,
                "unit": "m/s^2",
                "normalization": "value / 25",
                "normalized_value": bound.normalized_value,
                "type_id": bound.type_id,
                "char_span": [bound.char_start, bound.char_end],
                "token_count_with_special_tokens": token_count,
                "token_indices": token_indices,
                "token_ids": token_ids,
                "token_text": token_text,
                "token_offsets": token_offsets,
                "video_path": str(video_path),
                "video_bytes": size,
                "video_sha256": actual_video_sha256,
                "legacy_split_ignored_for_admission": row.get("split"),
            }
            write_jsonl(manifest_out, training_row)
            write_jsonl(sidecar_out, sidecar_row)
            if len(examples) < 12:
                examples.append(
                    {
                        "video_id": video_id,
                        "gravity": value,
                        "char_span": [bound.char_start, bound.char_end],
                        "token_indices": token_indices,
                        "token_text": token_text,
                        "token_count": token_count,
                    }
                )

    assert manifest_ids == train_ids, (
        len(manifest_ids - train_ids),
        len(train_ids - manifest_ids),
    )
    assert not (admitted_ids & consumed)
    assert not (admitted_ids & historical)
    assert not (admitted_ids & held_ids)
    assert counts["strict_admitted"] == len(admitted_ids)
    classified = sum(
        counts[key]
        for key in (
            "training_split_excluded",
            "protection_excluded",
            "already_consumed_excluded",
            "historical_t15_excluded",
            "no_clear_physical_value",
            "medium_over_512",
            "token_binding_failed",
            "final_media_not_present",
            "video_unreadable",
            "strict_admitted",
        )
    )
    assert classified == counts["scanned"], (classified, counts["scanned"])

    manifest_tmp.replace(manifest_path)
    sidecar_tmp.replace(sidecar_path)
    exclusions_tmp.replace(exclusions_path)

    output_sha256 = {
        manifest_path.name: sha256_file(manifest_path),
        sidecar_path.name: sha256_file(sidecar_path),
        exclusions_path.name: sha256_file(exclusions_path),
    }
    tokenizer_info = {
        "path": str(TOKENIZER_PATH),
        "class": tokenizer.__class__.__name__,
        "is_fast": tokenizer.is_fast,
        "vocab_size": tokenizer.vocab_size,
        "files_sha256": tokenizer_fingerprint(),
    }
    receipt = {
        "status": "STRICT_ADMITTED_M2_MEDIUM_GRAVITY_PROMPT_V2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "manifest": str(MANIFEST),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "train_ids": str(TRAIN_IDS),
            "train_ids_sha256": EXPECTED_TRAIN_IDS_SHA256,
            "held_ids": str(HELD_IDS),
            "held_ids_sha256": EXPECTED_HELD_IDS_SHA256,
        },
        "rules": {
            "prompt": "medium only",
            "gravity_patterns": ["Gravity is <value> m/s^2", "The gravity strength is <value> m/s^2"],
            "allowed_gravity_magnitudes": ALLOWED_GRAVITY,
            "direction_convention": "positive downward magnitude; identical to the prior gravity-vector norm",
            "normalization": "value / 25",
            "type_id": 0,
            "token_limit": MAX_TOKENS,
            "legacy_split_field": "recorded but ignored",
            "config_sha256": "not used for admission",
            "media_policy": "only frozen H3 media/<video_id>.mp4 snapshot; exact source_video_sha256",
        },
        "counts": dict(counts),
        "consumed_source_counts": consumed_source_counts,
        "consumed_unique": len(consumed),
        "historical_t15_unique": len(historical),
        "train_ids": len(train_ids),
        "held_ids": len(held_ids),
        "train_held_overlap": len(train_ids & held_ids),
        "admitted_duplicate_ids": counts["strict_admitted"] - len(admitted_ids),
        "available_media_snapshot": {
            "count": available_snapshot_count,
            "ids_sha256": available_snapshot_sha256,
        },
        "legacy_splits_scanned": dict(legacy_splits_scanned),
        "legacy_splits_admitted": dict(legacy_splits_admitted),
        "scenes_admitted": dict(scenes),
        "gravity_values_admitted": dict(gravity_values),
        "tokenizer": tokenizer_info,
        "outputs_sha256": output_sha256,
        "examples": examples,
    }
    receipt_path = OUT / "m2-medium-gravity-prompt-v2.receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_sha256[receipt_path.name] = sha256_file(receipt_path)

    checksum_path = OUT / "SHA256SUMS"
    with checksum_path.open("w", encoding="utf-8") as handle:
        for name, digest in sorted(output_sha256.items()):
            handle.write(f"{digest}  {name}\n")

    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
