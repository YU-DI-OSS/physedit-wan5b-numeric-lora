"""Fail-closed Wave01 M2 numeric schema (CPU-side preparation only).

This module does not patch the trainer or write a training manifest.  It turns
only uniquely bound, explicitly allow-listed physical facts into normalized
Harmonic inputs.  The unreliable initial-speed fact is always excluded and is
never replaced by zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re


SCHEMA_VERSION = "wave01_t15_m2_numeric_v1"
HARMONICS = 8


@dataclass(frozen=True)
class ParameterSpec:
    type_id: int
    unit: str
    scale: float


# Fixed, simple unit scales.  Wave01 observed ranges are checked by the test.
PARAMETERS = {
    "gravity": ParameterSpec(0, "m/s^2", 25.0),
    "t15v4_driver_book.mass": ParameterSpec(1, "kg", 2.0),
    "t15v4_driver_book.friction": ParameterSpec(2, "", 1.0),
    "t15v4_closed_laptop.mass": ParameterSpec(3, "kg", 2.0),
    "t15v4_closed_laptop.friction": ParameterSpec(4, "", 1.0),
    "t15v4_floor_phone.mass": ParameterSpec(5, "kg", 2.0),
    "t15v4_floor_phone.friction": ParameterSpec(6, "", 1.0),
    "t15v4_tabletop_collision.friction": ParameterSpec(7, "", 1.0),
}
EXCLUDED_PARAMETERS = frozenset({"t15v4_driver_book.initial_speed"})

NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class BoundNumericFact:
    name: str
    raw_value: float
    unit: str
    normalized_value: float
    char_start: int
    char_end: int
    type_id: int


def _unit_matches(text_after_number: str, unit: str) -> bool:
    if not unit:
        return True
    canonical = text_after_number.lower().replace("²", "^2").replace("2", "^2")
    wanted = unit.lower().replace("²", "^2").replace("2", "^2")
    return wanted in canonical[: max(20, len(wanted) + 8)]


def bind_parameter(prompt: str, name: str, value: float, unit: str) -> BoundNumericFact | None:
    """Return one validated binding, skip excluded keys, fail on all ambiguity."""
    if name in EXCLUDED_PARAMETERS:
        return None
    if name not in PARAMETERS:
        raise ValueError(f"parameter is not allow-listed: {name}")
    spec = PARAMETERS[name]
    if unit != spec.unit:
        raise ValueError(f"unit mismatch for {name}: {unit!r} != {spec.unit!r}")
    matches = []
    for match in NUMBER_RE.finditer(prompt):
        parsed = float(match.group(0))
        if not math.isclose(parsed, float(value), rel_tol=1e-9, abs_tol=1e-9):
            continue
        if _unit_matches(prompt[match.end() :], unit):
            matches.append(match)
    if len(matches) != 1:
        raise ValueError(f"{name} expected exactly one semantic numeric span, got {len(matches)}")
    normalized = float(value) / spec.scale
    if not math.isfinite(normalized):
        raise ValueError(f"non-finite normalized value for {name}")
    match = matches[0]
    return BoundNumericFact(
        name=name,
        raw_value=float(value),
        unit=unit,
        normalized_value=normalized,
        char_start=match.start(),
        char_end=match.end(),
        type_id=spec.type_id,
    )


def harmonic_features(value: float, harmonics: int = HARMONICS) -> tuple[float, ...]:
    if not math.isfinite(value):
        raise ValueError("harmonic input must be finite")
    out = [value]
    for k in range(harmonics):
        phase = (2**k) * math.pi * value
        out.extend((math.sin(phase), math.cos(phase)))
    return tuple(out)

