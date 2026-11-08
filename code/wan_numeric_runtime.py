"""Per-process, per-sample Wave01 numeric conditioning state.

The run-local train entrypoint sets this immediately before pipeline execution.
The text projection consumes it once or during checkpoint recomputation for the
same sample. Prompt SHA and lengths are checked fail-closed.
"""
from __future__ import annotations

import hashlib
import threading


_state = threading.local()


def set_current_from_data(data) -> None:
    prompt = data["prompt"]
    if isinstance(prompt, (list, tuple)):
        if len(prompt) != 1:
            raise RuntimeError("numeric runtime requires per-device batch size one")
        prompt = prompt[0]
    expected = data["numeric_prompt_sha256"]
    if isinstance(expected, (list, tuple)):
        expected = expected[0]
    actual = hashlib.sha256(prompt.encode()).hexdigest()
    if actual != expected:
        raise RuntimeError("numeric prompt identity mismatch")
    values = list(data["numeric_values"])
    type_ids = list(data["numeric_type_ids"])
    token_indices = list(data["numeric_token_indices"])
    names = list(data["numeric_parameter_names"])
    if not values or not (len(values) == len(type_ids) == len(token_indices) == len(names)):
        raise RuntimeError("invalid numeric sidecar lengths")
    if any(str(name).endswith("initial_speed") for name in names):
        raise RuntimeError("initial_speed entered numeric supervision")
    _state.current = {
        "values": [float(x) for x in values],
        "type_ids": [int(x) for x in type_ids],
        "token_indices": [int(x) for x in token_indices],
        "names": names,
        "prompt_sha256": actual,
    }


def get_current():
    current = getattr(_state, "current", None)
    if current is None:
        raise RuntimeError("numeric sample sidecar was not set before model forward")
    return current
