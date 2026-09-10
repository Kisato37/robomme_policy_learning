"""Read-only replay validation for the new 48-slot selector trace contract.

This validates trace structure, deterministic selection and digest consistency;
it does not recover feature values from hashes or certify the real checkpoint.
It neither accepts old-family traces nor changes any old-family validator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np

from mme_vla_suite.shared.uniform_keyframe_config import payload_digest
from mme_vla_suite.shared.uniform_keyframe_expansion import (
    MAX_POLICY_CALLS, SEED_FAMILY, ExpansionInvariantError,
    derive_expansion_seed, select_expansion_indices,
)


class ExpansionTraceError(ValueError):
    """Missing, malformed or inconsistent evidence; never an episode fail."""


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExpansionTraceError("Trace must contain finite JSON-serializable values") from exc


def _equal(actual: Any, expected: Any, name: str) -> None:
    if _json(actual) != _json(expected):
        raise ExpansionTraceError(f"Trace {name} differs from the frozen/replayed value")


def _get(trace: Mapping[str, Any], field: str) -> Any:
    if field not in trace:
        raise ExpansionTraceError(f"Missing trace field: {field}")
    return trace[field]


def _integer(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ExpansionTraceError(f"Trace {name} must be an integer in [{minimum}, {maximum}]")
    return value


def _digest(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ExpansionTraceError(f"Trace {name} must be a lowercase SHA-256 digest")
    return value


def _array_digest(array: np.ndarray) -> str:
    """Identical to MME_VLA_Policy._array_digest, without model dependencies."""
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _latency(value: Any, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ExpansionTraceError(f"Trace {name} must be a finite nonnegative duration")


def _floating_dtype(value: Any, name: str) -> None:
    # bfloat16 may be emitted by JAX although NumPy-only environments cannot
    # resolve that dtype without ml_dtypes. No model import is needed here.
    if value not in ("bfloat16", "float16", "float32", "float64"):
        raise ExpansionTraceError(f"Trace {name} must name a supported floating dtype")


def validate_selector_trace(
    trace: Mapping[str, Any], *, require_final_memory: bool = True,
) -> dict[str, Any]:
    """Validate a complete inference trace (or an explicitly pre-final fixture).

    Only the true-valid-prefix/false-padding mask and its digest are replayed;
    they cannot prove zero feature values. The real runtime checks zero padding
    before normalization. Tensor hashes are bound consistently here, but do not
    substitute for archived tensor/feature data.
    """
    if not isinstance(trace, Mapping) or not trace:
        raise ExpansionTraceError("Missing selector trace: expected a nonempty mapping")
    _json(dict(trace))
    if type(require_final_memory) is not bool:
        raise ExpansionTraceError("require_final_memory must be boolean")
    _equal(_get(trace, "schema_version"), 1, "schema_version")
    _equal(_get(trace, "experiment_family"), SEED_FAMILY, "experiment_family")
    step = _integer(_get(trace, "step_idx"), "step_idx")
    boundaries = _get(trace, "visible_boundary_indices")
    if not isinstance(boundaries, list):
        raise ExpansionTraceError("visible_boundary_indices must be a JSON list")
    normalized = [_integer(index, "boundary_index", maximum=step) for index in boundaries]
    if normalized != sorted(set(normalized)):
        raise ExpansionTraceError("Boundary indices must be unique and chronological")
    boundary_set = set(boundaries)
    # Reconstruct only the already visible labels. This is a checker, not the
    # runtime U sampler or a generator of new/future boundary observations.
    flags = [index in boundary_set for index in range(step + 1)]
    try:
        selected, decision = select_expansion_indices(
            _get(trace, "arm"), step_idx=step,
            base_uniform_indices=_get(trace, "base_uniform_indices"), boundary_flags=flags,
            split=_get(trace, "split"), task=_get(trace, "task"),
            episode_id=_get(trace, "episode_id"),
            policy_call_index=_get(trace, "policy_call_index"),
        )
    except (ExpansionInvariantError, TypeError, KeyError) as exc:
        raise ExpansionTraceError(f"Selector trace cannot replay under the frozen rules: {exc}") from exc
    # Context normalization in the helper is not permission for loose JSON IDs.
    for key, expected in decision.items():
        _equal(_get(trace, key), expected, key)
    if (decision["split"] == "val" and decision["episode_id"] != 0
            or decision["split"] == "test" and not 0 <= decision["episode_id"] < 50):
        raise ExpansionTraceError("Trace is outside the frozen split/episode population")

    seed_table = [derive_expansion_seed(decision["split"], decision["task"],
                                       decision["episode_id"], index)
                  for index in range(MAX_POLICY_CALLS)]
    expected_values = {
        "selector_name": decision["arm"], "seed_table_scope": SEED_FAMILY,
        "seed_table_dataset": decision["split"], "seed_table_sha256": payload_digest(seed_table),
        "history_length": step + 1, "current_history_index": step,
        "selected_frame_indices": selected, "selected_indices_sha256": payload_digest(selected),
        "valid_memory_token_count": len(selected) * 16,
        "mask_shape": [768], "mask_dtype": "bool", "mask_valid_prefix_all_true": True,
        "mask_padding_all_false": True, "effective_memory_budget": 768,
        "base_uniform_token_budget": 512,
        "image_tensor_shape": [768, 2048], "position_tensor_shape": [768, 768],
        "state_tensor_shape": [768, 8],
        "prepared_memory_component_shapes": [[768, 2048], [768, 768], [768, 8], [768]],
        "age_distribution": [step - index for index in selected],
        "maximum_temporal_gap": max((right - left for left, right in zip(selected, selected[1:])), default=0),
        "boundary_recall": decision["selected_boundary_count"] / decision["visible_boundary_count"],
    }
    for key, expected in expected_values.items():
        _equal(_get(trace, key), expected, key)
    expected_mask = np.zeros(768, dtype=np.bool_)
    expected_mask[:len(selected) * 16] = True
    _equal(_get(trace, "mask_sha256"), _array_digest(expected_mask), "mask_sha256")

    component_hashes = []
    for prefix in ("image", "position", "state"):
        _floating_dtype(_get(trace, f"{prefix}_tensor_dtype"), f"{prefix}_tensor_dtype")
        component_hashes.append(_digest(_get(trace, f"{prefix}_tensor_sha256"), f"{prefix}_tensor_sha256"))
    component_hashes.append(trace["mask_sha256"])
    combined = hashlib.sha256(b"".join(bytes.fromhex(value) for value in component_hashes)).hexdigest()
    _equal(_get(trace, "prepared_memory_components_sha256"), combined,
           "prepared_memory_components_sha256")

    for key in ("boundary_lookup_latency_ms", "selector_decision_latency_ms",
                "selector_bookkeeping_latency_ms", "selector_latency_ms"):
        _latency(_get(trace, key), key)
    expected_latency = sum(trace[key] for key in ("boundary_lookup_latency_ms",
                           "selector_decision_latency_ms", "selector_bookkeeping_latency_ms"))
    if not math.isclose(trace["selector_latency_ms"], expected_latency, rel_tol=1e-9, abs_tol=1e-9):
        raise ExpansionTraceError("Selector latency components do not sum to total")

    prepared_fields = ("prepared_memory_input_shape", "prepared_memory_input_dtype",
                       "prepared_memory_input_sha256")
    if require_final_memory or any(key in trace for key in prepared_fields):
        _equal(_get(trace, prepared_fields[0]), [768, 2824], prepared_fields[0])
        _floating_dtype(_get(trace, prepared_fields[1]), prepared_fields[1])
        _digest(_get(trace, prepared_fields[2]), prepared_fields[2])
    final_fields = ("final_memory_tensor_shape", "final_memory_tensor_dtype",
                    "final_memory_tensor_is_floating", "final_memory_tensor_finite",
                    "final_memory_tensor_sha256")
    if require_final_memory or any(key in trace for key in final_fields):
        _equal(_get(trace, final_fields[0]), [1, 768, 1024], final_fields[0])
        _floating_dtype(_get(trace, final_fields[1]), final_fields[1])
        _equal(_get(trace, final_fields[2]), True, final_fields[2])
        _equal(_get(trace, final_fields[3]), True, final_fields[3])
        _digest(_get(trace, final_fields[4]), final_fields[4])
    if require_final_memory or "environment_step" in trace:
        _integer(_get(trace, "environment_step"), "environment_step", maximum=1300)
    if require_final_memory or "model_latency_ms" in trace:
        _latency(_get(trace, "model_latency_ms"), "model_latency_ms")
    return {
        "status": "passed", "scope": "selector_trace_contract_only",
        "final_memory_required": require_final_memory, "arm": decision["arm"],
        "task": decision["task"], "split": decision["split"], "episode_id": decision["episode_id"],
        "policy_call_index": decision["policy_call_index"], "history_length": step + 1,
        "valid_frame_count": len(selected),
        "feature_values_verified_from_hashes": False,
    }


def validate_trace_sequence(
    traces: Sequence[Mapping[str, Any]], *, require_final_memory: bool = True,
    expected_call_count: int | None = None,
) -> dict[str, Any]:
    """Replay one episode's trace prefix with no missing/interchanged calls.

    Set expected_call_count from an independent result artifact to detect a
    missing final suffix. Without it, contiguous traces cannot prove episode
    completion. Terminal correctness and full-run completeness are out of scope.
    """
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)) or not traces:
        raise ExpansionTraceError("Missing episode selector traces")
    if len(traces) > MAX_POLICY_CALLS:
        raise ExpansionTraceError("Trace sequence exceeds policy call table 0..81")
    if expected_call_count is not None:
        _integer(expected_call_count, "expected_call_count", minimum=1, maximum=MAX_POLICY_CALLS)
        if len(traces) != expected_call_count:
            raise ExpansionTraceError("Missing or extra trace calls relative to the independent expected count")
    prior = None
    context = None
    for index, trace in enumerate(traces):
        validate_selector_trace(trace, require_final_memory=require_final_memory)
        _equal(trace["policy_call_index"], index, "contiguous policy_call_index")
        current_context = [trace[key] for key in ("experiment_family", "arm", "split", "task", "episode_id")]
        if context is None:
            context = current_context
        else:
            _equal(current_context, context, "single-episode context")
        if prior is not None:
            if trace["step_idx"] <= prior["step_idx"]:
                raise ExpansionTraceError("Observed history must strictly grow between policy calls")
            old_prefix = [value for value in trace["visible_boundary_indices"] if value <= prior["step_idx"]]
            _equal(old_prefix, prior["visible_boundary_indices"], "immutable historical boundary prefix")
            if ("environment_step" in prior and "environment_step" in trace
                    and trace["environment_step"] <= prior["environment_step"]):
                raise ExpansionTraceError("Environment steps must strictly increase between calls")
        prior = trace
    return {
        "status": "passed", "scope": "single_episode_trace_sequence_only",
        "call_count": len(traces), "expected_call_count_verified": expected_call_count is not None,
        "context": context, "final_history_length": traces[-1]["history_length"],
        "feature_values_verified_from_hashes": False,
        "terminal_outcome_verified": False,
    }
