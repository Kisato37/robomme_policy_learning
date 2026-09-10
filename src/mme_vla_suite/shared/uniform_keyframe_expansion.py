"""Pure, causal selectors for the separately versioned UK48/UN48 experiment.

The runtime must supply the *literal* released U32 sampler output.  The fixture
function imported below only verifies that input; it never replaces the runtime
sampler.  This module reads no simulator, model, outcome, or future observation.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json

import numpy as np

from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_HISTORY_FRAMES
from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_POLICY_CALLS
from mme_vla_suite.shared.keyframe_oracle_sampling import official_uniform_indices
from mme_vla_suite.shared.keyframe_oracle_sampling import validate_selector_output


BASE_FRAME_CAPACITY = MAX_HISTORY_FRAMES
FRAME_CAPACITY = 48
TOKENS_PER_FRAME = 16
MEMORY_TOKEN_CAPACITY = FRAME_CAPACITY * TOKENS_PER_FRAME
MASTER_SELECTOR_SEED = 2026091001
SEED_FAMILY = "uniform_keyframe_expansion-v1"
EXPANSION_ARMS = ("UK48", "UN48")
# Keep the lightweight runtime independent of the experiments package.  A parity
# test pins this tuple to the existing, immutable sixteen-task contract.
CANONICAL_TASKS = (
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    "ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
)


class ExpansionInvariantError(ValueError):
    """Protocol/implementation failure, never a scientific task failure.

    ``code`` and JSON-serializable ``evidence`` are available for append-only
    error artifacts.  In particular, capacity overflow and insufficient random
    candidates stop *either* arm instead of silently changing its definition.
    """

    def __init__(self, code: str, evidence: dict[str, object]):
        self.code = code
        self.evidence = {"error_code": code, **evidence}
        super().__init__(f"{code}: {json.dumps(self.evidence, sort_keys=True)}")


def _integer(value: object, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ExpansionInvariantError("invalid_integer", {"field": name, "value_repr": repr(value)})
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ExpansionInvariantError("integer_out_of_range", {
            "field": name, "value": result, "minimum": minimum, "maximum": maximum,
        })
    return result


def _context(split: str, task: str, episode_id: int, policy_call_index: int) -> dict[str, object]:
    if split not in ("val", "test"):
        raise ExpansionInvariantError("invalid_split", {"value_repr": repr(split)})
    if task not in CANONICAL_TASKS:
        raise ExpansionInvariantError("invalid_task", {"value_repr": repr(task)})
    return {
        "split": split,
        "task": task,
        "episode_id": _integer(episode_id, "episode_id"),
        "policy_call_index": _integer(policy_call_index, "policy_call_index", maximum=MAX_POLICY_CALLS - 1),
    }


def derive_expansion_seed(split: str, task: str, episode_id: int, policy_call_index: int) -> int:
    """UN48 seed; compact JSON UTF-8, first eight SHA256 bytes, big-endian."""
    context = _context(split, task, episode_id, policy_call_index)
    payload = [MASTER_SELECTOR_SEED, SEED_FAMILY, split, task,
               context["episode_id"], context["policy_call_index"], "UN48"]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def derive_selector_seed(*, split: str, task: str, episode_id: int,
                         policy_call_index: int, arm: str = "UN48") -> int:
    """Keyword-only seed-table helper; UK48 deliberately has no RNG seed."""
    if arm != "UN48":
        raise ExpansionInvariantError("seed_requested_for_nonrandom_arm", {"arm_repr": repr(arm)})
    return derive_expansion_seed(split, task, episode_id, policy_call_index)


def select_expansion_indices(
    arm: str,
    *,
    step_idx: int,
    base_uniform_indices: Sequence[int],
    boundary_flags: Sequence[bool],
    split: str,
    task: str,
    episode_id: int,
    policy_call_index: int,
) -> tuple[list[int], dict[str, object]]:
    """Retain exact U32 and add causal boundaries or equal-count random nonkeys.

    Flags must describe exactly the observed prefix ``[0, step_idx]``; accepting
    a longer sequence and filtering future entries is intentionally prohibited.
    Matching extra counts is a same-history invariant, not a promise about two
    diverging closed-loop trajectories.  No remaining slots are coverage-filled.
    """
    context = _context(split, task, episode_id, policy_call_index)
    if arm not in EXPANSION_ARMS:
        raise ExpansionInvariantError("invalid_arm", {"arm_repr": repr(arm), **context})
    step_idx = _integer(step_idx, "step_idx")
    evidence: dict[str, object] = {"arm": arm, "step_idx": step_idx, **context}
    try:
        base = validate_selector_output(
            base_uniform_indices, step_idx, max_frames=BASE_FRAME_CAPACITY,
            expected_count=min(step_idx + 1, BASE_FRAME_CAPACITY),
        )
    except (TypeError, ValueError) as exc:
        raise ExpansionInvariantError("invalid_base_uniform_indices", {
            **evidence, "detail": str(exc),
        }) from exc
    # A validation oracle only: actual output is built from caller-supplied base.
    if base != official_uniform_indices(step_idx):
        raise ExpansionInvariantError("base_is_not_original_uniform32", {
            **evidence, "base_uniform_indices": base,
        })
    evidence["base_uniform_indices"] = list(base)
    try:
        flag_count = len(boundary_flags)
    except TypeError as exc:
        raise ExpansionInvariantError("invalid_boundary_flags", evidence) from exc
    if flag_count != step_idx + 1:
        raise ExpansionInvariantError("boundary_history_alignment", {
            **evidence, "boundary_flag_count": flag_count, "expected_flag_count": step_idx + 1,
        })
    boundaries: list[int] = []
    for index, flag in enumerate(boundary_flags):
        if not isinstance(flag, (bool, np.bool_)):
            raise ExpansionInvariantError("non_boolean_boundary_flag", {
                **evidence, "flag_index": index, "flag_value_repr": repr(flag),
            })
        if flag:
            boundaries.append(index)
    if not boundaries or boundaries[0] != 0:
        raise ExpansionInvariantError("initial_frame_must_be_boundary", evidence)

    base_set, boundary_set = set(base), set(boundaries)
    new_keys = sorted(boundary_set - base_set)
    candidates = sorted(set(range(step_idx + 1)) - (base_set | boundary_set))
    extra_count = len(new_keys)
    evidence.update({
        "visible_boundary_indices": boundaries,
        "visible_boundary_count": len(boundaries),
        "new_key_indices": new_keys,
        "nonkey_candidate_count": len(candidates),
        "extra_count": extra_count,
        "base_frame_capacity": BASE_FRAME_CAPACITY,
        "frame_capacity": FRAME_CAPACITY,
        "tokens_per_frame": TOKENS_PER_FRAME,
        "memory_token_capacity": MEMORY_TOKEN_CAPACITY,
        "required_frame_count": len(base) + extra_count,
    })
    if len(base) + extra_count > FRAME_CAPACITY:
        raise ExpansionInvariantError("capacity_overflow", evidence)
    if len(candidates) < extra_count:
        raise ExpansionInvariantError("nonkey_candidate_shortage", evidence)

    seed = None
    if arm == "UK48":
        extras = new_keys
    else:
        seed = derive_expansion_seed(split, task, episode_id, policy_call_index)
        rng = np.random.Generator(np.random.PCG64(seed))
        extras = sorted(int(value) for value in rng.choice(candidates, size=extra_count, replace=False))
    selected = sorted(base_set | set(extras))
    # Independent final validation protects callers from future refactors.
    selected = validate_selector_output(
        selected, step_idx, max_frames=FRAME_CAPACITY,
        expected_count=len(base) + extra_count,
    )
    selected_boundaries = sorted(set(selected) & boundary_set)
    valid_count = len(selected)
    decision = {
        **evidence,
        "selected_extra_indices": list(extras),
        "selected_indices": list(selected),
        "selected_boundary_indices": selected_boundaries,
        "selected_boundary_count": len(selected_boundaries),
        "selector_seed": seed,
        "selector_rng": "PCG64" if arm == "UN48" else None,
        "selector_seed_family": SEED_FAMILY if arm == "UN48" else None,
        "valid_frame_count": valid_count,
        "padding_frame_count": FRAME_CAPACITY - valid_count,
        "valid_token_count": valid_count * TOKENS_PER_FRAME,
        "padding_token_count": (FRAME_CAPACITY - valid_count) * TOKENS_PER_FRAME,
    }
    return selected, decision
