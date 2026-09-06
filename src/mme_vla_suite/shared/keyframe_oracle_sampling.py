"""Pure selectors for the causal keyframe-oracle sampling experiment.

This module deliberately depends only on the Python standard library and NumPy.
It contains no model, simulator, H5, or outcome access.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import enum
import hashlib
import json

import numpy as np

MAX_HISTORY_FRAMES = 32
MASTER_SELECTOR_SEED = 2026082501
MAX_POLICY_CALLS = 82
RANDOM_SELECTOR_LABEL = "RandomSamp"
FORMAL_SEED_SCOPE = "formal-evaluation-v1"
FORMAL_SEED_DATASET = "test"
SMOKE_SEED_SCOPE = "development-smoke-v1"
SMOKE_SEED_DATASET = "val"


class SelectorArm(str, enum.Enum):
    OFFICIAL_UNIFORM = "U"
    ORACLE_ONLY = "O"
    ORACLE_COVERAGE = "OC"
    RANDOM = "R"
    ORACLE_NEIGHBORHOOD_3 = "OC3"
    ORACLE_NEIGHBORHOOD_5 = "OC5"


def parse_arm(value: str | SelectorArm) -> SelectorArm:
    if isinstance(value, SelectorArm):
        return value
    aliases = {
        "U": SelectorArm.OFFICIAL_UNIFORM,
        "Official Uniform": SelectorArm.OFFICIAL_UNIFORM,
        "OfficialUniform": SelectorArm.OFFICIAL_UNIFORM,
        "O": SelectorArm.ORACLE_ONLY,
        "Oracle-only": SelectorArm.ORACLE_ONLY,
        "OracleOnly": SelectorArm.ORACLE_ONLY,
        "OC": SelectorArm.ORACLE_COVERAGE,
        "Oracle+Coverage": SelectorArm.ORACLE_COVERAGE,
        "OracleCoverage": SelectorArm.ORACLE_COVERAGE,
        "OC3": SelectorArm.ORACLE_NEIGHBORHOOD_3,
        "OC-3": SelectorArm.ORACLE_NEIGHBORHOOD_3,
        "Oracle+Neighborhood3+Coverage": SelectorArm.ORACLE_NEIGHBORHOOD_3,
        "OC5": SelectorArm.ORACLE_NEIGHBORHOOD_5,
        "OC-5": SelectorArm.ORACLE_NEIGHBORHOOD_5,
        "Oracle+Neighborhood5+Coverage": SelectorArm.ORACLE_NEIGHBORHOOD_5,
        "R": SelectorArm.RANDOM,
        "RandomSamp": SelectorArm.RANDOM,
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"Unknown selector arm: {value!r}") from exc


def _validate_step_idx(step_idx: int) -> int:
    if isinstance(step_idx, (bool, np.bool_)) or not isinstance(step_idx, (int, np.integer)):
        raise TypeError("step_idx must be an integer")
    step_idx = int(step_idx)
    if step_idx < 0:
        raise ValueError("step_idx must be non-negative")
    return step_idx


def _ordered_unique_indices(indices: Iterable[int], *, step_idx: int | None = None) -> list[int]:
    result: set[int] = set()
    for value in indices:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"Frame index must be an integer, got {value!r}")
        value = int(value)
        if value < 0:
            raise ValueError(f"Frame index must be non-negative, got {value}")
        if step_idx is not None and value > step_idx:
            # A caller may hold a complete future fixture.  Future labels are
            # causally invisible rather than grounds for reading them.
            continue
        result.add(value)
    return sorted(result)


def validate_selector_output(
    indices: Sequence[int],
    step_idx: int,
    *,
    max_frames: int = MAX_HISTORY_FRAMES,
    expected_count: int | None = None,
) -> list[int]:
    """Return a normalized list after enforcing every protocol output invariant."""
    step_idx = _validate_step_idx(step_idx)
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    normalized: list[int] = []
    for value in indices:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"Frame index must be an integer, got {value!r}")
        normalized.append(int(value))
    if normalized != sorted(normalized):
        raise ValueError("Selector output must be strictly increasing")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Selector output contains duplicate frame indices")
    if len(normalized) > max_frames:
        raise ValueError(f"Selector output has {len(normalized)} frames, exceeding {max_frames}")
    if any(value < 0 or value > step_idx for value in normalized):
        raise ValueError(f"Selector output must stay inside causal history [0, {step_idx}]")
    if expected_count is not None and len(normalized) != expected_count:
        raise ValueError(
            f"Selector output has {len(normalized)} frames; expected {expected_count}"
        )
    return normalized


def even_thin(indices: Iterable[int], count: int) -> list[int]:
    """Protocol EVEN_THIN over an ordered, unique version of ``indices``."""
    ordered = _ordered_unique_indices(indices)
    if count <= 0:
        return []
    if len(ordered) <= count:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, count, dtype=np.int32)
    return [ordered[int(position)] for position in positions]


def official_uniform_indices(step_idx: int, max_frames: int = MAX_HISTORY_FRAMES) -> list[int]:
    """Pure expected behavior used only for fixtures, never the runtime U path."""
    step_idx = _validate_step_idx(step_idx)
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if step_idx < max_frames:
        selected = list(range(step_idx + 1))
    else:
        selected = np.linspace(0, step_idx, max_frames, dtype=np.int32).tolist()
    return validate_selector_output(
        selected,
        step_idx,
        max_frames=max_frames,
        expected_count=min(max_frames, step_idx + 1),
    )


def oracle_only_indices(
    step_idx: int,
    boundary_indices: Iterable[int],
    max_frames: int = MAX_HISTORY_FRAMES,
) -> list[int]:
    step_idx = _validate_step_idx(step_idx)
    visible_boundaries = _ordered_unique_indices(boundary_indices, step_idx=step_idx)
    selected = even_thin(visible_boundaries, max_frames)
    return validate_selector_output(selected, step_idx, max_frames=max_frames)


def oracle_coverage_indices(
    step_idx: int,
    boundary_indices: Iterable[int],
    max_frames: int = MAX_HISTORY_FRAMES,
) -> list[int]:
    step_idx = _validate_step_idx(step_idx)
    target_count = min(max_frames, step_idx + 1)
    selected = set(oracle_only_indices(step_idx, boundary_indices, max_frames))
    return _farthest_time_coverage_fill(
        selected,
        step_idx=step_idx,
        target_count=target_count,
        max_frames=max_frames,
    )


def _farthest_time_coverage_fill(
    selected: Iterable[int],
    *,
    step_idx: int,
    target_count: int,
    max_frames: int,
) -> list[int]:
    """Fill remaining slots using the frozen OC farthest-time rule."""
    selected = set(_ordered_unique_indices(selected, step_idx=step_idx))
    if not selected:
        selected.add(0)

    while len(selected) < target_count:
        candidates = (value for value in range(step_idx + 1) if value not in selected)
        # ``max`` sees candidates chronologically.  Negating the index makes an
        # earlier frame win equal-distance ties explicitly.
        next_index = max(
            candidates,
            key=lambda value: (min(abs(value - chosen) for chosen in selected), -value),
        )
        selected.add(next_index)

    return validate_selector_output(
        sorted(selected),
        step_idx,
        max_frames=max_frames,
        expected_count=target_count,
    )


def _boundary_neighborhood_core(
    boundary_core: Sequence[int],
    *,
    step_idx: int,
    offsets: Sequence[int],
    max_frames: int,
) -> list[int]:
    """Build a causal boundary neighborhood, preserving boundaries before context.

    When the requested neighborhood exceeds the frame budget, every boundary in
    ``boundary_core`` is retained and the remaining neighborhood candidates are
    deterministically thinned across their chronological order.
    """
    core = _ordered_unique_indices(boundary_core, step_idx=step_idx)
    if len(core) > max_frames:  # pragma: no cover - callers construct a capped core
        raise ValueError("Boundary core exceeds the memory frame budget")
    neighborhood = {
        boundary + offset
        for boundary in core
        for offset in offsets
        if 0 <= boundary + offset <= step_idx
    }
    neighborhood.update(core)
    if len(neighborhood) <= max_frames:
        return sorted(neighborhood)

    context_candidates = sorted(neighborhood.difference(core))
    remaining = max_frames - len(core)
    return sorted([*core, *even_thin(context_candidates, remaining)])


def _causal_neighborhood_indices(
    boundaries: Iterable[int],
    *,
    step_idx: int,
    offsets: Sequence[int],
) -> list[int]:
    visible = _ordered_unique_indices(boundaries, step_idx=step_idx)
    return sorted(
        {
            boundary + offset
            for boundary in visible
            for offset in offsets
            if 0 <= boundary + offset <= step_idx
        }
    )


def oracle_neighborhood_coverage_decision(
    step_idx: int,
    boundary_indices: Iterable[int],
    *,
    neighborhood_frames: int,
    max_frames: int = MAX_HISTORY_FRAMES,
) -> tuple[list[int], dict[str, object]]:
    """Boundary neighborhoods followed by the frozen OC temporal coverage fill.

    ``neighborhood_frames=3`` preserves offsets ``(-2, 0, +2)`` around each
    causal boundary. ``neighborhood_frames=5`` first requests offsets
    ``(-4, -2, 0, +2, +4)``; if their unique union exceeds the budget, it
    falls back wholesale to the three-frame neighborhood. In either arm, a
    still-overfull three-frame neighborhood retains the boundary core first and
    evenly thins only the +/-2 context candidates.
    """
    step_idx = _validate_step_idx(step_idx)
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if neighborhood_frames not in (3, 5):
        raise ValueError("neighborhood_frames must be 3 or 5")

    target_count = min(max_frames, step_idx + 1)
    visible_boundaries = _ordered_unique_indices(boundary_indices, step_idx=step_idx)
    boundary_core = even_thin(visible_boundaries, max_frames)
    three_offsets = (-2, 0, 2)
    requested_offsets = (-4, -2, 0, 2, 4) if neighborhood_frames == 5 else three_offsets
    requested_candidates = _causal_neighborhood_indices(
        visible_boundaries,
        step_idx=step_idx,
        offsets=requested_offsets,
    )
    fell_back_to_three = neighborhood_frames == 5 and len(requested_candidates) > max_frames

    if fell_back_to_three:
        effective_candidates = _causal_neighborhood_indices(
            visible_boundaries,
            step_idx=step_idx,
            offsets=three_offsets,
        )
        effective_neighborhood_frames = 3
    else:
        effective_candidates = requested_candidates
        effective_neighborhood_frames = neighborhood_frames

    secondary_thinning = len(effective_candidates) > max_frames
    if secondary_thinning:
        selected = _boundary_neighborhood_core(
            boundary_core,
            step_idx=step_idx,
            offsets=three_offsets,
            max_frames=max_frames,
        )
    else:
        selected = effective_candidates

    final_indices = _farthest_time_coverage_fill(
        selected,
        step_idx=step_idx,
        target_count=target_count,
        max_frames=max_frames,
    )
    return final_indices, {
        "requested_neighborhood_frames": neighborhood_frames,
        "requested_neighborhood_offsets": list(requested_offsets),
        "causal_neighborhood_candidates": requested_candidates,
        "effective_neighborhood_frames": effective_neighborhood_frames,
        "effective_neighborhood_candidates": effective_candidates,
        "oc5_fell_back_to_oc3": fell_back_to_three,
        "oc3_secondary_thinning": secondary_thinning,
        "boundary_core_indices": boundary_core,
        "boundary_core_thinned": len(boundary_core) < len(visible_boundaries),
    }


def oracle_neighborhood_coverage_indices(
    step_idx: int,
    boundary_indices: Iterable[int],
    *,
    neighborhood_frames: int,
    max_frames: int = MAX_HISTORY_FRAMES,
) -> list[int]:
    selected, _ = oracle_neighborhood_coverage_decision(
        step_idx,
        boundary_indices,
        neighborhood_frames=neighborhood_frames,
        max_frames=max_frames,
    )
    return selected


def oracle_neighborhood_3_coverage_indices(
    step_idx: int,
    boundary_indices: Iterable[int],
    max_frames: int = MAX_HISTORY_FRAMES,
) -> list[int]:
    return oracle_neighborhood_coverage_indices(
        step_idx,
        boundary_indices,
        neighborhood_frames=3,
        max_frames=max_frames,
    )


def oracle_neighborhood_5_coverage_indices(
    step_idx: int,
    boundary_indices: Iterable[int],
    max_frames: int = MAX_HISTORY_FRAMES,
) -> list[int]:
    return oracle_neighborhood_coverage_indices(
        step_idx,
        boundary_indices,
        neighborhood_frames=5,
        max_frames=max_frames,
    )


def random_sampling_indices(
    step_idx: int,
    seed: int,
    max_frames: int = MAX_HISTORY_FRAMES,
) -> list[int]:
    step_idx = _validate_step_idx(step_idx)
    if max_frames != MAX_HISTORY_FRAMES:
        raise ValueError("The frozen RandomSamp procedure requires max_frames=32")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("RandomSamp seed must be an integer")
    seed = int(seed)
    if seed < 0 or seed > np.iinfo(np.uint64).max:
        raise ValueError("RandomSamp seed must fit in an unsigned 64-bit integer")
    if step_idx + 1 <= max_frames:
        return list(range(step_idx + 1))

    rng = np.random.Generator(np.random.PCG64(seed))
    interior = np.arange(1, step_idx, dtype=np.int64)
    bins = np.array_split(interior, max_frames - 2)
    selected = [0]
    selected.extend(int(rng.choice(bin_values)) for bin_values in bins)
    selected.append(step_idx)
    return validate_selector_output(
        sorted(selected),
        step_idx,
        max_frames=max_frames,
        expected_count=max_frames,
    )


def select_indices(
    arm: str | SelectorArm,
    step_idx: int,
    *,
    boundary_indices: Iterable[int] = (),
    random_seed: int | None = None,
) -> list[int]:
    arm = parse_arm(arm)
    if arm is SelectorArm.OFFICIAL_UNIFORM:
        return official_uniform_indices(step_idx)
    if arm is SelectorArm.ORACLE_ONLY:
        return oracle_only_indices(step_idx, boundary_indices)
    if arm is SelectorArm.ORACLE_COVERAGE:
        return oracle_coverage_indices(step_idx, boundary_indices)
    if arm is SelectorArm.ORACLE_NEIGHBORHOOD_3:
        return oracle_neighborhood_3_coverage_indices(step_idx, boundary_indices)
    if arm is SelectorArm.ORACLE_NEIGHBORHOOD_5:
        return oracle_neighborhood_5_coverage_indices(step_idx, boundary_indices)
    if random_seed is None:
        raise ValueError("RandomSamp requires a preregistered seed")
    return random_sampling_indices(step_idx, random_seed)


def keyframe_timeout_reached(
    completed_steps: int,
    maximum_steps: int,
    official_stop: bool,
) -> bool:
    """Apply the local cap without overwriting an official exact-limit terminal."""
    return completed_steps >= maximum_steps and not official_stop


def canonical_seed_payload(task_name: str, episode_id: int, policy_call_index: int) -> bytes:
    """Canonical UTF-8 JSON encoding of the protocol's SHA-256 input list."""
    if not task_name:
        raise ValueError("task_name must be non-empty")
    if episode_id < 0:
        raise ValueError("episode_id must be non-negative")
    if policy_call_index < 0 or policy_call_index >= MAX_POLICY_CALLS:
        raise IndexError(
            f"policy_call_index {policy_call_index} is outside preregistered table 0..{MAX_POLICY_CALLS - 1}"
        )
    values = [
        MASTER_SELECTOR_SEED,
        task_name,
        int(episode_id),
        int(policy_call_index),
        RANDOM_SELECTOR_LABEL,
    ]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def derive_random_seed(task_name: str, episode_id: int, policy_call_index: int) -> int:
    """Derive the frozen formal-evaluation RandomSamp seed.

    This intentionally preserves the pre-amendment Section 7.4 payload, which
    A-001 leaves unchanged. Development smoke must use
    :func:`derive_smoke_random_seed` instead.
    """
    digest = hashlib.sha256(
        canonical_seed_payload(task_name, episode_id, policy_call_index)
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def canonical_smoke_seed_payload(
    task_name: str,
    episode_id: int,
    policy_call_index: int,
) -> bytes:
    """Canonical payload for the disjoint development-smoke seed namespace."""
    if not task_name:
        raise ValueError("task_name must be non-empty")
    if episode_id < 0:
        raise ValueError("episode_id must be non-negative")
    if policy_call_index < 0 or policy_call_index >= MAX_POLICY_CALLS:
        raise IndexError(
            f"policy_call_index {policy_call_index} is outside preregistered table "
            f"0..{MAX_POLICY_CALLS - 1}"
        )
    values = [
        MASTER_SELECTOR_SEED,
        SMOKE_SEED_SCOPE,
        SMOKE_SEED_DATASET,
        task_name,
        int(episode_id),
        int(policy_call_index),
        RANDOM_SELECTOR_LABEL,
    ]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def derive_smoke_random_seed(
    task_name: str,
    episode_id: int,
    policy_call_index: int,
) -> int:
    """Derive a RandomSamp seed reserved exclusively for development smoke."""
    digest = hashlib.sha256(
        canonical_smoke_seed_payload(task_name, episode_id, policy_call_index)
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def boundary_flags_from_stages(
    stages: Sequence[int],
    *,
    previous_stage: int | None = None,
    first_history_index: int = 0,
) -> list[bool]:
    """Derive adjacent stage boundaries for one non-overlapping live segment."""
    if first_history_index < 0:
        raise ValueError("first_history_index must be non-negative")
    normalized: list[int] = []
    for value in stages:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"current_task_index must be an integer, got {value!r}")
        normalized.append(int(value))
    if first_history_index > 0 and normalized and previous_stage is None:
        raise ValueError("A non-initial segment requires the preceding live stage")
    flags: list[bool] = []
    prior = None if previous_stage is None else int(previous_stage)
    for offset, stage in enumerate(normalized):
        history_index = first_history_index + offset
        flags.append(history_index == 0 or stage != prior)
        prior = stage
    return flags


def boundary_indices_from_flags(flags: Sequence[bool], *, start_index: int = 0) -> list[int]:
    return [start_index + offset for offset, value in enumerate(flags) if bool(value)]
