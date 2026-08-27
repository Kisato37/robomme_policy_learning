from __future__ import annotations

import numpy as np
import pytest

from mme_vla_suite.shared.keyframe_oracle_sampling import (
    MAX_HISTORY_FRAMES,
    boundary_flags_from_stages,
    derive_random_seed,
    derive_smoke_random_seed,
    even_thin,
    official_uniform_indices,
    keyframe_timeout_reached,
    oracle_coverage_indices,
    oracle_only_indices,
    random_sampling_indices,
    validate_selector_output,
)


HISTORY_LENGTHS = (1, 2, 16, 32, 33, 64, 1301)


@pytest.mark.parametrize("history_length", HISTORY_LENGTHS)
def test_all_selectors_enforce_protocol_invariants(history_length):
    step_idx = history_length - 1
    boundaries = [0, step_idx // 3, (2 * step_idx) // 3, step_idx]
    seed = derive_random_seed("InsertPeg", 0, min(step_idx, 81))
    outputs = (
        official_uniform_indices(step_idx),
        oracle_only_indices(step_idx, boundaries),
        oracle_coverage_indices(step_idx, boundaries),
        random_sampling_indices(step_idx, seed),
    )
    for selected in outputs:
        assert selected == sorted(set(selected))
        assert all(0 <= index <= step_idx for index in selected)
        assert len(selected) <= MAX_HISTORY_FRAMES
    assert len(outputs[0]) == min(history_length, MAX_HISTORY_FRAMES)
    assert len(outputs[2]) == min(history_length, MAX_HISTORY_FRAMES)
    assert len(outputs[3]) == min(history_length, MAX_HISTORY_FRAMES)


def test_even_thin_is_exact_numpy_int32_positional_rule():
    source = list(range(0, 1301, 3))
    expected_positions = np.linspace(0, len(source) - 1, 32, dtype=np.int32)
    assert even_thin(source, 32) == [source[int(position)] for position in expected_positions]
    assert even_thin(source, 0) == []
    assert even_thin([4, 4, 1, 9], 10) == [1, 4, 9]


def test_no_dense_duplicate_and_overflow_boundary_fixtures():
    assert oracle_only_indices(64, []) == []
    assert oracle_coverage_indices(4, []) == [0, 1, 2, 3, 4]
    assert oracle_only_indices(10, [0, 0, 5, 5, 10]) == [0, 5, 10]

    dense = oracle_only_indices(1300, range(1301))
    expected = np.linspace(0, 1300, 32, dtype=np.int32).tolist()
    assert dense == expected
    assert dense[0] == 0 and dense[-1] == 1300


def test_oracle_only_never_adds_latest_anchor_and_coverage_prioritizes_boundaries():
    assert oracle_only_indices(64, [0, 10]) == [0, 10]
    selected = oracle_coverage_indices(64, [0, 10])
    assert len(selected) == 32
    assert {0, 10}.issubset(selected)
    assert selected == [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 16, 19, 21, 23, 26,
        28, 30, 33, 35, 37, 40, 43, 46, 48, 50, 53, 55, 57, 60, 62, 64,
    ]


def test_coverage_ties_choose_earlier_frames_deterministically():
    # With only endpoints selected, the first farthest candidate in 0..64 is
    # 32.  Repeating the frozen greedy rule yields this exact final set.
    assert oracle_coverage_indices(64, [0, 64]) == [
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 64,
    ]


def test_random_sampling_bins_endpoints_reproducibility_and_seed_sensitivity():
    step_idx = 1300
    seed_a = derive_random_seed("InsertPeg", 0, 0)
    seed_b = derive_random_seed("InsertPeg", 0, 1)
    selected_a = random_sampling_indices(step_idx, seed_a)
    assert selected_a == random_sampling_indices(step_idx, seed_a)
    assert selected_a != random_sampling_indices(step_idx, seed_b)
    assert selected_a[0] == 0 and selected_a[-1] == step_idx
    assert len(selected_a) == len(set(selected_a)) == 32
    bins = np.array_split(np.arange(1, step_idx), 30)
    for selected, bin_values in zip(selected_a[1:-1], bins, strict=True):
        assert selected in bin_values


def test_seed_derivation_is_pinned_and_out_of_table_is_rejected():
    # The formal Section 7.4 values are a backward-compatibility contract.
    assert derive_random_seed("InsertPeg", 0, 0) == 12410663345555581359
    assert derive_random_seed("InsertPeg", 0, 81) == 204551496613866343
    assert derive_smoke_random_seed("InsertPeg", 0, 0) == 1726467455513403504
    assert derive_smoke_random_seed("InsertPeg", 0, 81) == 17684041232018081866
    assert derive_smoke_random_seed("InsertPeg", 0, 0) != derive_random_seed(
        "InsertPeg", 0, 0
    )
    with pytest.raises(IndexError, match="outside preregistered table"):
        derive_random_seed("InsertPeg", 0, 82)
    with pytest.raises(IndexError, match="outside preregistered table"):
        derive_smoke_random_seed("InsertPeg", 0, 82)


def test_future_boundary_changes_are_causally_invisible():
    visible = [0, 7, 31, 60]
    future_a = visible + [65, 70, 100]
    future_b = visible + [66, 99, 1300]
    assert oracle_only_indices(64, future_a) == oracle_only_indices(64, future_b)
    assert oracle_coverage_indices(64, future_a) == oracle_coverage_indices(64, future_b)


def test_stage_boundaries_span_non_overlapping_segments():
    first = boundary_flags_from_stages([2, 2, 3], first_history_index=0)
    second = boundary_flags_from_stages(
        [3, 4, 4], previous_stage=3, first_history_index=3
    )
    assert first == [True, False, True]
    assert second == [False, True, False]
    with pytest.raises(ValueError, match="preceding live stage"):
        boundary_flags_from_stages([4], first_history_index=3)


def test_strict_output_validation_rejects_invalid_selectors():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_selector_output([1, 0], 2)
    with pytest.raises(ValueError, match="duplicate"):
        validate_selector_output([0, 0], 2)
    with pytest.raises(ValueError, match="causal history"):
        validate_selector_output([0, 3], 2)
    with pytest.raises(TypeError, match="integer"):
        validate_selector_output([0, 1.5], 2)


def test_exact_step_limit_preserves_official_terminal_outcome():
    assert keyframe_timeout_reached(64, 64, official_stop=False)
    assert not keyframe_timeout_reached(64, 64, official_stop=True)
    assert not keyframe_timeout_reached(63, 64, official_stop=False)
