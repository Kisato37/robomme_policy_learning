from __future__ import annotations

import copy

import pytest

from experiments.keyframe_oracle_sampling.analysis import ANALYSIS_SEED
from experiments.keyframe_oracle_sampling.analysis import DEFAULT_BOOTSTRAP_REPLICATES
from experiments.keyframe_oracle_sampling.analysis import DEFAULT_RANDOMIZATION_REPLICATES
from experiments.keyframe_oracle_sampling.analysis import AnalysisProtocolError
from experiments.keyframe_oracle_sampling.analysis import build_point_analysis
from experiments.keyframe_oracle_sampling.analysis import classify_decision
from experiments.keyframe_oracle_sampling.analysis import discordant_counts
from experiments.keyframe_oracle_sampling.analysis import hierarchical_percentile_bootstrap_ci
from experiments.keyframe_oracle_sampling.analysis import holm_adjust
from experiments.keyframe_oracle_sampling.analysis import paired_effect
from experiments.keyframe_oracle_sampling.analysis import paired_randomization_test
from experiments.keyframe_oracle_sampling.analysis import require_resolved_inferential_protocol
from experiments.keyframe_oracle_sampling.analysis import validate_formal_records
from experiments.keyframe_oracle_sampling.artifacts import ALL_ARMS
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError


def _formal_records() -> list[dict]:
    success_limits = {"U": 10, "O": 20, "OC": 30, "R": 25}
    return [
        {
            "scientific_key": {
                "task": task,
                "episode_id": episode_id,
                "arm": arm,
                "trajectory_kind": "formal",
            },
            "dataset": "test",
            "success": episode_id < success_limits[arm],
        }
        for task in FORMAL_TASKS
        for episode_id in range(50)
        for arm in ALL_ARMS
    ]


def test_exact_formal_census_and_equal_task_weighted_effects():
    table = validate_formal_records(_formal_records())
    assert table.cell_count == 3200

    primary = paired_effect(table, "OC", "U")
    assert primary["task_count"] == 16
    assert primary["paired_episode_count"] == 800
    assert primary["treatment_equal_task_weighted_success_rate"] == pytest.approx(0.6)
    assert primary["control_equal_task_weighted_success_rate"] == pytest.approx(0.2)
    assert primary["estimate"] == pytest.approx(0.4)
    assert primary["estimate_pp"] == pytest.approx(40.0)
    assert {row["paired_episode_count"] for row in primary["per_task"]} == {50}


def test_discordant_counts_keep_both_ties_and_full_denominators():
    counts = discordant_counts(validate_formal_records(_formal_records()))
    assert counts["pooled"] == {
        "treatment_success_control_fail": 320,
        "treatment_fail_control_success": 0,
        "both_success": 160,
        "both_fail": 320,
        "denominator": 800,
    }
    assert all(row["denominator"] == 50 for row in counts["per_task"])
    assert all(row["treatment_success_control_fail"] == 20 for row in counts["per_task"])


@pytest.mark.parametrize("mutation", [("missing"), ("duplicate"), ("extra")])
def test_formal_validation_fails_closed_for_non_exact_cells(mutation):
    records = _formal_records()
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    else:
        records[-1]["scientific_key"]["episode_id"] = 50
    with pytest.raises(ArtifactContractError):
        validate_formal_records(records)


def test_formal_validation_rejects_wrong_split_kind_and_non_bool_outcome():
    records = _formal_records()
    records[0]["dataset"] = "val"
    with pytest.raises(ArtifactContractError, match="dataset"):
        validate_formal_records(records)

    records = _formal_records()
    records[0]["scientific_key"]["trajectory_kind"] = "short"
    with pytest.raises(ArtifactContractError, match="trajectory_kind"):
        validate_formal_records(records)

    records = _formal_records()
    records[0]["success"] = 1
    with pytest.raises(ArtifactContractError, match="binary bool"):
        validate_formal_records(records)


def test_frozen_exposure_sensitivity_excludes_blocks_but_preserves_task_weights():
    exposure_entries = [
        {"task": FORMAL_TASKS[0], "split": "test", "episode_id": episode_id}
        for episode_id in range(40)
    ]
    exposure_entries.append(
        {"task": FORMAL_TASKS[1], "split": "val", "episode_id": 0}
    )
    report = build_point_analysis(
        _formal_records(),
        prior_exposure_manifest={"entries": exposure_entries},
        bootstrap_replicates=200,
        randomization_replicates=200,
        allow_nonconfirmatory_replicate_override=True,
    )
    sensitivity = report["prior_exposure_sensitivity"]
    assert sensitivity["manifest_entry_count"] == 41
    assert sensitivity["excluded_formal_block_count"] == 40
    assert sensitivity["effect"]["paired_episode_count"] == 760
    assert sensitivity["effect"]["estimate"] == pytest.approx(0.375)
    per_task = sensitivity["effect"]["per_task"]
    assert per_task[0]["paired_episode_count"] == 10
    assert {row["paired_episode_count"] for row in per_task[1:]} == {50}


def test_report_freezes_declared_seed_and_production_replicates():
    with pytest.raises(AnalysisProtocolError, match="exactly 100,000"):
        build_point_analysis(
            _formal_records(),
            bootstrap_replicates=100,
            randomization_replicates=100,
        )

    report = build_point_analysis(
        _formal_records(),
        bootstrap_replicates=200,
        randomization_replicates=200,
        allow_nonconfirmatory_replicate_override=True,
    )
    assert report["analysis_seed"] == ANALYSIS_SEED == 2026082502
    assert report["frozen_bootstrap_replicates"] == DEFAULT_BOOTSTRAP_REPLICATES == 100_000
    assert (
        report["frozen_randomization_replicates"]
        == DEFAULT_RANDOMIZATION_REPLICATES
        == 100_000
    )
    assert set(report["secondary_unadjusted_point_estimates"]) == {
        "O - U",
        "OC - O",
        "R - U",
        "OC - R",
    }
    assert report["bootstrap_replicates"] == 200
    assert report["randomization_replicates"] == 200
    assert report["frozen_replicate_contract_met"] is False
    assert report["inferential_analysis_status"] == "complete_nonconfirmatory_test_override"
    assert report["decision_classification"] is None
    assert report["nonconfirmatory_decision_preview"] == "semantic_selection_go"
    assert require_resolved_inferential_protocol() is None


def test_hierarchical_percentile_bootstrap_is_reproducible_and_task_equal():
    table = validate_formal_records(_formal_records())
    first = hierarchical_percentile_bootstrap_ci(
        table, "OC", "U", seed=ANALYSIS_SEED, replicates=250
    )
    second = hierarchical_percentile_bootstrap_ci(
        table, "OC", "U", seed=ANALYSIS_SEED, replicates=250
    )
    assert first == second
    assert first["lower_pp"] < 40.0 < first["upper_pp"]
    assert first["task_count"] == 16
    assert set(first["per_task_episode_counts"].values()) == {50}
    assert first["percentile_bounds"] == [2.5, 97.5]


def test_paired_two_sided_randomization_uses_registered_plus_one_formula():
    table = validate_formal_records(_formal_records())
    first = paired_randomization_test(
        table, "OC", "U", seed=ANALYSIS_SEED, replicates=300
    )
    second = paired_randomization_test(
        table, "OC", "U", seed=ANALYSIS_SEED, replicates=300
    )
    assert first == second
    assert first["tail"] == "two-sided"
    assert first["paired_block_count"] == 800
    assert first["p_value"] == pytest.approx(
        (1 + first["extreme_permutation_count"]) / 301
    )


def test_holm_adjustment_is_step_down_monotone_and_named():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20})
    assert adjusted["a"]["holm_adjusted_p_value"] == pytest.approx(0.04)
    assert adjusted["c"]["holm_adjusted_p_value"] == pytest.approx(0.09)
    assert adjusted["b"]["holm_adjusted_p_value"] == pytest.approx(0.09)
    assert adjusted["d"]["holm_adjusted_p_value"] == pytest.approx(0.20)


def _decision_effect(estimate_pp: float, lower_pp: float, upper_pp: float) -> dict:
    return {
        "estimate_pp": estimate_pp,
        "confidence_interval_95": {"lower_pp": lower_pp, "upper_pp": upper_pp},
    }


def test_decision_rule_uses_ci_lower_and_does_not_invent_label_precedence():
    semantic_go = classify_decision(
        _decision_effect(4.0, 0.5, 7.0),
        _decision_effect(2.0, 0.1, 4.0),
        _decision_effect(2.0, 0.1, 4.0),
    )
    assert semantic_go["classification"] == "semantic_selection_go"
    assert semantic_go["coverage_complementarity_evidence"] is True

    no_go = classify_decision(
        _decision_effect(1.0, 0.2, 2.5),
        _decision_effect(0.1, -1.0, 1.0),
        _decision_effect(0.1, -1.0, 1.0),
    )
    assert no_go["classification"] == "overlapping_prespecified_conclusions"
    assert no_go["sampling_sensitivity_only"] is True
    assert no_go["satisfied_prespecified_conclusions"] == [
        "decisive_test_time_no_go",
        "sampling_sensitivity_only",
    ]

    sampling_only = classify_decision(
        _decision_effect(4.0, 0.2, 8.0),
        _decision_effect(0.1, -1.0, 1.0),
        _decision_effect(0.1, -1.0, 1.0),
    )
    assert sampling_only["classification"] == "sampling_sensitivity_only"

    random_strictly_better = classify_decision(
        _decision_effect(4.0, 0.2, 8.0),
        _decision_effect(-2.0, -4.0, -0.1),
        _decision_effect(0.1, -1.0, 1.0),
    )
    assert random_strictly_better["oc_beats_r"] is False
    assert random_strictly_better["oc_not_distinguished_from_r"] is False
    assert random_strictly_better["sampling_sensitivity_only"] is False
    assert random_strictly_better["classification"] == "inconclusive"

    inconclusive = classify_decision(
        _decision_effect(4.0, -0.5, 8.0),
        _decision_effect(2.0, 0.1, 4.0),
        _decision_effect(0.1, -1.0, 1.0),
    )
    assert inconclusive["classification"] == "inconclusive"
