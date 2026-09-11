from __future__ import annotations

import inspect

import pytest

from experiments.keyframe_oracle_sampling import analysis as parent
from experiments.uniform_keyframe_expansion import analysis as a


def _records(episodes=2, outcome=None):
    outcome = outcome or (lambda task_index, episode, arm: (task_index + episode + a.ARMS.index(arm)) % 3 == 0)
    return [{"task": task, "episode_id": episode, "arm": arm, "dataset": "test",
             "trajectory_kind": "formal", "success": bool(outcome(index, episode, arm))}
            for index, task in enumerate(a.FORMAL_TASKS) for episode in range(episodes) for arm in a.ARMS]


def _small_analysis(records, **kwargs):
    return a.analyze_matched_outcomes(records, allow_fixture=True, bootstrap_replicates=101,
                                     randomization_replicates=127, **kwargs)


def test_full_matrix_exact_and_record_order_independent():
    records = _records(50)
    table = a.validate_outcome_records(records)
    assert table.cell_count == 2400 and len(table.blocks) == 800
    assert table.excluded_blocks == frozenset()
    assert a.validate_outcome_records({"records": list(reversed(records))}) == table
    assert a.outcome_binding_hashes(table) == a.outcome_binding_hashes(a.validate_outcome_records(reversed(records)))
    with pytest.raises(a.ExpansionAnalysisError, match="2400"):
        a.validate_outcome_records(records[:-1])


@pytest.mark.parametrize("field,value", [
    ("episode_id", False), ("episode_id", 1.0), ("episode_id", 50),
    ("task", "Unknown"), ("arm", "OC"), ("dataset", "val"),
    ("trajectory_kind", "short"), ("success", 1), ("success", "false"),
    ("terminal_status", "infrastructure_failure"), ("infrastructure_failure", True),
    ("protocol_invariant_failure", True),
])
def test_bad_fields_cannot_become_binary_scientific_results(field, value):
    records = _records()
    records[0][field] = value
    with pytest.raises(a.ExpansionAnalysisError):
        a.validate_outcome_records(records, allow_fixture=True)


def test_duplicate_missing_arm_empty_task_and_nested_conflicts_are_rejected():
    records = _records()
    with pytest.raises(a.ExpansionAnalysisError, match="Duplicate"):
        a.validate_outcome_records(records + [records[0]], allow_fixture=True)
    with pytest.raises(a.ExpansionAnalysisError, match="Every paired block"):
        a.validate_outcome_records(records[1:], allow_fixture=True)
    with pytest.raises(a.ExpansionAnalysisError, match="16|canonical task"):
        a.validate_outcome_records(records[6:], allow_fixture=True)
    records[0]["scientific_key"] = {"episode_id": False}
    with pytest.raises(a.ExpansionAnalysisError, match="Conflicting"):
        a.validate_outcome_records(records, allow_fixture=True)


def test_timeout_fail_and_scientific_error_keep_denominators():
    records = _records(1, lambda *_: False)
    for index, record in enumerate(records):
        record["terminal_status"] = ("fail", "timeout", "error")[index % 3]
    result = _small_analysis(records)
    assert result["paired_block_count"] == 16 and result["cell_count"] == 48
    assert all(row == {"success": 0, "total": 16} for row in result["arm_counts"].values())
    assert result["primary"]["discordant_counts"]["pooled"]["both_fail"] == 16
    assert result["primary"]["randomization_test"]["p_value"] == 1.0
    records[0]["success"] = True
    with pytest.raises(a.ExpansionAnalysisError, match="contradicts"):
        a.validate_outcome_records(records, allow_fixture=True)


def test_fixture_is_explicit_nonformal_even_with_full_census():
    records = _records()
    with pytest.raises(a.ExpansionAnalysisError, match="100000"):
        a.analyze_matched_outcomes(records, bootstrap_replicates=1, randomization_replicates=1)
    with pytest.raises(a.ExpansionAnalysisError, match="2400"):
        a.analyze_matched_outcomes(records)
    report = _small_analysis(records)
    assert report["analysis_status"] == "nonformal_fixture_only"
    assert report["formal_analysis_contract_met"] is False
    assert report["decision"] is None
    assert report["nonformal_decision_preview"] is not None
    assert report["analysis_scope"] == "outcome-statistics; raw-launch-provenance-external-required"
    assert report["raw_artifacts_reverified_by_this_function"] is False
    assert report["exposure_based_exclusions"] == []
    signature = inspect.signature(a.analyze_matched_outcomes)
    assert signature.parameters["bootstrap_replicates"].default == 100000
    assert signature.parameters["randomization_replicates"].default == 100000
    assert a.ANALYSIS_SEED == 2026082502


def test_equal_task_weights_not_pooling_when_fixture_counts_differ():
    records = _records(2, lambda index, episode, arm: index == 0 and arm == "UK48")
    # Task zero retains one rather than two episodes; do not halve its weight.
    records = [r for r in records if not (r["task"] == a.FORMAL_TASKS[0] and r["episode_id"] == 1)]
    result = _small_analysis(records)
    assert result["paired_block_count"] == 31
    assert result["primary"]["estimate_pp"] == pytest.approx(100 / 16)
    assert result["primary"]["discordant_counts"]["pooled"]["denominator"] == 31
    assert result["primary"]["confidence_interval_95"]["per_task_episode_counts"][a.FORMAL_TASKS[0]] == 1


def test_adapter_numerically_reuses_parent_algorithm_without_mutating_parent():
    records = _records(50)
    table = a.validate_outcome_records(records)
    values = {(r["task"], r["episode_id"], r["arm"]): r["success"] for r in records}
    parent_table = parent.FormalOutcomeTable(tuple(tuple((
        values[(task, ep, "U")], False, values[(task, ep, "UK48")], values[(task, ep, "UN48")]
    ) for ep in range(50)) for task in a.FORMAL_TASKS))
    parent_arms_before = parent.ALL_ARMS
    effect = a._effect(table, "UK48", "U", 31, 41)
    assert effect["confidence_interval_95"] == parent.hierarchical_percentile_bootstrap_ci(parent_table, "OC", "U", replicates=31)
    assert effect["randomization_test"] == parent.paired_randomization_test(parent_table, "OC", "U", replicates=41)
    assert effect["estimate_pp"] == parent.paired_effect(parent_table, "OC", "U")["estimate_pp"]
    assert parent.ALL_ARMS == parent_arms_before == ("U", "O", "OC", "R")


def test_two_auxiliary_holm_only_and_reproducible_shared_schedule():
    records = _records()
    one = _small_analysis(records)
    two = _small_analysis(list(reversed(records)))
    assert one == two
    assert one["primary"]["comparison"] == "UK48 - U"
    assert "holm_adjusted_p_value" not in one["primary"]["randomization_test"]
    assert one["auxiliary_holm_family"]["family_size"] == 2
    assert one["auxiliary_holm_family"]["comparisons"] == ["UN48 - U", "UK48 - UN48"]
    raw = {name: value["randomization_test"]["p_value"] for name, value in one["auxiliary"].items()}
    assert one["auxiliary_holm_family"]["results"] == parent.holm_adjust(raw)
    for effect in [one["primary"], *one["auxiliary"].values()]:
        assert effect["confidence_interval_multiplicity_adjusted"] is False
        assert effect["randomization_test"]["seed"] == effect["confidence_interval_95"]["seed"] == 2026082502
        ptest = effect["randomization_test"]
        assert ptest["p_value"] == (1 + ptest["extreme_permutation_count"]) / 128


@pytest.mark.parametrize("estimate,lower,upper,stat,practical,ruled", [
    (4, 1, 7, True, True, False), (2, 1, 2.5, True, False, True),
    (3, 0, 6, False, False, False), (3, 0.1, 6, True, True, False),
    (1, -2, 4, False, False, False), (0, -1, 3, False, False, False),
])
def test_primary_decision_thresholds_without_random_control(estimate, lower, upper, stat, practical, ruled):
    result = a.classify_primary_result({"comparison": "UK48 - U", "estimate_pp": estimate,
                                       "confidence_interval_95": {"lower_pp": lower, "upper_pp": upper}})
    assert result["statistically_positive_primary"] is stat
    assert result["practically_positive_primary"] is practical
    assert result["practical_benefit_ruled_out"] is ruled
    assert result["requires_beating_random_control"] is False


def test_beating_random_only_never_counts_as_primary_success():
    records = _records(1, lambda _, __, arm: arm in ("U", "UK48"))
    result = _small_analysis(records)
    assert result["auxiliary"]["UK48 - UN48"]["estimate_pp"] == 100
    assert result["primary"]["estimate_pp"] == 0
    assert result["nonformal_decision_preview"]["practically_positive_primary"] is False


def _attestation(table):
    return {"status": "verified", "all_2400_outcomes_audited": True,
            "initial_inputs_states_text_matched": True, "seed_difficulty_mapping_matched": True,
            "checkpoint_control_pipeline_matched": True, "execution_provenance_reviewed": True,
            "same_run_three_arm_pairing_verified": True, "source_run_id": "synthetic-only",
            "comparison_audit_id": "synthetic-only", "raw_provenance_manifest_sha256": "a" * 64,
            "comparison_audit_sha256": "b" * 64, **a.outcome_binding_hashes(table)}


@pytest.mark.parametrize("bad", [None, True, {"status": "verified"}])
def test_formal_rejects_unverified_or_flag_only_run_attestation_before_statistics(bad, monkeypatch):
    monkeypatch.setattr(a, "_effect", lambda *args: pytest.fail("Statistics ran before run-attestation gate"))
    with pytest.raises(a.ExpansionAnalysisError):
        a.analyze_matched_outcomes(_records(50), run_attestation=bad)


def test_attestation_must_bind_exact_outcomes_and_provenance():
    table = a.validate_outcome_records(_records(50))
    valid = _attestation(table)
    assert a._validate_run_attestation(valid, table) == valid
    for field, value in (("u_outcomes_sha256", "f" * 64), ("all_outcomes_sha256", "f" * 64),
                         ("raw_provenance_manifest_sha256", None), ("comparison_audit_sha256", "PASS"),
                         ("initial_inputs_states_text_matched", 1), ("source_run_id", "")):
        invalid = {**valid, field: value}
        with pytest.raises(a.ExpansionAnalysisError):
            a._validate_run_attestation(invalid, table)
    changed = _records(50)
    changed[0]["success"] = not changed[0]["success"]
    with pytest.raises(a.ExpansionAnalysisError, match="bind"):
        a._validate_run_attestation(valid, a.validate_outcome_records(changed))
