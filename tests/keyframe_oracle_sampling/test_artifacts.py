from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest

from experiments.keyframe_oracle_sampling.artifacts import (
    ArtifactContractError,
    EpisodeAttemptWriter,
    RunArtifactStore,
    ScientificKey,
    atomic_write_json,
    audit_attempt,
    audit_initial_condition_fairness,
    audit_paired_manifest_invariants,
    build_seed_table,
    build_smoke_seed_table,
    completeness_report,
    expected_keys,
    is_retryable_infrastructure_exception,
    to_jsonable,
    validate_seed_table,
    validate_smoke_formal_seed_disjointness,
)
from mme_vla_suite.shared.keyframe_oracle_sampling import (
    FORMAL_SEED_SCOPE,
    SMOKE_SEED_SCOPE,
)


def test_structured_logging_normalizes_nested_arrays_scalars_and_nonfinite_values():
    payload = {
        "nested": [np.asarray([1, 2]), {"x": np.float32(3.5)}],
        "nan": float("nan"),
        "positive": float("inf"),
        "negative": float("-inf"),
    }
    normalized = to_jsonable(payload)
    assert normalized["nested"] == [[1, 2], {"x": 3.5}]
    assert normalized["nan"] == {"__nonfinite_float__": "NaN"}
    assert normalized["positive"] == {"__nonfinite_float__": "+Infinity"}
    assert normalized["negative"] == {"__nonfinite_float__": "-Infinity"}
    json.dumps(normalized, allow_nan=False)


def test_atomic_artifacts_are_write_once(tmp_path):
    target = tmp_path / "artifact.json"
    atomic_write_json(target, {"value": 1})
    with pytest.raises(FileExistsError, match="overwrite"):
        atomic_write_json(target, {"value": 2})
    assert json.loads(target.read_text()) == {"value": 1}
    assert not list(tmp_path.glob(".artifact.json.*"))


def test_seed_table_is_deterministic_complete_and_rejects_duplicates():
    first = build_seed_table(["InsertPeg"], [0])
    second = build_seed_table(["InsertPeg"], [0])
    assert first == second
    assert first["scope"] == FORMAL_SEED_SCOPE
    lookup = validate_seed_table(first, expected_scope=FORMAL_SEED_SCOPE)
    assert len(lookup) == 82

    duplicated = json.loads(json.dumps(first))
    duplicated["entries"].append(dict(duplicated["entries"][0]))
    from experiments.keyframe_oracle_sampling.artifacts import sha256_payload

    duplicated["entry_count"] = len(duplicated["entries"])
    duplicated["entries_sha256"] = sha256_payload(duplicated["entries"])
    with pytest.raises(ArtifactContractError, match="Duplicate"):
        validate_seed_table(duplicated)


def test_smoke_seed_table_scope_formula_and_dataset_are_strict():
    smoke = build_smoke_seed_table(["InsertPeg"], [0])
    assert smoke["scope"] == SMOKE_SEED_SCOPE
    lookup = validate_seed_table(smoke, expected_scope=SMOKE_SEED_SCOPE)
    assert lookup[("InsertPeg", 0, 0)] == 1726467455513403504

    with pytest.raises(ArtifactContractError, match="does not match required"):
        validate_seed_table(smoke, expected_scope=FORMAL_SEED_SCOPE)

    wrong_formula = json.loads(json.dumps(smoke))
    wrong_formula["derivation"] = "SHA256(wrong-formula)"
    with pytest.raises(ArtifactContractError, match="wrong derivation formula"):
        validate_seed_table(wrong_formula)

    wrong_dataset = json.loads(json.dumps(smoke))
    wrong_dataset["dataset"] = "test"
    with pytest.raises(ArtifactContractError, match="requires dataset"):
        validate_seed_table(wrong_dataset)


def test_complete_smoke_seed_universe_is_unique_and_disjoint_from_formal():
    from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS

    smoke = build_smoke_seed_table(FORMAL_TASKS, [0])
    formal = build_seed_table(FORMAL_TASKS, range(50))
    assert validate_smoke_formal_seed_disjointness(smoke, formal) == {
        "smoke_seed_count": 1312,
        "formal_seed_count": 65600,
        "intersection_count": 0,
    }


def test_attempt_resume_trace_immutability_and_digest_validation(tmp_path):
    key = ScientificKey("InsertPeg", 0, "OC", "short")
    writer = EpisodeAttemptWriter(tmp_path / "attempt_00", key, 0)
    writer.create({"dataset": "val"})
    assert writer.validate_resume() == "incomplete"
    writer.record_initial_conditions({"front": "a"})
    writer.append_trace({"selected": np.asarray([0, 4, 9])})
    writer.finalize({"success": False, "terminal_reason": "timeout"})
    assert writer.validate_resume() == "complete"
    with pytest.raises(ArtifactContractError, match="immutable"):
        writer.append_trace({"selected": [0]})
    with pytest.raises(FileExistsError, match="overwrite"):
        writer.finalize({"success": True})

    writer.trace_path.write_text('{"tampered":true}\n')
    with pytest.raises(ArtifactContractError, match="digest mismatch"):
        writer.validate_resume()


def test_attempt_requires_benchmark_evidence_for_a_scientific_error(tmp_path):
    key = ScientificKey("InsertPeg", 0, "U", "short")
    writer = EpisodeAttemptWriter(tmp_path / "attempt_00", key, 0)
    writer.create({"dataset": "val"})
    writer.record_initial_conditions({"front": "a"})

    with pytest.raises(ArtifactContractError, match="FailAwareWrapper error evidence"):
        writer.finalize({"success": False, "terminal_reason": "error"})

    writer.finalize(
        {
            "success": False,
            "terminal_reason": "error",
            "benchmark_error_message": "caught IK failure",
            "benchmark_exception_type": "RuntimeError",
        }
    )


def test_run_store_rejects_duplicate_completed_scientific_keys(tmp_path):
    store = RunArtifactStore(tmp_path)
    key = ScientificKey("InsertPeg", 0, "U", "short")
    first = store.new_attempt(key, 0, {"dataset": "val"})
    first.record_initial_conditions({"front": "a"})
    first.append_trace({"call": 0})
    first.finalize({"success": False, "terminal_reason": "timeout"})
    with pytest.raises(ArtifactContractError, match="already has"):
        store.new_attempt(key, 1, {"dataset": "val"})

    duplicate = EpisodeAttemptWriter(store.attempt_dir(key, 2), key, 2)
    duplicate.create({"dataset": "val"})
    duplicate.record_initial_conditions({"front": "a"})
    duplicate.append_trace({"call": 0})
    duplicate.finalize({"success": False, "terminal_reason": "timeout"})
    with pytest.raises(ArtifactContractError, match="Duplicate completed"):
        store.scan_completed_keys()


def test_failure_ledger_is_separate_and_fsynced_append_only(tmp_path):
    store = RunArtifactStore(tmp_path)
    store.record_failure({"task": "InsertPeg", "error": "node lost"})
    store.record_failure({"task": "RouteStick", "error": "transport"})
    records = [json.loads(line) for line in store.failures_path.read_text().splitlines()]
    assert [record["error"] for record in records] == ["node lost", "transport"]
    assert store.scan_completed_keys() == {}


def test_completeness_requires_exact_matrix_without_unexpected_keys(tmp_path):
    expected = expected_keys(["InsertPeg"], [0], ["U", "OC"], trajectory_kind="short")
    completed_key = ScientificKey("InsertPeg", 0, "U", "short")
    report = completeness_report(expected, {completed_key: tmp_path / "result.json"})
    assert report["complete"] is False
    assert report["completed_count"] == 1
    assert report["missing"] == [ScientificKey("InsertPeg", 0, "OC", "short").as_dict()]


def test_trace_duplicate_rejection_and_attempt_audit(tmp_path):
    key = ScientificKey("InsertPeg", 0, "U", "short")
    writer = EpisodeAttemptWriter(tmp_path / "attempt_00", key, 0)
    writer.create({"dataset": "val"})
    writer.record_initial_conditions({"front": "a"})
    selected = [0, 1]
    selected_hash = hashlib.sha256(b"[0,1]").hexdigest()
    trace = {
        "policy_call_index": 0,
        "history_length": 2,
        "current_history_index": 1,
        "selected_frame_indices": selected,
        "selected_indices_sha256": selected_hash,
        "valid_frame_count": 2,
        "padding_frame_count": 30,
        "valid_memory_token_count": 32,
        "selector_latency_ms": 0.25,
    }
    writer.append_trace(trace)
    with pytest.raises(ArtifactContractError, match="Duplicate selector-trace"):
        writer.append_trace(trace)
    writer.finalize({"success": False, "terminal_reason": "timeout"})
    report = audit_attempt(writer)
    assert report["status"] == "complete"
    assert report["policy_call_count"] == 1
    assert report["selector_latency"] == {
        "count": 1,
        "mean_ms": 0.25,
        "p50_ms": 0.25,
        "p95_ms": 0.25,
        "p99_ms": 0.25,
        "max_ms": 0.25,
    }


def test_initial_condition_fairness_audit_requires_all_arms_and_identical_hashes():
    hashes = {"front": "a", "wrist": "b", "robot": "c", "task": "d"}
    manifests = [
        {
            "scientific_key": ScientificKey("InsertPeg", 0, arm, "short").as_dict(),
            "initial_condition_hashes": hashes,
        }
        for arm in ("U", "O", "OC", "R")
    ]
    assert audit_initial_condition_fairness(manifests) == {
        "paired_blocks": 1,
        "fair": True,
    }
    manifests[-1]["initial_condition_hashes"] = {**hashes, "front": "changed"}
    with pytest.raises(ArtifactContractError, match="hash mismatch"):
        audit_initial_condition_fairness(manifests)


def test_paired_manifest_invariants_reject_treatment_confounds():
    common = {
        "dataset": "val",
        "max_steps": 64,
        "executed_action_horizon": 16,
        "evaluation_policy_seed": 7,
        "checkpoint_id": 79999,
        "seed_table_sha256": "a" * 64,
        "resolved_environment_seed": 123,
        "resolved_difficulty_hint": "hard",
        "difficulty": "hard",
    }
    manifests = [
        {
            **common,
            "scientific_key": ScientificKey("InsertPeg", 0, arm, "short").as_dict(),
        }
        for arm in ("U", "O", "OC", "R")
    ]
    assert audit_paired_manifest_invariants(manifests) == {
        "paired_blocks": 1,
        "invariants_match": True,
    }
    manifests[-1]["checkpoint_id"] = 1
    with pytest.raises(ArtifactContractError, match="Non-treatment"):
        audit_paired_manifest_invariants(manifests)


def test_retry_requires_immediately_preceding_allowed_failure(tmp_path):
    store = RunArtifactStore(tmp_path)
    key = ScientificKey("InsertPeg", 0, "U", "short")
    attempt_0 = store.new_attempt(key, 0, {"dataset": "val"})
    with pytest.raises(ArtifactContractError, match="failure ledger"):
        store.new_attempt(key, 1, {"dataset": "val"})
    store.record_failure(
        {
            **key.as_dict(),
            "attempt_id": 0,
            "classification": "infrastructure",
            "retry_allowed": True,
        }
    )
    attempt_1 = store.new_attempt(key, 1, {"dataset": "val"})
    assert attempt_0.attempt_dir.name == "attempt_00"
    assert attempt_1.attempt_dir.name == "attempt_01"


class ConnectionClosedFake(Exception):
    __module__ = "websockets.exceptions"


def test_retryable_infrastructure_classifier_includes_websocket_disconnects():
    assert is_retryable_infrastructure_exception(ConnectionError("gone"))
    assert is_retryable_infrastructure_exception(ConnectionClosedFake("gone"))
    assert not is_retryable_infrastructure_exception(ValueError("selector bug"))
