from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments.keyframe_neighborhood_sampling import recovery
from experiments.keyframe_neighborhood_sampling import aggregate_recovery
from experiments.keyframe_neighborhood_sampling import submit_direct as runner
from experiments.keyframe_neighborhood_sampling.runner_contract import write_once_record
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError, sha256_file


@pytest.fixture
def recovery_plan(tmp_path):
    root = tmp_path / "recovery"
    (root / "protocol").mkdir(parents=True)
    parent = tmp_path / recovery.PARENT_RUN
    (parent / "protocol").mkdir(parents=True)
    write_once_record(parent / "protocol/evidence.json", {"original": True})
    write_once_record(root / "protocol/launch_manifest.json", {"repository": {"commit_sha": "b" * 40}})
    missing = sorted([1352, 1354, *range(1398, 1600)])
    retained = sorted(set(range(1600)) - set(missing))
    inventory = recovery.inventory(parent)
    plan = {"schema": "keyframe-neighborhood-recovery-v1", "authorized": True,
            "run_root": str(root), "parent_root": str(parent), "repository_commit_sha": "b" * 40,
            "launch_manifest_sha256": sha256_file(root / "protocol/launch_manifest.json"),
            "parent_audit": {"passed": True, "retained_row_ids": retained, "missing_row_ids": missing,
                             "inventory": inventory, "inventory_sha256": recovery.digest(inventory)},
            "row_ids": missing, "global_retry_offsets": {str(i): int(i in recovery.INCIDENT_ROWS) for i in missing}}
    write_once_record(root / "protocol/recovery_plan.json", plan)
    return root, parent, plan


def test_original_full_matrix_is_default(tmp_path):
    assert recovery.initial_rows(tmp_path) == list(range(1600))


def test_recovery_authorizes_only_missing_complement(recovery_plan):
    root, _, plan = recovery_plan
    assert recovery.initial_rows(root, verify_parent=True) == plan["row_ids"]
    assert len(plan["row_ids"]) == 204
    assert not set(plan["row_ids"]) & set(plan["parent_audit"]["retained_row_ids"])


@pytest.mark.parametrize("fault", ["overlap", "duplicate", "offset", "commit", "authorization", "launch", "inventory"])
def test_recovery_plan_rejects_tampering(recovery_plan, fault):
    root, _, plan = recovery_plan
    if fault == "overlap":
        plan["row_ids"][0] = plan["parent_audit"]["retained_row_ids"][0]
    elif fault == "duplicate":
        plan["row_ids"][1] = plan["row_ids"][0]
    elif fault == "offset":
        plan["global_retry_offsets"]["1447"] = 0
    elif fault == "commit":
        plan["repository_commit_sha"] = "c" * 40
    elif fault == "authorization":
        plan["authorized"] = False
    elif fault == "launch":
        plan["launch_manifest_sha256"] = "d" * 64
    else:
        plan["parent_audit"]["inventory_sha256"] = "e" * 64
    (root / "protocol/recovery_plan.json").write_text(json.dumps(plan))
    with pytest.raises(ArtifactContractError):
        recovery.validate_recovery_plan(root)


def test_sealed_parent_may_not_change(recovery_plan):
    root, parent, _ = recovery_plan
    (parent / "protocol/evidence.json").write_text('{"original":false}')
    with pytest.raises(ArtifactContractError, match="Sealed parent changed"):
        recovery.validate_recovery_plan(root, verify_parent=True)


def test_new_root_does_not_reset_retry_allowance(recovery_plan):
    root, _, _ = recovery_plan
    recovery.validate_retry_budget(root, [1447], 1)
    recovery.validate_retry_budget(root, [1500], 2)
    with pytest.raises(ArtifactContractError, match="global retry"):
        recovery.validate_retry_budget(root, [1447], 2)
    with pytest.raises(ArtifactContractError):
        recovery.validate_retry_budget(root, [0], 0)


def test_completed_results_need_exact_full_union():
    with pytest.raises(ArtifactContractError, match="1,600"):
        aggregate_recovery.summarize({}, plan={}, provenance={}, child_context={})


def test_incomplete_child_cannot_publish(recovery_plan):
    root, _, plan = recovery_plan
    with pytest.raises(ArtifactContractError, match="no partial"):
        aggregate_recovery.audit_recovery_attempts(root, plan, {}, {})


def test_cleanup_before_reset_records_null_not_fake_success(tmp_path):
    execution = runner.Execution.__new__(runner.Execution)
    execution.directory = tmp_path
    execution.resident = object()
    execution.digest = "a" * 64
    execution.dispatch = SimpleNamespace(execution_id="failure-fixture")
    execution.roles = {}
    cleaned = []
    execution.cleanup = lambda: cleaned.append(True)
    write_once_record(tmp_path / "policy_session.json", {"fixture": "binding"})
    execution.complete()
    completion = json.loads((tmp_path / "completion.json").read_text())
    assert cleaned == [True]
    assert completion["resident_policy"] == {"binding_sha256": sha256_file(tmp_path / "policy_session.json"),
                                              "reset_sha256": None}
    assert not (tmp_path / "resident_reset.json").exists()


def test_checkpoint_drift_is_rejected_before_source_audit(tmp_path):
    parent = {"repository": {"commit_sha": recovery.PARENT_COMMIT}, "checkpoint_path": "frozen"}
    child = {**parent, "checkpoint_path": "different"}
    with pytest.raises(ArtifactContractError, match="checkpoint_path"):
        recovery.source_contract(parent, child, tmp_path)
