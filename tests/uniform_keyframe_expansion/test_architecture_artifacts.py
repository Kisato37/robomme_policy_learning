"""Synthetic CPU evidence tests; these never certify an actual GPU run."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.uniform_keyframe_expansion import architecture_artifacts as aa
from experiments.uniform_keyframe_expansion import launch_contract as lc
from tests.uniform_keyframe_expansion.test_launch_contract import Fixture


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.architecture()
    gate = json.loads(Path(fixture.evidence["architecture_gate"]["path"]).read_text())
    raw = json.loads(Path(gate["measurements"]["path"]).read_text())
    plan = lc.validate_execution_plan(json.loads(Path(gate["execution"]["plan"]["path"]).read_text()))
    return fixture, gate, raw, plan


def test_bound_architecture_raw_arrays_and_execution(evidence):
    _, gate, raw, plan = evidence
    assert aa.validate_execution_record(gate, raw, plan.binding, deep=True)["raw_artifact_count"] == 9
    assert aa.validate_raw_arrays(raw) == 9


@pytest.mark.parametrize("mutation", ["missing_execution", "cleanup", "worker_pid", "model_gpu", "checkpoint", "input_scope", "worker_hash", "wrong_path"])
def test_architecture_execution_cannot_be_replaced_by_claims(evidence, mutation):
    _, gate, raw, plan = evidence
    if mutation == "missing_execution":
        gate.pop("execution")
    elif mutation == "input_scope":
        raw["no_task_execution"] = False
    elif mutation == "wrong_path":
        gate["execution"]["plan"]["path"] = gate["execution"]["controller_result"]["path"]
    else:
        key = {"cleanup": "controller_result", "worker_pid": "worker_ownership", "model_gpu": "policy_bootstrap",
               "checkpoint": "policy_bootstrap", "worker_hash": "worker_complete"}[mutation]
        previous = gate["execution"][key]
        path = Path(previous["path"])
        value = json.loads(path.read_text())
        if mutation == "cleanup": value["owned_process_cleanup_confirmed"] = False
        if mutation == "worker_pid": value["worker_identity"]["pid"] += 1
        if mutation == "model_gpu": value["model_gpu_uuid"] = "GPU-wrong"
        if mutation == "checkpoint": value["checkpoint_content_verified_in_this_process"] = False
        if mutation == "worker_hash": value["artifact_sha256"]["architecture_measurements.json"] = "a" * 64
        path.write_text(json.dumps(value))
        replacement = lc.file_reference(path)
        gate["sources"] = [replacement if item == previous else item for item in gate["sources"]]
        gate["execution"][key] = replacement
    with pytest.raises(lc.ExpansionLaunchError):
        aa.validate_execution_record(gate, raw, plan.binding)


def test_failed_controller_cannot_publish_even_with_complete_record(evidence):
    _, gate, raw, plan = evidence
    Path(gate["execution"]["plan"]["path"]).with_name("controller_failure.json").write_text("{}")
    with pytest.raises(lc.ExpansionLaunchError, match="Failed architecture"):
        aa.validate_execution_record(gate, raw, plan.binding)


def test_deep_audit_detects_corrupted_bytes_not_just_manifest(evidence):
    _, gate, raw, plan = evidence
    Path(raw["raw_artifacts"][0]["path"]).write_bytes(b"corrupted original array file")
    assert aa.validate_execution_record(gate, raw, plan.binding)["deep_checked"] is False
    with pytest.raises(lc.ExpansionLaunchError):
        aa.validate_execution_record(gate, raw, plan.binding, deep=True)


def _replace_array(raw, reference, name, transform):
    path = Path(reference["path"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[name] = transform(arrays[name])
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    fresh = lc.file_reference(path)
    reference.update(fresh)
    for entry in raw["raw_artifacts"]:
        if entry["path"] == str(path):
            entry.update(fresh, size_bytes=path.stat().st_size)


@pytest.mark.parametrize("mutation", ["different_action", "different_noise", "padding", "mask", "shape", "nonfinite", "dishonest_difference", "different_diag_input"])
def test_actual_arrays_override_passing_boolean_claims(evidence, mutation):
    _, _, raw, _ = evidence
    if mutation == "dishonest_difference":
        raw["padded_u_diagnostic"]["velocity_difference_linf"] += 3
    elif mutation == "different_diag_input":
        _replace_array(raw, raw["padded_u_diagnostic"]["artifact"], "image_512", lambda array: array + 1)
    else:
        reference = raw["cases"][0]["repeat_artifact"]
        if mutation == "different_action": _replace_array(raw, reference, "actions", lambda array: array + 1)
        if mutation == "different_noise": _replace_array(raw, reference, "initial_noise", lambda array: array + 1)
        if mutation == "padding": _replace_array(raw, reference, "raw_image", lambda array: array + 1)
        if mutation == "mask": _replace_array(raw, reference, "raw_mask", lambda array: array.astype(np.int32))
        if mutation == "shape": _replace_array(raw, reference, "actions", lambda array: array[:1])
        if mutation == "nonfinite": _replace_array(raw, reference, "actions", lambda array: np.full_like(array, np.nan))
    with pytest.raises(lc.ExpansionLaunchError):
        aa.validate_raw_arrays(raw)


def test_publisher_writes_once_after_all_real_evidence_checks(tmp_path, monkeypatch):
    from tests.uniform_keyframe_expansion.architecture_fixtures import create_architecture_fixture
    fixture = Fixture(tmp_path, monkeypatch)
    source = create_architecture_fixture(fixture, write_gate=False)
    plan, directory = source["plan"], source["directory"]
    target = directory / "architecture_gate.json"
    assert not target.exists()
    assert aa.publish_architecture_gate(plan, directory) == target
    lc._architecture_gate(lc.file_reference(target), plan.binding)
    original = target.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        aa.publish_architecture_gate(plan, directory)
    assert target.read_bytes() == original


def test_publisher_rejects_non_architecture_authority(evidence):
    fixture, gate, _, _ = evidence
    plan = lc.validate_execution_plan(fixture.plan("end_to_end_smoke"))
    with pytest.raises(lc.ExpansionLaunchError):
        aa.publish_architecture_gate(plan, Path(gate["measurements"]["path"]).parent)
