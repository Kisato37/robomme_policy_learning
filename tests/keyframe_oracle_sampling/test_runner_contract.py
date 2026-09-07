from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json

import pytest

from experiments.keyframe_neighborhood_sampling.runner_contract import DirectDispatch
from experiments.keyframe_neighborhood_sampling.runner_contract import RunnerContractError
from experiments.keyframe_neighborhood_sampling.runner_contract import process_exit_record
from experiments.keyframe_neighborhood_sampling.runner_contract import process_start_record
from experiments.keyframe_neighborhood_sampling.runner_contract import record_digest
from experiments.keyframe_neighborhood_sampling.runner_contract import validate_runtime_binding
from experiments.keyframe_neighborhood_sampling.runner_contract import write_once_record

GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture
def dispatch(tmp_path):
    return DirectDispatch(
        execution_id="12345678-1234-4234-8234-123456789abc",
        run_root=str(tmp_path / "runs/keyframe_neighborhood_sampling/fresh-run"),
        repository_commit_sha="a" * 40,
        stage="development_smoke",
        launch_manifest_sha256="b" * 64,
        submission_plan_sha256="c" * 64,
        matrix_sha256="d" * 64,
        attempt_id=0,
        row_id=0,
        shard_id=None,
        gpu_uuids=(GPU_A, GPU_B),
        host_name="fixture-host",
        host_boot_id="12345678-1234-4234-8234-123456789def",
        policy_port=30000,
    )


def test_dispatch_is_explicit_and_round_trips_without_slurm_or_launch_permission(dispatch):
    record = dispatch.as_record()
    assert record["backend"] == "direct"
    assert record["grants_launch_authority"] is False
    assert not any("slurm" in field.lower() for field in record)
    assert DirectDispatch.from_record(json.loads(json.dumps(record))) == dispatch
    assert validate_runtime_binding(record, expected_sha256=record_digest(record), observed=dispatch) == dispatch


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "slurm"),
        ("backend", None),
        ("execution_schema", "future-unknown"),
        ("grants_launch_authority", True),
        ("grants_launch_authority", 0),
        ("protocol_family", "keyframe_oracle_sampling"),
        ("row_id", True),
        ("attempt_id", 3),
        ("gpu_uuids", [GPU_A, GPU_A]),
        ("gpu_uuids", ["0", "1"]),
        ("gpu_uuids", [GPU_A]),
        ("gpu_layout", None),
        ("gpu_layout", "automatic"),
        ("gpu_layout", "colocated"),
        ("policy_port", True),
        ("policy_port", 80),
        ("execution_id", "PID-123"),
        ("run_root", None),
        ("run_root", "/tmp/runs/keyframe_neighborhood_sampling/../old-run"),
    ],
)
def test_dispatch_rejects_ambiguous_or_invalid_identity(dispatch, field, value):
    record = {**dispatch.as_record(), field: value}
    with pytest.raises(RunnerContractError):
        DirectDispatch.from_record(record)


def test_missing_backend_and_extra_slurm_field_fail_closed(dispatch):
    record = dispatch.as_record()
    record.pop("backend")
    with pytest.raises(RunnerContractError, match="missing or unknown"):
        DirectDispatch.from_record(record)
    with pytest.raises(RunnerContractError, match="Slurm"):
        DirectDispatch.from_record({**dispatch.as_record(), "slurm_job_id": "123"})


def test_changed_dispatch_bytes_fail_even_when_observed_values_match(dispatch):
    altered = replace(dispatch, policy_port=30001)
    with pytest.raises(RunnerContractError, match="changed after"):
        validate_runtime_binding(
            altered.as_record(), expected_sha256=record_digest(dispatch.as_record()), observed=altered
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"row_id": 1},
        {"attempt_id": 1},
        {"repository_commit_sha": "f" * 40},
        {"host_name": "another-host"},
        {"host_boot_id": "99999999-1234-4234-8234-123456789def"},
        {"gpu_uuids": (GPU_B, GPU_A)},
    ],
)
def test_runtime_must_match_all_dispatch_identity(dispatch, updates):
    with pytest.raises(RunnerContractError, match="Live execution"):
        validate_runtime_binding(
            dispatch.as_record(),
            expected_sha256=record_digest(dispatch.as_record()),
            observed=replace(dispatch, **updates),
        )


def test_architecture_and_formal_have_explicit_distinct_resource_shapes(dispatch):
    architecture = replace(dispatch, stage="architecture_smoke", gpu_uuids=(GPU_A,), row_id=None, policy_port=None)
    assert DirectDispatch.from_record(architecture.as_record()) == architecture
    with pytest.raises(RunnerContractError, match="Architecture"):
        replace(architecture, attempt_id=1)
    with pytest.raises(RunnerContractError, match="Formal"):
        replace(dispatch, stage="formal")
    assert replace(dispatch, stage="formal", shard_id=0).shard_id == 0


def test_colocated_layout_explicitly_records_both_roles_on_one_physical_gpu(dispatch):
    colocated = replace(dispatch, gpu_layout="colocated", gpu_uuids=(GPU_A, GPU_A))
    assert colocated.as_record()["gpu_layout"] == "colocated"
    assert colocated.as_record()["gpu_uuids"] == [GPU_A, GPU_A]
    assert DirectDispatch.from_record(colocated.as_record()) == colocated
    assert replace(colocated, stage="formal", shard_id=0).gpu_uuids == (GPU_A, GPU_A)
    architecture = replace(colocated, stage="architecture_smoke", gpu_uuids=(GPU_A,), row_id=None, policy_port=None)
    assert DirectDispatch.from_record(architecture.as_record()).gpu_layout == "colocated"
    record = colocated.as_record()
    del record["gpu_layout"]
    with pytest.raises(RunnerContractError, match="layout"):
        DirectDispatch.from_record(record)


def test_legacy_separate_dispatch_keeps_original_bytes_and_lifecycle_digest(dispatch):
    assert dispatch.as_record()["gpu_layout"] == "separate"
    legacy = dispatch.as_record()
    del legacy["gpu_layout"]
    restored = DirectDispatch.from_record(legacy)
    assert restored == dispatch
    assert restored.gpu_layout == "separate"
    assert restored.as_record() == legacy
    exited = process_exit_record(
        restored,
        started_record_sha256="e" * 64,
        returncode=0,
        wall_clock_limit_reached=False,
    )
    assert exited["dispatch_sha256"] == record_digest(legacy)
    assert validate_runtime_binding(legacy, expected_sha256=record_digest(legacy), observed=dispatch) == dispatch


def test_write_once_is_exclusive_across_competing_writers(tmp_path):
    path = tmp_path / "dispatch.json"

    def publish(value):
        try:
            return write_once_record(path, {"writer": value})
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(publish, range(4)))
    winners = [digest for digest in outcomes if digest is not None]
    assert len(winners) == 1
    assert winners[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text())["writer"] in range(4)
    assert list(tmp_path.iterdir()) == [path]


def test_start_and_exit_are_linked_process_facts_not_scientific_outcomes(dispatch):
    process = {
        "boot_id": dispatch.host_boot_id,
        "pid": 999,
        "parent_pid": 998,
        "process_group": 999,
        "session": 999,
        "start_ticks": 100,
        "uid": 1000,
    }
    started = process_start_record(
        dispatch, process_identity=process, command=["/fixture/python", "worker.py"], working_directory="/fixture"
    )
    assert started["dispatch_sha256"] == record_digest(dispatch.as_record())
    with pytest.raises(RunnerContractError, match="different host boot"):
        process_start_record(
            dispatch,
            process_identity={**process, "boot_id": "another-boot"},
            command=["python"],
            working_directory="/fixture",
        )
    for returncode in (0, 1, -15, -9):
        exit_record = process_exit_record(
            dispatch,
            started_record_sha256=record_digest(started),
            returncode=returncode,
            wall_clock_limit_reached=returncode == -9,
        )
        assert exit_record["started_record_sha256"] == record_digest(started)
        assert exit_record["scientific_result_validated"] is False
        assert exit_record["retry_authorized"] is False
        assert "success" not in exit_record
