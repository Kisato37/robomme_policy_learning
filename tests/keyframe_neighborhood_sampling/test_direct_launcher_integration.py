"""CPU-only integration of exact direct plans, production audits and scheduling.

The synthetic evidence tests submission/lifecycle contracts, not scientific
results or GPU readiness. No production provenance validator is replaced.
"""
# ruff: noqa: SLF001
# Tests intentionally call the production attempt-authorization audit seam.
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import threading
import uuid

import pytest

from experiments.keyframe_neighborhood_sampling import aggregate_formal
from experiments.keyframe_neighborhood_sampling import direct_provenance
from experiments.keyframe_neighborhood_sampling import submit_direct
from experiments.keyframe_neighborhood_sampling.formal_matrix import build_formal_matrix
from experiments.keyframe_neighborhood_sampling.formal_matrix import validate_formal_runtime_row_binding
from experiments.keyframe_neighborhood_sampling.resident_policy import RESET_STATE
from experiments.keyframe_neighborhood_sampling.runner_contract import DirectDispatch
from experiments.keyframe_neighborhood_sampling.runner_contract import canonical_bytes
from experiments.keyframe_neighborhood_sampling.runner_contract import process_exit_record
from experiments.keyframe_neighborhood_sampling.runner_contract import process_start_record
from experiments.keyframe_neighborhood_sampling.runner_contract import write_once_record
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now

GPU_PAIRS = [[f"GPU-{digit * 8}-{digit * 4}-{digit * 4}-{digit * 4}-{digit * 12}" for digit in pair] for pair in ("12", "34")]
BOOT_ID = "55555555-5555-4555-8555-555555555555"


@dataclasses.dataclass
class FullPlan:
    root: Path
    launch: dict
    plan: dict
    records: list[tuple[Path, dict]]
    matrix: dict


@pytest.fixture
def full_plan(tmp_path, request):
    resident = getattr(request, "param", False)
    pairs = [[pair[0], pair[0]] for pair in GPU_PAIRS] if resident else GPU_PAIRS
    layout = {"gpu_layout": "colocated"} if resident else {}
    root = tmp_path / "runs/keyframe_neighborhood_sampling/direct-integration"
    protocol = root / "protocol"
    protocol.mkdir(parents=True)
    matrix = build_formal_matrix()
    write_once_record(protocol / "formal_matrix.json", matrix)
    profile = {
        "environment_sh": "/fixture/environment.sh", "environment_sh_sha256": "e" * 64,
        "graphics_wrapper": "/fixture/graphics.sh", "graphics_wrapper_sha256": "f" * 64,
        "lock_directory": "/fixture/stable-gpu-locks",
        **layout, **({"policy_lifetime": "resident"} if resident else {}),
    }
    identity = {
        "schema_version": 1, "protocol_version": "v1.0",
        "protocol_family": "keyframe_neighborhood_sampling_v1",
        "runner_backend": "direct", "repository_commit_sha": "a" * 40,
        "run_root": str(root), "formal_launch_authorized": True,
        **layout,
    }
    launch = {
        **identity, "repository": {"commit_sha": "a" * 40},
        "formal_matrix_sha256": sha256_file(protocol / "formal_matrix.json"),
        "architecture_report_sha256": "b" * 64,
        "development_smoke_audit_sha256": "c" * 64,
        "checkpoint_unpacked_metadata_sha256": "d" * 64,
        "checkpoint_content_tree_algorithm": "sha256-canonical-file-content-tree-v1",
        "checkpoint_unpacked_content_tree_sha256": "e" * 64,
    }
    write_once_record(protocol / "launch_manifest.json", launch)
    shards = []
    records = []
    command = ["/fixture/policy/python", "-m", "experiments.keyframe_neighborhood_sampling.submit_direct", "--stage", "formal"]
    for shard_id, start in enumerate((0, 1000)):
        rows = list(range(start, min(start + 1000, 1600)))
        shard = {
            "shard_id": shard_id, "shard_count": 2, "row_ids": rows,
            "array_task_ids": list(range(len(rows))),
            "array": f"0-{len(rows) - 1}%1", "command": command,
        }
        shards.append(shard)
        record = {
            **identity, **shard, "direct_run_id": str(uuid.uuid4()),
            "attempt_id": 0, "trajectory_count": len(rows), "max_concurrent": 1,
            "global_max_concurrent": 2, "gpu_pairs": pairs, "runtime_profile": profile,
            "launch_manifest_sha256": sha256_file(protocol / "launch_manifest.json"),
            **{key: value for key, value in launch.items() if key.startswith(("checkpoint_", "formal_matrix_", "architecture_report_", "development_smoke_audit_"))},
        }
        records.append((protocol / f"submission_record_shard_{shard_id:02d}.json", record))
    plan = {
        **identity, "direct_run_id": str(uuid.uuid4()), "attempt_id": 0,
        "trajectory_count": 1600, "shard_count": 2, "max_rows_per_array": 1000,
        "global_max_concurrent": 2, "gpu_pairs": pairs, "runtime_profile": profile,
        "shards": shards,
    }
    # Exercise the real publication boundary: plan first, then exact record
    # bytes with the plan SHA injected by production code.
    submit_direct.publish_submission(root, records, plan)
    return FullPlan(root, launch, plan, records, matrix)


def _authorizations(bundle):
    return aggregate_formal.validate_submission_attempt(
        bundle.root, attempt_id=0, expected_row_ids=list(range(1600)), launch=bundle.launch,
    )


def _completed_manifest(bundle, row_id):
    shard_id = int(row_id >= 1000)
    path, submission = bundle.records[shard_id]
    dispatch = DirectDispatch(
        execution_id=str(uuid.uuid4()), run_root=str(bundle.root), repository_commit_sha="a" * 40,
        stage="formal", launch_manifest_sha256=submission["launch_manifest_sha256"],
        submission_plan_sha256=sha256_file(path), matrix_sha256=bundle.launch["formal_matrix_sha256"],
        attempt_id=0, row_id=row_id, shard_id=shard_id, gpu_uuids=tuple(submission["gpu_pairs"][shard_id]),
        host_name="synthetic-integration-host", host_boot_id=BOOT_ID, policy_port=30000 + row_id,
        gpu_layout=submission.get("gpu_layout", "separate"),
    )
    dispatch_path = direct_provenance.dispatch_path(bundle.root, stage="formal", attempt_id=0, row_id=row_id)
    dispatch_path.parent.mkdir(parents=True)
    dispatch_hash = write_once_record(dispatch_path, dispatch.as_record())
    runner = {
        "backend": "direct", "dispatch_path": str(dispatch_path),
        "dispatch_sha256": dispatch_hash, "dispatch": dispatch.as_record(),
    }
    resident = submission["runtime_profile"].get("policy_lifetime") == "resident"
    resident_links = {}
    if resident:
        session = dataclasses.replace(dispatch, execution_id=str(uuid.uuid4()), stage="policy_session",
                                      row_id=None, shard_id=None, policy_port=29999)
        session_path = direct_provenance.dispatch_path(bundle.root, stage="policy_session", attempt_id=0,
                                                       row_id=None, execution_id=session.execution_id)
        session_path.parent.mkdir(parents=True)
        session_hash = write_once_record(session_path, session.as_record())
        model_start_hash = write_once_record(session_path.parent / "policy_start.json", process_start_record(
            session, process_identity={"boot_id": BOOT_ID, "pid": 800, "parent_pid": os.getpid(),
                                      "process_group": 800, "session": 800, "start_ticks": 1, "uid": os.getuid()},
            command=["/fixture/python", "real-resident-model"], working_directory="/fixture"))
        binding_hash = write_once_record(dispatch_path.parent / "policy_session.json", {
            "schema": "resident-policy-binding-v1", "row_execution_id": dispatch.execution_id,
            "row_dispatch_sha256": dispatch_hash, "policy_start_sha256": model_start_hash,
            "session": {"backend": "direct", "dispatch_path": str(session_path),
                        "dispatch_sha256": session_hash, "dispatch": session.as_record()},
        })
        write_once_record(session_path.parent / "row_plan.json", {
            "stage": "formal", "submission_shard_id": shard_id, "row_ids": submission["row_ids"],
        })
        reset_hash = write_once_record(dispatch_path.parent / "resident_reset.json", {
            "schema": "resident-reset-v1", "row_execution_id": dispatch.execution_id,
            "session_execution_id": session.execution_id, "binding_sha256": binding_hash,
            "state": dict(RESET_STATE), "model_process_pid": 801,
            "selector_config": {key: bundle.matrix["rows"][row_id][key] for key in ("task", "episode_id", "arm")},
            "recorded_utc": utc_now(),
        })
        resident_links = {"resident_policy": {"binding_sha256": binding_hash, "reset_sha256": reset_hash}}
    links = {}
    for index, role in enumerate(("preflight", "policy", "evaluator", "reconcile")):
        pid = 900 + index
        start = process_start_record(
            dispatch,
            process_identity={
                "boot_id": BOOT_ID, "pid": pid, "parent_pid": os.getpid(),
                "process_group": pid, "session": pid, "start_ticks": 100 + index, "uid": os.getuid(),
            },
            command=["/fixture/python", role], working_directory="/fixture",
        )
        start_hash = write_once_record(dispatch_path.parent / f"{role}_start.json", start)
        exit_record = process_exit_record(
            dispatch, started_record_sha256=start_hash, returncode=-15 if role == "policy" else 0,
            wall_clock_limit_reached=False,
        )
        exit_hash = write_once_record(dispatch_path.parent / f"{role}_exit.json", exit_record)
        links[role] = {"start_sha256": start_hash, "exit_sha256": exit_hash}
    write_once_record(dispatch_path.parent / "completion.json", {
        "backend": "direct", "execution_id": dispatch.execution_id,
        "dispatch_sha256": dispatch_hash, "cleanup_confirmed": True, "roles": links,
        "finished_utc": utc_now(),
        **resident_links,
    })
    if resident:
        exit_hash = write_once_record(session_path.parent / "policy_exit.json", process_exit_record(
            session, started_record_sha256=model_start_hash, returncode=-15, wall_clock_limit_reached=False))
        write_once_record(session_path.parent / "completion.json", {
            "backend": "direct", "execution_id": session.execution_id, "dispatch_sha256": session_hash,
            "cleanup_confirmed": True, "roles": {"policy": {"start_sha256": model_start_hash, "exit_sha256": exit_hash}},
            "finished_utc": utc_now(),
        })
    manifest = {
        "runner_backend": "direct", "runner": runner, "environment_setup_completed": True,
        "renderer_device": {
            "render_backend": "sapien_cuda", "simulation_backend": "physx_cpu",
            "cuda_device_id": 0, "pci_bus_id": "0000:41:00.0",
            "gpu_uuid": dispatch.gpu_uuids[1], "expected_gpu_uuid": dispatch.gpu_uuids[1],
            "can_render": True, "is_cuda": True, "matches_dispatch": True,
        },
    }
    return manifest, dispatch


def test_full_1600_direct_plan_and_both_shards_pass_production_authorization_audit(full_plan):
    summary, rows = _authorizations(full_plan)
    assert summary["trajectory_count"] == 1600
    assert summary["shard_count"] == 2
    assert summary["runner_backend"] == "direct"
    assert sorted(rows) == list(range(1600))
    assert [len(record["row_ids"]) for _, record in full_plan.records] == [1000, 600]
    assert [rows[row]["array_task_id"] for row in (0, 999, 1000, 1599)] == [0, 999, 0, 599]
    assert [rows[row]["shard_id"] for row in (0, 999, 1000, 1599)] == [0, 0, 1, 1]
    for row_id, authorization in rows.items():
        assert authorization["submission_record_sha256"] == sha256_file(full_plan.records[int(row_id >= 1000)][0])
        assert "slurm_array_job_id" not in authorization


@pytest.mark.parametrize("full_plan", [False, True], indirect=True)
@pytest.mark.parametrize("row_id", [0, 999, 1000, 1599])
def test_both_shard_boundaries_bind_real_runtime_then_offline_completion(full_plan, monkeypatch, row_id):
    _, authorizations = _authorizations(full_plan)
    manifest, dispatch = _completed_manifest(full_plan, row_id)
    runner = manifest["runner"]
    monkeypatch.setattr(direct_provenance, "live_host_identity", lambda: (dispatch.host_name, BOOT_ID))
    environ = {
        "KEYFRAME_RUNNER_BACKEND": "direct", "KEYFRAME_DIRECT_DISPATCH_PATH": runner["dispatch_path"],
        "KEYFRAME_DIRECT_DISPATCH_SHA256": runner["dispatch_sha256"],
        "KEYFRAME_FORMAL_ROW_ID": str(row_id), "CUDA_VISIBLE_DEVICES": dispatch.gpu_uuids[1],
    }
    row = full_plan.matrix["rows"][row_id]
    assert validate_formal_runtime_row_binding(full_plan.root, attempt_id=0, environ=environ, **row) == row
    aggregate_formal._validate_attempt_submission_binding(
        manifest, row_id=row_id, authorization=authorizations[row_id], run_root=full_plan.root,
    )
    wrong_authorization = {**authorizations[row_id], "submission_record_sha256": sha256_file(full_plan.records[1 - dispatch.shard_id][0])}
    with pytest.raises(ArtifactContractError, match="exact submission authorization"):
        aggregate_formal._validate_attempt_submission_binding(
            manifest, row_id=row_id, authorization=wrong_authorization, run_root=full_plan.root,
        )


@pytest.mark.parametrize("fault", ["plan_row_duplicate", "record_local_remap", "wrong_gpu_pair", "runtime_profile", "missing_second_shard", "slurm_impersonation"])
def test_full_plan_rejects_structural_or_backend_tampering(full_plan, fault):
    path = full_plan.records[1][0]
    record = json.loads(path.read_text())
    if fault == "plan_row_duplicate":
        path = full_plan.root / "protocol/submission_plan.json"
        record = json.loads(path.read_text())
        record["shards"][1]["row_ids"][0] = 999
    elif fault == "record_local_remap":
        record["array_task_ids"][0] = 999
    elif fault == "wrong_gpu_pair":
        record["gpu_pairs"] = [GPU_PAIRS[0]]
    elif fault == "runtime_profile":
        record["runtime_profile"]["environment_sh_sha256"] = "0" * 64
    elif fault == "missing_second_shard":
        path.unlink()
    elif fault == "slurm_impersonation":
        record["slurm_array_job_id"] = "fabricated"
    if fault != "missing_second_shard":
        path.write_bytes(canonical_bytes(record))
    with pytest.raises(ArtifactContractError):
        _authorizations(full_plan)


@pytest.mark.parametrize("fault", ["cleanup", "process_exit", "renderer"])
def test_formal_offline_attempt_gate_rejects_lifecycle_or_renderer_tampering(full_plan, fault):
    _, authorizations = _authorizations(full_plan)
    manifest, _ = _completed_manifest(full_plan, 1000)
    directory = Path(manifest["runner"]["dispatch_path"]).parent
    if fault == "renderer":
        manifest["renderer_device"]["gpu_uuid"] = GPU_PAIRS[1][0]
    else:
        path = directory / ("completion.json" if fault == "cleanup" else "evaluator_exit.json")
        record = json.loads(path.read_text())
        record["cleanup_confirmed" if fault == "cleanup" else "returncode"] = False if fault == "cleanup" else 1
        path.write_bytes(canonical_bytes(record))
    with pytest.raises(ArtifactContractError):
        aggregate_formal._validate_attempt_submission_binding(
            manifest, row_id=1000, authorization=authorizations[1000], run_root=full_plan.root,
        )


def test_controller_schedules_exact_full_plan_once_without_crossing_shards(full_plan, monkeypatch):
    calls = []
    lock = threading.Lock()
    def cpu_only_row(root, stage, path, row, gpu_uuids, stop):
        with lock:
            calls.append((row["row_id"], path, gpu_uuids))
        assert root == full_plan.root
        assert stage == "formal"
        assert not stop.is_set()
    monkeypatch.setattr(submit_direct, "execute_one", cpu_only_row)
    submit_direct.run_controller(full_plan.root, "formal", full_plan.records, GPU_PAIRS)
    assert [row_id for row_id, _, _ in calls] == list(range(1600))
    assert [path for _, path, _ in calls[:1000]] == [full_plan.records[0][0]] * 1000
    assert [path for _, path, _ in calls[1000:]] == [full_plan.records[1][0]] * 600
    assert all(pair == tuple(GPU_PAIRS[0]) for _, _, pair in calls)


def test_controller_stops_remaining_rows_after_known_row_failure(full_plan, monkeypatch):
    calls = []
    def failing_cpu_row(root, stage, path, row, gpu_uuids, stop):
        calls.append(row["row_id"])
        if row["row_id"] == 3:
            raise RuntimeError("synthetic evaluator failure after reconciled evidence")
    monkeypatch.setattr(submit_direct, "execute_one", failing_cpu_row)
    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        submit_direct.run_controller(full_plan.root, "formal", full_plan.records, GPU_PAIRS)
    assert calls == [0, 1, 2, 3]


def test_controller_preserves_other_rows_after_audited_infrastructure_failure_without_retry(full_plan, monkeypatch):
    calls = []

    def infrastructure_failure_cpu_row(root, stage, path, row, gpu_uuids, stop):
        assert not stop.is_set()
        calls.append(row["row_id"])
        if row["row_id"] == 3:
            raise submit_direct.RowInfrastructureError("synthetic audited infrastructure failure")

    monkeypatch.setattr(submit_direct, "execute_one", infrastructure_failure_cpu_row)
    with pytest.raises(RuntimeError, match="1 audited infrastructure failures; audit and explicitly select retries"):
        submit_direct.run_controller(full_plan.root, "formal", full_plan.records, GPU_PAIRS)
    # Both 1,000/600 shards finish their authorized queues. The failed row is
    # counted once, never silently retried, and the batch still exits nonzero.
    assert calls == list(range(1600))
    assert calls.count(3) == 1


@pytest.mark.parametrize("full_plan", [True], indirect=True)
def test_resident_controller_covers_1600_once_in_shard_queues(full_plan, monkeypatch):
    from experiments.keyframe_neighborhood_sampling import resident_policy  # noqa: PLC0415
    calls = []
    monkeypatch.setattr(resident_policy, "run_resident_rows", lambda *args: calls.append(args))
    submit_direct.run_controller(full_plan.root, "formal", full_plan.records, full_plan.records[0][1]["gpu_pairs"])
    assert [len(args[3]) for args in calls] == [1000, 600]
    assert [row["row_id"] for args in calls for row in args[3]] == list(range(1600))
    assert [args[2] for args in calls] == [path for path, _ in full_plan.records]


@pytest.mark.parametrize("full_plan", [True], indirect=True)
@pytest.mark.parametrize("fault", ["missing_reset", "wrong_shard", "missing_session_cleanup"])
def test_formal_completion_rejects_resident_evidence_faults(full_plan, fault):
    _, authorizations = _authorizations(full_plan)
    manifest, _ = _completed_manifest(full_plan, 1000)
    row_dir = Path(manifest["runner"]["dispatch_path"]).parent
    binding = json.loads((row_dir / "policy_session.json").read_bytes())
    session_dir = Path(binding["session"]["dispatch_path"]).parent
    if fault == "missing_reset":
        (row_dir / "resident_reset.json").unlink()
    elif fault == "missing_session_cleanup":
        (session_dir / "completion.json").unlink()
    else:
        path = session_dir / "row_plan.json"
        payload = json.loads(path.read_bytes())
        payload["submission_shard_id"] = 0
        path.write_bytes(canonical_bytes(payload))
    with pytest.raises((ArtifactContractError, OSError)):
        aggregate_formal._validate_attempt_submission_binding(
            manifest, row_id=1000, authorization=authorizations[1000], run_root=full_plan.root)
