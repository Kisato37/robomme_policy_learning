from __future__ import annotations

import json
import os

import pytest

from experiments.keyframe_neighborhood_sampling import architecture_smoke
from experiments.keyframe_neighborhood_sampling import direct_provenance as provenance
from experiments.keyframe_neighborhood_sampling.aggregate_formal import _validate_attempt_submission_binding
from experiments.keyframe_neighborhood_sampling.architecture_smoke import validate_architecture_pass_report
from experiments.keyframe_neighborhood_sampling.formal_artifacts import validate_extension_runtime_location
from experiments.keyframe_neighborhood_sampling.formal_matrix import build_formal_matrix
from experiments.keyframe_neighborhood_sampling.formal_matrix import validate_formal_runtime_row_binding
from experiments.keyframe_neighborhood_sampling.record_launcher_failure import reconcile_evaluator_exit
from experiments.keyframe_neighborhood_sampling.record_launcher_failure import record_launcher_failure
from experiments.keyframe_neighborhood_sampling.record_launcher_failure import validate_extension_failure_record
from experiments.keyframe_neighborhood_sampling.runner_contract import DirectDispatch
from experiments.keyframe_neighborhood_sampling.runner_contract import canonical_bytes
from experiments.keyframe_neighborhood_sampling.runner_contract import process_exit_record
from experiments.keyframe_neighborhood_sampling.runner_contract import process_start_record
from experiments.keyframe_neighborhood_sampling.runner_contract import write_once_record
from experiments.keyframe_neighborhood_sampling.smoke_matrix import build_smoke_matrix
from experiments.keyframe_neighborhood_sampling.smoke_matrix import validate_runtime_row_binding
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import RunArtifactStore
from experiments.keyframe_oracle_sampling.artifacts import ScientificKey
from experiments.keyframe_oracle_sampling.artifacts import read_jsonl
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now

GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
BOOT = "12345678-1234-4234-8234-123456789def"


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "live_host_identity", lambda: ("fixture-host", BOOT))

    def prepare(stage="development_smoke", row_id=0, attempt_id=0, gpu_layout=None):
        root = tmp_path / "runs/keyframe_neighborhood_sampling" / f"{stage}-{row_id}-{attempt_id}"
        protocol = root / "protocol"
        protocol.mkdir(parents=True)
        formal = stage == "formal"
        architecture = stage == "architecture_smoke"
        matrix = build_formal_matrix() if formal else build_smoke_matrix()
        matrix_file = protocol / ("formal_matrix.json" if formal else "smoke_matrix.json")
        write_once_record(matrix_file, matrix)
        launch = {"runner_backend": "direct", "repository": {"commit_sha": "a" * 40}}
        if gpu_layout is not None:
            launch["gpu_layout"] = gpu_layout
        write_once_record(protocol / "launch_manifest.json", launch)
        if architecture:
            filename = "architecture_submission_record.json"
        elif formal:
            filename = "submission_record_shard_01.json"
        else:
            filename = (
                "submission_record.json" if attempt_id == 0 else f"submission_record_attempt_{attempt_id:02d}.json"
            )
        submission_file = protocol / filename
        submission = {
            "schema_version": 1,
            "runner_backend": "direct",
            "direct_run_id": "12345678-1234-4234-8234-123456789abc",
            "protocol_version": "v1.0",
            "protocol_family": "keyframe_neighborhood_sampling_v1",
            "repository_commit_sha": "a" * 40,
            "run_root": str(root),
            "attempt_id": attempt_id,
            "launch_manifest_sha256": sha256_file(protocol / "launch_manifest.json"),
            "formal_matrix_sha256" if formal else "smoke_matrix_sha256": sha256_file(matrix_file),
            "row_ids": [row_id],
            "trajectory_count": 1,
            "checkpoint_unpacked_metadata_sha256": "b" * 64,
            "checkpoint_content_tree_algorithm": "sha256-canonical-file-content-tree-v1",
            "checkpoint_unpacked_content_tree_sha256": "c" * 64,
        }
        if gpu_layout is not None:
            submission.update(gpu_layout=gpu_layout, runtime_profile={"gpu_layout": gpu_layout})
        pair = (GPU_A, GPU_A) if gpu_layout == "colocated" else (GPU_A, GPU_B)
        if architecture:
            submission.update(gpu_uuids=[GPU_A], formal_started=False, arms=["OC3", "OC5"])
        else:
            submission["gpu_pairs"] = [list(pair)]
        if formal:
            submission.update(shard_id=1, array_task_ids=[0])
        write_once_record(submission_file, submission)
        dispatch = DirectDispatch(
            execution_id="12345678-1234-4234-8234-123456789123",
            run_root=str(root),
            repository_commit_sha="a" * 40,
            stage=stage,
            launch_manifest_sha256=submission["launch_manifest_sha256"],
            submission_plan_sha256=sha256_file(submission_file),
            matrix_sha256=sha256_file(matrix_file),
            attempt_id=attempt_id,
            row_id=None if architecture else row_id,
            shard_id=1 if formal else None,
            gpu_uuids=(GPU_A,) if architecture else pair,
            gpu_layout=gpu_layout or "separate",
            host_name="fixture-host",
            host_boot_id=BOOT,
            policy_port=None if architecture else 30000,
        )
        path = provenance.dispatch_path(root, stage=stage, attempt_id=attempt_id, row_id=dispatch.row_id)
        path.parent.mkdir(parents=True)
        digest = write_once_record(path, dispatch.as_record())
        environ = {
            "KEYFRAME_RUNNER_BACKEND": "direct",
            "KEYFRAME_DIRECT_DISPATCH_PATH": str(path),
            "KEYFRAME_DIRECT_DISPATCH_SHA256": digest,
            "CUDA_VISIBLE_DEVICES": ",".join(dict.fromkeys(dispatch.gpu_uuids)),
            "KEYFRAME_FORMAL_ROW_ID" if formal else "KEYFRAME_SMOKE_ROW_ID": str(row_id),
        }
        runner = provenance.direct_runner_for_runtime(root, environ=environ)
        return root, submission_file, submission, dispatch, environ, runner, matrix["rows"][row_id]

    return prepare


def _completion(root, dispatch, runner, roles):
    directory = provenance.dispatch_path(
        root, stage=dispatch.stage, attempt_id=dispatch.attempt_id, row_id=dispatch.row_id
    ).parent
    links = {}
    for index, role in enumerate(sorted(roles)):
        pid = 900 + index
        started = process_start_record(
            dispatch,
            process_identity={
                "boot_id": BOOT,
                "pid": pid,
                "parent_pid": os.getpid(),
                "process_group": pid,
                "session": pid,
                "start_ticks": 100 + index,
                "uid": os.getuid(),
            },
            command=["/fixture/python", role],
            working_directory="/fixture",
        )
        start_hash = write_once_record(directory / f"{role}_start.json", started)
        exited = process_exit_record(
            dispatch,
            started_record_sha256=start_hash,
            returncode=-15 if role == "policy" else 0,
            wall_clock_limit_reached=False,
        )
        exit_hash = write_once_record(directory / f"{role}_exit.json", exited)
        links[role] = {"start_sha256": start_hash, "exit_sha256": exit_hash}
    write_once_record(
        directory / "completion.json",
        {
            "backend": "direct",
            "execution_id": dispatch.execution_id,
            "dispatch_sha256": runner["dispatch_sha256"],
            "cleanup_confirmed": True,
            "roles": links,
            "finished_utc": utc_now(),
        },
    )
    return directory


def test_smoke_row_uses_direct_dispatch_without_any_slurm_variable(bundle):
    root, _, _, _, environ, _, row = bundle()
    assert validate_runtime_row_binding(root, attempt_id=0, environ=environ, **row) == row
    with pytest.raises(ArtifactContractError, match="differ"):
        validate_runtime_row_binding(root, attempt_id=0, environ=environ, **{**row, "arm": "OC5"})


@pytest.mark.parametrize("stage", ["architecture_smoke", "development_smoke", "formal"])
def test_colocated_direct_dispatch_binds_root_submission_roles_and_unique_visibility(bundle, stage):
    root, source, submission, dispatch, environ, runner, row = bundle(
        stage, row_id=1000 if stage == "formal" else 0, gpu_layout="colocated"
    )
    assert environ["CUDA_VISIBLE_DEVICES"] == GPU_A
    assert (
        provenance.validate_direct_runner(
            runner,
            root,
            stage=stage,
            attempt_id=0,
            row_id=dispatch.row_id,
            submission=submission,
            submission_path=source,
        )
        == dispatch
    )
    if stage == "development_smoke":
        assert validate_runtime_row_binding(root, attempt_id=0, environ=environ, **row) == row
    elif stage == "formal":
        assert validate_formal_runtime_row_binding(root, attempt_id=0, environ=environ, **row) == row
    with pytest.raises(ArtifactContractError, match="visibility"):
        provenance.direct_runner_for_runtime(root, environ={**environ, "CUDA_VISIBLE_DEVICES": f"{GPU_A},{GPU_A}"})


def test_colocated_slots_deduplicate_within_pair_but_never_overlap_other_slots(bundle):
    _, _, submission, _, _, _, _ = bundle(gpu_layout="colocated")
    provenance.validate_direct_submission(
        {**submission, "gpu_pairs": [[GPU_A, GPU_A], [GPU_B, GPU_B]]}, stage="development_smoke"
    )
    with pytest.raises(ArtifactContractError, match="across execution slots"):
        provenance.validate_direct_submission(
            {**submission, "gpu_pairs": [[GPU_A, GPU_A], [GPU_A, GPU_A]]}, stage="development_smoke"
        )
    with pytest.raises(ArtifactContractError, match="layout"):
        provenance.validate_direct_submission({**submission, "gpu_pairs": [[GPU_A, GPU_B]]}, stage="development_smoke")
    with pytest.raises(ArtifactContractError, match="layouts differ"):
        provenance.validate_direct_submission({**submission, "runtime_profile": {}}, stage="development_smoke")


def test_colocated_simulator_renderer_still_binds_the_second_recorded_role(bundle):
    root, _, _, dispatch, _, runner, _ = bundle(gpu_layout="colocated")
    assert dispatch.gpu_uuids == (GPU_A, GPU_A)
    manifest = {
        "runner_backend": "direct",
        "runner": runner,
        "environment_setup_completed": True,
        "renderer_device": {
            "render_backend": "sapien_cuda",
            "simulation_backend": "physx_cpu",
            "cuda_device_id": 0,
            "pci_bus_id": "0000:41:00.0",
            "gpu_uuid": GPU_A,
            "expected_gpu_uuid": GPU_A,
            "can_render": True,
            "is_cuda": True,
            "matches_dispatch": True,
        },
    }
    provenance.validate_direct_attempt(manifest, root, attempt_id=0, row_id=0, trajectory_kind="short")
    manifest["renderer_device"]["gpu_uuid"] = GPU_B
    with pytest.raises(ArtifactContractError, match="renderer identity"):
        provenance.validate_direct_attempt(manifest, root, attempt_id=0, row_id=0, trajectory_kind="short")


@pytest.mark.parametrize("changed", ["launch", "dispatch"])
def test_architecture_chosen_layout_must_match_root_and_submission(bundle, changed):
    root, source, submission, dispatch, _, runner, _ = bundle("architecture_smoke", gpu_layout="colocated")
    if changed == "launch":
        path = root / "protocol/launch_manifest.json"
        payload = json.loads(path.read_text())
        payload["gpu_layout"] = "separate"
        path.write_bytes(canonical_bytes(payload))
    else:
        path = provenance.dispatch_path(root, stage=dispatch.stage, attempt_id=0, row_id=None)
        runner["dispatch"]["gpu_layout"] = "separate"
        path.write_bytes(canonical_bytes(runner["dispatch"]))
        runner["dispatch_sha256"] = sha256_file(path)
    with pytest.raises(ArtifactContractError, match=r"layouts differ|artifact identity"):
        provenance.validate_direct_runner(
            runner,
            root,
            stage=dispatch.stage,
            attempt_id=0,
            row_id=None,
            submission=submission,
            submission_path=source,
        )


def _evaluator_exit_evidence(root, dispatch, *, returncode, deadline):
    directory = provenance.dispatch_path(
        root, stage=dispatch.stage, attempt_id=dispatch.attempt_id, row_id=dispatch.row_id
    ).parent
    started = process_start_record(
        dispatch,
        process_identity={
            "boot_id": BOOT,
            "pid": 900,
            "parent_pid": os.getpid(),
            "process_group": 900,
            "session": 900,
            "start_ticks": 100,
            "uid": os.getuid(),
        },
        command=["/fixture/python", "evaluator"],
        working_directory="/fixture",
    )
    started_hash = write_once_record(directory / "evaluator_start.json", started)
    path = directory / "evaluator_exit.json"
    write_once_record(
        path,
        process_exit_record(
            dispatch,
            started_record_sha256=started_hash,
            returncode=returncode,
            wall_clock_limit_reached=deadline,
        ),
    )
    return path


@pytest.mark.parametrize(
    ("returncode", "deadline"), [(124, True), (143, True), (-15, True), (137, True), (143, False), (124, False)]
)
def test_direct_reconciliation_deadline_restricts_retry_before_completion(bundle, returncode, deadline):
    root, _, _, dispatch, _, runner, row = bundle()
    path = _evaluator_exit_evidence(root, dispatch, returncode=returncode, deadline=deadline)
    shell_status = 128 - returncode if returncode < 0 else returncode
    _, disposition = reconcile_evaluator_exit(root, attempt_id=0, runner=runner, exit_status=shell_status, **row)
    assert disposition == "recorded_failure"
    assert not (path.parent / "completion.json").exists()
    record = read_jsonl(root / "failures/failure_ledger.jsonl")[0]
    retryable = not deadline and shell_status in {130, 137, 143}
    assert record["classification"] == ("infrastructure" if retryable else "hard_stop")
    assert record["retry_allowed"] is retryable
    if deadline:
        assert record["error_type"] == "DirectEvaluatorWallClockLimit"
        with pytest.raises(ArtifactContractError, match="cannot authorize a retry"):
            validate_extension_failure_record(
                root,
                {**record, "classification": "infrastructure", "retry_allowed": True},
                require_direct_completion=False,
            )


@pytest.mark.parametrize("fault", ["status_mismatch", "start_digest"])
def test_direct_deadline_reconciliation_requires_linked_exit_not_caller_claim(bundle, fault):
    root, _, _, dispatch, _, runner, row = bundle()
    path = _evaluator_exit_evidence(root, dispatch, returncode=124, deadline=True)
    if fault == "start_digest":
        record = json.loads(path.read_text())
        record["started_record_sha256"] = "0" * 64
        path.write_bytes(canonical_bytes(record))
    with pytest.raises(ArtifactContractError, match=r"status differs|mismatched identity"):
        reconcile_evaluator_exit(
            root, attempt_id=0, runner=runner, exit_status=143 if fault == "status_mismatch" else 124, **row
        )
    assert not (root / "failures/failure_ledger.jsonl").exists()


def test_direct_deadline_preserves_already_completed_result(bundle):
    root, _, _, dispatch, _, runner, row = bundle()
    _evaluator_exit_evidence(root, dispatch, returncode=124, deadline=True)
    key = ScientificKey(row["task"], row["episode_id"], row["arm"], row["trajectory_kind"])
    store = RunArtifactStore(root)
    writer = store.new_attempt(key, 0, {"runner_backend": "direct", "runner": runner})
    writer.record_initial_conditions({"fixture": "unchanged"})
    writer.append_trace({"policy_call_index": 0})
    writer.finalize({"success": False, "terminal_reason": "timeout"})
    before = writer.result_path.read_bytes()
    _, disposition = reconcile_evaluator_exit(root, attempt_id=0, runner=runner, exit_status=124, **row)
    assert disposition == "complete"
    assert writer.result_path.read_bytes() == before
    assert not store.failures_path.exists()


def test_formal_global_row_binds_exact_direct_shard_and_local_mapping(bundle):
    root, _, _, _, environ, _, row = bundle("formal", row_id=1000)
    assert validate_formal_runtime_row_binding(root, attempt_id=0, environ=environ, **row) == row
    with pytest.raises(ArtifactContractError, match="differs"):
        validate_formal_runtime_row_binding(root, attempt_id=0, environ=environ, **{**row, "row_id": 1001})


@pytest.mark.parametrize(
    "update",
    [
        {"KEYFRAME_RUNNER_BACKEND": "slurm"},
        {"KEYFRAME_DIRECT_DISPATCH_SHA256": "0" * 64},
        {"SLURM_ARRAY_JOB_ID": "fake"},
        {"CUDA_VISIBLE_DEVICES": "0,1"},
        {"CUDA_VISIBLE_DEVICES": f"{GPU_B},{GPU_A}"},
    ],
)
def test_direct_runtime_rejects_ambiguous_or_forged_context(bundle, update):
    root, _, _, _, environ, _, _ = bundle()
    with pytest.raises(ArtifactContractError):
        provenance.direct_runner_for_runtime(root, environ={**environ, **update})


def test_submission_argument_must_match_actual_file_and_gpu_authorization(bundle):
    root, source, submission, dispatch, _, runner, _ = bundle()
    assert (
        provenance.validate_direct_runner(
            runner,
            root,
            stage=dispatch.stage,
            attempt_id=0,
            row_id=0,
            submission=submission,
            submission_path=source,
        )
        == dispatch
    )
    with pytest.raises(ArtifactContractError, match="recorded bytes"):
        provenance.validate_direct_runner(
            runner,
            root,
            stage=dispatch.stage,
            attempt_id=0,
            row_id=0,
            submission={**submission, "row_ids": [0, 1]},
            submission_path=source,
        )
    with pytest.raises(ArtifactContractError, match="overlap"):
        provenance.validate_direct_submission({**submission, "gpu_pairs": [[GPU_A, GPU_A]]}, stage=dispatch.stage)


def test_process_completion_requires_exit_links_and_confirmed_cleanup(bundle):
    root, _, _, dispatch, _, runner, _ = bundle()
    roles = {"preflight", "policy", "evaluator", "reconcile"}
    with pytest.raises(ArtifactContractError, match="completion"):
        provenance.audit_direct_completion(runner, root, required_roles=roles)
    directory = _completion(root, dispatch, runner, roles)
    assert provenance.audit_direct_completion(runner, root, required_roles=roles)["cleanup_confirmed"] is True
    # Preserve a valid process identity but break the immutable exit link.
    altered = process_exit_record(
        dispatch, started_record_sha256="0" * 64, returncode=0, wall_clock_limit_reached=False
    )
    (directory / "evaluator_exit.json").write_text(json.dumps(altered, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ArtifactContractError, match="digests disagree"):
        provenance.audit_direct_completion(runner, root, required_roles=roles)


def test_wrong_host_boot_fails_even_with_valid_dispatch_hash(bundle, monkeypatch):
    root, _, _, _, environ, _, _ = bundle()
    monkeypatch.setattr(provenance, "live_host_identity", lambda: ("fixture-host", "another-boot"))
    with pytest.raises(ArtifactContractError, match="host or boot"):
        provenance.direct_runner_for_runtime(root, environ=environ)


@pytest.mark.parametrize("fault", ["missing_time", "naive_time", "before_exit", "reused_identity", "mixed_roles"])
def test_completion_rejects_invalid_clock_or_replayed_role_identity(bundle, fault):
    root, _, _, dispatch, _, runner, _ = bundle()
    roles = {"preflight", "policy", "evaluator", "reconcile"}
    directory = _completion(root, dispatch, runner, roles)
    path = directory / "completion.json"
    completion = json.loads(path.read_text())
    if fault == "missing_time":
        del completion["finished_utc"]
    elif fault == "naive_time":
        completion["finished_utc"] = "2026-01-01T00:00:00"
    elif fault == "before_exit":
        completion["finished_utc"] = "2000-01-01T00:00:00+00:00"
    elif fault == "mixed_roles":
        completion["roles"]["architecture"] = completion["roles"]["evaluator"]
    else:
        start_path = directory / "policy_start.json"
        start = json.loads(start_path.read_text())
        start["process_identity"] = json.loads((directory / "evaluator_start.json").read_text())["process_identity"]
        start_path.write_bytes(canonical_bytes(start))
        exit_path = directory / "policy_exit.json"
        exited = json.loads(exit_path.read_text())
        exited["started_record_sha256"] = sha256_file(start_path)
        exit_path.write_bytes(canonical_bytes(exited))
        completion["roles"]["policy"] = {"start_sha256": sha256_file(start_path), "exit_sha256": sha256_file(exit_path)}
    path.write_bytes(canonical_bytes(completion))
    with pytest.raises(ArtifactContractError, match=r"timestamp|identity|roles"):
        provenance.audit_direct_completion(runner, root, required_roles=roles)


def test_architecture_self_report_then_offline_exit_gate(bundle):
    root, _, submission, dispatch, _, runner, _ = bundle("architecture_smoke")
    report = {
        key: submission[key]
        for key in (
            "protocol_version",
            "protocol_family",
            "repository_commit_sha",
            "run_root",
            "arms",
            "formal_started",
            "checkpoint_unpacked_metadata_sha256",
            "checkpoint_content_tree_algorithm",
            "checkpoint_unpacked_content_tree_sha256",
        )
    }
    report.update(
        runner_backend="direct",
        runner=runner,
        passed=True,
        case_count=4,
        cases=[{"arm": arm, "history_length": length, "passed": True} for arm in ("OC3", "OC5") for length in (16, 64)],
        final_reset_evidence={"passed": True},
        formal_test_outcomes_opened=False,
    )
    validate_architecture_pass_report(
        report,
        run_root=root,
        architecture_submission=submission,
        require_direct_completion=False,
    )
    with pytest.raises(ArtifactContractError, match="completion"):
        validate_architecture_pass_report(report, run_root=root, architecture_submission=submission)
    _completion(root, dispatch, runner, {"architecture"})
    validate_architecture_pass_report(report, run_root=root, architecture_submission=submission)


def test_readiness_failure_is_retryable_only_after_its_owned_processes_exit(bundle):
    root, _, _, dispatch, _, runner, row = bundle()
    record_launcher_failure(
        root,
        attempt_id=0,
        runner=runner,
        error_type="PolicyServerReadinessTimeout",
        error="fixture startup failure",
        **row,
    )
    failure = read_jsonl(root / "failures/failure_ledger.jsonl")[0]
    assert failure["retry_allowed"] is True
    assert "slurm" not in failure
    with pytest.raises(ArtifactContractError, match="completion"):
        validate_extension_failure_record(root, failure)
    _completion(root, dispatch, runner, {"preflight", "policy", "reconcile"})
    validated = validate_extension_failure_record(root, failure)
    assert validated["row_id"] == 0


def test_formal_result_cannot_self_authorize_readiness_lifecycle_exemption(bundle):
    root, source, _, dispatch, _, runner, _ = bundle("formal", row_id=1000)
    _completion(root, dispatch, runner, {"preflight", "policy", "reconcile"})
    manifest = {
        "runner_backend": "direct",
        "runner": runner,
        "execution_phase": "policy_server_readiness",
        "launcher_failure_only": True,
    }
    authorization = {
        "runner_backend": "direct",
        "attempt_id": 0,
        "submission_record_sha256": sha256_file(source),
        "shard_id": 1,
    }
    with pytest.raises(ArtifactContractError, match="required process roles"):
        _validate_attempt_submission_binding(manifest, row_id=1000, authorization=authorization, run_root=root)


def test_completed_simulator_must_prove_actual_renderer_gpu(bundle):
    root, _, _, _, _, runner, _ = bundle()
    manifest = {"runner_backend": "direct", "runner": runner, "environment_setup_completed": True}
    with pytest.raises(ArtifactContractError, match="renderer"):
        provenance.validate_direct_attempt(manifest, root, attempt_id=0, row_id=0, trajectory_kind="short")
    manifest["renderer_device"] = {
        "render_backend": "Vulkan",
        "simulation_backend": "PhysX",
        "cuda_device_id": 0,
        "gpu_uuid": GPU_B,
        "expected_gpu_uuid": GPU_B,
        "pci_bus_id": "0000:02:00.0",
        "can_render": True,
        "is_cuda": True,
        "matches_dispatch": True,
    }
    provenance.validate_direct_attempt(manifest, root, attempt_id=0, row_id=0, trajectory_kind="short")
    manifest["renderer_device"]["gpu_uuid"] = GPU_A
    with pytest.raises(ArtifactContractError, match="renderer"):
        provenance.validate_direct_attempt(manifest, root, attempt_id=0, row_id=0, trajectory_kind="short")


def test_designated_runs_symlink_is_supported_without_accepting_arbitrary_external_roots(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    repo.mkdir()
    mount = tmp_path / "storage/runs"
    mount.mkdir(parents=True)
    (repo / "runs").symlink_to(mount, target_is_directory=True)
    root = mount / "keyframe_neighborhood_sampling/fresh-run"
    root.mkdir(parents=True)
    validate_extension_runtime_location(root, repo, source="fixture")
    validate_extension_runtime_location(repo / "runs/keyframe_neighborhood_sampling/fresh-run", repo, source="fixture")
    with pytest.raises(ArtifactContractError, match="direct child"):
        validate_extension_runtime_location(root / "nested", repo, source="fixture")
    with pytest.raises(ArtifactContractError, match="direct child"):
        validate_extension_runtime_location(
            tmp_path / "other/runs/keyframe_neighborhood_sampling/run", repo, source="fixture"
        )
    monkeypatch.setattr(architecture_smoke, "REPO", repo)
    assert architecture_smoke.dry_run_contract()["report_contract_valid"] is True
