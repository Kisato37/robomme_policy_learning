from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.keyframe_oracle_sampling import artifacts
from experiments.keyframe_oracle_sampling import submit_smoke
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import build_seed_table
from experiments.keyframe_oracle_sampling.artifacts import build_smoke_seed_table
from experiments.keyframe_oracle_sampling.artifacts import read_jsonl
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import validate_architecture_pass_report
from experiments.keyframe_oracle_sampling.artifacts import validate_prepared_smoke_root
from experiments.keyframe_oracle_sampling.architecture_smoke import (
    _dry_run_contract_example,
)
from experiments.keyframe_oracle_sampling.record_launcher_failure import record_launcher_failure
from experiments.keyframe_oracle_sampling.record_launcher_failure import reconcile_evaluator_exit
from experiments.keyframe_oracle_sampling.smoke_matrix import MATRIX_PATH
from experiments.keyframe_oracle_sampling.smoke_matrix import expand_rows
from experiments.keyframe_oracle_sampling.smoke_matrix import load_frozen_matrix
from experiments.keyframe_oracle_sampling.smoke_matrix import validate_runtime_row_binding
from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_POLICY_CALLS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _architecture_pass(run_root: Path, *, job_id: str = "arch-123") -> dict:
    report = _dry_run_contract_example(run_root.resolve())
    report.update(
        {
            "repository_commit_sha": "commit",
            "slurm_job_id": job_id,
            "checkpoint_unpacked_metadata_sha256": "c" * 64,
            "checkpoint_unpacked_content_tree_sha256": "d" * 64,
        }
    )
    return report


def _architecture_submission(run_root: Path, *, job_id: str = "arch-123") -> dict:
    return {
        "run_root": str(run_root.resolve()),
        "slurm_job_id": job_id,
        "checkpoint_content_tree_algorithm": (
            "sha256-canonical-file-content-tree-v1"
        ),
        "checkpoint_unpacked_content_tree_sha256": "d" * 64,
    }


def _smoke_seed_table() -> dict:
    return build_smoke_seed_table(FORMAL_TASKS, [0])


def _seed_manifest_fields(seed_table: dict) -> dict:
    return {
        "seed_table_scope": seed_table["scope"],
        "seed_table_dataset": seed_table["dataset"],
        "seed_table_derivation": seed_table["derivation"],
        "seed_table_entries_sha256": seed_table["entries_sha256"],
        "seed_disjointness_audit": {
            "smoke_seed_count": len(FORMAL_TASKS) * MAX_POLICY_CALLS,
            "formal_seed_count": len(FORMAL_TASKS) * 50 * MAX_POLICY_CALLS,
            "intersection_count": 0,
        },
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report.update(case_count=0, cases=[]),
            "exactly eight cases",
        ),
        (
            lambda report: report["cases"].pop(),
            "exactly eight cases",
        ),
        (
            lambda report: report["cases"].__setitem__(
                -1, dict(report["cases"][0])
            ),
            "unique U/O/OC/R",
        ),
    ],
)
def test_architecture_pass_validator_rejects_empty_missing_or_duplicate_grid(
    tmp_path, mutate, message
):
    report = _architecture_pass(tmp_path)
    mutate(report)
    with pytest.raises(ArtifactContractError, match=message):
        validate_architecture_pass_report(
            report,
            run_root=tmp_path,
            architecture_submission=_architecture_submission(tmp_path),
        )


def test_architecture_pass_validator_binds_run_root_and_slurm_job(tmp_path):
    report = _architecture_pass(tmp_path)
    with pytest.raises(ArtifactContractError, match="report run_root"):
        validate_architecture_pass_report(
            {**report, "run_root": str(tmp_path / "other")},
            run_root=tmp_path,
            architecture_submission=_architecture_submission(tmp_path),
        )
    with pytest.raises(ArtifactContractError, match="Slurm job ID"):
        validate_architecture_pass_report(
            report,
            run_root=tmp_path,
            architecture_submission=_architecture_submission(
                tmp_path, job_id="arch-other"
            ),
        )


def test_architecture_pass_validator_rejects_truthy_cases_without_absolute_evidence(
    tmp_path,
):
    report = _architecture_pass(tmp_path)
    report["cases"] = [
        {"arm": arm, "history_length": history_length, "passed": True}
        for arm in ("U", "O", "OC", "R")
        for history_length in (16, 64)
    ]
    with pytest.raises(ArtifactContractError, match="repeat/isolation evidence"):
        validate_architecture_pass_report(
            report,
            run_root=tmp_path,
            architecture_submission=_architecture_submission(tmp_path),
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("component_dtypes", ["bfloat16", "float32", "float32", "bool"]),
        ("action_dtype", "float32"),
    ],
)
def test_architecture_pass_validator_rejects_stale_unpadded_or_action_contract(
    tmp_path, field, bad_value
):
    report = _architecture_pass(tmp_path)
    u16 = next(
        case
        for case in report["cases"]
        if case["arm"] == "U" and case["history_length"] == 16
    )
    u16["first"][field] = bad_value
    with pytest.raises(ArtifactContractError, match="frozen smoke contract"):
        validate_architecture_pass_report(
            report,
            run_root=tmp_path,
            architecture_submission=_architecture_submission(tmp_path),
        )


def test_architecture_compile_cache_allows_only_two_released_dtype_specializations(
    tmp_path,
):
    report = _architecture_pass(tmp_path)
    u16, u64, o16 = report["cases"][:3]
    assert u16["first"]["compile_cache"] == {
        "vision_before": 0,
        "vision_after": 1,
        "memory_before": 0,
        "memory_after": 1,
        "sample_before": 0,
        "sample_after": 1,
    }
    assert u64["first"]["compile_cache"] == {
        "vision_before": 1,
        "vision_after": 1,
        "memory_before": 1,
        "memory_after": 2,
        "sample_before": 1,
        "sample_after": 2,
    }
    assert o16["first"]["compile_cache"]["memory_before"] == 2
    assert o16["first"]["compile_cache"]["memory_after"] == 2
    validate_architecture_pass_report(
        report,
        run_root=tmp_path,
        architecture_submission=_architecture_submission(tmp_path),
    )


def test_architecture_validator_rejects_selector_induced_third_specialization(
    tmp_path,
):
    report = _architecture_pass(tmp_path)
    selector_case_index = 2
    selector_case = report["cases"][selector_case_index]
    selector_case["first"]["compile_cache"]["memory_after"] += 1
    selector_case["repeat"]["compile_cache"]["memory_before"] += 1
    selector_case["repeat"]["compile_cache"]["memory_after"] += 1
    for case in report["cases"][selector_case_index + 1 :]:
        for repetition in ("first", "repeat"):
            case[repetition]["compile_cache"]["memory_before"] += 1
            case[repetition]["compile_cache"]["memory_after"] += 1
    report["stable_compile_cache"]["perceptual_memory"] += 1
    with pytest.raises(ArtifactContractError, match="unexpected compilation-cache"):
        validate_architecture_pass_report(
            report,
            run_root=tmp_path,
            architecture_submission=_architecture_submission(tmp_path),
        )


def test_smoke_submitter_rejects_vacuous_architecture_pass(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    protocol = run_root / "protocol"
    architecture_dir = run_root / "architecture_smoke"
    protocol.mkdir(parents=True)
    architecture_dir.mkdir()
    seed_table = _smoke_seed_table()
    formal_seed_table = build_seed_table(FORMAL_TASKS, range(50))
    seed_path = protocol / "seed_table.json"
    formal_seed_path = protocol / "formal_seed_audit_table.json"
    _write_json(seed_path, seed_table)
    _write_json(formal_seed_path, formal_seed_table)
    _write_json(
        protocol / "launch_manifest.json",
        {
            "repository": {"commit_sha": "commit"},
            "seed_table_file_sha256": sha256_file(seed_path),
            "formal_seed_audit_file_sha256": sha256_file(formal_seed_path),
            **_seed_manifest_fields(seed_table),
        },
    )
    _write_json(
        protocol / "architecture_submission_record.json",
        _architecture_submission(run_root),
    )
    report = _architecture_pass(run_root)
    report.update(case_count=0, cases=[])
    _write_json(architecture_dir / "report.json", report)
    monkeypatch.setattr(
        submit_smoke,
        "repository_state",
        lambda: {"commit_sha": "commit"},
    )
    monkeypatch.setattr(submit_smoke, "require_clean_repository", lambda _state: None)

    with pytest.raises(ArtifactContractError, match="exactly eight cases"):
        submit_smoke.build_submission(
            run_root,
            max_concurrent=1,
            attempt_id=0,
            row_ids=None,
        )


def _runtime_root(tmp_path: Path, *, row_ids: list[int] | None = None) -> Path:
    run_root = tmp_path / "run"
    protocol = run_root / "protocol"
    protocol.mkdir(parents=True)
    (protocol / "smoke_matrix.json").write_bytes(MATRIX_PATH.read_bytes())
    _write_json(
        protocol / "submission_record.json",
        {
            "attempt_id": 0,
            "slurm_array_job_id": "123",
            "row_ids": [0] if row_ids is None else row_ids,
        },
    )
    return run_root


def test_runtime_row_is_bound_to_slurm_submission_and_frozen_parameters(tmp_path):
    run_root = _runtime_root(tmp_path)
    row = expand_rows(load_frozen_matrix())[0]
    validated = validate_runtime_row_binding(
        run_root,
        attempt_id=0,
        row_id=0,
        task=row["task"],
        episode_id=row["episode_id"],
        arm=row["arm"],
        trajectory_kind=row["trajectory_kind"],
        max_steps=row["max_steps"],
        dataset=row["dataset"],
        environ={"SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0"},
    )
    assert validated == row

    with pytest.raises(ArtifactContractError, match="frozen row"):
        validate_runtime_row_binding(
            run_root,
            attempt_id=0,
            row_id=0,
            task=row["task"],
            episode_id=0,
            arm=row["arm"],
            trajectory_kind="terminal",
            max_steps=64,
            dataset="val",
            environ={"SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0"},
        )


def test_runtime_row_rejects_unsubmitted_or_different_array_task(tmp_path):
    row = expand_rows(load_frozen_matrix())[0]
    run_root = _runtime_root(tmp_path, row_ids=[1])
    common = {
        "run_root": run_root,
        "attempt_id": 0,
        "row_id": 0,
        "task": row["task"],
        "episode_id": 0,
        "arm": row["arm"],
        "trajectory_kind": row["trajectory_kind"],
        "max_steps": row["max_steps"],
        "dataset": row["dataset"],
    }
    with pytest.raises(ArtifactContractError, match="does not authorize"):
        validate_runtime_row_binding(
            **common,
            environ={"SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0"},
        )
    with pytest.raises(ArtifactContractError, match="differs from SLURM_ARRAY_TASK_ID"):
        validate_runtime_row_binding(
            **common,
            environ={"SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "1"},
        )


def test_prepared_gate_binds_exact_architecture_report_digest(tmp_path, monkeypatch):
    run_root = tmp_path / "runs" / "keyframe_oracle_sampling" / "run"
    protocol = run_root / "protocol"
    architecture_dir = run_root / "architecture_smoke"
    protocol.mkdir(parents=True)
    architecture_dir.mkdir()
    protocol_snapshot = protocol / "protocol_snapshot.md"
    seed_table = protocol / "seed_table.json"
    formal_seed_table = protocol / "formal_seed_audit_table.json"
    smoke_matrix = protocol / "smoke_matrix.json"
    protocol_snapshot.write_text("protocol")
    seed_payload = _smoke_seed_table()
    formal_seed_payload = build_seed_table(FORMAL_TASKS, range(50))
    _write_json(seed_table, seed_payload)
    _write_json(formal_seed_table, formal_seed_payload)
    smoke_matrix.write_bytes(MATRIX_PATH.read_bytes())
    checkpoint_digest = "c" * 64
    report_path = architecture_dir / "report.json"
    architecture_report = _architecture_pass(run_root)
    architecture_report["checkpoint_unpacked_metadata_sha256"] = checkpoint_digest
    _write_json(report_path, architecture_report)
    _write_json(
        protocol / "architecture_submission_record.json",
        _architecture_submission(run_root),
    )
    _write_json(
        protocol / "launch_manifest.json",
        {
            "protocol_version": "v0.9.1",
            "protocol_sha256": sha256_file(protocol_snapshot),
            "seed_table_file_sha256": sha256_file(seed_table),
            "formal_seed_audit_file_sha256": sha256_file(formal_seed_table),
            "smoke_matrix_sha256": sha256_file(smoke_matrix),
            "repository": {"commit_sha": "commit"},
            "checkpoint_unpacked_metadata_sha256": checkpoint_digest,
            "checkpoint_content_tree_algorithm": (
                "sha256-canonical-file-content-tree-v1"
            ),
            "checkpoint_unpacked_content_tree_sha256": "d" * 64,
            **_seed_manifest_fields(seed_payload),
        },
    )
    _write_json(
        protocol / "submission_record.json",
        {
            "repository_commit_sha": "commit",
            "attempt_id": 0,
            "architecture_report_sha256": sha256_file(report_path),
            "slurm_array_job_id": "123",
            "trajectory_count": 1,
            "row_ids": [0],
            "checkpoint_content_tree_algorithm": (
                "sha256-canonical-file-content-tree-v1"
            ),
            "checkpoint_unpacked_content_tree_sha256": "d" * 64,
        },
    )

    def fake_check_output(command, **_kwargs):
        return "commit\n" if command[1:3] == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(artifacts.subprocess, "check_output", fake_check_output)
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    validate_prepared_smoke_root(
        run_root, seed_table, tmp_path, attempt_id=0
    )

    report_path.write_text(report_path.read_text() + "\n")
    with pytest.raises(ArtifactContractError, match="changed after"):
        validate_prepared_smoke_root(
            run_root, seed_table, tmp_path, attempt_id=0
        )


def test_server_readiness_failure_reserves_attempt_and_retry_ledger(tmp_path):
    attempt_dir = record_launcher_failure(
        tmp_path,
        attempt_id=0,
        row_id=3,
        task="InsertPeg",
        episode_id=0,
        arm="OC",
        trajectory_kind="short",
        max_steps=64,
        dataset="val",
        error_type="PolicyServerReadinessTimeout",
        error="timed out",
        slurm={"array_job_id": "123", "array_task_id": "3", "policy_port": 20123},
    )
    manifest = json.loads((attempt_dir / "episode_manifest.json").read_text())
    assert manifest["scientific_actions_started"] is False
    assert manifest["launcher_failure_only"] is True
    failures = read_jsonl(tmp_path / "failures" / "failure_ledger.jsonl")
    assert len(failures) == 1
    assert failures[0]["classification"] == "infrastructure"
    assert failures[0]["retry_allowed"] is True
    assert failures[0]["episode_manifest_sha256"] == sha256_file(
        attempt_dir / "episode_manifest.json"
    )


def _reconcile(tmp_path: Path, *, exit_status: int, attempt_id: int = 0):
    return reconcile_evaluator_exit(
        tmp_path,
        attempt_id=attempt_id,
        row_id=3,
        task="InsertPeg",
        episode_id=0,
        arm="OC",
        trajectory_kind="short",
        max_steps=64,
        dataset="val",
        exit_status=exit_status,
        slurm={"array_job_id": "123", "array_task_id": "3", "policy_port": 20123},
    )


def test_evaluator_non_signal_failure_is_fail_closed_and_not_retryable(tmp_path):
    attempt_dir, disposition = _reconcile(tmp_path, exit_status=1)
    assert disposition == "recorded_failure"
    manifest = json.loads((attempt_dir / "episode_manifest.json").read_text())
    assert manifest["execution_phase"] == "evaluator_lifecycle"
    assert manifest["scientific_actions_started"] is False
    failures = read_jsonl(tmp_path / "failures" / "failure_ledger.jsonl")
    assert len(failures) == 1
    assert failures[0]["classification"] == "hard_stop"
    assert failures[0]["retry_allowed"] is False
    assert failures[0]["error_type"] == "UnhandledEvaluatorLifecycleFailure"


def test_evaluator_arbitrary_high_exit_is_not_misclassified_as_signal(tmp_path):
    _, disposition = _reconcile(tmp_path, exit_status=200)
    assert disposition == "recorded_failure"
    failure = read_jsonl(tmp_path / "failures" / "failure_ledger.jsonl")[0]
    assert failure["classification"] == "hard_stop"
    assert failure["retry_allowed"] is False


def test_evaluator_signal_failure_is_retryable_but_only_before_retry_limit(tmp_path):
    attempt_dir, disposition = _reconcile(tmp_path, exit_status=143)
    assert disposition == "recorded_failure"
    failures = read_jsonl(tmp_path / "failures" / "failure_ledger.jsonl")
    assert failures[0]["classification"] == "infrastructure"
    assert failures[0]["retry_allowed"] is True
    assert failures[0]["evaluator_exit_status"] == 143

    key = artifacts.ScientificKey("InsertPeg", 0, "OC", "short")
    store = artifacts.RunArtifactStore(tmp_path)
    store.new_attempt(key, 1, {"environment_setup_completed": False})
    store.record_failure(
        {
            **key.as_dict(),
            "attempt_id": 1,
            "classification": "infrastructure",
            "retry_allowed": True,
        }
    )
    store.new_attempt(key, 2, {"environment_setup_completed": False})
    _, final_disposition = _reconcile(tmp_path, exit_status=137, attempt_id=2)
    assert final_disposition == "recorded_failure"
    final_failure = read_jsonl(store.failures_path)[-1]
    assert final_failure["classification"] == "infrastructure"
    assert final_failure["retry_allowed"] is False


def test_evaluator_fallback_is_idempotent_when_evaluator_already_wrote_failure(tmp_path):
    store = artifacts.RunArtifactStore(tmp_path)
    key = artifacts.ScientificKey("InsertPeg", 0, "OC", "short")
    store.new_attempt(key, 0, {"environment_setup_completed": False})
    store.record_failure(
        {
            **key.as_dict(),
            "attempt_id": 0,
            "classification": "hard_stop",
            "retry_allowed": False,
        }
    )
    _, disposition = _reconcile(tmp_path, exit_status=1)
    assert disposition == "existing_failure"
    assert len(read_jsonl(store.failures_path)) == 1


def test_evaluator_fallback_does_not_contradict_completed_episode(tmp_path):
    store = artifacts.RunArtifactStore(tmp_path)
    key = artifacts.ScientificKey("InsertPeg", 0, "OC", "short")
    writer = store.new_attempt(key, 0, {"environment_setup_completed": True})
    writer.record_initial_conditions({"front": "abc"})
    writer.append_trace({"policy_call_index": 0})
    writer.finalize({"success": True, "terminal_reason": "success"})
    _, disposition = _reconcile(tmp_path, exit_status=1)
    assert disposition == "complete"
    assert not store.failures_path.exists()


def test_evaluator_zero_exit_without_artifact_is_hard_stop(tmp_path):
    _, disposition = _reconcile(tmp_path, exit_status=0)
    assert disposition == "recorded_failure"
    failure = read_jsonl(tmp_path / "failures" / "failure_ledger.jsonl")[0]
    assert failure["error_type"] == "EvaluatorExitedWithoutEpisodeArtifact"
    assert failure["classification"] == "hard_stop"
    assert failure["retry_allowed"] is False


def test_evaluator_interrupted_between_attempt_mkdir_and_manifest_gets_ledger(tmp_path):
    key = artifacts.ScientificKey("InsertPeg", 0, "OC", "short")
    store = artifacts.RunArtifactStore(tmp_path)
    store.attempt_dir(key, 0).mkdir(parents=True)
    attempt_dir, disposition = _reconcile(tmp_path, exit_status=143)
    assert disposition == "recorded_failure"
    assert attempt_dir.is_dir()
    failure = read_jsonl(store.failures_path)[0]
    assert failure["error_type"] == "InvalidOrPartialEpisodeArtifact"
    assert failure["classification"] == "infrastructure"
    assert failure["retry_allowed"] is True
    assert failure["episode_manifest_sha256"] is None


def test_evaluator_invalid_result_is_hard_stop_even_with_signal_status(tmp_path):
    store = artifacts.RunArtifactStore(tmp_path)
    key = artifacts.ScientificKey("InsertPeg", 0, "OC", "short")
    writer = store.new_attempt(key, 0, {"environment_setup_completed": True})
    writer.result_path.write_text("{}")
    _, disposition = _reconcile(tmp_path, exit_status=143)
    assert disposition == "recorded_failure"
    failure = read_jsonl(store.failures_path)[0]
    assert failure["error_type"] == "InvalidOrPartialEpisodeArtifact"
    assert failure["classification"] == "hard_stop"
    assert failure["retry_allowed"] is False


def test_launchers_use_pinned_policy_python_and_runtime_gate():
    repo = Path(__file__).resolve().parents[2]
    smoke = (repo / "experiments/keyframe_oracle_sampling/run_smoke.sbatch").read_text()
    architecture = (
        repo / "experiments/keyframe_oracle_sampling/run_architecture_smoke.sbatch"
    ).read_text()
    assert "preflight_smoke_row" in smoke
    assert "record_launcher_failure" in smoke
    assert smoke.index("preflight_smoke_row") < smoke.index("scripts/serve_policy.py")
    assert smoke.index("record_launcher_failure") > smoke.index("READINESS_STATUS")
    assert "--evaluator-exit-status" in smoke
    assert "trap 'handle_signal TERM 143' TERM" in smoke
    assert 'wait "${EVALUATOR_PID}"' in smoke
    assert ':${PYTHONPATH:-}' not in smoke
    assert (
        'export PYTHONPATH="${REPO_ROOT}/src:'
        '${REPO_ROOT}/packages/openpi-client/src:${REPO_ROOT}"' in smoke
    )
    assert "python -m experiments.keyframe_oracle_sampling" not in smoke
    assert "python3 -m experiments.keyframe_oracle_sampling" not in architecture
    assert '"${POLICY_PYTHON}" -m experiments.keyframe_oracle_sampling' in smoke
    assert '"${POLICY_PYTHON}" -m experiments.keyframe_oracle_sampling' in architecture


def test_clean_worktree_gates_exclude_only_protocol_artifact_namespaces():
    repo = Path(__file__).resolve().parents[2]
    sources = [
        (repo / "experiments/keyframe_oracle_sampling/prepare_smoke.py").read_text(),
        (repo / "experiments/keyframe_oracle_sampling/architecture_smoke.py").read_text(),
        (repo / "experiments/keyframe_oracle_sampling/artifacts.py").read_text(),
    ]
    assert ":(exclude)runs/keyframe_oracle_sampling" in sources[0]
    assert ":(exclude)runs/keyframe_oracle_sampling" in sources[1]
    assert 'runtime_artifact_root = Path("runs/keyframe_oracle_sampling")' in sources[2]
    for source in sources:
        assert "runs/test_time_scaling/checkpoints" in source
        assert "perceptual-framesamp-modul/79999" in source
        assert ":(exclude)src" not in source
        assert ":(exclude)examples" not in source
