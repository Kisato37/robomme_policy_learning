from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.keyframe_oracle_sampling import formal_artifacts
from experiments.keyframe_oracle_sampling.analysis import normalize_prior_exposure_manifest
from experiments.keyframe_oracle_sampling.architecture_smoke import _dry_run_contract_example
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import build_seed_table
from experiments.keyframe_oracle_sampling.artifacts import build_smoke_seed_table
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.formal_artifacts import validate_formal_prepared_root
from experiments.keyframe_oracle_sampling.formal_artifacts import validate_prepared_formal_root
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_oracle_sampling.formal_matrix import build_formal_matrix
from experiments.keyframe_oracle_sampling.formal_matrix import load_formal_matrix
from experiments.keyframe_oracle_sampling.formal_matrix import validate_formal_runtime_row_binding
from experiments.keyframe_oracle_sampling.prepare_formal import dry_run_contract

REPO = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def test_formal_matrix_is_exact_unique_and_canonical():
    matrix = build_formal_matrix()
    rows = matrix["rows"]
    assert matrix["trajectory_count"] == FORMAL_TRAJECTORY_COUNT == 3200
    assert rows[0] == {
        "row_id": 0,
        "task": "BinFill",
        "episode_id": 0,
        "arm": "U",
        "trajectory_kind": "formal",
        "max_steps": 1300,
        "dataset": "test",
    }
    assert rows[-1]["row_id"] == 3199
    assert rows[-1]["task"] == "RouteStick"
    assert rows[-1]["episode_id"] == 49
    assert rows[-1]["arm"] == "R"
    keys = {(row["task"], row["episode_id"], row["arm"], row["trajectory_kind"]) for row in rows}
    assert len(keys) == len(rows)


def test_formal_matrix_loader_and_runtime_binding_fail_closed(tmp_path):
    run_root = tmp_path / "run"
    protocol = run_root / "protocol"
    protocol.mkdir(parents=True)
    matrix_path = protocol / "formal_matrix.json"
    _write_json(matrix_path, build_formal_matrix())
    _write_json(
        protocol / "submission_record.json",
        {
            "attempt_id": 0,
            "slurm_array_job_id": "formal-123",
            "row_ids": [3199],
        },
    )
    row = load_formal_matrix(matrix_path)["rows"][3199]
    assert (
        validate_formal_runtime_row_binding(
            run_root,
            attempt_id=0,
            row_id=3199,
            task=row["task"],
            episode_id=row["episode_id"],
            arm=row["arm"],
            trajectory_kind=row["trajectory_kind"],
            max_steps=row["max_steps"],
            dataset=row["dataset"],
            environ={
                "SLURM_ARRAY_JOB_ID": "formal-123",
                "SLURM_ARRAY_TASK_ID": "3199",
            },
        )
        == row
    )
    with pytest.raises(ArtifactContractError, match="frozen row"):
        validate_formal_runtime_row_binding(
            run_root,
            attempt_id=0,
            row_id=3199,
            task=row["task"],
            episode_id=48,
            arm=row["arm"],
            trajectory_kind=row["trajectory_kind"],
            max_steps=row["max_steps"],
            dataset=row["dataset"],
            environ={
                "SLURM_ARRAY_JOB_ID": "formal-123",
                "SLURM_ARRAY_TASK_ID": "3199",
            },
        )


def _prepared_formal_root(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    run_root = tmp_path / "runs" / "keyframe_oracle_sampling" / "formal-run"
    protocol = run_root / "protocol"
    protocol.mkdir(parents=True)
    protocol_snapshot = protocol / "protocol_snapshot.md"
    protocol_snapshot.write_text("frozen protocol")

    formal_seed = build_seed_table(FORMAL_TASKS, range(50))
    development_seed = build_smoke_seed_table(FORMAL_TASKS, [0])
    seed_path = protocol / "seed_table.json"
    development_seed_path = protocol / "development_seed_audit_table.json"
    matrix_path = protocol / "formal_matrix.json"
    exposure_path = protocol / "prior_exposure_manifest.json"
    _write_json(seed_path, formal_seed)
    _write_json(development_seed_path, development_seed)
    matrix = build_formal_matrix()
    _write_json(matrix_path, matrix)
    exposure_payload = {
        "frozen_before_formal_execution": True,
        "present_formal_outcomes_inspected": False,
        "entries": [{"task": "PickXtimes", "split": "test", "episode_id": 0}],
    }
    _write_json(exposure_path, exposure_payload)

    architecture_source = tmp_path / "architecture-source"
    report = _dry_run_contract_example(architecture_source.resolve())
    report.update(
        {
            "repository_commit_sha": "commit",
            "slurm_job_id": "arch-123",
            "checkpoint_unpacked_metadata_sha256": "c" * 64,
            "checkpoint_unpacked_content_tree_sha256": "d" * 64,
        }
    )
    architecture_submission = {
        "run_root": str(architecture_source.resolve()),
        "slurm_job_id": "arch-123",
        "checkpoint_content_tree_algorithm": ("sha256-canonical-file-content-tree-v1"),
        "checkpoint_unpacked_content_tree_sha256": "d" * 64,
    }
    architecture_report_path = protocol / "architecture_pass_report.json"
    architecture_submission_path = protocol / "architecture_submission_record.json"
    _write_json(architecture_report_path, report)
    _write_json(architecture_submission_path, architecture_submission)

    development_smoke = {
        "passed": True,
        "formal_started": False,
        "failure_ledger_present": False,
        "short_trajectories_respect_official_terminal_or_64_cap": True,
        "completeness": {
            "expected_count": 80,
            "completed_count": 80,
            "complete": True,
            "missing": [],
            "unexpected": [],
        },
        "initial_condition_fairness": {"paired_blocks": 16, "fair": True},
        "paired_manifest_invariants": {
            "paired_blocks": 16,
            "invariants_match": True,
        },
        "seed_disjointness_audit": {
            "smoke_seed_count": 1312,
            "formal_seed_count": 65600,
            "intersection_count": 0,
        },
    }
    development_smoke_path = protocol / "development_smoke_audit.json"
    _write_json(development_smoke_path, development_smoke)
    monkeypatch.setattr(
        formal_artifacts,
        "ACCEPTED_DEVELOPMENT_SMOKE_AUDIT_SHA256",
        sha256_file(development_smoke_path),
    )

    exposure = normalize_prior_exposure_manifest(exposure_payload)
    _write_json(
        protocol / "launch_manifest.json",
        {
            "protocol_version": "v1.0",
            "run_kind": "formal",
            "dataset": "test",
            "formal_launch_authorized": True,
            "trajectory_count": 3200,
            "repository": {"commit_sha": "commit"},
            "protocol_sha256": sha256_file(protocol_snapshot),
            "seed_table_file_sha256": sha256_file(seed_path),
            "development_seed_audit_file_sha256": sha256_file(development_seed_path),
            "formal_matrix_sha256": sha256_file(matrix_path),
            "prior_exposure_manifest_sha256": sha256_file(exposure_path),
            "architecture_report_sha256": sha256_file(architecture_report_path),
            "architecture_submission_record_sha256": sha256_file(architecture_submission_path),
            "development_smoke_audit_sha256": sha256_file(development_smoke_path),
            "seed_table_scope": formal_seed["scope"],
            "seed_table_dataset": formal_seed["dataset"],
            "seed_table_derivation": formal_seed["derivation"],
            "seed_table_entries_sha256": formal_seed["entries_sha256"],
            "seed_disjointness_audit": {
                "smoke_seed_count": 1312,
                "formal_seed_count": 65600,
                "intersection_count": 0,
            },
            "matrix": matrix["rows"],
            "prior_exposure_entry_count": len(exposure.entries),
            "prior_exposure_test_block_count": len(exposure.excluded_formal_blocks),
            "architecture_source_run_root": str(architecture_source.resolve()),
            "checkpoint_unpacked_metadata_sha256": "c" * 64,
            "checkpoint_content_tree_algorithm": ("sha256-canonical-file-content-tree-v1"),
            "checkpoint_unpacked_content_tree_sha256": "d" * 64,
            "development_smoke_commit_sha": (formal_artifacts.ACCEPTED_DEVELOPMENT_SMOKE_COMMIT),
        },
    )

    def fake_check_output(command, **_kwargs):
        return "commit\n" if command[1:3] == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(formal_artifacts.subprocess, "check_output", fake_check_output)
    return run_root, seed_path


def test_formal_root_requires_exact_submission_digest_and_slurm_array(tmp_path, monkeypatch):
    run_root, seed_path = _prepared_formal_root(tmp_path, monkeypatch)
    manifest = validate_formal_prepared_root(run_root, seed_path, tmp_path)
    submission_path = run_root / "protocol" / "submission_record.json"
    _write_json(
        submission_path,
        {
            "repository_commit_sha": "commit",
            "attempt_id": 0,
            "formal_launch_authorized": True,
            "launch_manifest_sha256": sha256_file(run_root / "protocol" / "launch_manifest.json"),
            "formal_matrix_sha256": manifest["formal_matrix_sha256"],
            "slurm_array_job_id": "formal-123",
            "trajectory_count": 3200,
            "row_ids": list(range(3200)),
        },
    )
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "formal-123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    authorization = sha256_file(submission_path)
    validate_prepared_formal_root(
        run_root,
        seed_path,
        tmp_path,
        attempt_id=0,
        authorization_digest=authorization,
    )
    with pytest.raises(ArtifactContractError, match="authorization digest"):
        validate_prepared_formal_root(
            run_root,
            seed_path,
            tmp_path,
            attempt_id=0,
            authorization_digest="0" * 64,
        )


def test_formal_dry_run_is_complete_and_non_submitting():
    report = dry_run_contract()
    assert report["valid"] is True
    assert report["trajectory_count"] == 3200
    assert report["formal_seed_count"] == 65600
    assert report["submits_jobs"] is False
    assert report["creates_run_root"] is False


def test_formal_slurm_launcher_preserves_hard_gates_and_two_gpu_isolation():
    script = (REPO / "experiments/keyframe_oracle_sampling/run_formal.sbatch").read_text()
    evaluator = (REPO / "examples/robomme/eval.py").read_text()
    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --no-requeue" in script
    assert "preflight_formal_row" in script
    assert script.index("preflight_formal_row") < script.index("scripts/serve_policy.py")
    assert "--args.keyframe-formal-authorization" in script
    assert "--formal-authorization" in script
    assert "validate_prepared_formal_root" in evaluator
    assert "validate_formal_runtime_row_binding" in evaluator
    assert "Direct formal keyframe evaluation is hard-disabled" in evaluator
    assert "formal launcher must supply its recorded submission digest" in evaluator
