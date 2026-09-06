from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.keyframe_neighborhood_sampling import submit_smoke
from experiments.keyframe_neighborhood_sampling.architecture_smoke import CASE_COUNT
from experiments.keyframe_neighborhood_sampling.architecture_smoke import dry_run_contract as architecture_dry_run
from experiments.keyframe_neighborhood_sampling.architecture_smoke import validate_architecture_pass_report
from experiments.keyframe_neighborhood_sampling.audit_smoke import expected_smoke_keys
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.prepare_smoke import dry_run_contract as prepare_dry_run
from experiments.keyframe_neighborhood_sampling.record_launcher_failure import record_launcher_failure
from experiments.keyframe_neighborhood_sampling.record_launcher_failure import validate_extension_failure_record
from experiments.keyframe_neighborhood_sampling.smoke_matrix import SMOKE_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.smoke_matrix import build_smoke_matrix
from experiments.keyframe_neighborhood_sampling.submit_smoke import build_submission
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_POLICY_CALLS
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_SCOPE

REPO = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def test_extension_smoke_dry_contracts_are_isolated_and_non_submitting():
    preparation = prepare_dry_run()
    architecture = architecture_dry_run()
    assert preparation["protocol_family"] == EXTENSION_PROTOCOL_FAMILY
    assert preparation["arms"] == list(EXTENSION_ARMS)
    assert preparation["trajectory_count"] == SMOKE_TRAJECTORY_COUNT == 48
    assert preparation["formal_launch_authorized"] is False
    assert preparation["creates_run_root"] is False
    assert preparation["submits_jobs"] is False
    assert architecture["case_count"] == CASE_COUNT == 4
    assert {case["arm"] for case in architecture["cases"]} == set(EXTENSION_ARMS)
    assert architecture["runs_gpu_inference"] is False
    assert architecture["submits_jobs"] is False
    assert len(expected_smoke_keys()) == 48


def _architecture_contract(run_root: Path) -> tuple[dict, dict]:
    common = {
        "protocol_version": "v1.0",
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "formal_started": False,
        "arms": list(EXTENSION_ARMS),
        "repository_commit_sha": "a" * 40,
        "run_root": str(run_root.resolve()),
        "slurm_job_id": "123",
        "checkpoint_unpacked_metadata_sha256": "b" * 64,
        "checkpoint_content_tree_algorithm": "sha256-canonical-file-content-tree-v1",
        "checkpoint_unpacked_content_tree_sha256": "c" * 64,
    }
    report = {
        **common,
        "passed": True,
        "case_count": 4,
        "cases": [
            {"arm": arm, "history_length": length, "passed": True} for arm in EXTENSION_ARMS for length in (16, 64)
        ],
        "final_reset_evidence": {"passed": True},
        "formal_test_outcomes_opened": False,
    }
    return report, common


def test_architecture_report_rejects_parent_identity_and_partial_cases(tmp_path):
    run_root = tmp_path / "run"
    report, submission = _architecture_contract(run_root)
    validate_architecture_pass_report(report, run_root=run_root, architecture_submission=submission)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        validate_architecture_pass_report(
            {**report, "protocol_family": "keyframe_oracle_sampling_v1"},
            run_root=run_root,
            architecture_submission=submission,
        )
    with pytest.raises(RuntimeError, match="exactly four cases"):
        validate_architecture_pass_report(
            {**report, "cases": report["cases"][:-1]},
            run_root=run_root,
            architecture_submission=submission,
        )


def _stub_submittable_root(tmp_path: Path, monkeypatch) -> Path:
    run_root = tmp_path / "runs/keyframe_neighborhood_sampling/smoke-run"
    protocol = run_root / "protocol"
    protocol.mkdir(parents=True)
    matrix_path = protocol / "smoke_matrix.json"
    seed_path = protocol / "seed_table.json"
    formal_seed_path = protocol / "formal_seed_audit_table.json"
    architecture_path = run_root / "architecture_smoke/report.json"
    architecture_submission_path = protocol / "architecture_submission_record.json"
    _write_json(matrix_path, build_smoke_matrix())
    _write_json(seed_path, {})
    _write_json(formal_seed_path, {})
    _write_json(
        architecture_path,
        {
            "repository_commit_sha": "commit",
            "checkpoint_unpacked_metadata_sha256": "b" * 64,
            "checkpoint_content_tree_algorithm": "tree-v1",
            "checkpoint_unpacked_content_tree_sha256": "c" * 64,
        },
    )
    _write_json(architecture_submission_path, {})
    smoke_seed = {
        "scope": SMOKE_SEED_SCOPE,
        "dataset": "val",
        "derivation": "smoke",
        "entries_sha256": "d" * 64,
    }
    formal_seed = {
        "scope": FORMAL_SEED_SCOPE,
        "dataset": "test",
        "derivation": "formal",
        "entries_sha256": "e" * 64,
    }
    manifest_path = protocol / "launch_manifest.json"
    _write_json(
        manifest_path,
        {
            "protocol_version": "v1.0",
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "run_kind": "development_smoke",
            "dataset": "val",
            "trajectory_count": 48,
            "formal_launch_authorized": False,
            "repository": {"commit_sha": "commit"},
            "matrix": build_smoke_matrix()["rows"],
            "smoke_matrix_sha256": sha256_file(matrix_path),
            "seed_table_file_sha256": sha256_file(seed_path),
            "formal_seed_audit_file_sha256": sha256_file(formal_seed_path),
            "seed_table_scope": smoke_seed["scope"],
            "seed_table_dataset": smoke_seed["dataset"],
            "seed_table_derivation": smoke_seed["derivation"],
            "seed_table_entries_sha256": smoke_seed["entries_sha256"],
            "seed_disjointness_audit": {"intersection_count": 0},
            "checkpoint_unpacked_metadata_sha256": "b" * 64,
            "checkpoint_content_tree_algorithm": "tree-v1",
            "checkpoint_unpacked_content_tree_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(submit_smoke, "repository_state", lambda: {"commit_sha": "commit", "dirty": False})
    monkeypatch.setattr(submit_smoke, "require_clean_repository", lambda _state: None)
    monkeypatch.setattr(submit_smoke, "validate_environment_contract", lambda _manifest: None)
    monkeypatch.setattr(
        submit_smoke,
        "validate_extension_failure_record",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(submit_smoke, "validate_architecture_pass_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        submit_smoke,
        "checkpoint_identity",
        lambda _path: {"metadata_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        submit_smoke,
        "checkpoint_content_tree_identity",
        lambda _path: {"algorithm": "tree-v1", "content_tree_sha256": "c" * 64},
    )

    def fake_load_seed(path, *, expected_scope, expected_dataset):
        if Path(path).name == "seed_table.json":
            lookup = {(task, 0, call): call for task in FORMAL_TASKS for call in range(MAX_POLICY_CALLS)}
            return smoke_seed, lookup
        lookup = {
            (task, episode, call): call
            for task in FORMAL_TASKS
            for episode in range(50)
            for call in range(MAX_POLICY_CALLS)
        }
        return formal_seed, lookup

    monkeypatch.setattr(submit_smoke, "load_seed_table", fake_load_seed)
    monkeypatch.setattr(
        submit_smoke,
        "validate_smoke_formal_seed_disjointness",
        lambda *_args: {"intersection_count": 0},
    )
    return run_root


def test_initial_smoke_submission_is_exact_48_and_dry_build_is_non_mutating(tmp_path, monkeypatch):
    run_root = _stub_submittable_root(tmp_path, monkeypatch)
    command, record = build_submission(run_root, max_concurrent=4, attempt_id=0, row_ids=None)
    assert record["protocol_family"] == EXTENSION_PROTOCOL_FAMILY
    assert record["row_ids"] == list(range(48))
    assert record["array"] == "0-47%4"
    assert record["smoke_launch_authorized"] is True
    assert record["formal_launch_authorized"] is False
    assert "keyframe_neighborhood_sampling/run_smoke.sbatch" in command[-4]
    assert not (run_root / "slurm").exists()
    assert not (run_root / "protocol/submission_record.json").exists()
    with pytest.raises(ValueError, match="all 48 rows"):
        build_submission(run_root, max_concurrent=4, attempt_id=0, row_ids=(0,))


def test_smoke_retry_requires_one_explicit_allowed_preceding_failure(tmp_path, monkeypatch):
    run_root = _stub_submittable_root(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="explicit --rows"):
        build_submission(run_root, max_concurrent=2, attempt_id=1, row_ids=None)
    row = build_smoke_matrix()["rows"][0]
    ledger = run_root / "failures/failure_ledger.jsonl"
    ledger.parent.mkdir()
    ledger.write_text(
        json.dumps(
            {
                "task": row["task"],
                "episode_id": row["episode_id"],
                "arm": row["arm"],
                "trajectory_kind": row["trajectory_kind"],
                "attempt_id": 0,
                "retry_allowed": True,
            }
        )
        + "\n"
    )
    _, record = build_submission(run_root, max_concurrent=2, attempt_id=1, row_ids=(0,))
    assert record["row_ids"] == [0]
    assert record["array"] == "0%2"


def test_extension_readiness_failure_manifest_carries_protocol_family(tmp_path):
    run_root = tmp_path / "run"
    attempt_dir = record_launcher_failure(
        run_root,
        attempt_id=0,
        row_id=0,
        task="BinFill",
        episode_id=0,
        arm="OC3",
        trajectory_kind="short",
        max_steps=64,
        dataset="val",
        error_type="PolicyServerReadinessTimeout",
        error="timeout",
        slurm={"array_job_id": "123", "array_task_id": "0"},
    )
    manifest = json.loads((attempt_dir / "episode_manifest.json").read_text())
    assert manifest["protocol_family"] == EXTENSION_PROTOCOL_FAMILY
    ledger = json.loads((run_root / "failures/failure_ledger.jsonl").read_text())
    assert ledger["protocol_family"] == EXTENSION_PROTOCOL_FAMILY
    validate_extension_failure_record(run_root, ledger, expected_row_id=0)
    bare = dict(ledger)
    bare.pop("protocol_family")
    with pytest.raises(Exception, match="provenance"):
        validate_extension_failure_record(run_root, bare, expected_row_id=0)


def test_extension_smoke_launchers_route_extension_modules_and_two_gpus():
    script = (REPO / "experiments/keyframe_neighborhood_sampling/run_smoke.sbatch").read_text()
    architecture = (REPO / "experiments/keyframe_neighborhood_sampling/run_architecture_smoke.sbatch").read_text()
    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --no-requeue" in script
    assert "keyframe_neighborhood_sampling.preflight_smoke_row" in script
    assert "keyframe_neighborhood_sampling.record_launcher_failure" in script
    assert "keyframe_neighborhood_sampling.smoke_matrix" in script
    assert script.index("preflight_smoke_row") < script.index("scripts/serve_policy.py")
    assert '--args.keyframe-selector-arm="${ARM}"' in script
    assert "--args.keyframe-formal-authorization" not in script
    assert 'export KEYFRAME_SMOKE_ROW_ID="${ROW_ID}"' in script
    assert "#SBATCH --gres=gpu:1" in architecture
    assert "keyframe_neighborhood_sampling.architecture_smoke" in architecture
