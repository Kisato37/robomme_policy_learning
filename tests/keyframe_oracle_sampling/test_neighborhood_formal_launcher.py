from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.keyframe_neighborhood_sampling import submit_formal
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.formal_matrix import build_formal_matrix
from experiments.keyframe_neighborhood_sampling.prepare_formal import _require_extension_identity
from experiments.keyframe_neighborhood_sampling.prepare_formal import dry_run_contract
from experiments.keyframe_neighborhood_sampling.submit_formal import MAX_ROWS_PER_ARRAY
from experiments.keyframe_neighborhood_sampling.submit_formal import build_submission
from experiments.keyframe_neighborhood_sampling.submit_formal import shard_rows

REPO = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def test_extension_formal_dry_run_is_exact_and_non_submitting():
    report = dry_run_contract()
    assert report["valid"] is True
    assert report["protocol_family"] == EXTENSION_PROTOCOL_FAMILY
    assert report["arms"] == ["OC3", "OC5"]
    assert report["trajectory_count"] == FORMAL_TRAJECTORY_COUNT == 1600
    assert report["formal_seed_count"] == 16 * 50 * 82
    assert report["fresh_extension_smoke_required"] is True
    assert report["submits_jobs"] is False
    assert report["creates_run_root"] is False


def test_extension_formal_rows_shard_without_exceeding_global_concurrency():
    shards, per_shard_concurrent = shard_rows(list(range(FORMAL_TRAJECTORY_COUNT)), max_concurrent=4)
    assert MAX_ROWS_PER_ARRAY == 1000
    assert [len(shard) for shard in shards] == [1000, 600]
    assert per_shard_concurrent == 2
    assert len(shards) * per_shard_concurrent == 4
    assert [row for shard in shards for row in shard] == list(range(FORMAL_TRAJECTORY_COUNT))
    with pytest.raises(ValueError, match="require --max-concurrent"):
        shard_rows(list(range(FORMAL_TRAJECTORY_COUNT)), max_concurrent=1)


def test_prepare_smoke_gate_rejects_parent_or_partial_evidence():
    checkpoint = {"metadata_sha256": "a" * 64}
    content_tree = {
        "algorithm": "sha256-canonical-file-content-tree-v1",
        "content_tree_sha256": "b" * 64,
    }
    common = {
        "protocol_version": "v1.0",
        "passed": True,
        "formal_started": False,
        "repository_commit_sha": "commit",
        "arms": ["OC3", "OC5"],
        "checkpoint_unpacked_metadata_sha256": "a" * 64,
        "checkpoint_content_tree_algorithm": content_tree["algorithm"],
        "checkpoint_unpacked_content_tree_sha256": "b" * 64,
    }
    with pytest.raises(RuntimeError, match="not fresh"):
        _require_extension_identity(
            {**common, "protocol_family": "keyframe_oracle_sampling_v1"},
            label="Architecture-smoke report",
            current_commit="commit",
            checkpoint=checkpoint,
            checkpoint_content_tree=content_tree,
            require_formal_not_started=True,
        )
    with pytest.raises(RuntimeError, match="exactly OC3 and OC5"):
        _require_extension_identity(
            {
                **common,
                "protocol_family": EXTENSION_PROTOCOL_FAMILY,
                "arms": ["OC3"],
            },
            label="Architecture-smoke report",
            current_commit="commit",
            checkpoint=checkpoint,
            checkpoint_content_tree=content_tree,
            require_formal_not_started=True,
        )


def _stub_prepared_root(tmp_path: Path, monkeypatch) -> Path:
    run_root = tmp_path / "runs" / "keyframe_neighborhood_sampling" / "formal-run"
    protocol = run_root / "protocol"
    protocol.mkdir(parents=True)
    matrix_path = protocol / "formal_matrix.json"
    manifest_path = protocol / "launch_manifest.json"
    _write_json(matrix_path, build_formal_matrix())
    (protocol / "seed_table.json").write_text("{}")
    _write_json(
        manifest_path,
        {
            "protocol_version": "v1.0",
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "repository": {"commit_sha": "commit"},
            "checkpoint_unpacked_metadata_sha256": "a" * 64,
            "checkpoint_content_tree_algorithm": "tree-v1",
            "checkpoint_unpacked_content_tree_sha256": "b" * 64,
            "architecture_report_sha256": "c" * 64,
            "development_smoke_audit_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        submit_formal,
        "repository_state",
        lambda: {"commit_sha": "commit", "dirty": False},
    )
    monkeypatch.setattr(submit_formal, "require_clean_repository", lambda _state: None)
    monkeypatch.setattr(
        submit_formal,
        "validate_formal_prepared_root",
        lambda *_args, **_kwargs: json.loads(manifest_path.read_text()),
    )
    monkeypatch.setattr(submit_formal, "validate_environment_contract", lambda _manifest: None)
    monkeypatch.setattr(
        submit_formal,
        "validate_extension_failure_record",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        submit_formal,
        "checkpoint_identity",
        lambda _path: {"metadata_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        submit_formal,
        "checkpoint_content_tree_identity",
        lambda _path: {"algorithm": "tree-v1", "content_tree_sha256": "b" * 64},
    )
    return run_root


def test_initial_submission_is_exact_1600_and_cannot_be_subset(tmp_path, monkeypatch):
    run_root = _stub_prepared_root(tmp_path, monkeypatch)
    submissions, plan = build_submission(run_root, max_concurrent=4, attempt_id=0, row_ids=None)
    assert plan["protocol_family"] == EXTENSION_PROTOCOL_FAMILY
    assert plan["trajectory_count"] == 1600
    assert [len(record["row_ids"]) for _, record in submissions] == [1000, 600]
    assert [record["array"] for _, record in submissions] == [
        "0-999%2",
        "0-599%2",
    ]
    assert all(
        "experiments/keyframe_neighborhood_sampling/run_formal.sbatch" in command[-5] for command, _ in submissions
    )
    with pytest.raises(ValueError, match="complete 1,600-row matrix"):
        build_submission(run_root, max_concurrent=4, attempt_id=0, row_ids=(0,))


def test_retry_requires_explicit_preceding_retry_allowed_failure(tmp_path, monkeypatch):
    run_root = _stub_prepared_root(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="require explicit --rows"):
        build_submission(run_root, max_concurrent=2, attempt_id=1, row_ids=None)

    row = build_formal_matrix()["rows"][0]
    failure_path = run_root / "failures" / "failure_ledger.jsonl"
    failure_path.parent.mkdir(parents=True)
    failure_path.write_text(
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
    submissions, plan = build_submission(run_root, max_concurrent=2, attempt_id=1, row_ids=(0,))
    assert plan["trajectory_count"] == 1
    assert submissions[0][1]["row_ids"] == [0]


def test_extension_slurm_launcher_routes_only_extension_formal_modules():
    script = (REPO / "experiments/keyframe_neighborhood_sampling/run_formal.sbatch").read_text()
    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --no-requeue" in script
    assert "keyframe_neighborhood_sampling.preflight_formal_row" in script
    assert "keyframe_neighborhood_sampling.record_launcher_failure" in script
    assert "keyframe_neighborhood_sampling.formal_matrix" in script
    assert script.index("preflight_formal_row") < script.index("scripts/serve_policy.py")
    assert '--args.keyframe-selector-arm="${ARM}"' in script
    assert "--args.keyframe-formal-authorization" in script
    assert 'export KEYFRAME_FORMAL_ROW_ID="${ROW_ID}"' in script
    assert "keyframe_oracle_sampling.preflight_formal_row" not in script
    assert "keyframe_oracle_sampling.record_launcher_failure" not in script


def test_second_shard_sbatch_failure_resumes_without_resubmitting_first(tmp_path, monkeypatch):
    run_root = _stub_prepared_root(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    outcomes: list[object] = [
        "11111\n",
        FileNotFoundError("sbatch executable disappeared before shard 1"),
    ]

    def fake_check_output(command, **_kwargs):
        calls.append(command)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(submit_formal.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_formal.py",
            "--run-root",
            str(run_root),
            "--max-concurrent",
            "4",
            "--confirm-authorized-extension-v1",
        ],
    )
    with pytest.raises(FileNotFoundError):
        submit_formal.main()
    first_record_path = submit_formal.submission_path(run_root, 0, 0)
    first_record_bytes = first_record_path.read_bytes()
    assert not submit_formal.submission_path(run_root, 0, 1).exists()
    assert len(submit_formal.submission_failure_paths(run_root, 0, 1)) == 1

    outcomes.append("22222\n")
    submit_formal.main()
    assert len(calls) == 3
    assert calls[0][-1] == "0"
    assert calls[1][-1] == "1"
    assert calls[2][-1] == "1"
    assert first_record_path.read_bytes() == first_record_bytes
    second_record = json.loads(submit_formal.submission_path(run_root, 0, 1).read_text())
    assert second_record["slurm_array_job_id"] == "22222"


def test_ambiguous_second_shard_failure_is_not_automatically_resubmitted(tmp_path, monkeypatch):
    run_root = _stub_prepared_root(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    outcomes: list[object] = [
        "11111\n",
        subprocess.CalledProcessError(1, ["sbatch"], output="ambiguous failure"),
    ]

    def fake_check_output(command, **_kwargs):
        calls.append(command)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(submit_formal.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_formal.py",
            "--run-root",
            str(run_root),
            "--max-concurrent",
            "4",
            "--confirm-authorized-extension-v1",
        ],
    )
    with pytest.raises(subprocess.CalledProcessError):
        submit_formal.main()
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="outcome is uncertain"):
        submit_formal.main()
    assert len(calls) == 2


def test_preparation_does_not_reuse_completed_parent_smoke_constants():
    source = (REPO / "experiments/keyframe_neighborhood_sampling/prepare_formal.py").read_text()
    assert "ACCEPTED_DEVELOPMENT_SMOKE_COMMIT" not in source
    assert "20260829T181633Z_899912b1_devsmoke" not in source
    assert "runs/keyframe_neighborhood_sampling" in source
    assert "fresh extension" in source
