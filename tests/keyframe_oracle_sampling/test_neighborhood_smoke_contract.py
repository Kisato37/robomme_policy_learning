from __future__ import annotations

import json

import pytest

from experiments.keyframe_neighborhood_sampling.smoke_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.smoke_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.smoke_matrix import MATRIX_PATH
from experiments.keyframe_neighborhood_sampling.smoke_matrix import SMOKE_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.smoke_matrix import build_smoke_matrix
from experiments.keyframe_neighborhood_sampling.smoke_matrix import load_smoke_matrix
from experiments.keyframe_neighborhood_sampling.smoke_matrix import validate_runtime_row_binding
from experiments.keyframe_oracle_sampling.artifacts import ALL_ARMS
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.smoke_matrix import expand_rows as expand_original_smoke_rows
from experiments.keyframe_oracle_sampling.smoke_matrix import load_frozen_matrix as load_original_smoke_matrix


def test_extension_smoke_matrix_is_exact_48_row_contract():
    matrix = load_smoke_matrix(MATRIX_PATH)
    assert matrix == build_smoke_matrix()
    assert EXTENSION_ARMS == ("OC3", "OC5")
    assert matrix["protocol_family"] == EXTENSION_PROTOCOL_FAMILY
    assert matrix["trajectory_count"] == SMOKE_TRAJECTORY_COUNT == 48

    rows = matrix["rows"]
    assert [row["row_id"] for row in rows] == list(range(48))
    short = [row for row in rows if row["trajectory_kind"] == "short"]
    terminal = [row for row in rows if row["trajectory_kind"] == "terminal"]
    assert len(short) == 32
    assert len(terminal) == 16

    for task_index, task in enumerate(FORMAL_TASKS):
        task_short = [row for row in short if row["task"] == task]
        task_terminal = [row for row in terminal if row["task"] == task]
        assert {row["arm"] for row in task_short} == set(EXTENSION_ARMS)
        assert all(row["episode_id"] == 0 and row["dataset"] == "val" and row["max_steps"] == 64 for row in task_short)
        assert len(task_terminal) == 1
        assert task_terminal[0]["arm"] == EXTENSION_ARMS[task_index % 2]
        assert task_terminal[0]["episode_id"] == 0
        assert task_terminal[0]["dataset"] == "val"
        assert task_terminal[0]["max_steps"] == 1300


def test_original_smoke_matrix_permanently_remains_80_rows():
    rows = expand_original_smoke_rows(load_original_smoke_matrix())
    assert ALL_ARMS == ("U", "O", "OC", "R")
    assert len(rows) == 80
    assert sum(row["trajectory_kind"] == "short" for row in rows) == 64
    assert sum(row["trajectory_kind"] == "terminal" for row in rows) == 16
    assert {row["arm"] for row in rows} == set(ALL_ARMS)
    assert not {"OC3", "OC5"} & {row["arm"] for row in rows}


def test_extension_smoke_loader_rejects_any_change(tmp_path):
    path = tmp_path / "smoke_matrix.json"
    payload = build_smoke_matrix()
    path.write_text(json.dumps(payload))
    assert load_smoke_matrix(path) == payload

    payload["rows"][0]["max_steps"] = 65
    path.write_text(json.dumps(payload))
    with pytest.raises(ArtifactContractError, match="frozen"):
        load_smoke_matrix(path)


def _runtime_root(tmp_path, *, submission_overrides=None):
    run_root = tmp_path / "run"
    protocol_dir = run_root / "protocol"
    protocol_dir.mkdir(parents=True)
    (protocol_dir / "smoke_matrix.json").write_text(json.dumps(build_smoke_matrix()))
    submission = {
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "attempt_id": 0,
        "slurm_array_job_id": "123",
        "row_ids": [0],
    }
    if submission_overrides:
        submission.update(submission_overrides)
    (protocol_dir / "submission_record.json").write_text(json.dumps(submission))
    return run_root


def _bind_first_row(run_root, *, environ=None, **runtime_overrides):
    row = build_smoke_matrix()["rows"][0]
    runtime = {
        "run_root": run_root,
        "attempt_id": 0,
        "row_id": row["row_id"],
        "task": row["task"],
        "episode_id": row["episode_id"],
        "arm": row["arm"],
        "trajectory_kind": row["trajectory_kind"],
        "max_steps": row["max_steps"],
        "dataset": row["dataset"],
        "environ": {
            "SLURM_ARRAY_JOB_ID": "123",
            "SLURM_ARRAY_TASK_ID": "0",
        },
    }
    runtime.update(runtime_overrides)
    if environ is not None:
        runtime["environ"] = environ
    return validate_runtime_row_binding(**runtime)


def test_extension_smoke_runtime_binding_accepts_only_exact_submitted_row(tmp_path):
    run_root = _runtime_root(tmp_path)
    assert _bind_first_row(run_root) == build_smoke_matrix()["rows"][0]

    with pytest.raises(ArtifactContractError, match="frozen row"):
        _bind_first_row(run_root, arm="OC5")
    with pytest.raises(ArtifactContractError, match="SLURM_ARRAY_TASK_ID"):
        _bind_first_row(
            run_root,
            environ={"SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "1"},
        )


@pytest.mark.parametrize(
    ("submission_overrides", "message"),
    [
        ({"protocol_family": "wrong-family"}, "not an OC3/OC5"),
        ({"slurm_array_job_id": "other"}, "Slurm array"),
        ({"row_ids": [1]}, "does not authorize"),
        ({"row_ids": [0, 0]}, "duplicates"),
    ],
)
def test_extension_smoke_runtime_binding_fails_closed_on_submission_mismatch(tmp_path, submission_overrides, message):
    run_root = _runtime_root(tmp_path, submission_overrides=submission_overrides)
    with pytest.raises(ArtifactContractError, match=message):
        _bind_first_row(run_root)
