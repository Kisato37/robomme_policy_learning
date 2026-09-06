from __future__ import annotations

import json

import pytest

from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.formal_matrix import (
    FORMAL_TRAJECTORY_COUNT as EXTENSION_TRAJECTORY_COUNT,
)
from experiments.keyframe_neighborhood_sampling.formal_matrix import build_formal_matrix as build_extension_matrix
from experiments.keyframe_neighborhood_sampling.formal_matrix import load_formal_matrix as load_extension_matrix
from experiments.keyframe_neighborhood_sampling.formal_matrix import (
    validate_formal_runtime_row_binding as validate_extension_row,
)
from experiments.keyframe_oracle_sampling.artifacts import ALL_ARMS
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT as ORIGINAL_TRAJECTORY_COUNT
from experiments.keyframe_oracle_sampling.formal_matrix import build_formal_matrix as build_original_matrix


def test_extension_does_not_expand_completed_original_matrix():
    original = build_original_matrix()
    assert ALL_ARMS == ("U", "O", "OC", "R")
    assert original["arms"] == ["U", "O", "OC", "R"]
    assert original["trajectory_count"] == ORIGINAL_TRAJECTORY_COUNT == 3200


def test_extension_matrix_is_exact_paired_1600_row_contract():
    matrix = build_extension_matrix()
    assert EXTENSION_ARMS == ("OC3", "OC5")
    assert matrix["protocol_family"] == EXTENSION_PROTOCOL_FAMILY
    assert matrix["trajectory_count"] == EXTENSION_TRAJECTORY_COUNT == 1600
    assert matrix["rows"][0] == {
        "row_id": 0,
        "task": "BinFill",
        "episode_id": 0,
        "arm": "OC3",
        "trajectory_kind": "formal",
        "max_steps": 1300,
        "dataset": "test",
    }
    assert matrix["rows"][1]["arm"] == "OC5"
    assert matrix["rows"][-1] == {
        "row_id": 1599,
        "task": "RouteStick",
        "episode_id": 49,
        "arm": "OC5",
        "trajectory_kind": "formal",
        "max_steps": 1300,
        "dataset": "test",
    }


def test_extension_matrix_loader_rejects_any_semantic_change(tmp_path):
    path = tmp_path / "formal_matrix.json"
    payload = build_extension_matrix()
    path.write_text(json.dumps(payload))
    assert load_extension_matrix(path) == payload
    payload["rows"][0]["arm"] = "OC"
    path.write_text(json.dumps(payload))
    with pytest.raises(Exception, match="frozen 16 x 50 x 2"):
        load_extension_matrix(path)


def test_extension_runtime_binding_requires_recorded_array_mapping(tmp_path):
    run_root = tmp_path / "run"
    protocol_dir = run_root / "protocol"
    protocol_dir.mkdir(parents=True)
    (protocol_dir / "formal_matrix.json").write_text(
        json.dumps(build_extension_matrix())
    )
    submission = {
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "slurm_array_job_id": "123",
        "attempt_id": 0,
        "trajectory_count": 1,
        "array_task_ids": [0],
        "row_ids": [0],
    }
    (protocol_dir / "submission_record_shard_00.json").write_text(
        json.dumps(submission)
    )
    row = validate_extension_row(
        run_root,
        attempt_id=0,
        row_id=0,
        task="BinFill",
        episode_id=0,
        arm="OC3",
        trajectory_kind="formal",
        max_steps=1300,
        dataset="test",
        environ={"SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0"},
    )
    assert row["arm"] == "OC3"
