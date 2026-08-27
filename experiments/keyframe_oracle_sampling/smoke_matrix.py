#!/usr/bin/env python3
"""Validate and expand the frozen 64-short + 16-terminal smoke matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.keyframe_oracle_sampling.artifacts import ALL_ARMS
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError

MATRIX_PATH = Path(__file__).with_name("SMOKE_MATRIX.json")


def load_frozen_matrix(path: Path = MATRIX_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("dataset") != "val" or int(payload.get("episode_id", -1)) != 0:
        raise ValueError("Smoke matrix must use val episode 0")
    if int(payload.get("short_max_steps", -1)) != 64:
        raise ValueError("Smoke short trajectories must run through 64 environment steps")
    if int(payload.get("terminal_max_steps", -1)) != 1300:
        raise ValueError("Smoke terminal paths must retain the official 1300-step limit")
    task_rows = payload.get("tasks", [])
    tasks = tuple(row.get("task") for row in task_rows)
    if tasks != FORMAL_TASKS:
        raise ValueError("Smoke tasks/order differ from the frozen 16-task protocol list")
    for index, row in enumerate(task_rows):
        expected_arm = ALL_ARMS[index % len(ALL_ARMS)]
        if row.get("terminal_arm") != expected_arm:
            raise ValueError(
                f"Terminal arm rotation mismatch for {row.get('task')}: "
                f"{row.get('terminal_arm')} != {expected_arm}"
            )
    return payload


def expand_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task_row in payload["tasks"]:
        for arm in ALL_ARMS:
            rows.append(
                {
                    "task": task_row["task"],
                    "episode_id": 0,
                    "arm": arm,
                    "trajectory_kind": "short",
                    "max_steps": 64,
                    "dataset": "val",
                }
            )
        rows.append(
            {
                "task": task_row["task"],
                "episode_id": 0,
                "arm": task_row["terminal_arm"],
                "trajectory_kind": "terminal",
                "max_steps": 1300,
                "dataset": "val",
            }
        )
    if len(rows) != 80:
        raise AssertionError(f"Expected 80 smoke trajectories, got {len(rows)}")
    keys = {
        (row["task"], row["episode_id"], row["arm"], row["trajectory_kind"])
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("Frozen smoke matrix contains duplicate trajectory keys")
    return rows


def validate_runtime_row_binding(
    run_root: str | Path,
    *,
    attempt_id: int,
    row_id: int,
    task: str,
    episode_id: int,
    arm: str,
    trajectory_kind: str,
    max_steps: int,
    dataset: str,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Bind one evaluator invocation to its recorded submission and frozen row."""
    run_root = Path(run_root).resolve()
    environ = dict(os.environ if environ is None else environ)
    matrix_path = run_root / "protocol" / "smoke_matrix.json"
    if not matrix_path.is_file():
        raise ArtifactContractError(f"Prepared smoke matrix is missing: {matrix_path}")
    rows = expand_rows(load_frozen_matrix(matrix_path))
    if row_id < 0 or row_id >= len(rows):
        raise ArtifactContractError(f"Smoke row {row_id} outside 0..{len(rows) - 1}")

    submission_name = (
        "submission_record.json"
        if attempt_id == 0
        else f"submission_record_attempt_{attempt_id:02d}.json"
    )
    submission_path = run_root / "protocol" / submission_name
    if not submission_path.is_file():
        raise ArtifactContractError(f"Smoke submission is missing: {submission_path}")
    submission = json.loads(submission_path.read_text())
    if int(submission.get("attempt_id", -1)) != attempt_id:
        raise ArtifactContractError("Runtime attempt ID differs from the submission record")

    active_array_task = environ.get("SLURM_ARRAY_TASK_ID")
    try:
        active_row_id = int(active_array_task) if active_array_task is not None else -1
    except ValueError as exc:
        raise ArtifactContractError("SLURM_ARRAY_TASK_ID is not an integer") from exc
    if active_row_id != row_id:
        raise ArtifactContractError(
            f"Runtime row {row_id} differs from SLURM_ARRAY_TASK_ID {active_array_task!r}"
        )
    if submission.get("slurm_array_job_id") != environ.get("SLURM_ARRAY_JOB_ID"):
        raise ArtifactContractError("Runtime Slurm array differs from the submission record")

    raw_row_ids = submission.get("row_ids")
    if not isinstance(raw_row_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_row_ids
    ):
        raise ArtifactContractError("Smoke submission row_ids must be an integer list")
    if len(raw_row_ids) != len(set(raw_row_ids)):
        raise ArtifactContractError("Smoke submission row_ids contain duplicates")
    if row_id not in raw_row_ids:
        raise ArtifactContractError(f"Smoke submission does not authorize row {row_id}")

    expected = rows[row_id]
    runtime = {
        "task": task,
        "episode_id": int(episode_id),
        "arm": arm,
        "trajectory_kind": trajectory_kind,
        "max_steps": int(max_steps),
        "dataset": dataset,
    }
    mismatches = {
        field: {"expected": expected[field], "runtime": runtime[field]}
        for field in expected
        if expected[field] != runtime[field]
    }
    if mismatches:
        raise ArtifactContractError(
            "Runtime smoke parameters differ from the frozen row: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int)
    parser.add_argument("--row-tsv", action="store_true")
    args = parser.parse_args()
    rows = expand_rows(load_frozen_matrix())
    if args.row is None:
        print(json.dumps({"trajectory_count": len(rows), "rows": rows}, indent=2))
        return
    if args.row < 0 or args.row >= len(rows):
        raise IndexError(f"Smoke row {args.row} outside 0..{len(rows) - 1}")
    row = rows[args.row]
    if args.row_tsv:
        print(
            "\t".join(
                str(row[key])
                for key in ("task", "arm", "trajectory_kind", "max_steps", "dataset")
            )
        )
    else:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
