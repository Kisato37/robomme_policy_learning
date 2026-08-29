#!/usr/bin/env python3
"""Build and bind the exact frozen 3,200-row formal evaluation matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.keyframe_oracle_sampling.artifacts import ALL_ARMS
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError

FORMAL_EPISODE_IDS = tuple(range(50))
FORMAL_DATASET = "test"
FORMAL_MAX_STEPS = 1300
FORMAL_TRAJECTORY_KIND = "formal"
FORMAL_TRAJECTORY_COUNT = len(FORMAL_TASKS) * len(FORMAL_EPISODE_IDS) * len(ALL_ARMS)


def build_formal_matrix() -> dict[str, Any]:
    """Return the canonical task-major, episode-major, arm-major matrix."""
    rows: list[dict[str, Any]] = []
    for task in FORMAL_TASKS:
        for episode_id in FORMAL_EPISODE_IDS:
            for arm in ALL_ARMS:
                rows.append(
                    {
                        "row_id": len(rows),
                        "task": task,
                        "episode_id": episode_id,
                        "arm": arm,
                        "trajectory_kind": FORMAL_TRAJECTORY_KIND,
                        "max_steps": FORMAL_MAX_STEPS,
                        "dataset": FORMAL_DATASET,
                    }
                )
    if len(rows) != FORMAL_TRAJECTORY_COUNT:
        raise AssertionError(f"Expected {FORMAL_TRAJECTORY_COUNT} formal trajectories, got {len(rows)}")
    return {
        "schema_version": 1,
        "order": ["task", "episode_id", "arm"],
        "tasks": list(FORMAL_TASKS),
        "episode_ids": list(FORMAL_EPISODE_IDS),
        "arms": list(ALL_ARMS),
        "dataset": FORMAL_DATASET,
        "max_steps": FORMAL_MAX_STEPS,
        "trajectory_kind": FORMAL_TRAJECTORY_KIND,
        "trajectory_count": FORMAL_TRAJECTORY_COUNT,
        "rows": rows,
    }


def load_formal_matrix(path: str | Path) -> dict[str, Any]:
    """Load a prepared matrix and require byte-independent exact semantics."""
    payload = json.loads(Path(path).read_text())
    expected = build_formal_matrix()
    if payload != expected:
        raise ArtifactContractError("Prepared formal matrix differs from the frozen 16 x 50 x 4 contract")
    return payload


def _mapped_submission_row_id(submission: dict[str, Any], array_task_id: int) -> int:
    raw_array_task_ids = submission.get("array_task_ids")
    raw_row_ids = submission.get("row_ids")
    for field, values in (("array_task_ids", raw_array_task_ids), ("row_ids", raw_row_ids)):
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ArtifactContractError(f"Formal submission {field} must be an integer list")
        if len(values) != len(set(values)):
            raise ArtifactContractError(f"Formal submission {field} contains duplicates")
    if len(raw_array_task_ids) != len(raw_row_ids):
        raise ArtifactContractError("Formal submission array-task/row mapping has different lengths")
    if int(submission.get("trajectory_count", -1)) != len(raw_row_ids):
        raise ArtifactContractError("Formal submission trajectory_count/row mapping mismatch")
    try:
        mapping_index = raw_array_task_ids.index(array_task_id)
    except ValueError as exc:
        raise ArtifactContractError(
            f"SLURM_ARRAY_TASK_ID {array_task_id} is not authorized by the formal submission"
        ) from exc
    return raw_row_ids[mapping_index]


def load_formal_submission_row(
    run_root: str | Path,
    *,
    attempt_id: int,
    shard_id: int,
    array_task_id: int,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Map a shard-local Slurm task ID to its frozen global matrix row."""
    run_root = Path(run_root).resolve()
    environ = dict(os.environ if environ is None else environ)
    submission_name = (
        f"submission_record_shard_{shard_id:02d}.json"
        if attempt_id == 0
        else f"submission_record_attempt_{attempt_id:02d}_shard_{shard_id:02d}.json"
    )
    submission_path = run_root / "protocol" / submission_name
    if not submission_path.is_file():
        raise ArtifactContractError(f"Formal shard submission is missing: {submission_path}")
    submission = json.loads(submission_path.read_text())
    if int(submission.get("attempt_id", -1)) != attempt_id:
        raise ArtifactContractError("Runtime attempt ID differs from formal submission")
    if int(submission.get("shard_id", -1)) != shard_id:
        raise ArtifactContractError("Runtime shard ID differs from formal submission")
    if submission.get("slurm_array_job_id") != environ.get("SLURM_ARRAY_JOB_ID"):
        raise ArtifactContractError("Runtime Slurm array differs from formal shard submission")
    row_id = _mapped_submission_row_id(submission, array_task_id)
    rows = load_formal_matrix(run_root / "protocol" / "formal_matrix.json")["rows"]
    if row_id < 0 or row_id >= len(rows):
        raise ArtifactContractError(f"Formal row {row_id} outside 0..{len(rows) - 1}")
    return rows[row_id]


def validate_formal_runtime_row_binding(
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
    """Bind one formal evaluator process to its recorded Slurm row."""
    run_root = Path(run_root).resolve()
    environ = dict(os.environ if environ is None else environ)
    matrix_path = run_root / "protocol" / "formal_matrix.json"
    if not matrix_path.is_file():
        raise ArtifactContractError(f"Prepared formal matrix is missing: {matrix_path}")
    rows = load_formal_matrix(matrix_path)["rows"]
    if row_id < 0 or row_id >= len(rows):
        raise ArtifactContractError(f"Formal row {row_id} outside 0..{len(rows) - 1}")

    submission_pattern = (
        "submission_record_shard_*.json"
        if attempt_id == 0
        else f"submission_record_attempt_{attempt_id:02d}_shard_*.json"
    )
    active_array_job = environ.get("SLURM_ARRAY_JOB_ID")
    matching_submissions = []
    for submission_path in sorted((run_root / "protocol").glob(submission_pattern)):
        submission = json.loads(submission_path.read_text())
        if submission.get("slurm_array_job_id") == active_array_job:
            matching_submissions.append(submission)
    if len(matching_submissions) != 1:
        raise ArtifactContractError("Runtime Slurm array does not match exactly one formal shard submission")
    submission = matching_submissions[0]
    if int(submission.get("attempt_id", -1)) != attempt_id:
        raise ArtifactContractError("Runtime attempt ID differs from formal submission")

    active_array_task = environ.get("SLURM_ARRAY_TASK_ID")
    try:
        array_task_id = int(active_array_task) if active_array_task is not None else -1
    except ValueError as exc:
        raise ArtifactContractError("SLURM_ARRAY_TASK_ID is not an integer") from exc
    mapped_row_id = _mapped_submission_row_id(submission, array_task_id)
    if mapped_row_id != row_id:
        raise ArtifactContractError(
            f"Runtime formal row {row_id} differs from the recorded mapping for "
            f"SLURM_ARRAY_TASK_ID {active_array_task!r}"
        )

    expected = rows[row_id]
    runtime = {
        "row_id": int(row_id),
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
            "Runtime formal parameters differ from the frozen row: " + json.dumps(mismatches, sort_keys=True)
        )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int)
    parser.add_argument("--row-tsv", action="store_true")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--attempt-id", type=int)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--array-task-id", type=int)
    parser.add_argument("--submission-row-tsv", action="store_true")
    args = parser.parse_args()
    payload = build_formal_matrix()
    rows = payload["rows"]
    if args.submission_row_tsv:
        required = (args.run_root, args.attempt_id, args.shard_id, args.array_task_id)
        if any(value is None for value in required):
            parser.error(
                "--submission-row-tsv requires --run-root, --attempt-id, --shard-id, and --array-task-id"
            )
        row = load_formal_submission_row(
            args.run_root,
            attempt_id=args.attempt_id,
            shard_id=args.shard_id,
            array_task_id=args.array_task_id,
        )
        print(
            "\t".join(
                str(row[key])
                for key in (
                    "row_id",
                    "task",
                    "episode_id",
                    "arm",
                    "trajectory_kind",
                    "max_steps",
                    "dataset",
                )
            )
        )
        return
    if args.row is None:
        print(json.dumps(payload, indent=2))
        return
    if args.row < 0 or args.row >= len(rows):
        raise IndexError(f"Formal row {args.row} outside 0..{len(rows) - 1}")
    row = rows[args.row]
    if args.row_tsv:
        print(
            "\t".join(
                str(row[key])
                for key in (
                    "task",
                    "episode_id",
                    "arm",
                    "trajectory_kind",
                    "max_steps",
                    "dataset",
                )
            )
        )
    else:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
