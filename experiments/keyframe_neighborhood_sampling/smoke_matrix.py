#!/usr/bin/env python3
"""Build, load, and bind the frozen 48-row OC3/OC5 smoke matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.keyframe_neighborhood_sampling.direct_provenance import require_runtime_backend
from experiments.keyframe_neighborhood_sampling.direct_provenance import runtime_direct_provenance
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError

MATRIX_PATH = Path(__file__).with_name("SMOKE_MATRIX.json")
SMOKE_EPISODE_ID = 0
SMOKE_DATASET = "val"
SHORT_MAX_STEPS = 64
TERMINAL_MAX_STEPS = 1300
SMOKE_TRAJECTORY_COUNT = len(FORMAL_TASKS) * (len(EXTENSION_ARMS) + 1)


def build_smoke_matrix() -> dict[str, Any]:
    """Return the exact 32-short + 16-terminal development-smoke matrix."""
    rows: list[dict[str, Any]] = []
    terminal_arms: dict[str, str] = {}
    for task_index, task in enumerate(FORMAL_TASKS):
        for arm in EXTENSION_ARMS:
            rows.append(
                {
                    "row_id": len(rows),
                    "task": task,
                    "episode_id": SMOKE_EPISODE_ID,
                    "arm": arm,
                    "trajectory_kind": "short",
                    "max_steps": SHORT_MAX_STEPS,
                    "dataset": SMOKE_DATASET,
                }
            )
        terminal_arm = EXTENSION_ARMS[task_index % len(EXTENSION_ARMS)]
        terminal_arms[task] = terminal_arm
        rows.append(
            {
                "row_id": len(rows),
                "task": task,
                "episode_id": SMOKE_EPISODE_ID,
                "arm": terminal_arm,
                "trajectory_kind": "terminal",
                "max_steps": TERMINAL_MAX_STEPS,
                "dataset": SMOKE_DATASET,
            }
        )

    if len(rows) != SMOKE_TRAJECTORY_COUNT:
        raise AssertionError(f"Expected {SMOKE_TRAJECTORY_COUNT} extension smoke trajectories, got {len(rows)}")
    keys = {(row["task"], row["episode_id"], row["arm"], row["trajectory_kind"]) for row in rows}
    if len(keys) != len(rows):
        raise AssertionError("Extension smoke matrix contains duplicate trajectory keys")

    return {
        "schema_version": 1,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "order": ["task", "trajectory_kind", "arm"],
        "tasks": list(FORMAL_TASKS),
        "episode_id": SMOKE_EPISODE_ID,
        "arms": list(EXTENSION_ARMS),
        "terminal_arms": terminal_arms,
        "dataset": SMOKE_DATASET,
        "short_max_steps": SHORT_MAX_STEPS,
        "terminal_max_steps": TERMINAL_MAX_STEPS,
        "trajectory_count": SMOKE_TRAJECTORY_COUNT,
        "rows": rows,
    }


def load_smoke_matrix(path: str | Path = MATRIX_PATH) -> dict[str, Any]:
    """Load a prepared smoke matrix and reject any semantic difference."""
    payload = json.loads(Path(path).read_text())
    expected = build_smoke_matrix()
    if payload != expected:
        raise ArtifactContractError(
            "Prepared extension smoke matrix differs from the frozen 16-task x (2 short + 1 terminal) contract"
        )
    return payload


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
    """Bind one evaluator invocation to an exact submitted extension-smoke row."""
    run_root = Path(run_root).resolve()
    environ = dict(os.environ if environ is None else environ)
    matrix_path = run_root / "protocol" / "smoke_matrix.json"
    if not matrix_path.is_file():
        raise ArtifactContractError(f"Prepared extension smoke matrix is missing: {matrix_path}")
    rows = load_smoke_matrix(matrix_path)["rows"]
    if row_id < 0 or row_id >= len(rows):
        raise ArtifactContractError(f"Extension smoke row {row_id} outside 0..{len(rows) - 1}")

    submission_name = (
        "submission_record.json" if attempt_id == 0 else f"submission_record_attempt_{attempt_id:02d}.json"
    )
    submission_path = run_root / "protocol" / submission_name
    if not submission_path.is_file():
        raise ArtifactContractError(f"Extension smoke submission is missing: {submission_path}")
    submission = json.loads(submission_path.read_text())
    if submission.get("protocol_family") != EXTENSION_PROTOCOL_FAMILY:
        raise ArtifactContractError("Submission is not an OC3/OC5 smoke record")
    if int(submission.get("attempt_id", -1)) != attempt_id:
        raise ArtifactContractError("Runtime attempt ID differs from the extension smoke submission")
    backend = require_runtime_backend(submission, environ)
    if backend == "direct":
        runtime_direct_provenance(
            run_root, stage="development_smoke", attempt_id=attempt_id, row_id=row_id,
            submission=submission, submission_path=submission_path, environ=environ,
        )
    elif submission.get("slurm_array_job_id") != environ.get("SLURM_ARRAY_JOB_ID"):
        raise ArtifactContractError("Runtime Slurm array differs from the extension smoke submission")

    active_array_task = environ.get("KEYFRAME_SMOKE_ROW_ID" if backend == "direct" else "SLURM_ARRAY_TASK_ID")
    try:
        active_row_id = int(active_array_task) if active_array_task is not None else -1
    except ValueError as exc:
        raise ArtifactContractError("SLURM_ARRAY_TASK_ID is not an integer") from exc
    if active_row_id != row_id:
        raise ArtifactContractError(
            f"Runtime extension smoke row {row_id} differs from SLURM_ARRAY_TASK_ID {active_array_task!r}"
        )

    raw_row_ids = submission.get("row_ids")
    if not isinstance(raw_row_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_row_ids
    ):
        raise ArtifactContractError("Extension smoke submission row_ids must be an integer list")
    if len(raw_row_ids) != len(set(raw_row_ids)):
        raise ArtifactContractError("Extension smoke submission row_ids contain duplicates")
    if row_id not in raw_row_ids:
        raise ArtifactContractError(f"Extension smoke submission does not authorize row {row_id}")

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
    if runtime != expected:
        raise ArtifactContractError(
            f"Runtime extension smoke arguments differ from the frozen row: {runtime} != {expected}"
        )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int)
    parser.add_argument("--row-tsv", action="store_true")
    args = parser.parse_args()
    rows = load_smoke_matrix()["rows"]
    if args.row is None:
        print(json.dumps(build_smoke_matrix(), indent=2, sort_keys=True))
        return
    if args.row < 0 or args.row >= len(rows):
        parser.error(f"--row must be inside 0..{len(rows) - 1}")
    row = rows[args.row]
    if args.row_tsv:
        print(
            "\t".join(
                str(row[field])
                for field in (
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
    else:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
