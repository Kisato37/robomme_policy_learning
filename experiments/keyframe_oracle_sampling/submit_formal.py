#!/usr/bin/env python3
"""Validate and submit the exact frozen formal Slurm array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from experiments.keyframe_oracle_sampling.artifacts import RunArtifactStore
from experiments.keyframe_oracle_sampling.artifacts import ScientificKey
from experiments.keyframe_oracle_sampling.artifacts import atomic_write_json
from experiments.keyframe_oracle_sampling.artifacts import read_jsonl
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from experiments.keyframe_oracle_sampling.environment_contract import validate_environment_contract
from experiments.keyframe_oracle_sampling.formal_artifacts import submission_path
from experiments.keyframe_oracle_sampling.formal_artifacts import submission_paths
from experiments.keyframe_oracle_sampling.formal_artifacts import submission_plan_path
from experiments.keyframe_oracle_sampling.formal_artifacts import validate_formal_prepared_root
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_oracle_sampling.formal_matrix import load_formal_matrix
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_RELATIVE
from experiments.keyframe_oracle_sampling.prepare_smoke import REPO
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_content_tree_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import repository_state
from experiments.keyframe_oracle_sampling.prepare_smoke import require_clean_repository

SBATCH_PATH = Path(__file__).with_name("run_formal.sbatch")
MAX_ROWS_PER_ARRAY = 1000


def _compress_row_ids(row_ids: list[int]) -> str:
    ranges: list[str] = []
    start = prior = row_ids[0]
    for row_id in row_ids[1:]:
        if row_id == prior + 1:
            prior = row_id
            continue
        ranges.append(str(start) if start == prior else f"{start}-{prior}")
        start = prior = row_id
    ranges.append(str(start) if start == prior else f"{start}-{prior}")
    return ",".join(ranges)


def shard_rows(row_ids: list[int], *, max_concurrent: int) -> tuple[list[list[int]], int]:
    shards = [row_ids[start : start + MAX_ROWS_PER_ARRAY] for start in range(0, len(row_ids), MAX_ROWS_PER_ARRAY)]
    if not shards:
        raise ValueError("Formal submission cannot contain zero rows")
    if len(shards) > max_concurrent:
        raise ValueError(f"{len(shards)} Slurm shards require --max-concurrent at least {len(shards)}")
    per_shard_concurrent = max_concurrent // len(shards)
    return shards, per_shard_concurrent


def build_submission(
    run_root: Path,
    *,
    max_concurrent: int,
    attempt_id: int,
    row_ids: tuple[int, ...] | None,
) -> tuple[list[tuple[list[str], dict]], dict]:
    run_root = run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Formal run root does not exist: {run_root}")
    if not 1 <= max_concurrent <= 4:
        raise ValueError("--max-concurrent must be in the frozen safe range 1..4")
    if attempt_id not in {0, 1, 2}:
        raise ValueError("--attempt-id must be 0, 1, or 2")
    plan_path = submission_plan_path(run_root, attempt_id)
    existing_records = submission_paths(run_root, attempt_id)
    if plan_path.exists() or existing_records:
        raise FileExistsError(
            "Refusing duplicate or partial formal submission state: "
            f"plan={plan_path.exists()}, records={[str(path) for path in existing_records]}"
        )

    seed_path = run_root / "protocol" / "seed_table.json"
    manifest_path = run_root / "protocol" / "launch_manifest.json"
    matrix_path = run_root / "protocol" / "formal_matrix.json"
    state = repository_state()
    require_clean_repository(state)
    manifest = validate_formal_prepared_root(run_root, seed_path, REPO)
    validate_environment_contract(manifest)
    if manifest.get("repository", {}).get("commit_sha") != state["commit_sha"]:
        raise RuntimeError("Prepared formal root and current checkout use different commits")

    checkpoint = checkpoint_identity(REPO / CHECKPOINT_RELATIVE)
    content_tree = checkpoint_content_tree_identity(REPO / CHECKPOINT_RELATIVE)
    if manifest.get("checkpoint_unpacked_metadata_sha256") != checkpoint["metadata_sha256"]:
        raise RuntimeError("Prepared formal checkpoint metadata differs from live checkpoint")
    if (
        manifest.get("checkpoint_content_tree_algorithm") != content_tree["algorithm"]
        or manifest.get("checkpoint_unpacked_content_tree_sha256") != content_tree["content_tree_sha256"]
    ):
        raise RuntimeError("Prepared formal checkpoint content differs from live bytes")

    rows = load_formal_matrix(matrix_path)["rows"]
    store = RunArtifactStore(run_root)
    completed = store.scan_completed_keys()
    if attempt_id == 0:
        if row_ids:
            raise ValueError("Initial formal submission must use the complete 3,200-row matrix")
        if completed or store.failures_path.exists() or (run_root / "trajectories").exists():
            raise RuntimeError("Initial formal submission requires an unused prepared run root")
        selected_rows = list(range(len(rows)))
        if len(selected_rows) != FORMAL_TRAJECTORY_COUNT:
            raise AssertionError("Formal matrix is not exactly 3,200 rows")
    else:
        if not row_ids:
            raise ValueError("Formal retry submissions require explicit --rows")
        selected_rows = sorted(set(row_ids))
        if len(selected_rows) != len(row_ids):
            raise ValueError("Retry --rows may not contain duplicates")
        if any(row_id < 0 or row_id >= len(rows) for row_id in selected_rows):
            raise ValueError(f"Retry row must be inside 0..{len(rows) - 1}")
        failures = read_jsonl(store.failures_path) if store.failures_path.exists() else []
        for row_id in selected_rows:
            row = rows[row_id]
            key = ScientificKey(row["task"], row["episode_id"], row["arm"], row["trajectory_kind"])
            if key in completed:
                raise RuntimeError(f"Retry row {row_id} already has a scientific result")
            matching = [
                record
                for record in failures
                if str(record.get("task")) == key.task
                and int(record.get("episode_id", -1)) == key.episode_id
                and str(record.get("arm")) == key.arm
                and str(record.get("trajectory_kind", "formal")) == key.trajectory_kind
                and int(record.get("attempt_id", -1)) == attempt_id - 1
                and record.get("retry_allowed") is True
            ]
            if len(matching) != 1:
                raise RuntimeError(f"Retry row {row_id} lacks exactly one allowed preceding failure")

    log_dir = run_root / "slurm"
    row_shards, per_shard_concurrent = shard_rows(selected_rows, max_concurrent=max_concurrent)
    submissions: list[tuple[list[str], dict]] = []
    shard_count = len(row_shards)
    for shard_id, shard_row_ids in enumerate(row_shards):
        array_task_ids = list(range(len(shard_row_ids)))
        row_spec = _compress_row_ids(array_task_ids)
        array_spec = f"{row_spec}%{per_shard_concurrent}"
        command = [
            "sbatch",
            "--parsable",
            f"--chdir={REPO}",
            f"--export=KEYFRAME_REPO_ROOT={REPO}",
            f"--array={array_spec}",
            f"--output={log_dir}/formal-%A_%a.out",
            f"--error={log_dir}/formal-%A_%a.err",
            str(SBATCH_PATH),
            str(run_root),
            str(seed_path),
            str(attempt_id),
            str(shard_id),
        ]
        record = {
            "schema_version": 1,
            "submitted_utc": utc_now(),
            "repository_commit_sha": state["commit_sha"],
            "run_root": str(run_root),
            "array": array_spec,
            "trajectory_count": len(shard_row_ids),
            "array_task_ids": array_task_ids,
            "row_ids": shard_row_ids,
            "attempt_id": attempt_id,
            "shard_id": shard_id,
            "shard_count": shard_count,
            "max_concurrent": per_shard_concurrent,
            "global_max_concurrent": max_concurrent,
            "formal_launch_authorized": True,
            "launch_manifest_sha256": sha256_file(manifest_path),
            "formal_matrix_sha256": sha256_file(matrix_path),
            "prior_exposure_manifest_sha256": manifest["prior_exposure_manifest_sha256"],
            "architecture_report_sha256": manifest["architecture_report_sha256"],
            "development_smoke_audit_sha256": manifest["development_smoke_audit_sha256"],
            "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
            "checkpoint_content_tree_algorithm": content_tree["algorithm"],
            "checkpoint_unpacked_content_tree_sha256": content_tree["content_tree_sha256"],
            "command": command,
        }
        submissions.append((command, record))
    plan = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "repository_commit_sha": state["commit_sha"],
        "run_root": str(run_root),
        "attempt_id": attempt_id,
        "trajectory_count": len(selected_rows),
        "shard_count": shard_count,
        "max_rows_per_array": MAX_ROWS_PER_ARRAY,
        "global_max_concurrent": max_concurrent,
        "formal_launch_authorized": True,
        "shards": [
            {
                field: record[field]
                for field in ("shard_id", "shard_count", "array_task_ids", "row_ids", "array", "command")
            }
            for _, record in submissions
        ],
    }
    return submissions, plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--attempt-id", type=int, default=0)
    parser.add_argument("--rows", help="Comma-separated row IDs; retries only")
    parser.add_argument("--confirm-authorized-formal-v1", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.confirm_authorized_formal_v1:
        parser.error(
            "live submission requires --confirm-authorized-formal-v1; use --dry-run for a non-submitting validation"
        )
    row_ids = tuple(int(value) for value in args.rows.split(",") if value.strip()) if args.rows else None
    submissions, plan = build_submission(
        args.run_root,
        max_concurrent=args.max_concurrent,
        attempt_id=args.attempt_id,
        row_ids=row_ids,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    **plan,
                    "submits_jobs": False,
                    "submission_commands": [command for command, _ in submissions],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    (args.run_root / "slurm").mkdir(parents=True, exist_ok=True)
    plan_path = submission_plan_path(args.run_root, args.attempt_id)
    atomic_write_json(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    job_ids = []
    for command, record in submissions:
        job_id = subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]
        record["slurm_array_job_id"] = job_id
        record["submission_plan_sha256"] = plan_sha256
        atomic_write_json(
            submission_path(
                args.run_root,
                args.attempt_id,
                int(record["shard_id"]),
            ),
            record,
        )
        job_ids.append(job_id)
    print(json.dumps({"slurm_array_job_ids": job_ids}, sort_keys=True))


if __name__ == "__main__":
    main()
