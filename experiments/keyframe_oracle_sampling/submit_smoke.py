#!/usr/bin/env python3
"""Submit the frozen 80-row development smoke and record its Slurm job ID."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.keyframe_oracle_sampling.artifacts import (
    RunArtifactStore,
    ScientificKey,
    atomic_write_json,
    load_seed_table,
    read_jsonl,
    sha256_file,
    utc_now,
    validate_architecture_pass_report,
    validate_smoke_formal_seed_disjointness,
)
from experiments.keyframe_oracle_sampling.environment_contract import (
    validate_environment_contract,
)
from experiments.keyframe_oracle_sampling.prepare_smoke import (
    CHECKPOINT_RELATIVE,
    REPO,
    SBATCH_PATH,
    checkpoint_content_tree_identity,
    checkpoint_identity,
    repository_state,
    require_clean_repository,
)
from experiments.keyframe_oracle_sampling.smoke_matrix import expand_rows, load_frozen_matrix
from mme_vla_suite.shared.keyframe_oracle_sampling import (
    FORMAL_SEED_DATASET,
    FORMAL_SEED_SCOPE,
    SMOKE_SEED_DATASET,
    SMOKE_SEED_SCOPE,
)


def build_submission(
    run_root: Path,
    *,
    max_concurrent: int,
    attempt_id: int,
    row_ids: tuple[int, ...] | None,
) -> tuple[list[str], dict]:
    if not run_root.is_dir():
        raise FileNotFoundError(f"Smoke run root does not exist: {run_root}")
    if max_concurrent < 1:
        raise ValueError("--max-concurrent must be positive")
    if attempt_id not in {0, 1, 2}:
        raise ValueError("--attempt-id must be 0, 1, or 2")

    manifest_path = run_root / "protocol" / "launch_manifest.json"
    architecture_path = run_root / "architecture_smoke" / "report.json"
    architecture_submission_path = (
        run_root / "protocol" / "architecture_submission_record.json"
    )
    seed_path = run_root / "protocol" / "seed_table.json"
    formal_seed_audit_path = run_root / "protocol" / "formal_seed_audit_table.json"
    submission_name = (
        "submission_record.json"
        if attempt_id == 0
        else f"submission_record_attempt_{attempt_id:02d}.json"
    )
    submission_path = run_root / "protocol" / submission_name
    for required in (
        manifest_path,
        architecture_path,
        architecture_submission_path,
        seed_path,
        formal_seed_audit_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Missing smoke prerequisite: {required}")
    if submission_path.exists():
        raise FileExistsError(f"Refusing duplicate smoke submission: {submission_path}")

    state = repository_state()
    require_clean_repository(state)
    manifest = json.loads(manifest_path.read_text())
    seed_payload, _ = load_seed_table(
        seed_path,
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    formal_seed_payload, _ = load_seed_table(
        formal_seed_audit_path,
        expected_scope=FORMAL_SEED_SCOPE,
        expected_dataset=FORMAL_SEED_DATASET,
    )
    validate_smoke_formal_seed_disjointness(seed_payload, formal_seed_payload)
    if (
        manifest.get("seed_table_file_sha256") != sha256_file(seed_path)
        or manifest.get("formal_seed_audit_file_sha256")
        != sha256_file(formal_seed_audit_path)
    ):
        raise RuntimeError("Prepared seed-table file digest mismatch")
    if (
        manifest.get("seed_table_scope") != seed_payload["scope"]
        or manifest.get("seed_table_dataset") != seed_payload["dataset"]
        or manifest.get("seed_table_derivation") != seed_payload["derivation"]
        or manifest.get("seed_table_entries_sha256") != seed_payload["entries_sha256"]
    ):
        raise RuntimeError("Prepared manifest and smoke seed-table contract differ")
    recorded_sha = manifest.get("repository", {}).get("commit_sha")
    if recorded_sha != state["commit_sha"]:
        raise RuntimeError(
            f"Prepared commit {recorded_sha} differs from current clean commit "
            f"{state['commit_sha']}"
        )
    architecture = json.loads(architecture_path.read_text())
    architecture_submission = json.loads(architecture_submission_path.read_text())
    validate_architecture_pass_report(
        architecture,
        run_root=run_root,
        architecture_submission=architecture_submission,
    )
    validate_environment_contract(manifest)
    architecture_sha = architecture.get("repository_commit_sha")
    if architecture_sha != state["commit_sha"]:
        raise RuntimeError("Architecture smoke was run from a different commit")

    checkpoint = checkpoint_identity(REPO / CHECKPOINT_RELATIVE)
    checkpoint_content_tree = checkpoint_content_tree_identity(REPO / CHECKPOINT_RELATIVE)
    if (
        manifest.get("checkpoint_unpacked_metadata_sha256")
        != checkpoint["metadata_sha256"]
    ):
        raise RuntimeError("Prepared checkpoint identity differs from the live checkpoint")
    if (
        architecture.get("checkpoint_unpacked_metadata_sha256")
        != checkpoint["metadata_sha256"]
    ):
        raise RuntimeError("Architecture smoke used a different checkpoint identity")
    if (
        manifest.get("checkpoint_content_tree_algorithm")
        != checkpoint_content_tree["algorithm"]
        or manifest.get("checkpoint_unpacked_content_tree_sha256")
        != checkpoint_content_tree["content_tree_sha256"]
    ):
        raise RuntimeError("Prepared checkpoint content-tree identity differs from live bytes")
    if (
        architecture.get("checkpoint_content_tree_algorithm")
        != checkpoint_content_tree["algorithm"]
        or architecture.get("checkpoint_unpacked_content_tree_sha256")
        != checkpoint_content_tree["content_tree_sha256"]
    ):
        raise RuntimeError("Architecture smoke used different checkpoint file contents")

    rows = expand_rows(load_frozen_matrix())
    if attempt_id == 0:
        if row_ids:
            raise ValueError("Initial smoke submission must use the complete frozen matrix")
        selected_rows = list(range(len(rows)))
    else:
        if not row_ids:
            raise ValueError("Retry submissions require explicit --rows")
        selected_rows = sorted(set(row_ids))
        if len(selected_rows) != len(row_ids):
            raise ValueError("Retry --rows may not contain duplicates")
        if any(row_id < 0 or row_id >= len(rows) for row_id in selected_rows):
            raise ValueError(f"Retry row must be inside 0..{len(rows) - 1}")
        store = RunArtifactStore(run_root)
        completed = store.scan_completed_keys()
        failures = read_jsonl(store.failures_path) if store.failures_path.exists() else []
        for row_id in selected_rows:
            row = rows[row_id]
            key = ScientificKey(
                row["task"], row["episode_id"], row["arm"], row["trajectory_kind"]
            )
            if key in completed:
                raise RuntimeError(f"Retry row {row_id} already has a scientific result")
            matching = [
                record
                for record in failures
                if str(record.get("task")) == key.task
                and int(record.get("episode_id", -1)) == key.episode_id
                and str(record.get("arm")) == key.arm
                and str(record.get("trajectory_kind")) == key.trajectory_kind
                and int(record.get("attempt_id", -1)) == attempt_id - 1
                and record.get("retry_allowed") is True
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    f"Retry row {row_id} lacks exactly one allowed preceding failure"
                )
    log_dir = run_root / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)
    row_spec = (
        f"0-{len(rows) - 1}"
        if attempt_id == 0
        else ",".join(str(row_id) for row_id in selected_rows)
    )
    array_spec = f"{row_spec}%{max_concurrent}"
    command = [
        "sbatch",
        "--parsable",
        f"--chdir={REPO}",
        f"--export=KEYFRAME_REPO_ROOT={REPO}",
        f"--array={array_spec}",
        f"--output={log_dir}/smoke-%A_%a.out",
        f"--error={log_dir}/smoke-%A_%a.err",
        str(SBATCH_PATH),
        str(run_root),
        str(seed_path),
        str(attempt_id),
    ]
    record = {
        "schema_version": 1,
        "submitted_utc": utc_now(),
        "repository_commit_sha": state["commit_sha"],
        "run_root": str(run_root),
        "array": array_spec,
        "trajectory_count": len(selected_rows),
        "row_ids": selected_rows,
        "attempt_id": attempt_id,
        "max_concurrent": max_concurrent,
        "architecture_report_sha256": sha256_file(architecture_path),
        "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
        "checkpoint_content_tree_algorithm": checkpoint_content_tree["algorithm"],
        "checkpoint_unpacked_content_tree_sha256": checkpoint_content_tree[
            "content_tree_sha256"
        ],
        "command": command,
    }
    return command, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--attempt-id", type=int, default=0)
    parser.add_argument("--rows", help="Comma-separated row IDs; required only for retries")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    row_ids = (
        tuple(int(value) for value in args.rows.split(",") if value.strip())
        if args.rows
        else None
    )
    command, record = build_submission(
        args.run_root.resolve(),
        max_concurrent=args.max_concurrent,
        attempt_id=args.attempt_id,
        row_ids=row_ids,
    )
    if args.dry_run:
        print(json.dumps({**record, "submits_jobs": False}, indent=2, sort_keys=True))
        return

    job_id = subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]
    record["slurm_array_job_id"] = job_id
    atomic_write_json(
        args.run_root / "protocol" / (
            "submission_record.json"
            if args.attempt_id == 0
            else f"submission_record_attempt_{args.attempt_id:02d}.json"
        ),
        record,
    )
    print(job_id)


if __name__ == "__main__":
    main()
