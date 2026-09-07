#!/usr/bin/env python3
"""Validate and submit the exact 48-row OC3/OC5 development smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from experiments.keyframe_neighborhood_sampling.architecture_smoke import validate_architecture_pass_report
from experiments.keyframe_neighborhood_sampling.formal_artifacts import smoke_submission_path
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.prepare_formal import repository_state
from experiments.keyframe_neighborhood_sampling.prepare_formal import require_clean_repository
from experiments.keyframe_neighborhood_sampling.record_launcher_failure import validate_extension_failure_record
from experiments.keyframe_neighborhood_sampling.smoke_matrix import SMOKE_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.smoke_matrix import load_smoke_matrix
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import RunArtifactStore
from experiments.keyframe_oracle_sampling.artifacts import ScientificKey
from experiments.keyframe_oracle_sampling.artifacts import atomic_write_json
from experiments.keyframe_oracle_sampling.artifacts import load_seed_table
from experiments.keyframe_oracle_sampling.artifacts import read_jsonl
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from experiments.keyframe_oracle_sampling.artifacts import validate_smoke_formal_seed_disjointness
from experiments.keyframe_oracle_sampling.environment_contract import validate_environment_contract
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_RELATIVE
from experiments.keyframe_oracle_sampling.prepare_smoke import REPO
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_content_tree_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_identity
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_POLICY_CALLS
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_SCOPE

SBATCH_PATH = Path(__file__).with_name("run_smoke.sbatch")


def _validate_retry_rows(
    run_root: Path,
    rows: list[dict],
    *,
    attempt_id: int,
    row_ids: tuple[int, ...] | None,
) -> list[int]:
    if not row_ids:
        raise ValueError("Extension smoke retries require explicit --rows")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Retry --rows may not contain duplicates")
    selected = sorted(row_ids)
    if any(row_id < 0 or row_id >= len(rows) for row_id in selected):
        raise ValueError(f"Retry row must be inside 0..{len(rows) - 1}")
    store = RunArtifactStore(run_root)
    completed = store.scan_completed_keys()
    failures = read_jsonl(store.failures_path) if store.failures_path.exists() else []
    for row_id in selected:
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
            and str(record.get("trajectory_kind")) == key.trajectory_kind
            and int(record.get("attempt_id", -1)) == attempt_id - 1
            and record.get("retry_allowed") is True
        ]
        if len(matching) != 1:
            raise RuntimeError(f"Retry row {row_id} lacks exactly one allowed preceding failure")
        validate_extension_failure_record(
            run_root,
            matching[0],
            expected_row_id=row_id,
        )
    return selected


def build_submission(
    run_root: Path,
    *,
    max_concurrent: int,
    attempt_id: int,
    row_ids: tuple[int, ...] | None,
) -> tuple[list[str], dict]:
    run_root = run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Extension smoke root does not exist: {run_root}")
    if not 1 <= max_concurrent <= 4:
        raise ValueError("--max-concurrent must be in the frozen safe range 1..4")
    if attempt_id not in {0, 1, 2}:
        raise ValueError("--attempt-id must be 0, 1, or 2")

    protocol_dir = run_root / "protocol"
    manifest_path = protocol_dir / "launch_manifest.json"
    matrix_path = protocol_dir / "smoke_matrix.json"
    architecture_path = run_root / "architecture_smoke/report.json"
    architecture_submission_path = protocol_dir / "architecture_submission_record.json"
    submission_path = smoke_submission_path(run_root, attempt_id)
    for path in (
        manifest_path,
        matrix_path,
        architecture_path,
        architecture_submission_path,
        protocol_dir / "seed_table.json",
        protocol_dir / "formal_seed_audit_table.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing extension smoke prerequisite: {path}")
    if submission_path.exists():
        raise FileExistsError(f"Refusing duplicate extension smoke submission: {submission_path}")

    state = repository_state()
    require_clean_repository(state)
    manifest = json.loads(manifest_path.read_text())
    expected_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "run_kind": "development_smoke",
        "dataset": "val",
        "trajectory_count": SMOKE_TRAJECTORY_COUNT,
        "formal_launch_authorized": False,
        "smoke_matrix_sha256": sha256_file(matrix_path),
    }
    if any(manifest.get(field) != value for field, value in expected_manifest.items()):
        raise RuntimeError("Prepared root is not the exact OC3/OC5 smoke contract")
    if manifest.get("repository", {}).get("commit_sha") != state["commit_sha"]:
        raise RuntimeError("Prepared extension root and current checkout use different commits")
    validate_environment_contract(manifest)
    rows = load_smoke_matrix(matrix_path)["rows"]
    if manifest.get("matrix") != rows:
        raise RuntimeError("Prepared extension smoke matrix differs from its manifest")
    smoke_seed_path = protocol_dir / "seed_table.json"
    formal_seed_path = protocol_dir / "formal_seed_audit_table.json"
    smoke_seed, smoke_lookup = load_seed_table(
        smoke_seed_path,
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    formal_seed, formal_lookup = load_seed_table(
        formal_seed_path,
        expected_scope=FORMAL_SEED_SCOPE,
        expected_dataset=FORMAL_SEED_DATASET,
    )
    if set(smoke_lookup) != {(task, 0, call) for task in FORMAL_TASKS for call in range(MAX_POLICY_CALLS)} or set(
        formal_lookup
    ) != {(task, episode, call) for task in FORMAL_TASKS for episode in range(50) for call in range(MAX_POLICY_CALLS)}:
        raise RuntimeError("Extension smoke seed universes are incomplete")
    disjointness = validate_smoke_formal_seed_disjointness(smoke_seed, formal_seed)
    seed_fields = {
        "seed_table_file_sha256": sha256_file(smoke_seed_path),
        "formal_seed_audit_file_sha256": sha256_file(formal_seed_path),
        "seed_table_scope": smoke_seed["scope"],
        "seed_table_dataset": smoke_seed["dataset"],
        "seed_table_derivation": smoke_seed["derivation"],
        "seed_table_entries_sha256": smoke_seed["entries_sha256"],
        "seed_disjointness_audit": disjointness,
    }
    if any(manifest.get(field) != value for field, value in seed_fields.items()):
        raise RuntimeError("Extension smoke seed provenance mismatch")

    architecture = json.loads(architecture_path.read_text())
    architecture_submission = json.loads(architecture_submission_path.read_text())
    validate_architecture_pass_report(
        architecture,
        run_root=run_root,
        architecture_submission=architecture_submission,
    )
    if architecture.get("repository_commit_sha") != state["commit_sha"]:
        raise RuntimeError("Extension architecture gate used a different commit")

    checkpoint = checkpoint_identity(REPO / CHECKPOINT_RELATIVE)
    content_tree = checkpoint_content_tree_identity(REPO / CHECKPOINT_RELATIVE)
    checkpoint_fields = {
        "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
        "checkpoint_content_tree_algorithm": content_tree["algorithm"],
        "checkpoint_unpacked_content_tree_sha256": content_tree["content_tree_sha256"],
    }
    for payload, label in ((manifest, "manifest"), (architecture, "architecture report")):
        if any(payload.get(field) != value for field, value in checkpoint_fields.items()):
            raise RuntimeError(f"Extension {label} differs from live checkpoint bytes")

    store = RunArtifactStore(run_root)
    if attempt_id == 0:
        if row_ids:
            raise ValueError("Initial extension smoke submission must use all 48 rows")
        if store.scan_completed_keys() or store.failures_path.exists() or (run_root / "trajectories").exists():
            raise RuntimeError("Initial extension smoke submission requires an unused run root")
        selected_rows = list(range(len(rows)))
        if len(selected_rows) != SMOKE_TRAJECTORY_COUNT:
            raise AssertionError("Extension smoke matrix is not exactly 48 rows")
    else:
        selected_rows = _validate_retry_rows(run_root, rows, attempt_id=attempt_id, row_ids=row_ids)

    row_spec = (
        f"0-{SMOKE_TRAJECTORY_COUNT - 1}" if attempt_id == 0 else ",".join(str(row_id) for row_id in selected_rows)
    )
    array_spec = f"{row_spec}%{max_concurrent}"
    seed_path = protocol_dir / "seed_table.json"
    log_dir = run_root / "slurm"
    command = [
        "sbatch",
        "--parsable",
        f"--chdir={REPO}",
        f"--export=KEYFRAME_NEIGHBORHOOD_REPO_ROOT={REPO}",
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
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "submitted_utc": utc_now(),
        "repository_commit_sha": state["commit_sha"],
        "run_root": str(run_root),
        "array": array_spec,
        "trajectory_count": len(selected_rows),
        "row_ids": selected_rows,
        "attempt_id": attempt_id,
        "max_concurrent": max_concurrent,
        "smoke_launch_authorized": True,
        "formal_launch_authorized": False,
        "launch_manifest_sha256": sha256_file(manifest_path),
        "smoke_matrix_sha256": sha256_file(matrix_path),
        "architecture_report_sha256": sha256_file(architecture_path),
        **checkpoint_fields,
        "command": command,
    }
    return command, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--attempt-id", type=int, default=0)
    parser.add_argument("--rows", help="Comma-separated row IDs; retries only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-authorized-extension-smoke-v1", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.confirm_authorized_extension_smoke_v1:
        parser.error(
            "live submission requires --confirm-authorized-extension-smoke-v1; "
            "use --dry-run for non-submitting validation"
        )
    row_ids = tuple(int(value) for value in args.rows.split(",") if value.strip()) if args.rows else None
    command, record = build_submission(
        args.run_root,
        max_concurrent=args.max_concurrent,
        attempt_id=args.attempt_id,
        row_ids=row_ids,
    )
    if args.dry_run:
        print(json.dumps({**record, "submits_jobs": False}, indent=2, sort_keys=True))
        return

    manifest = json.loads((args.run_root / "protocol/launch_manifest.json").read_text())
    if manifest.get("runner_backend") == "direct":
        parser.error("This root requires submit_direct, not a Slurm submission")
    (args.run_root / "slurm").mkdir(parents=True, exist_ok=True)
    job_id = subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]
    record["slurm_array_job_id"] = job_id
    atomic_write_json(smoke_submission_path(args.run_root, args.attempt_id), record)
    print(job_id)


if __name__ == "__main__":
    main()
