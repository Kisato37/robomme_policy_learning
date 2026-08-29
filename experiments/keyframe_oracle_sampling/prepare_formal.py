#!/usr/bin/env python3
"""Dry-run or prepare one immutable formal root; never submit jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from experiments.keyframe_oracle_sampling.analysis import normalize_prior_exposure_manifest
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import atomic_write_bytes
from experiments.keyframe_oracle_sampling.artifacts import atomic_write_json
from experiments.keyframe_oracle_sampling.artifacts import build_seed_table
from experiments.keyframe_oracle_sampling.artifacts import build_smoke_seed_table
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from experiments.keyframe_oracle_sampling.artifacts import validate_architecture_pass_report
from experiments.keyframe_oracle_sampling.artifacts import validate_smoke_formal_seed_disjointness
from experiments.keyframe_oracle_sampling.environment_contract import validate_environment_contract
from experiments.keyframe_oracle_sampling.formal_artifacts import ACCEPTED_DEVELOPMENT_SMOKE_AUDIT_SHA256
from experiments.keyframe_oracle_sampling.formal_artifacts import ACCEPTED_DEVELOPMENT_SMOKE_COMMIT
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_DATASET
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_MAX_STEPS
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_oracle_sampling.formal_matrix import build_formal_matrix
from experiments.keyframe_oracle_sampling.prepare_smoke import BENCHMARK_UV_LOCK
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_RELATIVE
from experiments.keyframe_oracle_sampling.prepare_smoke import EXPECTED_CHECKPOINT_ARCHIVE_SHA256
from experiments.keyframe_oracle_sampling.prepare_smoke import POLICY_PYTHON
from experiments.keyframe_oracle_sampling.prepare_smoke import POLICY_UV_LOCK
from experiments.keyframe_oracle_sampling.prepare_smoke import POLICY_VENV
from experiments.keyframe_oracle_sampling.prepare_smoke import PROTOCOL_PATH
from experiments.keyframe_oracle_sampling.prepare_smoke import REPO
from experiments.keyframe_oracle_sampling.prepare_smoke import SIMULATOR_PYTHON
from experiments.keyframe_oracle_sampling.prepare_smoke import SIMULATOR_VENV
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_content_tree_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import environment_lock_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import python_environment_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import repository_state
from experiments.keyframe_oracle_sampling.prepare_smoke import require_clean_repository
from experiments.keyframe_oracle_sampling.prepare_smoke import verify_checkpoint_archive
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE

PRIOR_EXPOSURE_PATH = Path(__file__).with_name("PRIOR_EXPOSURE_MANIFEST.json")
ACCEPTED_DEVELOPMENT_SMOKE_RELATIVE = Path("runs/keyframe_oracle_sampling/20260829T181633Z_899912b1_devsmoke")


def _validate_development_smoke_source(run_root: Path) -> tuple[Path, dict]:
    audit_path = run_root / "aggregate" / "smoke_audit.json"
    launch_path = run_root / "protocol" / "launch_manifest.json"
    for path in (audit_path, launch_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing accepted development-smoke evidence: {path}")
    if sha256_file(audit_path) != ACCEPTED_DEVELOPMENT_SMOKE_AUDIT_SHA256:
        raise RuntimeError("Development-smoke audit digest differs from PREFORMAL_AUDIT.md")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("passed") is not True
        or audit.get("formal_started") is not False
        or audit.get("completeness", {}).get("complete") is not True
        or audit.get("completeness", {}).get("completed_count") != 80
        or audit.get("initial_condition_fairness") != {"paired_blocks": 16, "fair": True}
        or audit.get("paired_manifest_invariants") != {"paired_blocks": 16, "invariants_match": True}
    ):
        raise RuntimeError("Accepted development-smoke audit is not the exact PASS gate")
    launch = json.loads(launch_path.read_text())
    if launch.get("repository", {}).get("commit_sha") != ACCEPTED_DEVELOPMENT_SMOKE_COMMIT:
        raise RuntimeError("Accepted development smoke used an unexpected commit")
    return audit_path, launch


def _validate_architecture_source(
    run_root: Path,
    *,
    current_commit: str,
    checkpoint: dict[str, int | str],
    checkpoint_content_tree: dict[str, int | str],
) -> tuple[Path, Path, dict]:
    report_path = run_root / "architecture_smoke" / "report.json"
    submission_path = run_root / "protocol" / "architecture_submission_record.json"
    launch_path = run_root / "protocol" / "launch_manifest.json"
    for path in (report_path, submission_path, launch_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing current-commit architecture evidence: {path}")
    report = json.loads(report_path.read_text())
    submission = json.loads(submission_path.read_text())
    launch = json.loads(launch_path.read_text())
    validate_architecture_pass_report(
        report,
        run_root=run_root,
        architecture_submission=submission,
    )
    validate_environment_contract(launch)
    if launch.get("repository", {}).get("commit_sha") != current_commit:
        raise RuntimeError("Architecture source was prepared from a different commit")
    if report.get("repository_commit_sha") != current_commit:
        raise RuntimeError("Architecture smoke ran from a different commit")
    expected_checkpoint_fields = {
        "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
        "checkpoint_content_tree_algorithm": checkpoint_content_tree["algorithm"],
        "checkpoint_unpacked_content_tree_sha256": checkpoint_content_tree["content_tree_sha256"],
    }
    for field, expected in expected_checkpoint_fields.items():
        if launch.get(field) != expected or report.get(field) != expected:
            raise RuntimeError(f"Architecture evidence differs on {field}")
    return report_path, submission_path, launch


def dry_run_contract() -> dict:
    matrix = build_formal_matrix()
    formal_seed_table = build_seed_table(FORMAL_TASKS, range(50))
    development_seed_table = build_smoke_seed_table(FORMAL_TASKS, [0])
    disjointness = validate_smoke_formal_seed_disjointness(development_seed_table, formal_seed_table)
    exposure_payload = json.loads(PRIOR_EXPOSURE_PATH.read_text())
    exposure = normalize_prior_exposure_manifest(exposure_payload)
    if exposure_payload.get("frozen_before_formal_execution") is not True:
        raise RuntimeError("Prior-exposure manifest is not frozen")
    if exposure_payload.get("present_formal_outcomes_inspected") is not False:
        raise RuntimeError("Prior-exposure manifest reports current formal outcome access")
    return {
        "valid": True,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": FORMAL_DATASET,
        "max_steps": FORMAL_MAX_STEPS,
        "trajectory_count": matrix["trajectory_count"],
        "formal_seed_count": formal_seed_table["entry_count"],
        "seed_scope": FORMAL_SEED_SCOPE,
        "seed_dataset": FORMAL_SEED_DATASET,
        "seed_disjointness_audit": disjointness,
        "prior_exposure_entry_count": len(exposure.entries),
        "prior_exposure_test_block_count": len(exposure.excluded_formal_blocks),
        "submits_jobs": False,
        "creates_run_root": False,
    }


def prepare(
    run_root: Path,
    checkpoint_archive: Path,
    architecture_run_root: Path,
    development_smoke_run_root: Path,
    prior_exposure_path: Path,
    *,
    authorization_note: str,
) -> Path:
    run_root = run_root.resolve()
    expected_parent = (REPO / "runs" / "keyframe_oracle_sampling").resolve()
    if run_root.parent != expected_parent:
        raise ValueError(f"Formal run root must be one direct run ID beneath {expected_parent}: {run_root}")
    if run_root.exists():
        raise FileExistsError(f"Refusing to reuse formal run root: {run_root}")
    if not authorization_note.strip():
        raise ValueError("Formal preparation requires a non-empty authorization note")

    repo_state = repository_state()
    require_clean_repository(repo_state)
    archive_sha256 = verify_checkpoint_archive(checkpoint_archive)
    checkpoint_dir = REPO / CHECKPOINT_RELATIVE
    checkpoint = checkpoint_identity(checkpoint_dir)
    checkpoint_content_tree = checkpoint_content_tree_identity(checkpoint_dir)
    if (
        checkpoint_content_tree["file_count"] != checkpoint["file_count"]
        or checkpoint_content_tree["total_bytes"] != checkpoint["total_bytes"]
    ):
        raise RuntimeError("Checkpoint content inventory differs from frozen metadata")

    environment_locks = environment_lock_identity()
    python_environments = {
        "policy": python_environment_identity(POLICY_PYTHON, POLICY_VENV),
        "simulator": python_environment_identity(SIMULATOR_PYTHON, SIMULATOR_VENV),
    }
    benchmark_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO / "third_party" / "robomme_benchmark",
        text=True,
    ).strip()
    architecture_report_path, architecture_submission_path, _ = _validate_architecture_source(
        architecture_run_root.resolve(),
        current_commit=str(repo_state["commit_sha"]),
        checkpoint=checkpoint,
        checkpoint_content_tree=checkpoint_content_tree,
    )
    development_audit_path, _ = _validate_development_smoke_source(development_smoke_run_root.resolve())

    if not prior_exposure_path.is_file():
        raise FileNotFoundError(f"Missing prior-exposure manifest: {prior_exposure_path}")
    exposure_payload = json.loads(prior_exposure_path.read_text())
    exposure = normalize_prior_exposure_manifest(exposure_payload)
    if exposure_payload.get("frozen_before_formal_execution") is not True:
        raise RuntimeError("Prior-exposure manifest must be frozen before formal execution")
    if exposure_payload.get("present_formal_outcomes_inspected") is not False:
        raise RuntimeError("Prior-exposure manifest indicates present formal outcome access")

    matrix = build_formal_matrix()
    formal_seed_table = build_seed_table(FORMAL_TASKS, range(50))
    development_seed_table = build_smoke_seed_table(FORMAL_TASKS, [0])
    disjointness = validate_smoke_formal_seed_disjointness(development_seed_table, formal_seed_table)

    run_root.mkdir(parents=True, exist_ok=False)
    protocol_dir = run_root / "protocol"
    protocol_dir.mkdir()
    protocol_snapshot = protocol_dir / "protocol_snapshot.md"
    atomic_write_bytes(protocol_snapshot, PROTOCOL_PATH.read_bytes())
    atomic_write_bytes(
        protocol_dir / "protocol_sha256.txt",
        (sha256_file(protocol_snapshot) + "\n").encode("ascii"),
    )
    seed_path = protocol_dir / "seed_table.json"
    atomic_write_json(seed_path, formal_seed_table)
    development_seed_path = protocol_dir / "development_seed_audit_table.json"
    atomic_write_json(development_seed_path, development_seed_table)
    matrix_path = protocol_dir / "formal_matrix.json"
    atomic_write_json(matrix_path, matrix)
    exposure_path = protocol_dir / "prior_exposure_manifest.json"
    atomic_write_bytes(exposure_path, prior_exposure_path.read_bytes())
    architecture_report_copy = protocol_dir / "architecture_pass_report.json"
    atomic_write_bytes(architecture_report_copy, architecture_report_path.read_bytes())
    architecture_submission_copy = protocol_dir / "architecture_submission_record.json"
    atomic_write_bytes(architecture_submission_copy, architecture_submission_path.read_bytes())
    development_audit_copy = protocol_dir / "development_smoke_audit.json"
    atomic_write_bytes(development_audit_copy, development_audit_path.read_bytes())

    atomic_write_json(
        protocol_dir / "launch_manifest.json",
        {
            "run_id": run_root.name,
            "run_kind": "formal",
            "created_utc": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": sha256_file(protocol_snapshot),
            "repository": repo_state,
            "benchmark_repository_commit": benchmark_commit,
            "environment_lock_sha256": environment_locks["policy_uv_lock_sha256"],
            "environment_locks": environment_locks,
            "python_environments": python_environments,
            "environment_lock_paths": {
                "policy": str(POLICY_UV_LOCK.relative_to(REPO)),
                "benchmark": str(BENCHMARK_UV_LOCK.relative_to(REPO)),
            },
            "checkpoint_path": str(CHECKPOINT_RELATIVE),
            "checkpoint_archive_path": str(checkpoint_archive.resolve()),
            "checkpoint_archive_sha256_expected": EXPECTED_CHECKPOINT_ARCHIVE_SHA256,
            "checkpoint_archive_sha256_actual": archive_sha256,
            "checkpoint_content_tree_algorithm": checkpoint_content_tree["algorithm"],
            "checkpoint_unpacked_content_tree_sha256": checkpoint_content_tree["content_tree_sha256"],
            "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
            "checkpoint_file_count": checkpoint["file_count"],
            "checkpoint_total_bytes": checkpoint["total_bytes"],
            "evaluation_policy_seed": 7,
            "master_selector_seed": 2026082501,
            "executed_action_horizon": 16,
            "dataset": FORMAL_DATASET,
            "max_steps": FORMAL_MAX_STEPS,
            "trajectory_count": FORMAL_TRAJECTORY_COUNT,
            "matrix": matrix["rows"],
            "formal_matrix_sha256": sha256_file(matrix_path),
            "seed_table_scope": formal_seed_table["scope"],
            "seed_table_dataset": formal_seed_table["dataset"],
            "seed_table_derivation": formal_seed_table["derivation"],
            "seed_table_entries_sha256": formal_seed_table["entries_sha256"],
            "seed_table_file_sha256": sha256_file(seed_path),
            "development_seed_audit_scope": development_seed_table["scope"],
            "development_seed_audit_dataset": development_seed_table["dataset"],
            "development_seed_audit_file_sha256": sha256_file(development_seed_path),
            "seed_disjointness_audit": disjointness,
            "prior_exposure_manifest_sha256": sha256_file(exposure_path),
            "prior_exposure_entry_count": len(exposure.entries),
            "prior_exposure_test_block_count": len(exposure.excluded_formal_blocks),
            "architecture_source_run_root": str(architecture_run_root.resolve()),
            "architecture_report_sha256": sha256_file(architecture_report_copy),
            "architecture_submission_record_sha256": sha256_file(architecture_submission_copy),
            "development_smoke_source_run_root": str(development_smoke_run_root.resolve()),
            "development_smoke_commit_sha": ACCEPTED_DEVELOPMENT_SMOKE_COMMIT,
            "development_smoke_audit_sha256": sha256_file(development_audit_copy),
            "formal_launch_authorized": True,
            "formal_launch_authorization": {
                "authorized_utc_recorded": utc_now(),
                "authorized_trajectory_count": FORMAL_TRAJECTORY_COUNT,
                "authorization_note": authorization_note.strip(),
            },
            "command_template": (
                "python -m experiments.keyframe_oracle_sampling.submit_formal "
                "--run-root <run_root> --confirm-authorized-formal-v1"
            ),
            "resources": {
                "nodes_per_row": 1,
                "gpus_per_row": 2,
                "max_concurrent_rows": 4,
                "ports": "20000 + (SLURM_JOB_ID mod 20000)",
            },
            "submitted": False,
        },
    )
    return seed_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--checkpoint-archive", type=Path)
    parser.add_argument("--architecture-run-root", type=Path)
    parser.add_argument(
        "--development-smoke-run-root",
        type=Path,
        default=REPO / ACCEPTED_DEVELOPMENT_SMOKE_RELATIVE,
    )
    parser.add_argument("--prior-exposure-manifest", type=Path, default=PRIOR_EXPOSURE_PATH)
    parser.add_argument("--authorization-note")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run_contract(), indent=2, sort_keys=True))
        return
    for name in (
        "run_root",
        "checkpoint_archive",
        "architecture_run_root",
        "authorization_note",
    ):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required unless --dry-run is used")
    seed_path = prepare(
        args.run_root,
        args.checkpoint_archive,
        args.architecture_run_root,
        args.development_smoke_run_root,
        args.prior_exposure_manifest,
        authorization_note=args.authorization_note,
    )
    print(
        "Prepared without submission. Validate and submit only through:\n"
        "python -m experiments.keyframe_oracle_sampling.submit_formal "
        f"--run-root {args.run_root} --confirm-authorized-formal-v1\n"
        f"Formal seed table: {seed_path}"
    )


if __name__ == "__main__":
    main()
