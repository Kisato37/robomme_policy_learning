#!/usr/bin/env python3
"""Prepare one immutable OC3/OC5 development-smoke root; never submit jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.prepare_formal import RUN_PARENT_RELATIVE
from experiments.keyframe_neighborhood_sampling.prepare_formal import repository_state
from experiments.keyframe_neighborhood_sampling.prepare_formal import require_clean_repository
from experiments.keyframe_neighborhood_sampling.smoke_matrix import MATRIX_PATH
from experiments.keyframe_neighborhood_sampling.smoke_matrix import SMOKE_DATASET
from experiments.keyframe_neighborhood_sampling.smoke_matrix import SMOKE_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.smoke_matrix import build_smoke_matrix
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import atomic_write_bytes
from experiments.keyframe_oracle_sampling.artifacts import atomic_write_json
from experiments.keyframe_oracle_sampling.artifacts import build_seed_table
from experiments.keyframe_oracle_sampling.artifacts import build_smoke_seed_table
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from experiments.keyframe_oracle_sampling.artifacts import validate_smoke_formal_seed_disjointness
from experiments.keyframe_oracle_sampling.prepare_smoke import BENCHMARK_UV_LOCK
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_RELATIVE
from experiments.keyframe_oracle_sampling.prepare_smoke import EXPECTED_CHECKPOINT_ARCHIVE_SHA256
from experiments.keyframe_oracle_sampling.prepare_smoke import POLICY_PYTHON
from experiments.keyframe_oracle_sampling.prepare_smoke import POLICY_UV_LOCK
from experiments.keyframe_oracle_sampling.prepare_smoke import POLICY_VENV
from experiments.keyframe_oracle_sampling.prepare_smoke import REPO
from experiments.keyframe_oracle_sampling.prepare_smoke import SIMULATOR_PYTHON
from experiments.keyframe_oracle_sampling.prepare_smoke import SIMULATOR_VENV
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_content_tree_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import environment_lock_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import python_environment_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import verify_checkpoint_archive
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_SCOPE

PROTOCOL_PATH = Path(__file__).with_name("EXPERIMENT_EXTENSION_PROTOCOL.md")


def dry_run_contract() -> dict[str, Any]:
    """Describe the exact smoke census without touching disk or Slurm."""
    smoke_seeds = build_smoke_seed_table(FORMAL_TASKS, [0])
    formal_seeds = build_seed_table(FORMAL_TASKS, range(50))
    return {
        "valid": True,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "run_kind": "development_smoke",
        "dataset": SMOKE_DATASET,
        "arms": list(EXTENSION_ARMS),
        "trajectory_count": build_smoke_matrix()["trajectory_count"],
        "smoke_seed_count": smoke_seeds["entry_count"],
        "formal_seed_count": formal_seeds["entry_count"],
        "seed_scope": SMOKE_SEED_SCOPE,
        "seed_dataset": SMOKE_SEED_DATASET,
        "seed_disjointness_audit": validate_smoke_formal_seed_disjointness(smoke_seeds, formal_seeds),
        "submits_jobs": False,
        "creates_run_root": False,
        "formal_launch_authorized": False,
    }


def prepare(run_root: Path, checkpoint_archive: Path) -> Path:
    """Create the fresh, write-once protocol bundle for the 48 smoke rows."""
    run_root = run_root.resolve()
    expected_parent = (REPO / RUN_PARENT_RELATIVE).resolve()
    if run_root.parent != expected_parent:
        raise ValueError(f"Smoke run root must be one direct run ID beneath {expected_parent}: {run_root}")
    if run_root.exists():
        raise FileExistsError(f"Refusing to reuse extension smoke root: {run_root}")

    state = repository_state()
    require_clean_repository(state)
    archive_sha256 = verify_checkpoint_archive(checkpoint_archive)
    checkpoint_dir = REPO / CHECKPOINT_RELATIVE
    checkpoint = checkpoint_identity(checkpoint_dir)
    content_tree = checkpoint_content_tree_identity(checkpoint_dir)
    if (
        content_tree["file_count"] != checkpoint["file_count"]
        or content_tree["total_bytes"] != checkpoint["total_bytes"]
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
    matrix = build_smoke_matrix()
    if matrix["trajectory_count"] != SMOKE_TRAJECTORY_COUNT:
        raise AssertionError("Extension smoke matrix is not exactly 48 rows")
    smoke_seeds = build_smoke_seed_table(FORMAL_TASKS, [0])
    formal_seeds = build_seed_table(FORMAL_TASKS, range(50))
    disjointness = validate_smoke_formal_seed_disjointness(smoke_seeds, formal_seeds)

    run_root.mkdir(parents=True, exist_ok=False)
    protocol_dir = run_root / "protocol"
    protocol_dir.mkdir()
    protocol_path = protocol_dir / "protocol_snapshot.md"
    seed_path = protocol_dir / "seed_table.json"
    formal_seed_path = protocol_dir / "formal_seed_audit_table.json"
    matrix_path = protocol_dir / "smoke_matrix.json"
    atomic_write_bytes(protocol_path, PROTOCOL_PATH.read_bytes())
    atomic_write_bytes(
        protocol_dir / "protocol_sha256.txt",
        (sha256_file(protocol_path) + "\n").encode("ascii"),
    )
    atomic_write_json(seed_path, smoke_seeds)
    atomic_write_json(formal_seed_path, formal_seeds)
    # Copy the reviewed, checked-in bytes rather than regenerating a near match.
    atomic_write_bytes(matrix_path, MATRIX_PATH.read_bytes())
    atomic_write_json(
        protocol_dir / "launch_manifest.json",
        {
            "schema_version": 1,
            "run_id": run_root.name,
            "run_kind": "development_smoke",
            "created_utc": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "protocol_sha256": sha256_file(protocol_path),
            "repository": state,
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
            "checkpoint_archive_verification": "actual archive bytes hashed before run-root creation",
            "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
            "checkpoint_content_tree_algorithm": content_tree["algorithm"],
            "checkpoint_unpacked_content_tree_sha256": content_tree["content_tree_sha256"],
            "checkpoint_file_count": checkpoint["file_count"],
            "checkpoint_total_bytes": checkpoint["total_bytes"],
            "evaluation_policy_seed": 7,
            "master_selector_seed": 2026082501,
            "executed_action_horizon": 16,
            "memory_frame_budget": 32,
            "memory_token_budget": 512,
            "tokens_per_frame": 16,
            "arms": list(EXTENSION_ARMS),
            "seed_table_scope": smoke_seeds["scope"],
            "seed_table_dataset": smoke_seeds["dataset"],
            "seed_table_derivation": smoke_seeds["derivation"],
            "seed_table_entries_sha256": smoke_seeds["entries_sha256"],
            "seed_table_file_sha256": sha256_file(seed_path),
            "formal_seed_audit_scope": FORMAL_SEED_SCOPE,
            "formal_seed_audit_dataset": FORMAL_SEED_DATASET,
            "formal_seed_audit_derivation": formal_seeds["derivation"],
            "formal_seed_audit_entries_sha256": formal_seeds["entries_sha256"],
            "formal_seed_audit_file_sha256": sha256_file(formal_seed_path),
            "seed_disjointness_audit": disjointness,
            "smoke_matrix_sha256": sha256_file(matrix_path),
            "dataset": SMOKE_DATASET,
            "trajectory_count": SMOKE_TRAJECTORY_COUNT,
            "matrix": matrix["rows"],
            "resources": {
                "nodes_per_row": 1,
                "gpus_per_row": 2,
                "max_concurrent": 4,
                "ports": "20000 + (SLURM_JOB_ID mod 20000)",
            },
            "formal_launch_authorized": False,
            "submitted": False,
            "slurm_jobs": [],
            "command_template": (
                "python -m experiments.keyframe_neighborhood_sampling.submit_smoke --run-root <run_root>"
            ),
        },
    )
    return seed_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--checkpoint-archive", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run_contract(), indent=2, sort_keys=True))
        return
    if args.run_root is None or args.checkpoint_archive is None:
        parser.error("--run-root and --checkpoint-archive are required unless --dry-run is used")
    prepare(args.run_root, args.checkpoint_archive)
    print(
        "Prepared without submission. Next run the fresh OC3/OC5 architecture gate via "
        "experiments.keyframe_neighborhood_sampling.submit_architecture_smoke."
    )


if __name__ == "__main__":
    main()
