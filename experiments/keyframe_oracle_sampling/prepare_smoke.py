#!/usr/bin/env python3
"""Dry-run or prepare an immutable smoke run root; never submit jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from experiments.keyframe_oracle_sampling.artifacts import (
    FORMAL_TASKS,
    PROTOCOL_VERSION,
    atomic_write_bytes,
    atomic_write_json,
    build_seed_table,
    build_smoke_seed_table,
    sha256_file,
    utc_now,
    validate_smoke_formal_seed_disjointness,
)
from mme_vla_suite.shared.keyframe_oracle_sampling import (
    SMOKE_SEED_DATASET,
    SMOKE_SEED_SCOPE,
)
from experiments.keyframe_oracle_sampling.smoke_matrix import (
    MATRIX_PATH,
    expand_rows,
    load_frozen_matrix,
)


REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("EXPERIMENT_PROTOCOL.md")
SBATCH_PATH = Path(__file__).with_name("run_smoke.sbatch")
CHECKPOINT_RELATIVE = Path(
    "runs/test_time_scaling/checkpoints/perceptual-framesamp-modul/79999"
)
POLICY_VENV = REPO / ".venv"
SIMULATOR_VENV = REPO / "third_party" / "robomme_benchmark" / ".venv"
POLICY_PYTHON = POLICY_VENV / "bin" / "python"
SIMULATOR_PYTHON = SIMULATOR_VENV / "bin" / "python"
POLICY_UV_LOCK = REPO / "uv.lock"
BENCHMARK_UV_LOCK = REPO / "third_party" / "robomme_benchmark" / "uv.lock"
EXPECTED_CHECKPOINT_ARCHIVE_SHA256 = (
    "2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"
)
EXPECTED_CHECKPOINT_METADATA_SHA256 = (
    "313e483ae32881e606402365a8b01a599eda22a367e436f9dfbd5456120dca26"
)
EXPECTED_CHECKPOINT_FILE_COUNT = 18
EXPECTED_CHECKPOINT_TOTAL_BYTES = 11_877_152_238
CHECKPOINT_CONTENT_TREE_ALGORITHM = "sha256-canonical-file-content-tree-v1"


def repository_state() -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--",
            ".",
            ":(exclude)runs/keyframe_oracle_sampling",
            f":(exclude){CHECKPOINT_RELATIVE.as_posix()}",
        ],
        cwd=REPO,
        text=True,
    )
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=REPO
    )
    return {
        "commit_sha": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def checkpoint_identity(checkpoint_dir: Path) -> dict[str, int | str]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing frozen checkpoint: {checkpoint_dir}")
    entries = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_file():
            entries.append(
                {
                    "path": str(path.relative_to(checkpoint_dir)),
                    "size": path.stat().st_size,
                }
            )
    if not entries:
        raise RuntimeError(f"Frozen checkpoint directory is empty: {checkpoint_dir}")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity = {
        "metadata_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
    }
    expected = {
        "metadata_sha256": EXPECTED_CHECKPOINT_METADATA_SHA256,
        "file_count": EXPECTED_CHECKPOINT_FILE_COUNT,
        "total_bytes": EXPECTED_CHECKPOINT_TOTAL_BYTES,
    }
    if identity != expected:
        raise RuntimeError(
            "Frozen checkpoint identity mismatch; refusing path fallback or smoke launch: "
            f"observed={identity}, expected={expected}"
        )
    return identity


def checkpoint_metadata_digest(checkpoint_dir: Path) -> str:
    """Backward-compatible accessor used by lightweight tests and audits."""
    return str(checkpoint_identity(checkpoint_dir)["metadata_sha256"])


def checkpoint_content_tree_identity(checkpoint_dir: Path) -> dict[str, int | str]:
    """Hash the actual bytes of every regular file in an unpacked checkpoint.

    The existing checkpoint identity intentionally remains a cheap path-and-size
    gate.  This second, stronger identity detects equal-size content changes by
    hashing each file and then hashing a canonical, path-sorted file manifest.
    It is deliberately called only by run-level gates, never by each smoke row.
    """
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing frozen checkpoint: {checkpoint_dir}")
    entries: list[dict[str, int | str]] = []
    for path in sorted(
        checkpoint_dir.rglob("*"),
        key=lambda candidate: candidate.relative_to(checkpoint_dir).as_posix(),
    ):
        relative = path.relative_to(checkpoint_dir).as_posix()
        if path.is_symlink():
            raise RuntimeError(
                "Frozen checkpoint content tree must contain only regular files and "
                f"directories; refusing symbolic link: {relative}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(
                "Frozen checkpoint content tree contains an unsupported entry: "
                f"{relative}"
            )
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise RuntimeError(f"Frozen checkpoint directory is empty: {checkpoint_dir}")
    canonical_tree = {
        "algorithm": CHECKPOINT_CONTENT_TREE_ALGORITHM,
        "files": entries,
    }
    encoded = json.dumps(
        canonical_tree,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "algorithm": CHECKPOINT_CONTENT_TREE_ALGORITHM,
        "content_tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
    }


def require_clean_repository(repo_state: dict) -> None:
    if repo_state["dirty"]:
        raise RuntimeError(
            "Smoke preparation requires a clean committed worktree; commit and push the "
            "reviewed implementation before creating an immutable run root"
        )


def verify_checkpoint_archive(checkpoint_archive: Path) -> str:
    if not checkpoint_archive.is_file():
        raise FileNotFoundError(f"Missing frozen checkpoint archive: {checkpoint_archive}")
    observed = sha256_file(checkpoint_archive)
    if observed != EXPECTED_CHECKPOINT_ARCHIVE_SHA256:
        raise RuntimeError(
            "Frozen checkpoint archive SHA-256 mismatch: "
            f"{observed} != {EXPECTED_CHECKPOINT_ARCHIVE_SHA256}"
        )
    return observed


def python_environment_identity(
    python_path: Path,
    expected_venv: Path,
) -> dict[str, object]:
    """Probe the exact interpreter and virtual environment used by a launcher."""
    python_path = python_path.absolute()
    expected_venv = expected_venv.resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Missing pinned Python interpreter: {python_path}")
    pyvenv_config = expected_venv / "pyvenv.cfg"
    if not pyvenv_config.is_file():
        raise FileNotFoundError(f"Missing virtual-environment identity: {pyvenv_config}")
    probe = (
        "import json, platform, sys; "
        "print(json.dumps({"
        "'executable': sys.executable, "
        "'prefix': sys.prefix, "
        "'base_prefix': sys.base_prefix, "
        "'version': sys.version, "
        "'version_info': list(sys.version_info[:5]), "
        "'implementation': platform.python_implementation(), "
        "'cache_tag': sys.implementation.cache_tag"
        "}, sort_keys=True))"
    )
    payload = json.loads(
        subprocess.check_output([str(python_path), "-c", probe], text=True)
    )
    observed_prefix = Path(str(payload["prefix"])).resolve()
    observed_base_prefix = Path(str(payload["base_prefix"])).resolve()
    if observed_prefix != expected_venv:
        raise RuntimeError(
            "Pinned Python does not belong to the expected virtual environment: "
            f"{observed_prefix} != {expected_venv}"
        )
    if observed_prefix == observed_base_prefix:
        raise RuntimeError(f"Pinned Python is not running inside a virtual environment: {python_path}")
    executable = Path(str(payload["executable"]))
    executable_realpath = executable.resolve()
    return {
        "launcher_python_path": str(python_path),
        "reported_executable": str(executable),
        "executable_realpath": str(executable_realpath),
        "interpreter_sha256": sha256_file(executable_realpath),
        "venv_prefix": str(observed_prefix),
        "base_prefix": str(observed_base_prefix),
        "pyvenv_cfg_sha256": sha256_file(pyvenv_config),
        "python_version": str(payload["version"]),
        "python_version_info": payload["version_info"],
        "implementation": str(payload["implementation"]),
        "cache_tag": str(payload["cache_tag"]),
    }


def environment_lock_identity() -> dict[str, str]:
    """Hash both independently resolved policy and benchmark lockfiles."""
    return {
        "policy_uv_lock_sha256": sha256_file(POLICY_UV_LOCK),
        "benchmark_uv_lock_sha256": sha256_file(BENCHMARK_UV_LOCK),
    }


def prepare(run_root: Path, checkpoint_archive: Path) -> Path:
    run_root = run_root.resolve()
    expected_parent = (REPO / "runs" / "keyframe_oracle_sampling").resolve()
    if run_root.parent != expected_parent:
        raise ValueError(
            f"Smoke run root must be one direct run ID beneath {expected_parent}: {run_root}"
        )
    if run_root.exists():
        raise FileExistsError(f"Refusing to reuse smoke run root: {run_root}")
    checkpoint_dir = REPO / CHECKPOINT_RELATIVE
    archive_sha256 = verify_checkpoint_archive(checkpoint_archive)
    checkpoint = checkpoint_identity(checkpoint_dir)
    checkpoint_content_tree = checkpoint_content_tree_identity(checkpoint_dir)
    if (
        checkpoint_content_tree["file_count"] != checkpoint["file_count"]
        or checkpoint_content_tree["total_bytes"] != checkpoint["total_bytes"]
    ):
        raise RuntimeError(
            "Checkpoint content-tree inventory differs from the frozen metadata identity"
        )
    benchmark_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO / "third_party/robomme_benchmark",
        text=True,
    ).strip()
    repo_state = repository_state()
    require_clean_repository(repo_state)
    environment_locks = environment_lock_identity()
    python_environments = {
        "policy": python_environment_identity(POLICY_PYTHON, POLICY_VENV),
        "simulator": python_environment_identity(SIMULATOR_PYTHON, SIMULATOR_VENV),
    }
    rows = expand_rows(load_frozen_matrix())
    seed_table = build_smoke_seed_table(FORMAL_TASKS, [0])
    formal_seed_audit_table = build_seed_table(FORMAL_TASKS, range(50))
    seed_disjointness = validate_smoke_formal_seed_disjointness(
        seed_table,
        formal_seed_audit_table,
    )

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
    atomic_write_json(seed_path, seed_table)
    formal_seed_audit_path = protocol_dir / "formal_seed_audit_table.json"
    atomic_write_json(formal_seed_audit_path, formal_seed_audit_table)
    frozen_matrix_path = protocol_dir / "smoke_matrix.json"
    atomic_write_bytes(frozen_matrix_path, MATRIX_PATH.read_bytes())
    atomic_write_json(
        protocol_dir / "launch_manifest.json",
        {
            "run_id": run_root.name,
            "created_utc": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "repository": repo_state,
            "benchmark_repository_commit": benchmark_commit,
            "environment_lock_sha256": environment_locks["policy_uv_lock_sha256"],
            "environment_locks": environment_locks,
            "python_environments": python_environments,
            "checkpoint_path": str(CHECKPOINT_RELATIVE),
            "checkpoint_archive_path": str(checkpoint_archive.resolve()),
            "checkpoint_archive_sha256_expected": EXPECTED_CHECKPOINT_ARCHIVE_SHA256,
            "checkpoint_archive_sha256_actual": archive_sha256,
            "checkpoint_archive_verification": "actual archive bytes hashed before run-root creation",
            "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
            "checkpoint_content_tree_algorithm": checkpoint_content_tree["algorithm"],
            "checkpoint_unpacked_content_tree_sha256": checkpoint_content_tree[
                "content_tree_sha256"
            ],
            "checkpoint_file_count": checkpoint["file_count"],
            "checkpoint_total_bytes": checkpoint["total_bytes"],
            "evaluation_policy_seed": 7,
            "master_selector_seed": 2026082501,
            "seed_table_scope": SMOKE_SEED_SCOPE,
            "seed_table_dataset": SMOKE_SEED_DATASET,
            "seed_table_derivation": seed_table["derivation"],
            "seed_table_entries_sha256": seed_table["entries_sha256"],
            "seed_table_file_sha256": sha256_file(seed_path),
            "formal_seed_audit_scope": formal_seed_audit_table["scope"],
            "formal_seed_audit_dataset": formal_seed_audit_table["dataset"],
            "formal_seed_audit_derivation": formal_seed_audit_table["derivation"],
            "formal_seed_audit_entries_sha256": formal_seed_audit_table[
                "entries_sha256"
            ],
            "formal_seed_audit_file_sha256": sha256_file(formal_seed_audit_path),
            "seed_disjointness_audit": seed_disjointness,
            "smoke_matrix_sha256": sha256_file(frozen_matrix_path),
            "dataset": "val",
            "matrix": rows,
            "command_template": (
                "python -m experiments.keyframe_oracle_sampling.submit_smoke "
                "--run-root <run_root>"
            ),
            "resources": {
                "nodes_per_row": 1,
                "gpus_per_row": 2,
                "ports": "20000 + (SLURM_JOB_ID mod 20000)",
            },
            "slurm_jobs": [],
            "submitted": False,
        },
    )
    return seed_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--checkpoint-archive", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = expand_rows(load_frozen_matrix())
    if args.dry_run:
        smoke_seed_table = build_smoke_seed_table(FORMAL_TASKS, [0])
        formal_seed_table = build_seed_table(FORMAL_TASKS, range(50))
        disjointness = validate_smoke_formal_seed_disjointness(
            smoke_seed_table,
            formal_seed_table,
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "trajectory_count": len(rows),
                    "submits_jobs": False,
                    "seed_scope": SMOKE_SEED_SCOPE,
                    "seed_dataset": SMOKE_SEED_DATASET,
                    "seed_disjointness_audit": disjointness,
                }
            )
        )
        return
    if args.run_root is None:
        parser.error("--run-root is required unless --dry-run is used")
    if args.checkpoint_archive is None:
        parser.error("--checkpoint-archive is required unless --dry-run is used")
    prepare(args.run_root, args.checkpoint_archive)
    print(
        "Prepared without submission. Submit the GPU architecture gate first:\n"
        "python -m experiments.keyframe_oracle_sampling.submit_architecture_smoke "
        f"--run-root {args.run_root}\n"
        "After its write-once PASS report, submit the 80-row smoke only through:\n"
        "python -m experiments.keyframe_oracle_sampling.submit_smoke "
        f"--run-root {args.run_root}"
    )


if __name__ == "__main__":
    main()
