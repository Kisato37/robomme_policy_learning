#!/usr/bin/env python3
"""Dry-run or prepare one immutable OC3/OC5 formal root; never submit jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_DATASET
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_MAX_STEPS
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.formal_matrix import build_formal_matrix
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

PROTOCOL_PATH = Path(__file__).with_name("EXPERIMENT_EXTENSION_PROTOCOL.md")
RUN_PARENT_RELATIVE = Path("runs/keyframe_neighborhood_sampling")
REFERENCE_RESULTS_RELATIVE = Path("results/keyframe_oracle_sampling/20260829T231425Z_7b594786_formal_v1")
REFERENCE_RUN_ID = "20260829T231425Z_7b594786_formal_v1"
REFERENCE_PER_EPISODE_SHA256 = "e1ec5d1273f3ba97b14d252a1a1668c00635a71477acca687f3e47d7e9f3a478"
REFERENCE_SUMMARY_SHA256 = "e3f40b7192fc1be9a3b280c30234170b63dfae26d389c9b99ba321e82e3efbe9"
REFERENCE_COMPLETENESS_SHA256 = "9f58713cb8da6762c3a81fcfa1776c25bb52a1be75ed26326325ce0cec655ccc"
ANALYSIS_SOURCE_RELATIVE = Path("experiments/keyframe_neighborhood_sampling/analysis.py")
AGGREGATOR_SOURCE_RELATIVE = Path("experiments/keyframe_neighborhood_sampling/aggregate_formal.py")


def repository_state() -> dict[str, Any]:
    """Return a clean-worktree identity while excluding immutable run artifacts."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--",
            ".",
            f":(exclude){RUN_PARENT_RELATIVE.as_posix()}",
            ":(exclude)runs/keyframe_oracle_sampling",
            f":(exclude){CHECKPOINT_RELATIVE.as_posix()}",
        ],
        cwd=REPO,
        text=True,
    )
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=REPO)
    return {
        "commit_sha": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def require_clean_repository(state: dict[str, Any]) -> None:
    if state.get("dirty"):
        raise RuntimeError(
            "OC3/OC5 formal preparation requires a clean committed worktree; "
            "commit the reviewed implementation and fresh extension-smoke evidence first"
        )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _require_extension_identity(
    payload: dict[str, Any],
    *,
    label: str,
    current_commit: str,
    checkpoint: dict[str, int | str],
    checkpoint_content_tree: dict[str, int | str],
    require_formal_not_started: bool,
) -> None:
    """Fail closed until a fresh OC3/OC5 smoke report satisfies this contract."""
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"{label} has the wrong extension protocol version")
    if payload.get("protocol_family") != EXTENSION_PROTOCOL_FAMILY:
        raise RuntimeError(f"{label} is not fresh {EXTENSION_PROTOCOL_FAMILY} evidence")
    if payload.get("passed") is not True:
        raise RuntimeError(f"{label} is not a PASS report")
    if require_formal_not_started and payload.get("formal_started") is not False:
        raise RuntimeError(f"{label} must explicitly report formal_started=false")
    if payload.get("repository_commit_sha") != current_commit:
        raise RuntimeError(f"{label} was produced from a different commit")
    if payload.get("arms") != list(EXTENSION_ARMS):
        raise RuntimeError(f"{label} must explicitly cover exactly OC3 and OC5")
    expected_checkpoint_fields = {
        "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
        "checkpoint_content_tree_algorithm": checkpoint_content_tree["algorithm"],
        "checkpoint_unpacked_content_tree_sha256": checkpoint_content_tree["content_tree_sha256"],
    }
    for field, expected in expected_checkpoint_fields.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"{label} differs from the live checkpoint on {field}")


def _validate_architecture_source(
    run_root: Path,
    *,
    current_commit: str,
    checkpoint: dict[str, int | str],
    checkpoint_content_tree: dict[str, int | str],
) -> tuple[Path, Path]:
    """Require a new extension architecture smoke; old U/O/OC/R smoke is invalid."""
    report_path = run_root / "architecture_smoke" / "report.json"
    submission_path = run_root / "protocol" / "architecture_submission_record.json"
    report = _read_json_object(report_path, label="architecture-smoke report")
    submission = _read_json_object(submission_path, label="architecture-smoke submission record")
    _require_extension_identity(
        report,
        label="Architecture-smoke report",
        current_commit=current_commit,
        checkpoint=checkpoint,
        checkpoint_content_tree=checkpoint_content_tree,
        require_formal_not_started=True,
    )
    if submission.get("protocol_family") != EXTENSION_PROTOCOL_FAMILY:
        raise RuntimeError("Architecture submission is not an extension submission")
    if submission.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Architecture submission has the wrong protocol version")
    if submission.get("repository_commit_sha") != current_commit:
        raise RuntimeError("Architecture submission used a different commit")
    if submission.get("run_root") != str(run_root.resolve()):
        raise RuntimeError("Architecture submission run_root mismatch")
    if report.get("run_root") != str(run_root.resolve()):
        raise RuntimeError("Architecture report run_root mismatch")
    if report.get("runner_backend") == "direct" or submission.get("runner_backend") == "direct":
        # Imported here because architecture_smoke reuses preparation identity helpers.
        from experiments.keyframe_neighborhood_sampling.architecture_smoke import (  # noqa: PLC0415
            validate_architecture_pass_report,
        )

        validate_architecture_pass_report(report, run_root=run_root, architecture_submission=submission)
    elif not report.get("slurm_job_id") or (report.get("slurm_job_id") != submission.get("slurm_job_id")):
        raise RuntimeError("Architecture report/submission Slurm job mismatch")
    return report_path, submission_path


def _validate_development_smoke_source(
    run_root: Path,
    *,
    current_commit: str,
    checkpoint: dict[str, int | str],
    checkpoint_content_tree: dict[str, int | str],
) -> Path:
    """Require a new OC3/OC5 end-to-end development smoke PASS report."""
    audit_path = run_root / "aggregate" / "smoke_audit.json"
    audit = _read_json_object(audit_path, label="development-smoke audit")
    _require_extension_identity(
        audit,
        label="Development-smoke audit",
        current_commit=current_commit,
        checkpoint=checkpoint,
        checkpoint_content_tree=checkpoint_content_tree,
        require_formal_not_started=True,
    )
    completeness = audit.get("completeness")
    if not isinstance(completeness, dict) or completeness.get("complete") is not True:
        raise RuntimeError("Development-smoke audit is not complete")
    fairness = audit.get("initial_condition_fairness")
    if not isinstance(fairness, dict) or fairness.get("fair") is not True:
        raise RuntimeError("Development-smoke audit lacks paired initial-condition fairness")
    return audit_path


def _validate_reference_results(
    checkpoint: dict[str, int | str],
    checkpoint_content_tree: dict[str, int | str],
) -> dict[str, str]:
    root = REPO / REFERENCE_RESULTS_RELATIVE
    per_episode = root / "aggregate" / "per_episode.csv"
    summary = root / "aggregate" / "summary.json"
    completeness = root / "aggregate" / "completeness_report.json"
    observed = {
        "per_episode_sha256": sha256_file(per_episode),
        "summary_sha256": sha256_file(summary),
        "completeness_sha256": sha256_file(completeness),
    }
    expected = {
        "per_episode_sha256": REFERENCE_PER_EPISODE_SHA256,
        "summary_sha256": REFERENCE_SUMMARY_SHA256,
        "completeness_sha256": REFERENCE_COMPLETENESS_SHA256,
    }
    if observed != expected:
        raise RuntimeError("Completed OC reference results differ from the frozen extension protocol")
    summary_payload = _read_json_object(summary, label="published OC reference summary")
    completeness_payload = _read_json_object(
        completeness,
        label="published OC reference completeness report",
    )
    reference_provenance = summary_payload.get("artifact_provenance")
    if not isinstance(reference_provenance, dict):
        raise RuntimeError("Published OC reference summary lacks artifact provenance")
    expected_reference_identity = {
        "checkpoint_id": 79999,
        "checkpoint_path": str(CHECKPOINT_RELATIVE),
        "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
        "checkpoint_content_tree_algorithm": checkpoint_content_tree["algorithm"],
        "checkpoint_unpacked_content_tree_sha256": checkpoint_content_tree["content_tree_sha256"],
    }
    if summary_payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Published OC reference uses a different protocol version")
    if any(reference_provenance.get(field) != value for field, value in expected_reference_identity.items()):
        raise RuntimeError("Published OC reference uses a different checkpoint identity")
    if (
        completeness_payload.get("formal_result_census_complete") is not True
        or completeness_payload.get("strict_selector_trace_audit_complete") is not True
        or completeness_payload.get("initial_condition_fairness", {}).get("fair") is not True
        or completeness_payload.get("paired_manifest_invariants", {}).get("invariants_match") is not True
    ):
        raise RuntimeError("Published OC reference completeness evidence is not a strict PASS")
    return observed


def dry_run_contract() -> dict[str, Any]:
    matrix = build_formal_matrix()
    formal_seed_table = build_seed_table(FORMAL_TASKS, range(50))
    development_seed_table = build_smoke_seed_table(FORMAL_TASKS, [0])
    disjointness = validate_smoke_formal_seed_disjointness(development_seed_table, formal_seed_table)
    return {
        "valid": True,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "dataset": FORMAL_DATASET,
        "max_steps": FORMAL_MAX_STEPS,
        "arms": list(EXTENSION_ARMS),
        "trajectory_count": matrix["trajectory_count"],
        "formal_seed_count": formal_seed_table["entry_count"],
        "seed_scope": FORMAL_SEED_SCOPE,
        "seed_dataset": FORMAL_SEED_DATASET,
        "seed_disjointness_audit": disjointness,
        "fresh_extension_smoke_required": True,
        "fresh_extension_smoke_schema": {
            "common_required_fields": [
                "protocol_version",
                "protocol_family",
                "passed",
                "formal_started",
                "repository_commit_sha",
                "arms",
                "checkpoint_unpacked_metadata_sha256",
                "checkpoint_content_tree_algorithm",
                "checkpoint_unpacked_content_tree_sha256",
            ],
            "architecture_report": "<root>/architecture_smoke/report.json",
            "architecture_submission": ("<root>/protocol/architecture_submission_record.json"),
            "development_audit": "<root>/aggregate/smoke_audit.json",
        },
        "submits_jobs": False,
        "creates_run_root": False,
    }


def prepare(
    run_root: Path,
    checkpoint_archive: Path,
    architecture_run_root: Path,
    development_smoke_run_root: Path,
    *,
    authorization_note: str,
    runner_backend: str = "slurm",
    gpu_layout: str = "separate",
) -> Path:
    if runner_backend not in {"slurm", "direct"}:
        raise ValueError("Unknown runner backend")
    if gpu_layout not in {"separate", "colocated"} or (runner_backend != "direct" and gpu_layout != "separate"):
        raise ValueError("Colocated GPU layout is supported only by the explicit direct backend")
    run_root = run_root.resolve()
    expected_parent = (REPO / RUN_PARENT_RELATIVE).resolve()
    if run_root.parent != expected_parent:
        raise ValueError(f"Formal run root must be one direct run ID beneath {expected_parent}: {run_root}")
    if run_root.exists():
        raise FileExistsError(f"Refusing to reuse extension formal run root: {run_root}")
    if not authorization_note.strip():
        raise ValueError("Formal preparation requires a non-empty authorization note")

    state = repository_state()
    require_clean_repository(state)
    archive_sha256 = verify_checkpoint_archive(checkpoint_archive)
    checkpoint_dir = REPO / CHECKPOINT_RELATIVE
    checkpoint = checkpoint_identity(checkpoint_dir)
    checkpoint_content_tree = checkpoint_content_tree_identity(checkpoint_dir)
    if (
        checkpoint_content_tree["file_count"] != checkpoint["file_count"]
        or checkpoint_content_tree["total_bytes"] != checkpoint["total_bytes"]
    ):
        raise RuntimeError("Checkpoint content inventory differs from frozen metadata")

    current_commit = str(state["commit_sha"])
    architecture_report_path, architecture_submission_path = _validate_architecture_source(
        architecture_run_root.resolve(),
        current_commit=current_commit,
        checkpoint=checkpoint,
        checkpoint_content_tree=checkpoint_content_tree,
    )
    development_audit_path = _validate_development_smoke_source(
        development_smoke_run_root.resolve(),
        current_commit=current_commit,
        checkpoint=checkpoint,
        checkpoint_content_tree=checkpoint_content_tree,
    )
    for evidence_path in (architecture_report_path, development_audit_path):
        evidence = json.loads(evidence_path.read_text())
        if evidence.get("runner_backend", "slurm") != runner_backend:
            raise RuntimeError("Formal backend must match its fresh architecture and development smoke")
    if runner_backend == "direct":
        for source_root in (architecture_run_root, development_smoke_run_root):
            source_manifest = _read_json_object(source_root / "protocol/launch_manifest.json", label="smoke manifest")
            if source_manifest.get("gpu_layout", "separate") != gpu_layout:
                raise RuntimeError("Formal GPU layout must match its fresh architecture and development smoke")
    reference_identity = _validate_reference_results(checkpoint, checkpoint_content_tree)
    frozen_source_identity = {
        "frozen_analysis_source_relative": ANALYSIS_SOURCE_RELATIVE.as_posix(),
        "frozen_analysis_source_sha256": sha256_file(REPO / ANALYSIS_SOURCE_RELATIVE),
        "frozen_aggregator_source_relative": AGGREGATOR_SOURCE_RELATIVE.as_posix(),
        "frozen_aggregator_source_sha256": sha256_file(REPO / AGGREGATOR_SOURCE_RELATIVE),
    }

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
    development_seed_path = protocol_dir / "development_seed_audit_table.json"
    matrix_path = protocol_dir / "formal_matrix.json"
    atomic_write_json(seed_path, formal_seed_table)
    atomic_write_json(development_seed_path, development_seed_table)
    atomic_write_json(matrix_path, matrix)
    architecture_report_copy = protocol_dir / "architecture_pass_report.json"
    architecture_submission_copy = protocol_dir / "architecture_submission_record.json"
    development_audit_copy = protocol_dir / "development_smoke_audit.json"
    atomic_write_bytes(architecture_report_copy, architecture_report_path.read_bytes())
    atomic_write_bytes(architecture_submission_copy, architecture_submission_path.read_bytes())
    atomic_write_bytes(development_audit_copy, development_audit_path.read_bytes())

    manifest_path = protocol_dir / "launch_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "run_id": run_root.name,
            **({"runner_backend": "direct", "gpu_layout": gpu_layout} if runner_backend == "direct" else {}),
            "run_kind": "formal",
            "created_utc": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "protocol_sha256": sha256_file(protocol_snapshot),
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
            "selector_seed_table_role": "randomsamp_policy_call_rng_audit_only",
            "development_seed_audit_scope": development_seed_table["scope"],
            "development_seed_audit_dataset": development_seed_table["dataset"],
            "development_seed_audit_file_sha256": sha256_file(development_seed_path),
            "seed_disjointness_audit": disjointness,
            "architecture_source_run_root": str(architecture_run_root.resolve()),
            "architecture_report_sha256": sha256_file(architecture_report_copy),
            "architecture_submission_record_sha256": sha256_file(architecture_submission_copy),
            "development_smoke_source_run_root": str(development_smoke_run_root.resolve()),
            "development_smoke_audit_sha256": sha256_file(development_audit_copy),
            "reference_run_id": REFERENCE_RUN_ID,
            "reference_results_relative": str(REFERENCE_RESULTS_RELATIVE),
            "reference_per_episode_sha256": reference_identity["per_episode_sha256"],
            "reference_summary_sha256": reference_identity["summary_sha256"],
            "reference_completeness_sha256": reference_identity["completeness_sha256"],
            "reference_cross_run_verification": {
                "pairing_key": ["task", "episode_id"],
                "directly_verified": [
                    "immutable_oc_per_episode_sha256",
                    "immutable_oc_summary_sha256",
                    "immutable_oc_completeness_sha256",
                    "checkpoint_identity",
                    "protocol_identity",
                ],
                "not_directly_verifiable_from_published_reference": [
                    "environment_seed",
                    "difficulty",
                    "raw_initial_condition_hashes",
                ],
            },
            **frozen_source_identity,
            "formal_launch_authorized": True,
            "formal_launch_authorization": {
                "authorized_utc_recorded": utc_now(),
                "authorized_trajectory_count": FORMAL_TRAJECTORY_COUNT,
                "authorization_note": authorization_note.strip(),
            },
            "command_template": (
                "python -m experiments.keyframe_neighborhood_sampling.submit_direct "
                "--run-root <run_root> --stage formal <reviewed-runtime-options> "
                "--confirm-authorized-extension-v1"
                if runner_backend == "direct" else
                "python -m experiments.keyframe_neighborhood_sampling.submit_formal "
                "--run-root <run_root> --confirm-authorized-extension-v1"
            ),
            "resources": {
                "nodes_per_row": 1,
                "gpus_per_row": 1 if gpu_layout == "colocated" else 2,
                "max_concurrent_rows": 4,
                "ports": ("inherited loopback listening socket" if runner_backend == "direct"
                          else "20000 + (SLURM_JOB_ID mod 20000)"),
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
    parser.add_argument("--development-smoke-run-root", type=Path)
    parser.add_argument("--authorization-note")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runner-backend", choices=("slurm", "direct"), default="slurm")
    parser.add_argument("--gpu-layout", choices=("separate", "colocated"), default="separate")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run_contract(), indent=2, sort_keys=True))
        return
    for name in (
        "run_root",
        "checkpoint_archive",
        "architecture_run_root",
        "development_smoke_run_root",
        "authorization_note",
    ):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required unless --dry-run is used")
    seed_path = prepare(
        args.run_root,
        args.checkpoint_archive,
        args.architecture_run_root,
        args.development_smoke_run_root,
        authorization_note=args.authorization_note,
        runner_backend=args.runner_backend,
        gpu_layout=args.gpu_layout,
    )
    entry = (
        "submit_direct --stage formal <reviewed-runtime-options>"
        if args.runner_backend == "direct" else "submit_formal"
    )
    print(
        "Prepared without submission. Validate and submit only through:\n"
        f"python -m experiments.keyframe_neighborhood_sampling.{entry} "
        f"--run-root {args.run_root} --confirm-authorized-extension-v1\n"
        f"Formal seed table: {seed_path}"
    )


if __name__ == "__main__":
    main()
