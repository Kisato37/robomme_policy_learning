#!/usr/bin/env python3
"""Reconcile launcher/evaluator failures into the immutable attempt ledger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import EpisodeAttemptWriter
from experiments.keyframe_oracle_sampling.artifacts import RunArtifactStore
from experiments.keyframe_oracle_sampling.artifacts import ScientificKey
from experiments.keyframe_oracle_sampling.artifacts import read_jsonl
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import validate_prepared_smoke_root
from experiments.keyframe_oracle_sampling.formal_artifacts import validate_prepared_formal_root
from experiments.keyframe_oracle_sampling.formal_matrix import validate_formal_runtime_row_binding
from experiments.keyframe_oracle_sampling.smoke_matrix import validate_runtime_row_binding


def _matrix_row_field(trajectory_kind: str) -> str:
    return (
        "formal_matrix_row_id"
        if trajectory_kind == "formal"
        else "smoke_matrix_row_id"
    )


def record_launcher_failure(
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
    error_type: str,
    error: str,
    slurm: dict[str, Any],
) -> Path:
    """Record a server-readiness failure without manufacturing a result."""
    key = ScientificKey(task, episode_id, arm, trajectory_kind)
    store = RunArtifactStore(run_root)
    writer = store.new_attempt(
        key,
        attempt_id,
        {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": dataset,
            "max_steps": int(max_steps),
            _matrix_row_field(trajectory_kind): int(row_id),
            "execution_phase": "policy_server_readiness",
            "scientific_actions_started": False,
            "environment_setup_completed": False,
            "launcher_failure_only": True,
            "slurm": slurm,
        },
    )
    retry_allowed = attempt_id < 2
    store.record_failure(
        {
            **key.as_dict(),
            "attempt_id": int(attempt_id),
            _matrix_row_field(trajectory_kind): int(row_id),
            "error_type": error_type,
            "error": error,
            "failure_phase": "policy_server_readiness",
            "classification": "infrastructure",
            "retry_allowed": retry_allowed,
            "scientific_actions_started": False,
            "episode_manifest_sha256": sha256_file(writer.manifest_path),
            "slurm": slurm,
        }
    )
    return writer.attempt_dir


def _matching_failures(
    store: RunArtifactStore,
    key: ScientificKey,
    attempt_id: int,
) -> list[dict[str, Any]]:
    if not store.failures_path.exists():
        return []
    return [
        record
        for record in read_jsonl(store.failures_path)
        if str(record.get("task")) == key.task
        and int(record.get("episode_id", -1)) == key.episode_id
        and str(record.get("arm")) == key.arm
        and str(record.get("trajectory_kind", "formal")) == key.trajectory_kind
        and int(record.get("attempt_id", -1)) == int(attempt_id)
    ]


def reconcile_evaluator_exit(
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
    exit_status: int,
    slurm: dict[str, Any],
) -> tuple[Path, str]:
    """Ensure an evaluator exit has either a valid result or exactly one failure.

    This is the launcher's final, fail-closed guard around evaluator lifecycle
    failures.  The evaluator remains the primary writer.  If it already wrote a
    result or failure, this function validates/accepts that evidence and never
    appends a duplicate.  Only shell signal-style statuses are retryable; an
    ordinary non-zero exit, or a zero exit without a completed artifact, is an
    unknown deterministic failure and therefore a hard stop.
    """
    exit_status = int(exit_status)
    if exit_status < 0 or exit_status > 255:
        raise ValueError(f"Evaluator exit status must be in 0..255, got {exit_status}")

    key = ScientificKey(task, episode_id, arm, trajectory_kind)
    store = RunArtifactStore(run_root)
    attempt_dir = store.attempt_dir(key, attempt_id)
    writer = EpisodeAttemptWriter(attempt_dir, key, attempt_id)
    existing_failures = _matching_failures(store, key, attempt_id)
    if len(existing_failures) > 1:
        raise ArtifactContractError(
            f"Attempt already has duplicate failure-ledger records: {attempt_dir}"
        )

    artifact_error: Exception | None = None
    if writer.result_path.exists():
        # A non-zero process status can occur after the immutable episode result
        # was finalized (for example while writing a legacy summary).  The
        # protocol-governed trajectory itself is complete, so do not manufacture
        # a contradictory failure record.
        try:
            if writer.validate_resume() != "complete":
                raise ArtifactContractError(f"Invalid completed attempt: {attempt_dir}")
        except Exception as exc:  # Preserve invalid artifact evidence in the ledger.
            artifact_error = exc
        else:
            return attempt_dir, "complete"

    if existing_failures:
        if not attempt_dir.is_dir():
            raise ArtifactContractError(
                "Failure ledger references a missing immutable attempt directory"
            )
        return attempt_dir, "existing_failure"

    if not attempt_dir.exists():
        writer = store.new_attempt(
            key,
            attempt_id,
            {
                "protocol_version": PROTOCOL_VERSION,
                "dataset": dataset,
                "max_steps": int(max_steps),
                _matrix_row_field(trajectory_kind): int(row_id),
                "execution_phase": "evaluator_lifecycle",
                "scientific_actions_started": False,
                "environment_setup_completed": False,
                "launcher_failure_only": True,
                "evaluator_exit_status": exit_status,
                "slurm": slurm,
            },
        )
    else:
        try:
            writer.validate_resume()
        except Exception as exc:
            artifact_error = artifact_error or exc

    # Retry only the explicit signal statuses handled/expected by this launcher.
    # Arbitrary programs may intentionally exit in 128..255, so treating that
    # entire range as infrastructure would silently retry deterministic defects.
    signal_exit = exit_status in {130, 137, 143}
    classification = "infrastructure" if signal_exit else "hard_stop"
    retry_allowed = signal_exit and attempt_id < 2
    if artifact_error is not None:
        # A malformed/partial immutable artifact is never silently accepted as
        # complete.  A signal-interrupted attempt that never reached a result is
        # retryable under the explicit infrastructure rule; an invalid result,
        # or any non-signal artifact defect, remains a hard stop.
        error_type = "InvalidOrPartialEpisodeArtifact"
        error = f"Immutable episode artifact validation failed: {artifact_error}"
        if writer.result_path.exists() or not signal_exit:
            classification = "hard_stop"
            retry_allowed = False
    elif exit_status == 0:
        error_type = "EvaluatorExitedWithoutEpisodeArtifact"
        error = (
            "Evaluator exited with status 0 but did not produce an immutable "
            "episode_result.json"
        )
    elif signal_exit:
        error_type = "EvaluatorProcessSignal"
        error = (
            f"Evaluator process ended with signal-style exit status {exit_status}; "
            "inspect immutable Slurm/server logs"
        )
    else:
        error_type = "UnhandledEvaluatorLifecycleFailure"
        error = (
            f"Evaluator process exited with status {exit_status} before producing "
            "a complete immutable episode artifact"
        )

    manifest: dict[str, Any] = {}
    manifest_sha256: str | None = None
    if writer.manifest_path.exists():
        try:
            manifest = json.loads(writer.manifest_path.read_text())
        except Exception:
            # The validation error above is already preserved verbatim.  Do not
            # rewrite a malformed write-once manifest merely to make it parse.
            manifest = {}
        manifest_sha256 = sha256_file(writer.manifest_path)
    store.record_failure(
        {
            **key.as_dict(),
            "attempt_id": int(attempt_id),
            _matrix_row_field(trajectory_kind): int(row_id),
            "error_type": error_type,
            "error": error,
            "failure_phase": "evaluator_lifecycle",
            "classification": classification,
            "retry_allowed": retry_allowed,
            "evaluator_exit_status": exit_status,
            "scientific_actions_started": bool(writer.trace_path.exists()),
            "environment_setup_completed": manifest.get(
                "environment_setup_completed"
            ),
            "episode_manifest_sha256": manifest_sha256,
            "slurm": slurm,
        }
    )
    return attempt_dir, "recorded_failure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed-table", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--attempt-id", type=int, required=True)
    parser.add_argument("--row-id", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--trajectory-kind", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--error-type")
    parser.add_argument("--error")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--evaluator-exit-status", type=int)
    parser.add_argument("--formal-authorization", default="")
    args = parser.parse_args()

    if args.trajectory_kind == "formal":
        validate_prepared_formal_root(
            args.run_root,
            args.seed_table,
            args.repo_root,
            attempt_id=args.attempt_id,
            authorization_digest=args.formal_authorization,
        )
        row_validator = validate_formal_runtime_row_binding
    else:
        if args.formal_authorization:
            parser.error("smoke reconciliation may not carry formal authorization")
        validate_prepared_smoke_root(
            args.run_root,
            args.seed_table,
            args.repo_root,
            attempt_id=args.attempt_id,
        )
        row_validator = validate_runtime_row_binding
    row_validator(
        args.run_root,
        attempt_id=args.attempt_id,
        row_id=args.row_id,
        task=args.task,
        episode_id=args.episode_id,
        arm=args.arm,
        trajectory_kind=args.trajectory_kind,
        max_steps=args.max_steps,
        dataset=args.dataset,
    )
    slurm = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "policy_port": args.policy_port,
    }
    if args.evaluator_exit_status is None:
        if not args.error_type or not args.error:
            parser.error("readiness failure mode requires --error-type and --error")
        attempt_dir = record_launcher_failure(
            args.run_root,
            attempt_id=args.attempt_id,
            row_id=args.row_id,
            task=args.task,
            episode_id=args.episode_id,
            arm=args.arm,
            trajectory_kind=args.trajectory_kind,
            max_steps=args.max_steps,
            dataset=args.dataset,
            error_type=args.error_type,
            error=args.error,
            slurm=slurm,
        )
        print(attempt_dir)
        return

    attempt_dir, disposition = reconcile_evaluator_exit(
        args.run_root,
        attempt_id=args.attempt_id,
        row_id=args.row_id,
        task=args.task,
        episode_id=args.episode_id,
        arm=args.arm,
        trajectory_kind=args.trajectory_kind,
        max_steps=args.max_steps,
        dataset=args.dataset,
        exit_status=args.evaluator_exit_status,
        slurm=slurm,
    )
    print(f"{disposition}\t{attempt_dir}")
    if disposition != "complete":
        raise SystemExit(80)


if __name__ == "__main__":
    main()
