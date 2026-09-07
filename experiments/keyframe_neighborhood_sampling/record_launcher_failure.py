#!/usr/bin/env python3
"""Reconcile OC3/OC5 smoke/formal launcher failures under extension gates."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

from experiments.keyframe_neighborhood_sampling.direct_provenance import direct_runner_for_runtime
from experiments.keyframe_neighborhood_sampling.direct_provenance import runner_backend
from experiments.keyframe_neighborhood_sampling.direct_provenance import validate_direct_attempt
from experiments.keyframe_neighborhood_sampling.direct_provenance import validate_direct_process_exit
from experiments.keyframe_neighborhood_sampling.formal_artifacts import validate_prepared_formal_root
from experiments.keyframe_neighborhood_sampling.formal_artifacts import validate_prepared_smoke_root
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.formal_matrix import validate_formal_runtime_row_binding
from experiments.keyframe_neighborhood_sampling.smoke_matrix import validate_runtime_row_binding
from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import EpisodeAttemptWriter
from experiments.keyframe_oracle_sampling.artifacts import RunArtifactStore
from experiments.keyframe_oracle_sampling.artifacts import ScientificKey
from experiments.keyframe_oracle_sampling.artifacts import read_jsonl
from experiments.keyframe_oracle_sampling.artifacts import sha256_file


def _matrix_row_field(trajectory_kind: str) -> str:
    return "formal_matrix_row_id" if trajectory_kind == "formal" else "smoke_matrix_row_id"


def _execution_provenance(
    run_root: Path, *, row_id: int, attempt_id: int, trajectory_kind: str,
    slurm: dict | None, runner: dict | None,
) -> dict[str, Any]:
    if runner is not None:
        if slurm is not None:
            raise ArtifactContractError("Direct attempt cannot contain Slurm provenance")
        payload = {"runner_backend": "direct", "runner": runner}
        validate_direct_attempt(
            payload, run_root, row_id=row_id, attempt_id=attempt_id, trajectory_kind=trajectory_kind,
        )
        return payload
    slurm = dict(slurm or {})
    row_field = _matrix_row_field(trajectory_kind)
    if row_field in slurm and str(slurm[row_field]) != str(row_id):
        raise ArtifactContractError("Launcher failure Slurm row binding conflicts with row ID")
    slurm[row_field] = str(row_id)
    return {"slurm": slurm}


def _manifest_execution(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if runner_backend(manifest) == "direct":
        return {
            "runner_backend": "direct", "runner": manifest.get("runner"),
            **({"renderer_device": manifest["renderer_device"]} if "renderer_device" in manifest else {}),
        }
    return {"slurm": manifest.get("slurm")}


def _direct_lifecycle_deadline(run_root: Path, manifest: Mapping[str, Any], exit_status: int) -> bool:
    if type(exit_status) is not int or not 0 <= exit_status <= 255:
        raise ArtifactContractError("Direct evaluator lifecycle lacks a valid shell exit status")
    exited = validate_direct_process_exit(manifest["runner"], run_root, role="evaluator")
    raw_status = exited["returncode"]
    if exit_status != (128 - raw_status if raw_status < 0 else raw_status):
        raise ArtifactContractError("Direct evaluator status differs from its linked process exit")
    return exited["wall_clock_limit_reached"] is True


def record_launcher_failure(
    run_root: Path,
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
    slurm: dict | None = None,
    runner: dict | None = None,
) -> Path:
    """Write an extension-identified readiness failure without a fake result."""
    key = ScientificKey(task, episode_id, arm, trajectory_kind)
    row_field = _matrix_row_field(trajectory_kind)
    execution = _execution_provenance(
        run_root, row_id=row_id, attempt_id=attempt_id, trajectory_kind=trajectory_kind, slurm=slurm, runner=runner,
    )
    store = RunArtifactStore(run_root)
    writer = store.new_attempt(
        key,
        attempt_id,
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "dataset": dataset,
            "max_steps": int(max_steps),
            row_field: int(row_id),
            "execution_phase": "policy_server_readiness",
            "scientific_actions_started": False,
            "environment_setup_completed": False,
            "launcher_failure_only": True,
            **execution,
        },
    )
    store.record_failure(
        {
            **key.as_dict(),
            "protocol_version": PROTOCOL_VERSION,
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "attempt_id": int(attempt_id),
            row_field: int(row_id),
            "error_type": error_type,
            "error": error,
            "failure_phase": "policy_server_readiness",
            "classification": "infrastructure",
            "retry_allowed": attempt_id < 2,
            "scientific_actions_started": False,
            "environment_setup_completed": False,
            "episode_manifest_sha256": sha256_file(writer.manifest_path),
            **execution,
        }
    )
    return writer.attempt_dir


def _ensure_extension_attempt_manifest(
    run_root: Path,
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
    slurm: dict | None = None,
    runner: dict | None = None,
) -> None:
    """Pre-create only a missing launcher-owned manifest with extension identity."""
    key = ScientificKey(task, episode_id, arm, trajectory_kind)
    row_field = _matrix_row_field(trajectory_kind)
    execution = _execution_provenance(
        run_root, row_id=row_id, attempt_id=attempt_id, trajectory_kind=trajectory_kind, slurm=slurm, runner=runner,
    )
    store = RunArtifactStore(run_root)
    attempt_dir = store.attempt_dir(key, attempt_id)
    if attempt_dir.exists():
        return
    store.new_attempt(
        key,
        attempt_id,
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "dataset": dataset,
            "max_steps": int(max_steps),
            row_field: int(row_id),
            "execution_phase": "evaluator_lifecycle",
            "scientific_actions_started": False,
            "environment_setup_completed": False,
            "launcher_failure_only": True,
            "evaluator_exit_status": int(exit_status),
            **execution,
        },
    )


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
        and int(record.get("attempt_id", -1)) == attempt_id
    ]


def validate_extension_failure_record(
    run_root: Path,
    record: Mapping[str, Any],
    *,
    expected_row_id: int | None = None,
    require_direct_completion: bool = True,
) -> dict[str, Any]:
    """Require a failure ledger row to bind exact OC3/OC5 attempt evidence."""
    try:
        key = ScientificKey(
            str(record["task"]),
            int(record["episode_id"]),
            str(record["arm"]),
            str(record["trajectory_kind"]),
        )
        attempt_id = int(record["attempt_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactContractError("Malformed extension failure identity") from exc
    if key.arm not in {"OC3", "OC5"} or attempt_id not in {0, 1, 2}:
        raise ArtifactContractError("Failure record is not an OC3/OC5 attempt")
    writer = EpisodeAttemptWriter(RunArtifactStore(run_root).attempt_dir(key, attempt_id), key, attempt_id)
    if not writer.manifest_path.is_file() or writer.result_path.exists():
        raise ArtifactContractError("Extension failure must bind one incomplete attempt")
    try:
        manifest = json.loads(writer.manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError("Extension failure attempt manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise ArtifactContractError("Extension failure attempt manifest must be an object")
    if (
        manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("protocol_family") != EXTENSION_PROTOCOL_FAMILY
    ):
        raise ArtifactContractError("Extension failure attempt has wrong protocol identity")
    row_field = _matrix_row_field(key.trajectory_kind)
    row_id = record.get(row_field)
    if type(row_id) is not int or (expected_row_id is not None and row_id != expected_row_id):
        raise ArtifactContractError("Extension failure has wrong matrix row binding")
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "task": key.task,
        "episode_id": key.episode_id,
        "arm": key.arm,
        "trajectory_kind": key.trajectory_kind,
        "attempt_id": attempt_id,
        "episode_manifest_sha256": sha256_file(writer.manifest_path),
        "scientific_actions_started": writer.trace_path.exists(),
        "environment_setup_completed": manifest.get("environment_setup_completed"),
        **_manifest_execution(manifest),
    }
    if any(record.get(field) != value for field, value in expected.items()):
        raise ArtifactContractError("Extension failure provenance does not match its attempt")
    backend = runner_backend(manifest)
    if runner_backend(record) != backend:
        raise ArtifactContractError("Failure and attempt runner backends differ")
    if backend == "direct":
        roles = {"preflight", "policy", "reconcile"}
        if record.get("failure_phase") != "policy_server_readiness":
            roles.add("evaluator")
        elif (
            manifest.get("execution_phase") != "policy_server_readiness"
            or manifest.get("launcher_failure_only") is not True
            or record.get("scientific_actions_started") is not False
        ):
            raise ArtifactContractError("Direct readiness failure lacks launcher-only provenance")
        dispatch = validate_direct_attempt(
            manifest, run_root, attempt_id=attempt_id, row_id=row_id, trajectory_kind=key.trajectory_kind,
            required_roles=roles if require_direct_completion else None,
        )
        manifest_row_id = dispatch.row_id
    else:
        manifest_row_id = manifest.get("slurm", {}).get(row_field)
    if str(row_id) != str(manifest_row_id):
        raise ArtifactContractError("Extension failure row differs from attempt Slurm evidence")
    classification = record.get("classification")
    retry_allowed = record.get("retry_allowed")
    if classification not in {"infrastructure", "hard_stop"} or type(retry_allowed) is not bool:
        raise ArtifactContractError("Extension failure classification is invalid")
    if retry_allowed and (classification != "infrastructure" or attempt_id >= 2):
        raise ArtifactContractError("Extension failure has an invalid retry authorization")
    if (
        backend == "direct"
        and record.get("failure_phase") == "evaluator_lifecycle"
        and _direct_lifecycle_deadline(run_root, manifest, record.get("evaluator_exit_status"))
        and (classification != "hard_stop" or retry_allowed)
    ):
        raise ArtifactContractError("Direct deadline-terminated lifecycle cannot authorize a retry")
    if not isinstance(record.get("error_type"), str) or not isinstance(record.get("error"), str):
        raise ArtifactContractError("Extension failure lacks error evidence")
    if record.get("failure_phase") not in {
        "policy_server_readiness",
        "evaluator_episode",
        "evaluator_lifecycle",
    }:
        raise ArtifactContractError("Extension failure phase is invalid")
    return {"key": key, "attempt_id": attempt_id, "row_id": row_id, "manifest": manifest}


def reconcile_evaluator_exit(run_root: Path, **kwargs):
    """Reconcile evaluator exit without accepting a provenance-free failure."""
    _ensure_extension_attempt_manifest(
        run_root,
        exit_status=kwargs["exit_status"],
        **{
            field: kwargs.get(field)
            for field in (
                "attempt_id",
                "row_id",
                "task",
                "episode_id",
                "arm",
                "trajectory_kind",
                "max_steps",
                "dataset",
                "slurm",
                "runner",
            )
        },
    )
    key = ScientificKey(
        kwargs["task"],
        kwargs["episode_id"],
        kwargs["arm"],
        kwargs["trajectory_kind"],
    )
    attempt_id = int(kwargs["attempt_id"])
    store = RunArtifactStore(run_root)
    writer = EpisodeAttemptWriter(store.attempt_dir(key, attempt_id), key, attempt_id)
    existing_failures = _matching_failures(store, key, attempt_id)
    if len(existing_failures) > 1:
        raise ArtifactContractError("Extension attempt has duplicate failure records")
    if writer.result_path.exists() and writer.validate_resume() == "complete":
        return writer.attempt_dir, "complete"
    if existing_failures:
        validate_extension_failure_record(
            run_root,
            existing_failures[0],
            expected_row_id=int(kwargs["row_id"]),
            require_direct_completion=False,
        )
        return writer.attempt_dir, "existing_failure"

    exit_status = int(kwargs["exit_status"])
    if exit_status < 0 or exit_status > 255:
        raise ValueError(f"Evaluator exit status must be in 0..255, got {exit_status}")
    manifest = json.loads(writer.manifest_path.read_text())
    direct_deadline = runner_backend(manifest) == "direct" and _direct_lifecycle_deadline(run_root, manifest, exit_status)
    artifact_error: Exception | None = None
    try:
        writer.validate_resume()
    except Exception as exc:
        artifact_error = exc
    signal_exit = exit_status in {130, 137, 143} and not direct_deadline
    classification = "infrastructure" if signal_exit else "hard_stop"
    retry_allowed = signal_exit and attempt_id < 2
    if artifact_error is not None:
        error_type = "InvalidOrPartialEpisodeArtifact"
        error = f"Immutable episode artifact validation failed: {artifact_error}"
        if writer.result_path.exists() or not signal_exit:
            classification = "hard_stop"
            retry_allowed = False
    elif direct_deadline:
        error_type = "DirectEvaluatorWallClockLimit"
        error = "Evaluator reached the recorded direct execution deadline without a complete result; review required"
    elif exit_status == 0:
        error_type = "EvaluatorExitedWithoutEpisodeArtifact"
        error = "Evaluator exited with status 0 but produced no immutable episode result"
    elif signal_exit:
        error_type = "EvaluatorProcessSignal"
        error = f"Evaluator process ended with signal-style exit status {exit_status}"
    else:
        error_type = "UnhandledEvaluatorLifecycleFailure"
        error = f"Evaluator exited with status {exit_status} before a complete result"
    record = {
        **key.as_dict(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "attempt_id": attempt_id,
        _matrix_row_field(key.trajectory_kind): int(kwargs["row_id"]),
        "error_type": error_type,
        "error": error,
        "failure_phase": "evaluator_lifecycle",
        "classification": classification,
        "retry_allowed": retry_allowed,
        "evaluator_exit_status": exit_status,
        "scientific_actions_started": writer.trace_path.exists(),
        "environment_setup_completed": manifest.get("environment_setup_completed"),
        "episode_manifest_sha256": sha256_file(writer.manifest_path),
        **_manifest_execution(manifest),
    }
    store.record_failure(record)
    validate_extension_failure_record(
        run_root,
        record,
        expected_row_id=int(kwargs["row_id"]),
        require_direct_completion=False,
    )
    return writer.attempt_dir, "recorded_failure"


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
        if not args.formal_authorization:
            parser.error("formal reconciliation requires --formal-authorization")
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
            parser.error("development smoke may not carry formal authorization")
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
        _matrix_row_field(args.trajectory_kind): str(args.row_id),
        "node": os.environ.get("SLURMD_NODENAME"),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "policy_port": args.policy_port,
    }
    runner = None
    if os.environ.get("KEYFRAME_RUNNER_BACKEND") == "direct":
        runner = direct_runner_for_runtime(args.run_root)
        if runner["dispatch"]["policy_port"] != args.policy_port:
            raise ArtifactContractError("Direct reconciliation uses a different policy port")
        slurm = None
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
            runner=runner,
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
        runner=runner,
    )
    print(f"{disposition}\t{attempt_dir}")
    if disposition != "complete":
        raise SystemExit(80)


if __name__ == "__main__":
    main()
