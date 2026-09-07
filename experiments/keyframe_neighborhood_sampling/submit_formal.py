#!/usr/bin/env python3
"""Validate and submit the exact frozen 1,600-row OC3/OC5 Slurm array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.keyframe_neighborhood_sampling.formal_artifacts import submission_path
from experiments.keyframe_neighborhood_sampling.formal_artifacts import submission_paths
from experiments.keyframe_neighborhood_sampling.formal_artifacts import submission_plan_path
from experiments.keyframe_neighborhood_sampling.formal_artifacts import validate_formal_prepared_root
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.formal_matrix import load_formal_matrix
from experiments.keyframe_neighborhood_sampling.prepare_formal import repository_state
from experiments.keyframe_neighborhood_sampling.prepare_formal import require_clean_repository
from experiments.keyframe_neighborhood_sampling.record_launcher_failure import validate_extension_failure_record
from experiments.keyframe_oracle_sampling.artifacts import RunArtifactStore
from experiments.keyframe_oracle_sampling.artifacts import ScientificKey
from experiments.keyframe_oracle_sampling.artifacts import atomic_write_json
from experiments.keyframe_oracle_sampling.artifacts import read_jsonl
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from experiments.keyframe_oracle_sampling.environment_contract import validate_environment_contract
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_RELATIVE
from experiments.keyframe_oracle_sampling.prepare_smoke import REPO
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_content_tree_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_identity

SBATCH_PATH = Path(__file__).with_name("run_formal.sbatch")
MAX_ROWS_PER_ARRAY = 1000


def submission_failure_paths(run_root: Path, attempt_id: int, shard_id: int) -> list[Path]:
    return sorted(
        (run_root / "protocol").glob(f"submission_failure_attempt_{attempt_id:02d}_shard_{shard_id:02d}_try_*.json")
    )


def submission_no_job_confirmation_path(run_root: Path, attempt_id: int, shard_id: int, try_id: int) -> Path:
    return (
        run_root
        / "protocol"
        / (f"submission_no_job_confirmation_attempt_{attempt_id:02d}_shard_{shard_id:02d}_try_{try_id:02d}.json")
    )


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_existing_record(
    path: Path,
    *,
    expected: dict[str, Any],
    plan_sha256: str,
) -> dict[str, Any]:
    record = _load_object(path, label="extension formal submission record")
    expected_keys = set(expected) | {"slurm_array_job_id", "submission_plan_sha256"}
    if set(record) != expected_keys:
        raise RuntimeError(f"Existing extension submission record has unexpected fields: {path}")
    for field, value in expected.items():
        if field != "submitted_utc" and record.get(field) != value:
            raise RuntimeError(f"Existing extension submission record changed {field}: {path}")
    if not isinstance(record.get("submitted_utc"), str) or not record["submitted_utc"]:
        raise RuntimeError(f"Existing extension submission record lacks timestamp: {path}")
    if record.get("submission_plan_sha256") != plan_sha256:
        raise RuntimeError(f"Existing extension submission record has wrong plan digest: {path}")
    if not isinstance(record.get("slurm_array_job_id"), str) or not record["slurm_array_job_id"]:
        raise RuntimeError(f"Existing extension submission record lacks job ID: {path}")
    return record


def _validate_known_submission_failures(
    paths: list[Path],
    *,
    run_root: Path,
    expected: dict[str, Any],
    plan_sha256: str,
    explicit_no_job_confirmation: bool,
) -> None:
    if not paths:
        raise RuntimeError("A partial extension submission may resume only after a recorded, known sbatch failure")
    for try_id, path in enumerate(paths):
        failure = _load_object(path, label="extension formal submission failure")
        required = {
            "schema_version": 1,
            "protocol_version": expected["protocol_version"],
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "attempt_id": expected["attempt_id"],
            "shard_id": expected["shard_id"],
            "try_id": try_id,
            "command": expected["command"],
            "submission_plan_sha256": plan_sha256,
        }
        if any(failure.get(field) != value for field, value in required.items()):
            raise RuntimeError(f"Extension submission failure provenance mismatch: {path}")
        if not isinstance(failure.get("failed_utc"), str) or not failure["failed_utc"]:
            raise RuntimeError(f"Extension submission failure lacks timestamp: {path}")
        error_type = failure.get("error_type")
        if error_type not in {"CalledProcessError", "FileNotFoundError", "PermissionError"}:
            raise RuntimeError(f"Extension submission failure type is not safely retryable: {path}")
        if error_type in {"FileNotFoundError", "PermissionError"}:
            if failure.get("known_no_job_id") is not True or failure.get("outcome_uncertain") is not False:
                raise RuntimeError(f"Safe extension submission failure has invalid outcome evidence: {path}")
        elif failure.get("known_no_job_id") is not False or failure.get("outcome_uncertain") is not True:
            raise RuntimeError(f"Ambiguous extension submission failure has invalid evidence: {path}")

    if any(_load_object(path, label="extension formal submission failure")["outcome_uncertain"] for path in paths):
        confirmation_path = submission_no_job_confirmation_path(
            run_root,
            int(expected["attempt_id"]),
            int(expected["shard_id"]),
            len(paths) - 1,
        )
        if confirmation_path.is_file():
            confirmation = _load_object(confirmation_path, label="manual no-job confirmation")
            expected_confirmation = {
                "schema_version": 1,
                "protocol_version": expected["protocol_version"],
                "protocol_family": EXTENSION_PROTOCOL_FAMILY,
                "attempt_id": expected["attempt_id"],
                "shard_id": expected["shard_id"],
                "submission_plan_sha256": plan_sha256,
                "failure_record_sha256": sha256_file(paths[-1]),
                "human_verified_no_job": True,
            }
            if any(confirmation.get(field) != value for field, value in expected_confirmation.items()):
                raise RuntimeError("Manual no-job confirmation provenance mismatch")
        elif not explicit_no_job_confirmation:
            raise RuntimeError(
                "sbatch outcome is uncertain; verify Slurm created no job, then explicitly confirm this shard"
            )


def _compress_row_ids(row_ids: list[int]) -> str:
    if not row_ids:
        raise ValueError("Cannot compress an empty row list")
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


def shard_rows(
    row_ids: list[int], *, max_concurrent: int, sequential: bool = False
) -> tuple[list[list[int]], int]:
    """Preserve 1,000-row shards under concurrent Slurm or sequential direct scheduling."""
    shards = [row_ids[start : start + MAX_ROWS_PER_ARRAY] for start in range(0, len(row_ids), MAX_ROWS_PER_ARRAY)]
    if not shards:
        raise ValueError("Formal extension submission cannot contain zero rows")
    if not sequential and len(shards) > max_concurrent:
        raise ValueError(f"{len(shards)} Slurm shards require --max-concurrent at least {len(shards)}")
    per_shard_concurrent = max_concurrent if sequential else max_concurrent // len(shards)
    if per_shard_concurrent < 1:
        raise AssertionError("Per-shard concurrency must remain positive")
    return shards, per_shard_concurrent


def _validate_retry_rows(
    run_root: Path,
    rows: list[dict],
    *,
    attempt_id: int,
    row_ids: tuple[int, ...] | None,
) -> list[int]:
    if not row_ids:
        raise ValueError("Formal extension retry submissions require explicit --rows")
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
        validate_extension_failure_record(
            run_root,
            matching[0],
            expected_row_id=row_id,
        )
    return selected_rows


def build_submission(
    run_root: Path,
    *,
    max_concurrent: int,
    attempt_id: int,
    row_ids: tuple[int, ...] | None,
    confirmed_no_job_shards: tuple[int, ...] | None = None,
) -> tuple[list[tuple[list[str], dict]], dict]:
    run_root = run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Extension formal run root does not exist: {run_root}")
    if not 1 <= max_concurrent <= 4:
        raise ValueError("--max-concurrent must be in the frozen safe range 1..4")
    if attempt_id not in {0, 1, 2}:
        raise ValueError("--attempt-id must be 0, 1, or 2")
    confirmed_no_job_shards = tuple(confirmed_no_job_shards or ())
    if len(set(confirmed_no_job_shards)) != len(confirmed_no_job_shards):
        raise ValueError("Manual no-job shard confirmations may not contain duplicates")
    plan_path = submission_plan_path(run_root, attempt_id)
    existing_records = submission_paths(run_root, attempt_id)
    if existing_records and not plan_path.is_file():
        raise FileExistsError("Extension submission records exist without their immutable plan")
    existing_plan = _load_object(plan_path, label="extension formal submission plan") if plan_path.is_file() else None

    seed_path = run_root / "protocol" / "seed_table.json"
    manifest_path = run_root / "protocol" / "launch_manifest.json"
    matrix_path = run_root / "protocol" / "formal_matrix.json"
    state = repository_state()
    require_clean_repository(state)
    manifest = validate_formal_prepared_root(run_root, seed_path, REPO)
    validate_environment_contract(manifest)
    if manifest.get("repository", {}).get("commit_sha") != state["commit_sha"]:
        raise RuntimeError("Prepared extension root and current checkout use different commits")

    checkpoint = checkpoint_identity(REPO / CHECKPOINT_RELATIVE)
    content_tree = checkpoint_content_tree_identity(REPO / CHECKPOINT_RELATIVE)
    if manifest.get("checkpoint_unpacked_metadata_sha256") != checkpoint["metadata_sha256"]:
        raise RuntimeError("Prepared extension checkpoint metadata differs from live checkpoint")
    if (
        manifest.get("checkpoint_content_tree_algorithm") != content_tree["algorithm"]
        or manifest.get("checkpoint_unpacked_content_tree_sha256") != content_tree["content_tree_sha256"]
    ):
        raise RuntimeError("Prepared extension checkpoint content differs from live bytes")

    rows = load_formal_matrix(matrix_path)["rows"]
    store = RunArtifactStore(run_root)
    if attempt_id == 0:
        if row_ids:
            raise ValueError("Initial extension formal submission must use the complete 1,600-row matrix")
        if existing_plan is None and (
            store.scan_completed_keys() or store.failures_path.exists() or (run_root / "trajectories").exists()
        ):
            raise RuntimeError("Initial extension formal submission requires an unused prepared run root")
        selected_rows = list(range(len(rows)))
        if len(selected_rows) != FORMAL_TRAJECTORY_COUNT:
            raise AssertionError("Extension formal matrix is not exactly 1,600 rows")
    else:
        selected_rows = _validate_retry_rows(run_root, rows, attempt_id=attempt_id, row_ids=row_ids)

    log_dir = run_root / "slurm"
    sequential = manifest.get("runner_backend") == "direct"
    row_shards, per_shard_concurrent = shard_rows(
        selected_rows, max_concurrent=max_concurrent, sequential=sequential
    )
    scheduling = {"shard_scheduling": "sequential"} if sequential else {}
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
            f"--export=KEYFRAME_NEIGHBORHOOD_REPO_ROOT={REPO}",
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
            **scheduling,
            "schema_version": 1,
            "protocol_version": manifest["protocol_version"],
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
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
            "architecture_report_sha256": manifest["architecture_report_sha256"],
            "development_smoke_audit_sha256": manifest["development_smoke_audit_sha256"],
            "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
            "checkpoint_content_tree_algorithm": content_tree["algorithm"],
            "checkpoint_unpacked_content_tree_sha256": content_tree["content_tree_sha256"],
            "command": command,
        }
        submissions.append((command, record))

    plan = {
        **scheduling,
        "schema_version": 1,
        "protocol_version": manifest["protocol_version"],
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "created_utc": existing_plan.get("created_utc") if existing_plan is not None else utc_now(),
        "repository_commit_sha": state["commit_sha"],
        "run_root": str(run_root),
        "attempt_id": attempt_id,
        "trajectory_count": len(selected_rows),
        "shard_count": shard_count,
        "max_rows_per_array": MAX_ROWS_PER_ARRAY,
        "global_max_concurrent": max_concurrent,
        "formal_launch_authorized": True,
        "launch_manifest_sha256": sha256_file(manifest_path),
        "formal_matrix_sha256": sha256_file(matrix_path),
        "shards": [
            {
                field: record[field]
                for field in (
                    "shard_id",
                    "shard_count",
                    "array_task_ids",
                    "row_ids",
                    "array",
                    "command",
                )
            }
            for _, record in submissions
        ],
    }
    if existing_plan is not None:
        if existing_plan != plan:
            raise RuntimeError("Existing extension submission plan differs from the canonical plan")
        plan_sha256 = sha256_file(plan_path)
        expected_by_shard = {int(record["shard_id"]): record for _, record in submissions}
        recorded_shards: set[int] = set()
        for path in existing_records:
            payload = _load_object(path, label="extension formal submission record")
            shard_id = payload.get("shard_id")
            if type(shard_id) is not int or shard_id not in expected_by_shard:
                raise RuntimeError(f"Existing extension submission record has invalid shard: {path}")
            if shard_id in recorded_shards:
                raise RuntimeError(f"Duplicate extension submission record for shard {shard_id}")
            _validate_existing_record(
                path,
                expected=expected_by_shard[shard_id],
                plan_sha256=plan_sha256,
            )
            recorded_shards.add(shard_id)
        pending: list[tuple[list[str], dict]] = []
        for command, record in submissions:
            shard_id = int(record["shard_id"])
            if shard_id in recorded_shards:
                continue
            _validate_known_submission_failures(
                submission_failure_paths(run_root, attempt_id, shard_id),
                run_root=run_root,
                expected=record,
                plan_sha256=plan_sha256,
                explicit_no_job_confirmation=shard_id in confirmed_no_job_shards,
            )
            pending.append((command, record))
        if set(confirmed_no_job_shards) - {int(record["shard_id"]) for _, record in pending}:
            raise ValueError("Manual no-job confirmation named a shard that is not pending")
        return pending, plan
    if confirmed_no_job_shards:
        raise ValueError("Manual no-job confirmations are valid only for partial recovery")
    return submissions, plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--attempt-id", type=int, default=0)
    parser.add_argument("--rows", help="Comma-separated row IDs; retries only")
    parser.add_argument(
        "--confirm-no-job-for-shards",
        help="Comma-separated shards manually verified to have no Slurm job after an uncertain sbatch outcome",
    )
    parser.add_argument("--confirm-authorized-extension-v1", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.confirm_authorized_extension_v1:
        parser.error(
            "live submission requires --confirm-authorized-extension-v1; use --dry-run for non-submitting validation"
        )
    row_ids = tuple(int(value) for value in args.rows.split(",") if value.strip()) if args.rows else None
    confirmed_no_job_shards = (
        tuple(int(value) for value in args.confirm_no_job_for_shards.split(",") if value.strip())
        if args.confirm_no_job_for_shards
        else None
    )
    submissions, plan = build_submission(
        args.run_root,
        max_concurrent=args.max_concurrent,
        attempt_id=args.attempt_id,
        row_ids=row_ids,
        confirmed_no_job_shards=confirmed_no_job_shards,
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

    manifest = json.loads((args.run_root / "protocol/launch_manifest.json").read_text())
    if manifest.get("runner_backend") == "direct":
        parser.error("This root requires submit_direct, not a Slurm submission")
    (args.run_root / "slurm").mkdir(parents=True, exist_ok=True)
    plan_path = submission_plan_path(args.run_root, args.attempt_id)
    if not plan_path.exists():
        atomic_write_json(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    for _, record in submissions:
        shard_id = int(record["shard_id"])
        failure_paths = submission_failure_paths(args.run_root, args.attempt_id, shard_id)
        if (
            shard_id in set(confirmed_no_job_shards or ())
            and failure_paths
            and any(
                _load_object(path, label="extension formal submission failure").get("outcome_uncertain") is True
                for path in failure_paths
            )
        ):
            confirmation_path = submission_no_job_confirmation_path(
                args.run_root, args.attempt_id, shard_id, len(failure_paths) - 1
            )
            if not confirmation_path.exists():
                atomic_write_json(
                    confirmation_path,
                    {
                        "schema_version": 1,
                        "protocol_version": record["protocol_version"],
                        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
                        "confirmed_utc": utc_now(),
                        "attempt_id": args.attempt_id,
                        "shard_id": shard_id,
                        "submission_plan_sha256": plan_sha256,
                        "failure_record_sha256": sha256_file(failure_paths[-1]),
                        "human_verified_no_job": True,
                    },
                )
    job_ids = [
        _load_object(path, label="extension formal submission record")["slurm_array_job_id"]
        for path in submission_paths(args.run_root, args.attempt_id)
    ]
    for command, record in submissions:
        shard_id = int(record["shard_id"])
        failures = submission_failure_paths(args.run_root, args.attempt_id, shard_id)
        try:
            job_id = subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as exc:
            known_no_job_id = isinstance(exc, (FileNotFoundError, PermissionError))
            failure_record = {
                "schema_version": 1,
                "protocol_version": record["protocol_version"],
                "protocol_family": EXTENSION_PROTOCOL_FAMILY,
                "failed_utc": utc_now(),
                "attempt_id": args.attempt_id,
                "shard_id": shard_id,
                "try_id": len(failures),
                "command": command,
                "submission_plan_sha256": plan_sha256,
                "known_no_job_id": known_no_job_id,
                "outcome_uncertain": not known_no_job_id,
                "error_type": type(exc).__name__,
                "returncode": getattr(exc, "returncode", None),
                "output": getattr(exc, "output", None),
                "stderr": getattr(exc, "stderr", None),
            }
            atomic_write_json(
                args.run_root
                / "protocol"
                / (
                    f"submission_failure_attempt_{args.attempt_id:02d}_"
                    f"shard_{shard_id:02d}_try_{len(failures):02d}.json"
                ),
                failure_record,
            )
            raise
        if not job_id:
            raise RuntimeError("sbatch returned an empty extension formal job ID")
        record["slurm_array_job_id"] = job_id
        record["submission_plan_sha256"] = plan_sha256
        atomic_write_json(
            submission_path(
                args.run_root,
                args.attempt_id,
                shard_id,
            ),
            record,
        )
        job_ids.append(job_id)
    print(json.dumps({"slurm_array_job_ids": job_ids}, sort_keys=True))


if __name__ == "__main__":
    main()
