#!/usr/bin/env python3
"""Fail-closed audit of the fresh 48-row OC3/OC5 development smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.keyframe_neighborhood_sampling.architecture_smoke import validate_architecture_pass_report
from experiments.keyframe_neighborhood_sampling.direct_provenance import runner_backend
from experiments.keyframe_neighborhood_sampling.direct_provenance import validate_direct_attempt
from experiments.keyframe_neighborhood_sampling.direct_provenance import validate_direct_submission
from experiments.keyframe_neighborhood_sampling.formal_artifacts import smoke_submission_path
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.record_launcher_failure import validate_extension_failure_record
from experiments.keyframe_neighborhood_sampling.smoke_matrix import SMOKE_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.smoke_matrix import load_smoke_matrix
from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import EpisodeAttemptWriter
from experiments.keyframe_oracle_sampling.artifacts import RunArtifactStore
from experiments.keyframe_oracle_sampling.artifacts import ScientificKey
from experiments.keyframe_oracle_sampling.artifacts import atomic_write_json
from experiments.keyframe_oracle_sampling.artifacts import audit_initial_condition_fairness
from experiments.keyframe_oracle_sampling.artifacts import audit_paired_manifest_invariants
from experiments.keyframe_oracle_sampling.artifacts import audit_smoke_attempt
from experiments.keyframe_oracle_sampling.artifacts import completeness_report
from experiments.keyframe_oracle_sampling.artifacts import load_seed_table
from experiments.keyframe_oracle_sampling.artifacts import read_jsonl
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from experiments.keyframe_oracle_sampling.artifacts import validate_smoke_formal_seed_disjointness
from experiments.keyframe_oracle_sampling.audit_smoke import audit_short_trajectory_length
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import oracle_neighborhood_coverage_decision


def expected_smoke_keys() -> set[ScientificKey]:
    return {
        ScientificKey(row["task"], row["episode_id"], row["arm"], row["trajectory_kind"])
        for row in load_smoke_matrix()["rows"]
    }


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError(f"{label} must be an object: {path}")
    return payload


def _validate_submission(
    path: Path,
    *,
    attempt_id: int,
    launch: dict[str, Any],
    run_root: Path,
    architecture_path: Path,
    row_count: int,
) -> dict[str, Any]:
    submission = _load_object(path, label="extension smoke submission")
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "repository_commit_sha": launch["repository"]["commit_sha"],
        "run_root": str(run_root.resolve()),
        "attempt_id": attempt_id,
        "smoke_launch_authorized": True,
        "formal_launch_authorized": False,
        "launch_manifest_sha256": sha256_file(run_root / "protocol/launch_manifest.json"),
        "smoke_matrix_sha256": launch["smoke_matrix_sha256"],
        "architecture_report_sha256": sha256_file(architecture_path),
    }
    if any(submission.get(field) != value for field, value in expected.items()):
        raise ArtifactContractError("Extension smoke submission provenance mismatch")
    rows = submission.get("row_ids")
    if (
        not isinstance(rows, list)
        or any(type(row_id) is not int for row_id in rows)
        or len(rows) != len(set(rows))
        or any(row_id < 0 or row_id >= row_count for row_id in rows)
    ):
        raise ArtifactContractError("Extension smoke submission has invalid row IDs")
    if submission.get("trajectory_count") != len(rows):
        raise ArtifactContractError("Extension smoke submission row count mismatch")
    backend = runner_backend(submission)
    if backend != runner_backend(launch):
        raise ArtifactContractError("Smoke submission and launch runner backends differ")
    if backend == "direct":
        validate_direct_submission(submission, stage="development_smoke")
    elif not isinstance(submission.get("slurm_array_job_id"), str) or not submission["slurm_array_job_id"]:
        raise ArtifactContractError("Extension smoke submission lacks a Slurm array ID")
    for field in (
        "checkpoint_unpacked_metadata_sha256",
        "checkpoint_content_tree_algorithm",
        "checkpoint_unpacked_content_tree_sha256",
    ):
        if submission.get(field) != launch.get(field):
            raise ArtifactContractError(f"Extension smoke submission differs from launch manifest on {field}")
    return submission


def _audit_neighborhood_trace(writer: EpisodeAttemptWriter, arm: str) -> dict[str, int]:
    requested_frames = 3 if arm == "OC3" else 5
    fallback_calls = 0
    secondary_thinning_calls = 0
    traces = read_jsonl(writer.trace_path)
    for call_index, trace in enumerate(traces):
        boundaries = trace.get("visible_boundary_indices")
        current_index = trace.get("current_history_index")
        if not isinstance(boundaries, list) or type(current_index) is not int:
            raise ArtifactContractError("Extension selector trace lacks causal boundary evidence")
        expected_selected, expected_decision = oracle_neighborhood_coverage_decision(
            current_index,
            boundaries,
            neighborhood_frames=requested_frames,
        )
        if trace.get("selected_frame_indices") != expected_selected:
            raise ArtifactContractError(f"Extension smoke trace call {call_index} does not replay as {arm}")
        for field, expected_value in expected_decision.items():
            if trace.get(field) != expected_value:
                raise ArtifactContractError(f"Extension smoke trace call {call_index} has wrong {field}")
        fallback_calls += int(bool(expected_decision["oc5_fell_back_to_oc3"]))
        secondary_thinning_calls += int(bool(expected_decision["oc3_secondary_thinning"]))
    return {
        "policy_call_count": len(traces),
        "oc5_fallback_calls": fallback_calls,
        "oc3_secondary_thinning_calls": secondary_thinning_calls,
    }


def build_smoke_audit(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    protocol_dir = run_root / "protocol"
    paths = {
        "manifest": protocol_dir / "launch_manifest.json",
        "protocol": protocol_dir / "protocol_snapshot.md",
        "protocol_digest": protocol_dir / "protocol_sha256.txt",
        "seed": protocol_dir / "seed_table.json",
        "formal_seed": protocol_dir / "formal_seed_audit_table.json",
        "matrix": protocol_dir / "smoke_matrix.json",
        "architecture_submission": protocol_dir / "architecture_submission_record.json",
        "architecture": run_root / "architecture_smoke/report.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise ArtifactContractError(f"Extension smoke audit prerequisite is missing: {path}")
    launch = _load_object(paths["manifest"], label="extension smoke manifest")
    expected_launch = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "run_kind": "development_smoke",
        "dataset": "val",
        "trajectory_count": SMOKE_TRAJECTORY_COUNT,
        "arms": list(EXTENSION_ARMS),
        "formal_launch_authorized": False,
    }
    if any(launch.get(field) != value for field, value in expected_launch.items()):
        raise ArtifactContractError("Launch root is not the fresh OC3/OC5 development smoke")
    digest_map = {
        "protocol_sha256": paths["protocol"],
        "seed_table_file_sha256": paths["seed"],
        "formal_seed_audit_file_sha256": paths["formal_seed"],
        "smoke_matrix_sha256": paths["matrix"],
    }
    if any(launch.get(field) != sha256_file(path) for field, path in digest_map.items()):
        raise ArtifactContractError("Extension smoke protocol bundle digest mismatch")
    if paths["protocol_digest"].read_text().strip() != sha256_file(paths["protocol"]):
        raise ArtifactContractError("Extension protocol digest sidecar mismatch")

    smoke_seed_payload, smoke_lookup = load_seed_table(
        paths["seed"], expected_scope=SMOKE_SEED_SCOPE, expected_dataset=SMOKE_SEED_DATASET
    )
    formal_seed_payload, _ = load_seed_table(
        paths["formal_seed"],
        expected_scope=FORMAL_SEED_SCOPE,
        expected_dataset=FORMAL_SEED_DATASET,
    )
    disjointness = validate_smoke_formal_seed_disjointness(smoke_seed_payload, formal_seed_payload)
    expected_seed_fields = {
        "seed_table_scope": smoke_seed_payload["scope"],
        "seed_table_dataset": smoke_seed_payload["dataset"],
        "seed_table_derivation": smoke_seed_payload["derivation"],
        "seed_table_entries_sha256": smoke_seed_payload["entries_sha256"],
        "seed_disjointness_audit": disjointness,
    }
    if any(launch.get(field) != value for field, value in expected_seed_fields.items()):
        raise ArtifactContractError("Extension smoke seed provenance mismatch")

    rows = load_smoke_matrix(paths["matrix"])["rows"]
    if launch.get("matrix") != rows:
        raise ArtifactContractError("Extension smoke launch manifest matrix mismatch")
    expected_rows: dict[ScientificKey, dict[str, Any]] = {}
    row_id_by_key: dict[ScientificKey, int] = {}
    for row_id, row in enumerate(rows):
        if row.get("row_id") != row_id:
            raise ArtifactContractError("Extension smoke row IDs are not canonical")
        key = ScientificKey(row["task"], row["episode_id"], row["arm"], row["trajectory_kind"])
        if key in expected_rows:
            raise ArtifactContractError(f"Duplicate extension smoke key: {key}")
        expected_rows[key] = row
        row_id_by_key[key] = row_id

    architecture_submission = _load_object(paths["architecture_submission"], label="extension architecture submission")
    architecture = _load_object(paths["architecture"], label="extension architecture report")
    if runner_backend(architecture) != runner_backend(launch):
        raise ArtifactContractError("Extension architecture and smoke runner backends differ")
    validate_architecture_pass_report(
        architecture,
        run_root=run_root,
        architecture_submission=architecture_submission,
    )
    repository_commit = launch.get("repository", {}).get("commit_sha")
    if architecture.get("repository_commit_sha") != repository_commit:
        raise ArtifactContractError("Extension architecture and smoke commit differ")
    for field in (
        "checkpoint_unpacked_metadata_sha256",
        "checkpoint_content_tree_algorithm",
        "checkpoint_unpacked_content_tree_sha256",
    ):
        if architecture.get(field) != launch.get(field):
            raise ArtifactContractError(f"Extension architecture differs on {field}")

    initial_submission = _validate_submission(
        smoke_submission_path(run_root, 0),
        attempt_id=0,
        launch=launch,
        run_root=run_root,
        architecture_path=paths["architecture"],
        row_count=len(rows),
    )
    if initial_submission["row_ids"] != list(range(SMOKE_TRAJECTORY_COUNT)):
        raise ArtifactContractError("Initial extension smoke submission is not all 48 rows")

    store = RunArtifactStore(run_root)
    completed = store.scan_completed_keys()
    completeness = completeness_report(set(expected_rows), completed)
    failures = read_jsonl(store.failures_path) if store.failures_path.exists() else []
    for failure in failures:
        try:
            failure_key = ScientificKey(
                str(failure["task"]),
                int(failure["episode_id"]),
                str(failure["arm"]),
                str(failure["trajectory_kind"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactContractError("Malformed extension smoke failure key") from exc
        if failure_key not in row_id_by_key:
            raise ArtifactContractError("Unexpected extension smoke failure key")
        validate_extension_failure_record(
            run_root,
            failure,
            expected_row_id=row_id_by_key[failure_key],
        )
    attempts: dict[str, Any] = {}
    short_audits: dict[str, Any] = {}
    paired_manifests: list[dict[str, Any]] = []
    diagnostics = {
        arm: {"policy_call_count": 0, "oc5_fallback_calls": 0, "oc3_secondary_thinning_calls": 0}
        for arm in EXTENSION_ARMS
    }
    retry_submissions: dict[int, dict[str, Any]] = {}
    policy_lifetimes: set[str] = set()
    used_failures: set[int] = set()
    for key, result_path in sorted(completed.items()):
        if key not in expected_rows:
            raise ArtifactContractError(f"Unexpected extension smoke result: {key}")
        result = _load_object(result_path, label="extension smoke result")
        attempt_id = result.get("attempt_id")
        if type(attempt_id) is not int or attempt_id not in {0, 1, 2}:
            raise ArtifactContractError("Extension smoke result has invalid attempt ID")
        submission = initial_submission
        if attempt_id > 0:
            submission = retry_submissions.get(attempt_id)
            if submission is None:
                submission = _validate_submission(
                    smoke_submission_path(run_root, attempt_id),
                    attempt_id=attempt_id,
                    launch=launch,
                    run_root=run_root,
                    architecture_path=paths["architecture"],
                    row_count=len(rows),
                )
                retry_submissions[attempt_id] = submission
            for prior_attempt in range(attempt_id):
                matching_indices = [
                    index
                    for index, failure in enumerate(failures)
                    if str(failure.get("task")) == key.task
                    and int(failure.get("episode_id", -1)) == key.episode_id
                    and str(failure.get("arm")) == key.arm
                    and str(failure.get("trajectory_kind")) == key.trajectory_kind
                    and int(failure.get("attempt_id", -1)) == prior_attempt
                    and failure.get("retry_allowed") is True
                ]
                if len(matching_indices) != 1:
                    raise ArtifactContractError("Extension smoke retry chain lacks one allowed failure")
                used_failures.add(matching_indices[0])
        row_id = row_id_by_key[key]
        if row_id not in submission["row_ids"]:
            raise ArtifactContractError("Extension smoke result was not authorized by its submission")
        writer = EpisodeAttemptWriter(result_path.parent, key, attempt_id)
        attempt_report = audit_smoke_attempt(
            writer,
            expected_key=key,
            expected_row=expected_rows[key],
            launch_manifest=launch,
            smoke_seed_payload=smoke_seed_payload,
            smoke_seed_lookup=smoke_lookup,
        )
        episode_manifest = _load_object(writer.manifest_path, label="extension episode manifest")
        if episode_manifest.get("protocol_family") != EXTENSION_PROTOCOL_FAMILY:
            raise ArtifactContractError("Extension episode manifest lacks protocol family")
        if runner_backend(episode_manifest) != runner_backend(submission):
            raise ArtifactContractError("Smoke attempt and submission runner backends differ")
        if runner_backend(submission) == "direct":
            policy_lifetimes.add(submission.get("runtime_profile", {}).get("policy_lifetime", "per_row"))
            if episode_manifest.get("environment_setup_completed") is not True:
                raise ArtifactContractError("Completed direct smoke result lacks confirmed simulator setup")
            validate_direct_attempt(
                episode_manifest, run_root, attempt_id=attempt_id, row_id=row_id,
                trajectory_kind=key.trajectory_kind, required_roles={"preflight", "policy", "evaluator", "reconcile"},
            )
        else:
            slurm = episode_manifest.get("slurm")
            if not isinstance(slurm, dict) or (
                slurm.get("array_job_id") != submission["slurm_array_job_id"] or slurm.get("array_task_id") != str(row_id)
            ):
                raise ArtifactContractError("Extension smoke attempt Slurm binding mismatch")
        decision_report = _audit_neighborhood_trace(writer, key.arm)
        for field, value in decision_report.items():
            diagnostics[key.arm][field] += value
        identity = json.dumps(key.as_dict(), sort_keys=True)
        attempts[identity] = {**attempt_report, "selector_diagnostics": decision_report}
        if key.trajectory_kind == "short":
            short_audits[identity] = audit_short_trajectory_length(result)
            paired = dict(episode_manifest)
            paired["initial_condition_hashes"] = _load_object(
                writer.initial_conditions_path, label="initial-condition hashes"
            )
            paired_manifests.append(paired)

    unconsumed_failure_count = len(set(range(len(failures))) - used_failures)
    failure_chain_complete = unconsumed_failure_count == 0
    fairness = (
        audit_initial_condition_fairness(paired_manifests, required_arms=EXTENSION_ARMS)
        if len(paired_manifests) == 32
        else {"paired_blocks": 0, "fair": False, "reason": "short matrix incomplete"}
    )
    invariants = (
        audit_paired_manifest_invariants(paired_manifests, required_arms=EXTENSION_ARMS)
        if len(paired_manifests) == 32
        else {"paired_blocks": 0, "invariants_match": False, "reason": "short matrix incomplete"}
    )
    short_ok = len(short_audits) == 32 and all(audit["valid"] for audit in short_audits.values())
    passed = bool(
        completeness["complete"]
        and failure_chain_complete
        and short_ok
        and fairness == {"paired_blocks": 16, "fair": True}
        and invariants == {"paired_blocks": 16, "invariants_match": True}
    )
    return {
        "schema_version": 1,
        **({"runner_backend": "direct", "policy_lifetimes": sorted(policy_lifetimes)}
           if runner_backend(launch) == "direct" else {}),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "audited_utc": utc_now(),
        "passed": passed,
        "formal_started": False,
        "arms": list(EXTENSION_ARMS),
        "repository_commit_sha": repository_commit,
        "checkpoint_unpacked_metadata_sha256": launch["checkpoint_unpacked_metadata_sha256"],
        "checkpoint_content_tree_algorithm": launch["checkpoint_content_tree_algorithm"],
        "checkpoint_unpacked_content_tree_sha256": launch["checkpoint_unpacked_content_tree_sha256"],
        "completeness": completeness,
        "short_trajectories_respect_official_terminal_or_64_cap": short_ok,
        "short_trajectory_termination_audits": short_audits,
        "initial_condition_fairness": fairness,
        "paired_manifest_invariants": invariants,
        "seed_disjointness_audit": disjointness,
        "selector_diagnostics": diagnostics,
        "attempts": attempts,
        "failure_ledger_present": store.failures_path.exists(),
        "failure_count": len(failures),
        "failure_chain_complete": failure_chain_complete,
        "unconsumed_failure_count": unconsumed_failure_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "protocol_family": EXTENSION_PROTOCOL_FAMILY,
                    "arms": list(EXTENSION_ARMS),
                    "expected_trajectory_count": len(expected_smoke_keys()),
                    "submits_jobs": False,
                    "opens_formal_results": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.run_root is None:
        parser.error("--run-root is required unless --dry-run is used")
    report = build_smoke_audit(args.run_root)
    if not report["passed"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(2)
    output = args.output or args.run_root / "aggregate/smoke_audit.json"
    atomic_write_json(output, report)
    print(output)


if __name__ == "__main__":
    main()
