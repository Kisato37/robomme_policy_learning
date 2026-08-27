#!/usr/bin/env python3
"""Audit the exact 80-row development smoke before any protocol freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.keyframe_oracle_sampling.artifacts import (
    ArtifactContractError,
    EpisodeAttemptWriter,
    RunArtifactStore,
    ScientificKey,
    atomic_write_json,
    audit_initial_condition_fairness,
    audit_paired_manifest_invariants,
    audit_smoke_attempt,
    completeness_report,
    load_seed_table,
    sha256_file,
    utc_now,
    validate_architecture_pass_report,
    validate_smoke_formal_seed_disjointness,
)
from mme_vla_suite.shared.keyframe_oracle_sampling import (
    FORMAL_SEED_DATASET,
    FORMAL_SEED_SCOPE,
    SMOKE_SEED_DATASET,
    SMOKE_SEED_SCOPE,
)
from experiments.keyframe_oracle_sampling.smoke_matrix import expand_rows, load_frozen_matrix


SHORT_STEP_CAP = 64
OFFICIAL_SHORT_TERMINAL_REASONS = frozenset({"success", "fail", "timeout"})


def audit_short_trajectory_length(result: dict) -> dict:
    """Validate ``min(official terminal, 64)`` without inventing a terminal.

    A trajectory may end before the development cap only when the benchmark
    emitted a recognized official terminal reason.  Reaching the cap remains
    valid even when the terminal reason is the experiment-imposed timeout.
    """
    steps = result.get("steps")
    terminal_reason = result.get("terminal_reason")
    valid_steps = isinstance(steps, int) and not isinstance(steps, bool)
    if not valid_steps or steps < 1 or steps > SHORT_STEP_CAP:
        return {
            "valid": False,
            "steps": steps,
            "terminal_reason": terminal_reason,
            "completion_mode": "invalid_step_count",
        }
    if steps < SHORT_STEP_CAP:
        valid = terminal_reason in OFFICIAL_SHORT_TERMINAL_REASONS
        return {
            "valid": valid,
            "steps": steps,
            "terminal_reason": terminal_reason,
            "completion_mode": (
                "official_terminal_before_cap"
                if valid
                else "unrecognized_early_termination"
            ),
        }
    valid = terminal_reason in OFFICIAL_SHORT_TERMINAL_REASONS
    return {
        "valid": valid,
        "steps": steps,
        "terminal_reason": terminal_reason,
        "completion_mode": (
            "reached_64_step_cap" if valid else "unrecognized_termination_at_cap"
        ),
    }


def expected_smoke_keys() -> set[ScientificKey]:
    return {
        ScientificKey(
            row["task"],
            int(row["episode_id"]),
            row["arm"],
            row["trajectory_kind"],
        )
        for row in expand_rows(load_frozen_matrix())
    }


def build_smoke_audit(run_root: Path) -> dict:
    manifest_path = run_root / "protocol" / "launch_manifest.json"
    submission_path = run_root / "protocol" / "submission_record.json"
    if not manifest_path.is_file() or not submission_path.is_file():
        raise ArtifactContractError("Smoke audit requires launch and submission records")

    launch_manifest = json.loads(manifest_path.read_text())
    protocol_dir = run_root / "protocol"
    protocol_snapshot_path = protocol_dir / "protocol_snapshot.md"
    protocol_digest_path = protocol_dir / "protocol_sha256.txt"
    architecture_submission_path = (
        protocol_dir / "architecture_submission_record.json"
    )
    architecture_report_path = run_root / "architecture_smoke" / "report.json"
    smoke_seed_path = protocol_dir / "seed_table.json"
    formal_seed_path = protocol_dir / "formal_seed_audit_table.json"
    matrix_path = protocol_dir / "smoke_matrix.json"
    for path in (
        protocol_snapshot_path,
        protocol_digest_path,
        architecture_submission_path,
        architecture_report_path,
        smoke_seed_path,
        formal_seed_path,
        matrix_path,
    ):
        if not path.is_file():
            raise ArtifactContractError(f"Smoke audit prerequisite is missing: {path}")
    protocol_digest = sha256_file(protocol_snapshot_path)
    if protocol_digest_path.read_text().strip() != protocol_digest:
        raise ArtifactContractError("Protocol digest sidecar differs from the snapshot")
    if launch_manifest.get("protocol_sha256") != protocol_digest:
        raise ArtifactContractError("Launch manifest protocol digest mismatch")
    if launch_manifest.get("seed_table_file_sha256") != sha256_file(smoke_seed_path):
        raise ArtifactContractError("Smoke seed-table file digest mismatch")
    if launch_manifest.get("formal_seed_audit_file_sha256") != sha256_file(
        formal_seed_path
    ):
        raise ArtifactContractError("Formal seed-audit file digest mismatch")
    if launch_manifest.get("smoke_matrix_sha256") != sha256_file(matrix_path):
        raise ArtifactContractError("Smoke matrix file digest mismatch")
    smoke_seed_payload, smoke_seed_lookup = load_seed_table(
        smoke_seed_path,
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    formal_seed_payload, _ = load_seed_table(
        formal_seed_path,
        expected_scope=FORMAL_SEED_SCOPE,
        expected_dataset=FORMAL_SEED_DATASET,
    )
    launch_seed_fields = {
        "seed_table_scope": smoke_seed_payload["scope"],
        "seed_table_dataset": smoke_seed_payload["dataset"],
        "seed_table_derivation": smoke_seed_payload["derivation"],
        "seed_table_entries_sha256": smoke_seed_payload["entries_sha256"],
        "formal_seed_audit_scope": formal_seed_payload["scope"],
        "formal_seed_audit_dataset": formal_seed_payload["dataset"],
        "formal_seed_audit_derivation": formal_seed_payload["derivation"],
        "formal_seed_audit_entries_sha256": formal_seed_payload["entries_sha256"],
    }
    if any(
        launch_manifest.get(field) != expected
        for field, expected in launch_seed_fields.items()
    ):
        raise ArtifactContractError("Launch manifest seed-table provenance mismatch")
    seed_disjointness = validate_smoke_formal_seed_disjointness(
        smoke_seed_payload,
        formal_seed_payload,
    )
    if launch_manifest.get("seed_disjointness_audit") != seed_disjointness:
        raise ArtifactContractError("Launch manifest seed-disjointness audit mismatch")

    frozen_rows = expand_rows(load_frozen_matrix(matrix_path))
    expected_rows = {
        ScientificKey(
            row["task"],
            int(row["episode_id"]),
            row["arm"],
            row["trajectory_kind"],
        ): row
        for row in frozen_rows
    }
    if launch_manifest.get("matrix") != frozen_rows:
        raise ArtifactContractError("Launch manifest matrix differs from frozen smoke rows")

    architecture_submission = json.loads(architecture_submission_path.read_text())
    architecture_report = json.loads(architecture_report_path.read_text())
    validate_architecture_pass_report(
        architecture_report,
        run_root=run_root,
        architecture_submission=architecture_submission,
    )
    repository_commit = launch_manifest.get("repository", {}).get("commit_sha")
    if not isinstance(repository_commit, str) or len(repository_commit) != 40:
        raise ArtifactContractError("Launch manifest lacks a repository commit SHA")
    if architecture_report.get("repository_commit_sha") != repository_commit:
        raise ArtifactContractError("Architecture report used a different commit")
    for field in (
        "checkpoint_unpacked_metadata_sha256",
        "checkpoint_content_tree_algorithm",
        "checkpoint_unpacked_content_tree_sha256",
    ):
        if architecture_report.get(field) != launch_manifest.get(field):
            raise ArtifactContractError(
                f"Architecture report/launch manifest mismatch for {field}"
            )

    def validate_submission_record(path: Path, *, expected_attempt_id: int) -> dict:
        if not path.is_file():
            raise ArtifactContractError(f"Missing smoke submission record: {path}")
        submission = json.loads(path.read_text())
        if submission.get("repository_commit_sha") != repository_commit:
            raise ArtifactContractError("Smoke submission used a different commit")
        if Path(str(submission.get("run_root", ""))).resolve() != run_root.resolve():
            raise ArtifactContractError("Smoke submission used a different run root")
        if submission.get("attempt_id") != expected_attempt_id:
            raise ArtifactContractError("Smoke submission attempt ID mismatch")
        row_ids = submission.get("row_ids")
        if (
            not isinstance(row_ids, list)
            or any(type(row_id) is not int for row_id in row_ids)
            or len(row_ids) != len(set(row_ids))
            or any(row_id < 0 or row_id >= len(frozen_rows) for row_id in row_ids)
        ):
            raise ArtifactContractError("Smoke submission has invalid row IDs")
        if submission.get("trajectory_count") != len(row_ids):
            raise ArtifactContractError("Smoke submission row count mismatch")
        if not isinstance(submission.get("slurm_array_job_id"), str) or not submission[
            "slurm_array_job_id"
        ]:
            raise ArtifactContractError("Smoke submission lacks a Slurm array job ID")
        if submission.get("architecture_report_sha256") != sha256_file(
            architecture_report_path
        ):
            raise ArtifactContractError("Architecture report changed after smoke submission")
        for field in (
            "checkpoint_unpacked_metadata_sha256",
            "checkpoint_content_tree_algorithm",
            "checkpoint_unpacked_content_tree_sha256",
        ):
            if submission.get(field) != launch_manifest.get(field):
                raise ArtifactContractError(
                    f"Smoke submission/launch manifest mismatch for {field}"
                )
        return submission

    initial_submission = validate_submission_record(
        submission_path,
        expected_attempt_id=0,
    )
    if initial_submission["row_ids"] != list(range(len(frozen_rows))):
        raise ArtifactContractError(
            "Initial smoke submission must contain the complete frozen matrix"
        )

    store = RunArtifactStore(run_root)
    completed = store.scan_completed_keys()
    expected = set(expected_rows)
    completeness = completeness_report(expected, completed)

    attempts = {}
    paired_manifests = []
    short_termination_audits = {}
    row_id_by_key = {key: row_id for row_id, key in enumerate(expected_rows)}
    validated_retry_submissions: dict[int, dict] = {}
    for key, result_path in sorted(completed.items()):
        if key not in expected_rows:
            raise ArtifactContractError(f"Unexpected completed smoke key: {key}")
        result = json.loads(result_path.read_text())
        attempt_id = int(result["attempt_id"])
        if attempt_id > 0:
            retry_submission = validated_retry_submissions.get(attempt_id)
            if retry_submission is None:
                retry_submission = validate_submission_record(
                    protocol_dir / f"submission_record_attempt_{attempt_id:02d}.json",
                    expected_attempt_id=attempt_id,
                )
                validated_retry_submissions[attempt_id] = retry_submission
            if row_id_by_key[key] not in retry_submission["row_ids"]:
                raise ArtifactContractError(
                    f"Completed retry was not authorized for smoke row {row_id_by_key[key]}"
                )
        writer = EpisodeAttemptWriter(result_path.parent, key, attempt_id)
        attempt_report = audit_smoke_attempt(
            writer,
            expected_key=key,
            expected_row=expected_rows[key],
            launch_manifest=launch_manifest,
            smoke_seed_payload=smoke_seed_payload,
            smoke_seed_lookup=smoke_seed_lookup,
        )
        attempts[json.dumps(key.as_dict(), sort_keys=True)] = attempt_report

        if key.trajectory_kind == "short":
            short_termination_audits[
                json.dumps(key.as_dict(), sort_keys=True)
            ] = audit_short_trajectory_length(result)
            manifest = json.loads(writer.manifest_path.read_text())
            manifest["initial_condition_hashes"] = json.loads(
                writer.initial_conditions_path.read_text()
            )
            paired_manifests.append(manifest)

    fairness = (
        audit_initial_condition_fairness(paired_manifests)
        if len(paired_manifests) == 64
        else {"paired_blocks": 0, "fair": False, "reason": "short matrix incomplete"}
    )
    paired_invariants = (
        audit_paired_manifest_invariants(paired_manifests)
        if len(paired_manifests) == 64
        else {
            "paired_blocks": 0,
            "invariants_match": False,
            "reason": "short matrix incomplete",
        }
    )
    short_termination_ok = bool(
        len(short_termination_audits) == 64
        and all(item["valid"] for item in short_termination_audits.values())
    )
    passed = bool(
        completeness["complete"]
        and short_termination_ok
        and fairness["fair"]
        and paired_invariants["invariants_match"]
    )
    return {
        "schema_version": 1,
        "audited_utc": utc_now(),
        "passed": passed,
        "formal_started": False,
        "completeness": completeness,
        "short_trajectories_respect_official_terminal_or_64_cap": (
            short_termination_ok
        ),
        "short_trajectory_termination_audits": short_termination_audits,
        "initial_condition_fairness": fairness,
        "paired_manifest_invariants": paired_invariants,
        "seed_disjointness_audit": seed_disjointness,
        "attempts": attempts,
        "failure_ledger_present": store.failures_path.exists(),
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
                    "expected_trajectory_count": len(expected_smoke_keys()),
                    "submits_jobs": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.run_root is None:
        parser.error("--run-root is required unless --dry-run is used")

    report = build_smoke_audit(args.run_root.resolve())
    if not report["passed"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(2)
    output = args.output or args.run_root / "aggregate" / "smoke_audit.json"
    atomic_write_json(output, report)
    print(output)


if __name__ == "__main__":
    main()
