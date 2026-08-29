"""Fail-closed validation for prepared and submitted formal run roots."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from experiments.keyframe_oracle_sampling.analysis import normalize_prior_exposure_manifest
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import load_seed_table
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import validate_architecture_pass_report
from experiments.keyframe_oracle_sampling.artifacts import validate_smoke_formal_seed_disjointness
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_DATASET
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_EPISODE_IDS
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_oracle_sampling.formal_matrix import load_formal_matrix
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_RELATIVE
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_POLICY_CALLS
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_SCOPE

ACCEPTED_DEVELOPMENT_SMOKE_COMMIT = "899912b11a379346b4c8f4d6c80f54f07c118ea3"
ACCEPTED_DEVELOPMENT_SMOKE_AUDIT_SHA256 = "fc1b7c95d0c4b7d7e5353e31a27f31dbef80a6e26c1872b6e5fba6f768d9729e"


def submission_path(run_root: str | Path, attempt_id: int) -> Path:
    if attempt_id not in {0, 1, 2}:
        raise ValueError("attempt_id must be 0, 1, or 2")
    name = "submission_record.json" if attempt_id == 0 else f"submission_record_attempt_{attempt_id:02d}.json"
    return Path(run_root) / "protocol" / name


def _require_exact_development_smoke(audit: dict[str, Any], audit_path: Path) -> None:
    if sha256_file(audit_path) != ACCEPTED_DEVELOPMENT_SMOKE_AUDIT_SHA256:
        raise ArtifactContractError("Development-smoke audit is not the immutable pre-formal PASS evidence")
    expected = {
        "passed": True,
        "formal_started": False,
        "failure_ledger_present": False,
        "short_trajectories_respect_official_terminal_or_64_cap": True,
        "completeness": {
            "expected_count": 80,
            "completed_count": 80,
            "complete": True,
            "missing": [],
            "unexpected": [],
        },
        "initial_condition_fairness": {"paired_blocks": 16, "fair": True},
        "paired_manifest_invariants": {
            "paired_blocks": 16,
            "invariants_match": True,
        },
        "seed_disjointness_audit": {
            "smoke_seed_count": 1312,
            "formal_seed_count": 65600,
            "intersection_count": 0,
        },
    }
    mismatches = {
        field: {"expected": value, "observed": audit.get(field)}
        for field, value in expected.items()
        if audit.get(field) != value
    }
    if mismatches:
        raise ArtifactContractError(
            "Development-smoke PASS evidence changed: " + json.dumps(mismatches, sort_keys=True)
        )


def validate_formal_prepared_root(
    run_root: str | Path,
    seed_table_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate all immutable pre-submission formal artifacts and live code SHA."""
    run_root = Path(run_root).resolve()
    repo_root = Path(repo_root).resolve()
    protocol_dir = run_root / "protocol"
    paths = {
        "manifest": protocol_dir / "launch_manifest.json",
        "protocol": protocol_dir / "protocol_snapshot.md",
        "seed": protocol_dir / "seed_table.json",
        "development_seed": protocol_dir / "development_seed_audit_table.json",
        "matrix": protocol_dir / "formal_matrix.json",
        "exposure": protocol_dir / "prior_exposure_manifest.json",
        "architecture_report": protocol_dir / "architecture_pass_report.json",
        "architecture_submission": (protocol_dir / "architecture_submission_record.json"),
        "development_smoke": protocol_dir / "development_smoke_audit.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise ArtifactContractError(f"Prepared formal artifact is missing: {path}")
    if Path(seed_table_path).resolve() != paths["seed"].resolve():
        raise ArtifactContractError("Formal evaluator seed table is not the prepared run-root table")

    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ArtifactContractError("Prepared formal protocol version mismatch")
    if manifest.get("run_kind") != "formal" or manifest.get("dataset") != FORMAL_DATASET:
        raise ArtifactContractError("Prepared root is not a formal test run")
    if manifest.get("formal_launch_authorized") is not True:
        raise ArtifactContractError("Prepared formal root lacks explicit launch authorization")
    if int(manifest.get("trajectory_count", -1)) != FORMAL_TRAJECTORY_COUNT:
        raise ArtifactContractError("Prepared formal trajectory count is not exactly 3,200")

    digest_pairs = (
        ("protocol_sha256", paths["protocol"]),
        ("seed_table_file_sha256", paths["seed"]),
        ("development_seed_audit_file_sha256", paths["development_seed"]),
        ("formal_matrix_sha256", paths["matrix"]),
        ("prior_exposure_manifest_sha256", paths["exposure"]),
        ("architecture_report_sha256", paths["architecture_report"]),
        (
            "architecture_submission_record_sha256",
            paths["architecture_submission"],
        ),
        ("development_smoke_audit_sha256", paths["development_smoke"]),
    )
    for field, path in digest_pairs:
        if manifest.get(field) != sha256_file(path):
            raise ArtifactContractError(f"Prepared formal digest mismatch for {path.name}")

    formal_payload, formal_lookup = load_seed_table(
        paths["seed"],
        expected_scope=FORMAL_SEED_SCOPE,
        expected_dataset=FORMAL_SEED_DATASET,
    )
    expected_formal_keys = {
        (task, episode_id, call_index)
        for task in FORMAL_TASKS
        for episode_id in FORMAL_EPISODE_IDS
        for call_index in range(MAX_POLICY_CALLS)
    }
    if set(formal_lookup) != expected_formal_keys:
        raise ArtifactContractError("Formal seed table is not the exact 16 x 50 x 82 universe")
    development_payload, development_lookup = load_seed_table(
        paths["development_seed"],
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    expected_development_keys = {
        (task, 0, call_index) for task in FORMAL_TASKS for call_index in range(MAX_POLICY_CALLS)
    }
    if set(development_lookup) != expected_development_keys:
        raise ArtifactContractError("Development seed audit is not the exact 16 x 1 x 82 universe")
    disjointness = validate_smoke_formal_seed_disjointness(development_payload, formal_payload)
    if manifest.get("seed_disjointness_audit") != disjointness:
        raise ArtifactContractError("Formal manifest seed-disjointness audit mismatch")
    seed_contract = {
        "seed_table_scope": formal_payload["scope"],
        "seed_table_dataset": formal_payload["dataset"],
        "seed_table_derivation": formal_payload["derivation"],
        "seed_table_entries_sha256": formal_payload["entries_sha256"],
    }
    if any(manifest.get(field) != value for field, value in seed_contract.items()):
        raise ArtifactContractError("Formal manifest and seed-table contract differ")

    matrix = load_formal_matrix(paths["matrix"])
    if manifest.get("matrix") != matrix["rows"]:
        raise ArtifactContractError("Launch manifest does not record the exact formal matrix")

    exposure_payload = json.loads(paths["exposure"].read_text())
    exposure = normalize_prior_exposure_manifest(exposure_payload)
    if exposure_payload.get("frozen_before_formal_execution") is not True:
        raise ArtifactContractError("Prior-exposure manifest was not frozen before formal")
    if exposure_payload.get("present_formal_outcomes_inspected") is not False:
        raise ArtifactContractError("Prior-exposure manifest indicates formal outcome leakage")
    if manifest.get("prior_exposure_entry_count") != len(exposure.entries):
        raise ArtifactContractError("Prior-exposure manifest entry count mismatch")
    if manifest.get("prior_exposure_test_block_count") != len(exposure.excluded_formal_blocks):
        raise ArtifactContractError("Prior-exposure formal-block count mismatch")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    try:
        relative_root = run_root.relative_to(repo_root)
    except ValueError as exc:
        raise ArtifactContractError("Formal run root must be inside the repository") from exc
    runtime_root = Path("runs/keyframe_oracle_sampling")
    if relative_root.parent != runtime_root:
        raise ArtifactContractError("Formal run root must be one direct child of runs/keyframe_oracle_sampling")
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--",
            ".",
            f":(exclude){runtime_root.as_posix()}",
            f":(exclude){CHECKPOINT_RELATIVE.as_posix()}",
        ],
        cwd=repo_root,
        text=True,
    )
    if status:
        raise ArtifactContractError("Formal evaluation requires a clean committed worktree")
    if manifest.get("repository", {}).get("commit_sha") != commit:
        raise ArtifactContractError("Live repository commit differs from prepared formal commit")

    architecture_report = json.loads(paths["architecture_report"].read_text())
    architecture_submission = json.loads(paths["architecture_submission"].read_text())
    architecture_source_root = manifest.get("architecture_source_run_root")
    if not isinstance(architecture_source_root, str) or not architecture_source_root:
        raise ArtifactContractError("Formal manifest lacks architecture source run root")
    validate_architecture_pass_report(
        architecture_report,
        run_root=architecture_source_root,
        architecture_submission=architecture_submission,
    )
    if architecture_report.get("repository_commit_sha") != commit:
        raise ArtifactContractError("Architecture smoke used a different formal-launch commit")
    for field in (
        "checkpoint_unpacked_metadata_sha256",
        "checkpoint_content_tree_algorithm",
        "checkpoint_unpacked_content_tree_sha256",
    ):
        if architecture_report.get(field) != manifest.get(field):
            raise ArtifactContractError(f"Architecture smoke and formal manifest differ on {field}")

    development_smoke = json.loads(paths["development_smoke"].read_text())
    _require_exact_development_smoke(development_smoke, paths["development_smoke"])
    if manifest.get("development_smoke_commit_sha") != ACCEPTED_DEVELOPMENT_SMOKE_COMMIT:
        raise ArtifactContractError("Formal manifest names the wrong development-smoke commit")
    return manifest


def validate_prepared_formal_root(
    run_root: str | Path,
    seed_table_path: str | Path,
    repo_root: str | Path,
    *,
    attempt_id: int,
    authorization_digest: str,
) -> dict[str, Any]:
    """Reject direct formal execution outside its exact recorded Slurm array."""
    manifest = validate_formal_prepared_root(run_root, seed_table_path, repo_root)
    run_root = Path(run_root).resolve()
    record_path = submission_path(run_root, attempt_id)
    if not record_path.is_file():
        raise ArtifactContractError(f"Formal submission record is missing: {record_path}")
    observed_digest = sha256_file(record_path)
    if authorization_digest != observed_digest:
        raise ArtifactContractError("Formal authorization digest does not match the recorded submission")
    submission = json.loads(record_path.read_text())
    commit = str(manifest["repository"]["commit_sha"])
    if submission.get("repository_commit_sha") != commit:
        raise ArtifactContractError("Formal submission used a different commit")
    if int(submission.get("attempt_id", -1)) != attempt_id:
        raise ArtifactContractError("Formal submission attempt ID mismatch")
    if submission.get("formal_launch_authorized") is not True:
        raise ArtifactContractError("Formal submission lacks the authorization gate")
    if submission.get("launch_manifest_sha256") != sha256_file(run_root / "protocol" / "launch_manifest.json"):
        raise ArtifactContractError("Formal launch manifest changed after submission")
    if submission.get("formal_matrix_sha256") != manifest.get("formal_matrix_sha256"):
        raise ArtifactContractError("Formal submission used a different matrix")

    active_array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    if not active_array_job or submission.get("slurm_array_job_id") != active_array_job:
        raise ArtifactContractError("Formal evaluator is not running inside the recorded Slurm array")
    active_array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    try:
        active_row_id = int(active_array_task) if active_array_task is not None else -1
    except ValueError as exc:
        raise ArtifactContractError("SLURM_ARRAY_TASK_ID is not an integer") from exc
    raw_row_ids = submission.get("row_ids")
    if not isinstance(raw_row_ids, list) or any(
        isinstance(row_id, bool) or not isinstance(row_id, int) for row_id in raw_row_ids
    ):
        raise ArtifactContractError("Formal submission row_ids must be an integer list")
    if len(raw_row_ids) != len(set(raw_row_ids)):
        raise ArtifactContractError("Formal submission row_ids contain duplicates")
    if int(submission.get("trajectory_count", -1)) != len(raw_row_ids):
        raise ArtifactContractError("Formal submission trajectory_count/row_ids mismatch")
    if active_row_id not in raw_row_ids:
        raise ArtifactContractError("Current SLURM_ARRAY_TASK_ID is not authorized by the formal submission")
    return manifest
