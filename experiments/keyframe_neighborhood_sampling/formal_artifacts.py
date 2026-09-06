"""Fail-closed artifact validation for the OC3/OC5 formal extension.

This module is deliberately separate from the completed U/O/OC/R experiment.
It reuses the parent's deterministic formal seed contract, but it accepts only
the 1,600-row neighborhood-extension matrix and a new write-once run root.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_DATASET
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_EPISODE_IDS
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.formal_matrix import load_formal_matrix
from experiments.keyframe_neighborhood_sampling.smoke_matrix import SMOKE_DATASET
from experiments.keyframe_neighborhood_sampling.smoke_matrix import SMOKE_EPISODE_ID
from experiments.keyframe_neighborhood_sampling.smoke_matrix import SMOKE_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.smoke_matrix import load_smoke_matrix
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import load_seed_table
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import validate_smoke_formal_seed_disjointness
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_RELATIVE
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_POLICY_CALLS
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_SCOPE

EXTENSION_PROTOCOL_VERSION = "v1.0"
EXTENSION_RUNTIME_ROOT = Path("runs/keyframe_neighborhood_sampling")
MAX_ROWS_PER_ARRAY = 1000
ANALYSIS_SOURCE_RELATIVE = Path("experiments/keyframe_neighborhood_sampling/analysis.py")
AGGREGATOR_SOURCE_RELATIVE = Path("experiments/keyframe_neighborhood_sampling/aggregate_formal.py")
REFERENCE_PER_EPISODE_SHA256 = "e1ec5d1273f3ba97b14d252a1a1668c00635a71477acca687f3e47d7e9f3a478"
REFERENCE_SUMMARY_SHA256 = "e3f40b7192fc1be9a3b280c30234170b63dfae26d389c9b99ba321e82e3efbe9"
REFERENCE_COMPLETENESS_SHA256 = "9f58713cb8da6762c3a81fcfa1776c25bb52a1be75ed26326325ce0cec655ccc"
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")

# Smoke semantics are validated by the dedicated smoke stage.  The formal
# runtime gate nevertheless binds the exact evidence bytes that the launcher
# reviewed, so they cannot be replaced between preparation and execution.
REQUIRED_PREPARED_ARTIFACTS = {
    "manifest": "launch_manifest.json",
    "protocol": "protocol_snapshot.md",
    "seed": "seed_table.json",
    "development_seed": "development_seed_audit_table.json",
    "matrix": "formal_matrix.json",
    "architecture_report": "architecture_pass_report.json",
    "architecture_submission": "architecture_submission_record.json",
    "development_smoke": "development_smoke_audit.json",
}
REQUIRED_DIGEST_FIELDS = (
    ("protocol_sha256", "protocol"),
    ("seed_table_file_sha256", "seed"),
    ("development_seed_audit_file_sha256", "development_seed"),
    ("formal_matrix_sha256", "matrix"),
    ("architecture_report_sha256", "architecture_report"),
    ("architecture_submission_record_sha256", "architecture_submission"),
    ("development_smoke_audit_sha256", "development_smoke"),
)
SMOKE_REQUIRED_PREPARED_ARTIFACTS = {
    "manifest": "launch_manifest.json",
    "protocol": "protocol_snapshot.md",
    "seed": "seed_table.json",
    "formal_seed": "formal_seed_audit_table.json",
    "matrix": "smoke_matrix.json",
}
SMOKE_REQUIRED_DIGEST_FIELDS = (
    ("protocol_sha256", "protocol"),
    ("seed_table_file_sha256", "seed"),
    ("formal_seed_audit_file_sha256", "formal_seed"),
    ("smoke_matrix_sha256", "matrix"),
)


def submission_plan_path(run_root: str | Path, attempt_id: int) -> Path:
    if attempt_id not in {0, 1, 2}:
        raise ValueError("attempt_id must be 0, 1, or 2")
    name = "submission_plan.json" if attempt_id == 0 else f"submission_plan_attempt_{attempt_id:02d}.json"
    return Path(run_root) / "protocol" / name


def submission_path(run_root: str | Path, attempt_id: int, shard_id: int) -> Path:
    if attempt_id not in {0, 1, 2}:
        raise ValueError("attempt_id must be 0, 1, or 2")
    if shard_id < 0:
        raise ValueError("shard_id must be non-negative")
    name = (
        f"submission_record_shard_{shard_id:02d}.json"
        if attempt_id == 0
        else f"submission_record_attempt_{attempt_id:02d}_shard_{shard_id:02d}.json"
    )
    return Path(run_root) / "protocol" / name


def submission_paths(run_root: str | Path, attempt_id: int) -> list[Path]:
    protocol_dir = Path(run_root) / "protocol"
    pattern = (
        "submission_record_shard_*.json"
        if attempt_id == 0
        else f"submission_record_attempt_{attempt_id:02d}_shard_*.json"
    )
    return sorted(protocol_dir.glob(pattern))


def smoke_submission_path(run_root: str | Path, attempt_id: int) -> Path:
    if attempt_id not in {0, 1, 2}:
        raise ValueError("attempt_id must be 0, 1, or 2")
    name = "submission_record.json" if attempt_id == 0 else f"submission_record_attempt_{attempt_id:02d}.json"
    return Path(run_root) / "protocol" / name


def _load_json_object(path: Path, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Invalid {source}: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError(f"{source} must be a JSON object: {path}")
    return payload


def _require_extension_identity(payload: Mapping[str, Any], *, source: str) -> None:
    if payload.get("protocol_version") != EXTENSION_PROTOCOL_VERSION:
        raise ArtifactContractError(f"{source} protocol version mismatch")
    if payload.get("protocol_family") != EXTENSION_PROTOCOL_FAMILY:
        raise ArtifactContractError(f"{source} protocol family mismatch")


def _validate_extension_run_root(
    run_root: Path,
    repo_root: Path,
    *,
    source: str,
) -> str:
    try:
        relative_root = run_root.relative_to(repo_root)
    except ValueError as exc:
        raise ArtifactContractError(f"{source} run root must be inside the repository") from exc
    if relative_root.parent != EXTENSION_RUNTIME_ROOT:
        raise ArtifactContractError(
            f"{source} run root must be one direct child of runs/keyframe_neighborhood_sampling"
        )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--",
            ".",
            f":(exclude){EXTENSION_RUNTIME_ROOT.as_posix()}",
            f":(exclude){CHECKPOINT_RELATIVE.as_posix()}",
        ],
        cwd=repo_root,
        text=True,
    )
    if status:
        raise ArtifactContractError(f"{source} requires a clean committed worktree")
    return commit


def validate_frozen_formal_sources(
    manifest: Mapping[str, Any],
    repo_root: str | Path,
    *,
    source: str,
) -> dict[str, str]:
    """Bind formal execution/analysis to the exact reviewed analysis source bytes."""
    repo_root = Path(repo_root).resolve()
    expected = {
        "analysis": (
            "frozen_analysis_source_relative",
            "frozen_analysis_source_sha256",
            ANALYSIS_SOURCE_RELATIVE,
        ),
        "aggregator": (
            "frozen_aggregator_source_relative",
            "frozen_aggregator_source_sha256",
            AGGREGATOR_SOURCE_RELATIVE,
        ),
    }
    observed: dict[str, str] = {}
    for label, (path_field, digest_field, expected_relative) in expected.items():
        if manifest.get(path_field) != expected_relative.as_posix():
            raise ArtifactContractError(f"{source} has the wrong frozen {label} source path")
        recorded_digest = manifest.get(digest_field)
        if (
            not isinstance(recorded_digest, str)
            or len(recorded_digest) != 64
            or any(character not in _SHA256_HEX_DIGITS for character in recorded_digest)
        ):
            raise ArtifactContractError(f"{source} lacks a valid frozen {label} source digest")
        live_digest = sha256_file(repo_root / expected_relative)
        if recorded_digest != live_digest:
            raise ArtifactContractError(f"{source} frozen {label} source digest mismatch")
        observed[digest_field] = live_digest
    return observed


def validate_prepared_smoke_root(
    run_root: str | Path,
    seed_table_path: str | Path,
    repo_root: str | Path,
    *,
    attempt_id: int,
) -> dict[str, Any]:
    """Validate and authorize one OC3/OC5 development-smoke invocation."""
    run_root = Path(run_root).resolve()
    repo_root = Path(repo_root).resolve()
    protocol_dir = run_root / "protocol"
    paths = {key: protocol_dir / name for key, name in SMOKE_REQUIRED_PREPARED_ARTIFACTS.items()}
    submission = smoke_submission_path(run_root, attempt_id)
    for path in (*paths.values(), submission):
        if not path.is_file():
            raise ArtifactContractError(f"Prepared extension smoke artifact is missing: {path}")
    if Path(seed_table_path).resolve() != paths["seed"].resolve():
        raise ArtifactContractError("Extension smoke evaluator seed table is not the prepared run-root table")

    manifest = _load_json_object(paths["manifest"], source="extension smoke manifest")
    _require_extension_identity(manifest, source="Prepared extension smoke")
    if manifest.get("run_kind") != "development_smoke":
        raise ArtifactContractError("Prepared extension root is not a development smoke")
    if manifest.get("dataset") != SMOKE_DATASET:
        raise ArtifactContractError("Prepared extension smoke does not use val")
    if int(manifest.get("trajectory_count", -1)) != SMOKE_TRAJECTORY_COUNT:
        raise ArtifactContractError("Prepared extension smoke trajectory count is not exactly 48")
    if manifest.get("formal_launch_authorized") is not False:
        raise ArtifactContractError("Extension smoke manifest must explicitly forbid formal launch")
    for field, path_key in SMOKE_REQUIRED_DIGEST_FIELDS:
        path = paths[path_key]
        if manifest.get(field) != sha256_file(path):
            raise ArtifactContractError(f"Prepared extension smoke digest mismatch for {path.name}")

    smoke_payload, smoke_lookup = load_seed_table(
        paths["seed"],
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    expected_smoke_keys = {
        (task, SMOKE_EPISODE_ID, call_index) for task in FORMAL_TASKS for call_index in range(MAX_POLICY_CALLS)
    }
    if set(smoke_lookup) != expected_smoke_keys:
        raise ArtifactContractError("Extension smoke seed table is not the exact 16 x 1 x 82 universe")
    formal_payload, formal_lookup = load_seed_table(
        paths["formal_seed"],
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
        raise ArtifactContractError("Extension formal seed audit is not the exact 16 x 50 x 82 universe")
    disjointness = validate_smoke_formal_seed_disjointness(smoke_payload, formal_payload)
    seed_contract = {
        "seed_table_scope": smoke_payload["scope"],
        "seed_table_dataset": smoke_payload["dataset"],
        "seed_table_derivation": smoke_payload["derivation"],
        "seed_table_entries_sha256": smoke_payload["entries_sha256"],
        "seed_disjointness_audit": disjointness,
    }
    if any(manifest.get(field) != value for field, value in seed_contract.items()):
        raise ArtifactContractError("Extension smoke manifest and seed-table contract differ")

    matrix = load_smoke_matrix(paths["matrix"])
    if manifest.get("matrix") != matrix["rows"]:
        raise ArtifactContractError("Extension smoke manifest does not record the exact 48-row matrix")

    commit = _validate_extension_run_root(run_root, repo_root, source="Extension smoke")
    if manifest.get("repository", {}).get("commit_sha") != commit:
        raise ArtifactContractError("Live repository commit differs from prepared extension smoke commit")

    submission_payload = _load_json_object(submission, source="extension smoke submission")
    _require_extension_identity(submission_payload, source="Extension smoke submission")
    if submission_payload.get("repository_commit_sha") != commit:
        raise ArtifactContractError("Extension smoke submission used a different commit")
    if int(submission_payload.get("attempt_id", -1)) != attempt_id:
        raise ArtifactContractError("Extension smoke submission attempt ID mismatch")
    if submission_payload.get("smoke_launch_authorized") is not True:
        raise ArtifactContractError("Extension smoke submission lacks authorization")
    if submission_payload.get("formal_launch_authorized") is not False:
        raise ArtifactContractError("Extension smoke submission must explicitly forbid formal launch")
    if submission_payload.get("launch_manifest_sha256") != sha256_file(paths["manifest"]):
        raise ArtifactContractError("Extension smoke manifest changed after submission")
    if submission_payload.get("smoke_matrix_sha256") != manifest.get("smoke_matrix_sha256"):
        raise ArtifactContractError("Extension smoke submission used a different matrix")
    row_ids = _require_integer_list(submission_payload, "row_ids")
    if any(row_id < 0 or row_id >= SMOKE_TRAJECTORY_COUNT for row_id in row_ids):
        raise ArtifactContractError("Extension smoke submission has an out-of-range row")
    if int(submission_payload.get("trajectory_count", -1)) != len(row_ids):
        raise ArtifactContractError("Extension smoke submission trajectory_count/row_ids mismatch")
    if attempt_id == 0 and row_ids != list(range(SMOKE_TRAJECTORY_COUNT)):
        raise ArtifactContractError("Initial extension smoke submission is not the exact 48-row matrix")
    active_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    if not active_job or submission_payload.get("slurm_array_job_id") != active_job:
        raise ArtifactContractError("Extension smoke evaluator is not inside the recorded Slurm array")
    active_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    try:
        active_row = int(active_task) if active_task is not None else -1
    except ValueError as exc:
        raise ArtifactContractError("SLURM_ARRAY_TASK_ID is not an integer") from exc
    if active_row not in row_ids:
        raise ArtifactContractError("Current SLURM_ARRAY_TASK_ID is not authorized by the extension smoke submission")
    return manifest


def _validate_bound_smoke_evidence(
    *,
    paths: Mapping[str, Path],
    manifest: Mapping[str, Any],
    commit: str,
) -> None:
    """Require fresh extension PASS markers without duplicating smoke schemas."""
    architecture = _load_json_object(paths["architecture_report"], source="extension architecture report")
    architecture_submission = _load_json_object(
        paths["architecture_submission"], source="extension architecture submission"
    )
    development = _load_json_object(paths["development_smoke"], source="extension development-smoke audit")
    for source, payload in (
        ("Extension architecture report", architecture),
        ("Extension architecture submission", architecture_submission),
        ("Extension development-smoke audit", development),
    ):
        if payload.get("protocol_family") != EXTENSION_PROTOCOL_FAMILY:
            raise ArtifactContractError(f"{source} protocol family mismatch")
    if architecture.get("passed") is not True:
        raise ArtifactContractError("Extension architecture smoke is not a recorded PASS")
    if development.get("passed") is not True:
        raise ArtifactContractError("Extension development smoke is not a recorded PASS")
    if development.get("formal_started") is not False:
        raise ArtifactContractError("Extension development-smoke evidence was not frozen before formal execution")
    for source, payload in (
        ("architecture report", architecture),
        ("architecture submission", architecture_submission),
        ("development-smoke audit", development),
    ):
        recorded_commit = payload.get("repository_commit_sha")
        if recorded_commit is not None and recorded_commit != commit:
            raise ArtifactContractError(f"Extension {source} used a different repository commit")
    for field in (
        "checkpoint_unpacked_metadata_sha256",
        "checkpoint_content_tree_algorithm",
        "checkpoint_unpacked_content_tree_sha256",
    ):
        expected = manifest.get(field)
        if expected is None:
            raise ArtifactContractError(f"Extension formal manifest lacks {field}")
        for source, payload in (
            ("architecture report", architecture),
            ("architecture submission", architecture_submission),
        ):
            observed = payload.get(field)
            if observed is not None and observed != expected:
                raise ArtifactContractError(f"Extension {source} and formal manifest differ on {field}")


def validate_formal_prepared_root(
    run_root: str | Path,
    seed_table_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate the immutable 1,600-cell extension root before submission."""
    run_root = Path(run_root).resolve()
    repo_root = Path(repo_root).resolve()
    protocol_dir = run_root / "protocol"
    paths = {key: protocol_dir / name for key, name in REQUIRED_PREPARED_ARTIFACTS.items()}
    for path in paths.values():
        if not path.is_file():
            raise ArtifactContractError(f"Prepared extension formal artifact is missing: {path}")
    if Path(seed_table_path).resolve() != paths["seed"].resolve():
        raise ArtifactContractError("Extension evaluator seed table is not the prepared run-root table")

    manifest = _load_json_object(paths["manifest"], source="extension launch manifest")
    _require_extension_identity(manifest, source="Prepared extension formal")
    if manifest.get("run_kind") != "formal" or manifest.get("dataset") != FORMAL_DATASET:
        raise ArtifactContractError("Prepared extension root is not a formal test run")
    if manifest.get("formal_launch_authorized") is not True:
        raise ArtifactContractError("Prepared extension formal root lacks explicit launch authorization")
    if int(manifest.get("trajectory_count", -1)) != FORMAL_TRAJECTORY_COUNT:
        raise ArtifactContractError("Prepared extension formal trajectory count is not exactly 1,600")
    if manifest.get("selector_seed_table_role") != "randomsamp_policy_call_rng_audit_only":
        raise ArtifactContractError(
            "Prepared extension formal seed table must be identified as RandomSamp policy-call RNG evidence"
        )
    expected_reference_digests = {
        "reference_per_episode_sha256": REFERENCE_PER_EPISODE_SHA256,
        "reference_summary_sha256": REFERENCE_SUMMARY_SHA256,
        "reference_completeness_sha256": REFERENCE_COMPLETENESS_SHA256,
    }
    if any(manifest.get(field) != digest for field, digest in expected_reference_digests.items()):
        raise ArtifactContractError("Prepared extension formal OC reference digest binding mismatch")
    cross_run = manifest.get("reference_cross_run_verification")
    if not isinstance(cross_run, Mapping) or cross_run.get("pairing_key") != ["task", "episode_id"]:
        raise ArtifactContractError("Prepared extension formal OC reference pairing contract is missing")
    unverifiable = cross_run.get("not_directly_verifiable_from_published_reference")
    if unverifiable != ["environment_seed", "difficulty", "raw_initial_condition_hashes"]:
        raise ArtifactContractError(
            "Prepared extension formal must disclose unavailable cross-run initial-condition evidence"
        )

    for field, path_key in REQUIRED_DIGEST_FIELDS:
        path = paths[path_key]
        if manifest.get(field) != sha256_file(path):
            raise ArtifactContractError(f"Prepared extension formal digest mismatch for {path.name}")

    seed_payload, seed_lookup = load_seed_table(
        paths["seed"],
        expected_scope=FORMAL_SEED_SCOPE,
        expected_dataset=FORMAL_SEED_DATASET,
    )
    expected_seed_keys = {
        (task, episode_id, call_index)
        for task in FORMAL_TASKS
        for episode_id in FORMAL_EPISODE_IDS
        for call_index in range(MAX_POLICY_CALLS)
    }
    if set(seed_lookup) != expected_seed_keys:
        raise ArtifactContractError("Extension formal seed table is not the exact 16 x 50 x 82 universe")
    seed_contract = {
        "seed_table_scope": seed_payload["scope"],
        "seed_table_dataset": seed_payload["dataset"],
        "seed_table_derivation": seed_payload["derivation"],
        "seed_table_entries_sha256": seed_payload["entries_sha256"],
    }
    if any(manifest.get(field) != value for field, value in seed_contract.items()):
        raise ArtifactContractError("Extension formal manifest and parent formal seed contract differ")

    matrix = load_formal_matrix(paths["matrix"])
    if manifest.get("matrix") != matrix["rows"]:
        raise ArtifactContractError("Extension launch manifest does not record the exact 1,600-row matrix")

    try:
        relative_root = run_root.relative_to(repo_root)
    except ValueError as exc:
        raise ArtifactContractError("Extension formal run root must be inside the repository") from exc
    if relative_root.parent != EXTENSION_RUNTIME_ROOT:
        raise ArtifactContractError(
            "Extension formal run root must be one direct child of runs/keyframe_neighborhood_sampling"
        )

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--",
            ".",
            f":(exclude){EXTENSION_RUNTIME_ROOT.as_posix()}",
            f":(exclude){CHECKPOINT_RELATIVE.as_posix()}",
        ],
        cwd=repo_root,
        text=True,
    )
    if status:
        raise ArtifactContractError("Extension formal evaluation requires a clean committed worktree")
    if manifest.get("repository", {}).get("commit_sha") != commit:
        raise ArtifactContractError("Live repository commit differs from prepared extension formal commit")
    validate_frozen_formal_sources(
        manifest,
        repo_root,
        source="Prepared extension formal",
    )

    _validate_bound_smoke_evidence(paths=paths, manifest=manifest, commit=commit)
    return manifest


def _require_integer_list(payload: Mapping[str, Any], field: str) -> list[int]:
    values = payload.get(field)
    if not isinstance(values, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ArtifactContractError(f"Extension submission {field} must be an integer list")
    if len(values) != len(set(values)):
        raise ArtifactContractError(f"Extension submission {field} contains duplicates")
    return values


def validate_prepared_formal_root(
    run_root: str | Path,
    seed_table_path: str | Path,
    repo_root: str | Path,
    *,
    attempt_id: int,
    authorization_digest: str,
) -> dict[str, Any]:
    """Reject extension execution outside its exact authorized Slurm shard."""
    manifest = validate_formal_prepared_root(run_root, seed_table_path, repo_root)
    run_root = Path(run_root).resolve()
    matching_paths = [
        path for path in submission_paths(run_root, attempt_id) if sha256_file(path) == authorization_digest
    ]
    if len(matching_paths) != 1:
        raise ArtifactContractError("Extension formal authorization digest does not match exactly one recorded shard")
    submission = _load_json_object(matching_paths[0], source="extension formal submission")
    _require_extension_identity(submission, source="Extension formal submission")
    commit = str(manifest["repository"]["commit_sha"])
    if submission.get("repository_commit_sha") != commit:
        raise ArtifactContractError("Extension formal submission used a different commit")
    if int(submission.get("attempt_id", -1)) != attempt_id:
        raise ArtifactContractError("Extension formal submission attempt ID mismatch")
    if submission.get("formal_launch_authorized") is not True:
        raise ArtifactContractError("Extension formal submission lacks authorization")
    if submission.get("launch_manifest_sha256") != sha256_file(run_root / "protocol" / "launch_manifest.json"):
        raise ArtifactContractError("Extension launch manifest changed after submission")
    if submission.get("formal_matrix_sha256") != manifest.get("formal_matrix_sha256"):
        raise ArtifactContractError("Extension formal submission used a different matrix")

    plan_path = submission_plan_path(run_root, attempt_id)
    if not plan_path.is_file():
        raise ArtifactContractError(f"Extension formal submission plan is missing: {plan_path}")
    if submission.get("submission_plan_sha256") != sha256_file(plan_path):
        raise ArtifactContractError("Extension formal submission plan changed after shard submission")
    plan = _load_json_object(plan_path, source="extension formal submission plan")
    _require_extension_identity(plan, source="Extension formal submission plan")
    if int(plan.get("attempt_id", -1)) != attempt_id:
        raise ArtifactContractError("Extension formal submission plan attempt ID mismatch")
    if int(plan.get("max_rows_per_array", -1)) != MAX_ROWS_PER_ARRAY:
        raise ArtifactContractError("Extension formal plan has the wrong shard limit")
    planned_shards = plan.get("shards")
    if not isinstance(planned_shards, list):
        raise ArtifactContractError("Extension formal plan shards must be a list")
    if int(plan.get("shard_count", -1)) != len(planned_shards):
        raise ArtifactContractError("Extension formal plan shard_count mismatch")

    submission_shard_id = submission.get("shard_id")
    matching_plans = [
        shard for shard in planned_shards if isinstance(shard, Mapping) and shard.get("shard_id") == submission_shard_id
    ]
    if len(matching_plans) != 1:
        raise ArtifactContractError("Extension formal shard is absent or duplicated in its plan")
    planned = matching_plans[0]
    for field in ("shard_count", "array_task_ids", "row_ids", "array", "command"):
        if submission.get(field) != planned.get(field):
            raise ArtifactContractError(f"Extension formal shard differs from its plan on {field}")

    all_rows: list[int] = []
    shard_ids: list[int] = []
    for shard in planned_shards:
        if not isinstance(shard, Mapping):
            raise ArtifactContractError("Extension formal plan contains an invalid shard")
        shard_id = shard.get("shard_id")
        if isinstance(shard_id, bool) or not isinstance(shard_id, int):
            raise ArtifactContractError("Extension formal shard ID must be an integer")
        rows = _require_integer_list(shard, "row_ids")
        array_ids = _require_integer_list(shard, "array_task_ids")
        if array_ids != list(range(len(rows))):
            raise ArtifactContractError("Extension formal shard must map local array tasks 0..N-1 in row order")
        if len(rows) > MAX_ROWS_PER_ARRAY:
            raise ArtifactContractError("Extension formal shard exceeds 1,000 rows")
        if int(shard.get("shard_count", -1)) != len(planned_shards):
            raise ArtifactContractError("Extension formal per-shard count mismatch")
        shard_ids.append(shard_id)
        all_rows.extend(rows)
    if sorted(shard_ids) != list(range(len(planned_shards))):
        raise ArtifactContractError("Extension formal shard IDs must be exactly 0..N-1")
    if len(all_rows) != len(set(all_rows)):
        raise ArtifactContractError("Extension formal plan assigns a row more than once")
    if int(plan.get("trajectory_count", -1)) != len(all_rows):
        raise ArtifactContractError("Extension formal plan trajectory count mismatch")
    if attempt_id == 0 and sorted(all_rows) != list(range(FORMAL_TRAJECTORY_COUNT)):
        raise ArtifactContractError("Initial extension formal plan is not the exact 1,600-row matrix")

    active_array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    if not active_array_job or submission.get("slurm_array_job_id") != active_array_job:
        raise ArtifactContractError("Extension evaluator is not inside the recorded Slurm array")
    active_array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    try:
        active_array_id = int(active_array_task) if active_array_task is not None else -1
    except ValueError as exc:
        raise ArtifactContractError("SLURM_ARRAY_TASK_ID is not an integer") from exc
    submission_rows = _require_integer_list(submission, "row_ids")
    submission_array_ids = _require_integer_list(submission, "array_task_ids")
    if submission_array_ids != list(range(len(submission_rows))):
        raise ArtifactContractError("Extension formal submission has an invalid array-task mapping")
    if int(submission.get("trajectory_count", -1)) != len(submission_rows):
        raise ArtifactContractError("Extension formal submission trajectory_count/row mapping mismatch")
    if active_array_id not in submission_array_ids:
        raise ArtifactContractError("Current SLURM_ARRAY_TASK_ID is not authorized by the extension submission")
    return manifest
