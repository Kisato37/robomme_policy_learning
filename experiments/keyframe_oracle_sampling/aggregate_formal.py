#!/usr/bin/env python3
"""Fail-closed audit and preregistered aggregation for one formal run root.

This module is intentionally an I/O and validation layer.  All estimands,
resampling procedures, seeds, replicate counts, and decision rules remain in
the pre-formal ``analysis.py`` implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np

from experiments.keyframe_oracle_sampling.analysis import build_point_analysis
from experiments.keyframe_oracle_sampling.analysis import normalize_prior_exposure_manifest
from experiments.keyframe_oracle_sampling.analysis import validate_formal_records
from experiments.keyframe_oracle_sampling.artifacts import ALL_ARMS
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_CHECKPOINT_ID
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_CHECKPOINT_PATH
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_EVALUATION_POLICY_SEED
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_EXECUTED_ACTION_HORIZON
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_FINAL_MEMORY_DTYPE
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_FINAL_MEMORY_SHAPE
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_INITIAL_CONDITION_HASH_FIELDS
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_PREPARED_COMPONENT_SHAPES
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import EpisodeAttemptWriter
from experiments.keyframe_oracle_sampling.artifacts import RunArtifactStore
from experiments.keyframe_oracle_sampling.artifacts import ScientificKey
from experiments.keyframe_oracle_sampling.artifacts import audit_initial_condition_fairness
from experiments.keyframe_oracle_sampling.artifacts import audit_paired_manifest_invariants
from experiments.keyframe_oracle_sampling.artifacts import canonical_json_bytes
from experiments.keyframe_oracle_sampling.artifacts import completeness_report
from experiments.keyframe_oracle_sampling.artifacts import load_seed_table
from experiments.keyframe_oracle_sampling.artifacts import read_jsonl
from experiments.keyframe_oracle_sampling.artifacts import released_prepared_component_dtypes
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from experiments.keyframe_oracle_sampling.artifacts import validate_architecture_pass_report
from experiments.keyframe_oracle_sampling.artifacts import validate_smoke_formal_seed_disjointness
from experiments.keyframe_oracle_sampling.formal_artifacts import ACCEPTED_DEVELOPMENT_SMOKE_AUDIT_SHA256
from experiments.keyframe_oracle_sampling.formal_artifacts import ACCEPTED_DEVELOPMENT_SMOKE_COMMIT
from experiments.keyframe_oracle_sampling.formal_artifacts import submission_path
from experiments.keyframe_oracle_sampling.formal_artifacts import submission_paths
from experiments.keyframe_oracle_sampling.formal_artifacts import submission_plan_path
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_DATASET
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_EPISODE_IDS
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_MAX_STEPS
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_oracle_sampling.formal_matrix import load_formal_matrix
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_POLICY_CALLS
from mme_vla_suite.shared.keyframe_oracle_sampling import SelectorArm
from mme_vla_suite.shared.keyframe_oracle_sampling import select_indices
from mme_vla_suite.shared.keyframe_oracle_sampling import validate_selector_output

AGGREGATE_SCHEMA_VERSION = 1
AGGREGATE_FILENAMES = (
    "completeness_report.json",
    "per_episode.csv",
    "per_task.csv",
    "summary.json",
    "analysis.md",
)
FORMAL_TERMINAL_REASONS = frozenset({"success", "fail", "timeout", "error"})
_SHA256_HEX = frozenset("0123456789abcdef")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError(f"JSON artifact must be an object: {path}")
    return payload


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise ArtifactContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_exact_fields(payload: Mapping[str, Any], expected: Mapping[str, Any], *, source: str) -> None:
    mismatches = {
        field: {"expected": expected_value, "observed": payload.get(field)}
        for field, expected_value in expected.items()
        if field not in payload or payload.get(field) != expected_value
    }
    if mismatches:
        raise ArtifactContractError(
            f"{source} differs from the frozen formal contract: " + json.dumps(mismatches, sort_keys=True)
        )


def _require_nonnegative_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | float | np.number):
        raise ArtifactContractError(f"{field} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ArtifactContractError(f"{field} must be a finite non-negative number")
    return normalized


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array < 0):
        raise ArtifactContractError("Latency summary requires finite non-negative values")
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
    }


def _expected_formal_keys(matrix: Mapping[str, Any]) -> dict[ScientificKey, dict[str, Any]]:
    rows = matrix.get("rows")
    if not isinstance(rows, list):
        raise ArtifactContractError("Formal matrix rows must be a list")
    expected: dict[ScientificKey, dict[str, Any]] = {}
    for position, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict) or raw_row.get("row_id") != position:
            raise ArtifactContractError("Formal matrix row IDs must be canonical and contiguous")
        key = ScientificKey(
            str(raw_row["task"]),
            int(raw_row["episode_id"]),
            str(raw_row["arm"]),
            str(raw_row["trajectory_kind"]),
        )
        if key in expected:
            raise ArtifactContractError(f"Duplicate key in formal matrix: {key}")
        expected[key] = raw_row
    if len(expected) != FORMAL_TRAJECTORY_COUNT:
        raise ArtifactContractError("Formal matrix does not contain exactly 3,200 unique rows")
    return expected


def validate_formal_protocol_bundle(
    run_root: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[tuple[str, int, int], int], dict[str, Any], dict[str, Any]
]:
    """Validate immutable pre-run provenance without requiring the old live checkout."""
    run_root = run_root.resolve()
    protocol_dir = run_root / "protocol"
    paths = {
        "manifest": protocol_dir / "launch_manifest.json",
        "protocol": protocol_dir / "protocol_snapshot.md",
        "protocol_sidecar": protocol_dir / "protocol_sha256.txt",
        "seed": protocol_dir / "seed_table.json",
        "development_seed": protocol_dir / "development_seed_audit_table.json",
        "matrix": protocol_dir / "formal_matrix.json",
        "exposure": protocol_dir / "prior_exposure_manifest.json",
        "architecture_report": protocol_dir / "architecture_pass_report.json",
        "architecture_submission": protocol_dir / "architecture_submission_record.json",
        "development_smoke": protocol_dir / "development_smoke_audit.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ArtifactContractError(f"Formal protocol bundle is incomplete: {missing}")

    launch = _load_json_object(paths["manifest"])
    _require_exact_fields(
        launch,
        {
            "protocol_version": PROTOCOL_VERSION,
            "run_kind": "formal",
            "dataset": FORMAL_DATASET,
            "formal_launch_authorized": True,
            "trajectory_count": FORMAL_TRAJECTORY_COUNT,
            "max_steps": FORMAL_MAX_STEPS,
            "executed_action_horizon": SMOKE_EXECUTED_ACTION_HORIZON,
            "evaluation_policy_seed": SMOKE_EVALUATION_POLICY_SEED,
            "checkpoint_path": SMOKE_CHECKPOINT_PATH,
        },
        source="Formal launch manifest",
    )
    digest_bindings = {
        "protocol_sha256": paths["protocol"],
        "seed_table_file_sha256": paths["seed"],
        "development_seed_audit_file_sha256": paths["development_seed"],
        "formal_matrix_sha256": paths["matrix"],
        "prior_exposure_manifest_sha256": paths["exposure"],
        "architecture_report_sha256": paths["architecture_report"],
        "architecture_submission_record_sha256": paths["architecture_submission"],
        "development_smoke_audit_sha256": paths["development_smoke"],
    }
    for field, path in digest_bindings.items():
        if launch.get(field) != sha256_file(path):
            raise ArtifactContractError(f"Launch-manifest digest mismatch for {path.name}")
    if paths["protocol_sidecar"].read_text().strip() != launch["protocol_sha256"]:
        raise ArtifactContractError("Protocol SHA-256 sidecar differs from launch manifest")

    formal_seed_payload, formal_seed_lookup = load_seed_table(
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
    if set(formal_seed_lookup) != expected_seed_keys:
        raise ArtifactContractError("Formal seed table is not the exact 16 x 50 x 82 universe")
    development_seed_payload, development_seed_lookup = load_seed_table(paths["development_seed"])
    expected_development_keys = {
        (task, 0, call_index) for task in FORMAL_TASKS for call_index in range(MAX_POLICY_CALLS)
    }
    if set(development_seed_lookup) != expected_development_keys:
        raise ArtifactContractError("Development seed audit is not the exact 16 x 1 x 82 universe")
    seed_disjointness = validate_smoke_formal_seed_disjointness(development_seed_payload, formal_seed_payload)
    _require_exact_fields(
        launch,
        {
            "seed_table_scope": formal_seed_payload["scope"],
            "seed_table_dataset": formal_seed_payload["dataset"],
            "seed_table_derivation": formal_seed_payload["derivation"],
            "seed_table_entries_sha256": formal_seed_payload["entries_sha256"],
            "seed_disjointness_audit": seed_disjointness,
        },
        source="Formal launch seed provenance",
    )

    matrix = load_formal_matrix(paths["matrix"])
    if launch.get("matrix") != matrix["rows"]:
        raise ArtifactContractError("Launch manifest differs from the frozen formal matrix")
    exposure = _load_json_object(paths["exposure"])
    normalized_exposure = normalize_prior_exposure_manifest(exposure)
    _require_exact_fields(
        exposure,
        {
            "frozen_before_formal_execution": True,
            "present_formal_outcomes_inspected": False,
        },
        source="Prior-exposure manifest",
    )
    _require_exact_fields(
        launch,
        {
            "prior_exposure_entry_count": len(normalized_exposure.entries),
            "prior_exposure_test_block_count": len(normalized_exposure.excluded_formal_blocks),
        },
        source="Prior-exposure launch binding",
    )

    repository_commit = launch.get("repository", {}).get("commit_sha")
    if (
        not isinstance(repository_commit, str)
        or len(repository_commit) != 40
        or any(character not in _SHA256_HEX for character in repository_commit)
    ):
        raise ArtifactContractError("Launch manifest lacks a valid repository commit SHA")
    architecture_report = _load_json_object(paths["architecture_report"])
    architecture_submission = _load_json_object(paths["architecture_submission"])
    architecture_source_root = launch.get("architecture_source_run_root")
    if not isinstance(architecture_source_root, str) or not architecture_source_root:
        raise ArtifactContractError("Launch manifest lacks its architecture-smoke source root")
    validate_architecture_pass_report(
        architecture_report,
        run_root=architecture_source_root,
        architecture_submission=architecture_submission,
    )
    if architecture_report.get("repository_commit_sha") != repository_commit:
        raise ArtifactContractError("Architecture smoke used a different formal commit")
    for field in (
        "checkpoint_unpacked_metadata_sha256",
        "checkpoint_content_tree_algorithm",
        "checkpoint_unpacked_content_tree_sha256",
    ):
        if architecture_report.get(field) != launch.get(field):
            raise ArtifactContractError(f"Architecture report and launch manifest differ on {field}")

    development_smoke = _load_json_object(paths["development_smoke"])
    if sha256_file(paths["development_smoke"]) != ACCEPTED_DEVELOPMENT_SMOKE_AUDIT_SHA256:
        raise ArtifactContractError("Development-smoke audit is not the accepted pre-formal PASS")
    if launch.get("development_smoke_commit_sha") != ACCEPTED_DEVELOPMENT_SMOKE_COMMIT:
        raise ArtifactContractError("Launch manifest names the wrong development-smoke commit")
    _require_exact_fields(
        development_smoke,
        {
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
            "seed_disjointness_audit": seed_disjointness,
        },
        source="Development-smoke PASS",
    )

    provenance = {
        "launch_manifest_sha256": sha256_file(paths["manifest"]),
        "protocol_sha256": launch["protocol_sha256"],
        "formal_matrix_sha256": launch["formal_matrix_sha256"],
        "formal_seed_table_file_sha256": launch["seed_table_file_sha256"],
        "formal_seed_table_entries_sha256": formal_seed_payload["entries_sha256"],
        "prior_exposure_manifest_sha256": launch["prior_exposure_manifest_sha256"],
        "architecture_report_sha256": launch["architecture_report_sha256"],
        "development_smoke_audit_sha256": launch["development_smoke_audit_sha256"],
        "formal_evaluation_commit_sha": repository_commit,
        "checkpoint_id": SMOKE_CHECKPOINT_ID,
        "checkpoint_path": launch["checkpoint_path"],
        "checkpoint_unpacked_metadata_sha256": launch["checkpoint_unpacked_metadata_sha256"],
        "checkpoint_content_tree_algorithm": launch["checkpoint_content_tree_algorithm"],
        "checkpoint_unpacked_content_tree_sha256": launch["checkpoint_unpacked_content_tree_sha256"],
        "seed_disjointness_audit": seed_disjointness,
    }
    return launch, matrix, formal_seed_payload, formal_seed_lookup, exposure, provenance


def _integer_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ArtifactContractError(f"{field} must be an integer list")
    if len(value) != len(set(value)):
        raise ArtifactContractError(f"{field} contains duplicates")
    return value


def validate_submission_attempt(
    run_root: Path,
    *,
    attempt_id: int,
    expected_row_ids: Sequence[int],
    launch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Validate one complete offline submission plan and all of its shard records."""
    plan_path = submission_plan_path(run_root, attempt_id)
    if not plan_path.is_file():
        raise ArtifactContractError(f"Missing formal submission plan: {plan_path}")
    plan = _load_json_object(plan_path)
    repository_commit = launch["repository"]["commit_sha"]
    _require_exact_fields(
        plan,
        {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "repository_commit_sha": repository_commit,
            "run_root": str(run_root.resolve()),
            "formal_launch_authorized": True,
            "max_rows_per_array": 1000,
        },
        source=f"Submission plan attempt {attempt_id}",
    )
    shards = plan.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ArtifactContractError("Submission plan must contain at least one shard")
    if plan.get("shard_count") != len(shards):
        raise ArtifactContractError("Submission plan shard count mismatch")
    if plan.get("global_max_concurrent") not in {1, 2, 3, 4}:
        raise ArtifactContractError("Submission plan has an invalid global concurrency")

    planned_rows: list[int] = []
    shard_by_id: dict[int, Mapping[str, Any]] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ArtifactContractError("Submission plan contains a non-object shard")
        shard_id = shard.get("shard_id")
        if type(shard_id) is not int or shard_id in shard_by_id:
            raise ArtifactContractError("Submission plan has an invalid or duplicate shard ID")
        row_ids = _integer_list(shard.get("row_ids"), field="planned shard row_ids")
        array_task_ids = _integer_list(shard.get("array_task_ids"), field="planned shard array_task_ids")
        if array_task_ids != list(range(len(row_ids))):
            raise ArtifactContractError("Submission shard local task IDs are not canonical")
        if not row_ids or len(row_ids) > 1000:
            raise ArtifactContractError("Submission shard must contain 1..1,000 rows")
        if shard.get("shard_count") != len(shards):
            raise ArtifactContractError("Submission shard records the wrong shard count")
        shard_by_id[shard_id] = shard
        planned_rows.extend(row_ids)
    if sorted(shard_by_id) != list(range(len(shards))):
        raise ArtifactContractError("Submission shard IDs must be contiguous from zero")
    if len(planned_rows) != len(set(planned_rows)):
        raise ArtifactContractError("Submission plan assigns a row more than once")
    if plan.get("trajectory_count") != len(planned_rows):
        raise ArtifactContractError("Submission plan trajectory count mismatch")
    if sorted(planned_rows) != sorted(expected_row_ids):
        raise ArtifactContractError(f"Submission attempt {attempt_id} does not authorize the exact expected rows")

    record_paths = submission_paths(run_root, attempt_id)
    if len(record_paths) != len(shards):
        raise ArtifactContractError(
            f"Submission attempt {attempt_id} has {len(record_paths)} records for {len(shards)} planned shards"
        )
    plan_sha256 = sha256_file(plan_path)
    row_authorizations: dict[int, dict[str, Any]] = {}
    record_digests = []
    job_ids = []
    for record_path in record_paths:
        record = _load_json_object(record_path)
        shard_id = record.get("shard_id")
        if type(shard_id) is not int or shard_id not in shard_by_id:
            raise ArtifactContractError("Submission record has an unknown shard ID")
        if record_path != submission_path(run_root, attempt_id, shard_id):
            raise ArtifactContractError("Submission record path is non-canonical")
        planned = shard_by_id[shard_id]
        _require_exact_fields(
            record,
            {
                "schema_version": 1,
                "attempt_id": attempt_id,
                "shard_id": shard_id,
                "repository_commit_sha": repository_commit,
                "run_root": str(run_root.resolve()),
                "formal_launch_authorized": True,
                "launch_manifest_sha256": sha256_file(run_root / "protocol" / "launch_manifest.json"),
                "formal_matrix_sha256": launch["formal_matrix_sha256"],
                "prior_exposure_manifest_sha256": launch["prior_exposure_manifest_sha256"],
                "architecture_report_sha256": launch["architecture_report_sha256"],
                "development_smoke_audit_sha256": launch["development_smoke_audit_sha256"],
                "checkpoint_unpacked_metadata_sha256": launch["checkpoint_unpacked_metadata_sha256"],
                "checkpoint_content_tree_algorithm": launch["checkpoint_content_tree_algorithm"],
                "checkpoint_unpacked_content_tree_sha256": launch["checkpoint_unpacked_content_tree_sha256"],
                "submission_plan_sha256": plan_sha256,
                "shard_count": len(shards),
                "global_max_concurrent": plan["global_max_concurrent"],
            },
            source=f"Submission record attempt {attempt_id} shard {shard_id}",
        )
        for field in ("array_task_ids", "row_ids", "array", "command", "shard_count"):
            if record.get(field) != planned.get(field):
                raise ArtifactContractError(
                    f"Submission record attempt {attempt_id} shard {shard_id} differs from its plan on {field}"
                )
        row_ids = _integer_list(record.get("row_ids"), field="submission row_ids")
        array_task_ids = _integer_list(record.get("array_task_ids"), field="submission array_task_ids")
        if record.get("trajectory_count") != len(row_ids):
            raise ArtifactContractError("Submission record trajectory count mismatch")
        job_id = record.get("slurm_array_job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ArtifactContractError("Submission record lacks a Slurm array job ID")
        job_ids.append(job_id)
        record_digests.append({"shard_id": shard_id, "sha256": sha256_file(record_path)})
        for local_id, row_id in zip(array_task_ids, row_ids, strict=True):
            if row_id in row_authorizations:
                raise ArtifactContractError("A row appears in two submission records")
            row_authorizations[row_id] = {
                "attempt_id": attempt_id,
                "shard_id": shard_id,
                "array_task_id": local_id,
                "slurm_array_job_id": job_id,
                "submission_record_sha256": sha256_file(record_path),
            }
    return (
        {
            "attempt_id": attempt_id,
            "trajectory_count": len(planned_rows),
            "shard_count": len(shards),
            "slurm_array_job_ids": job_ids,
            "submission_plan_sha256": plan_sha256,
            "submission_record_digests": sorted(record_digests, key=lambda item: item["shard_id"]),
            "row_ids_sha256": hashlib.sha256(canonical_json_bytes(sorted(planned_rows))).hexdigest(),
        },
        row_authorizations,
    )


def audit_formal_attempt(
    writer: EpisodeAttemptWriter,
    *,
    expected_key: ScientificKey,
    expected_row: Mapping[str, Any],
    launch: Mapping[str, Any],
    seed_payload: Mapping[str, Any],
    seed_lookup: Mapping[tuple[str, int, int], int],
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, Any], dict[str, Any]]:
    """Recompute and validate every selector decision in one completed trajectory."""
    if writer.key != expected_key:
        raise ArtifactContractError("Attempt writer is bound to the wrong formal key")
    _require_exact_fields(
        expected_row,
        {
            "task": expected_key.task,
            "episode_id": expected_key.episode_id,
            "arm": expected_key.arm,
            "trajectory_kind": "formal",
            "dataset": FORMAL_DATASET,
            "max_steps": FORMAL_MAX_STEPS,
        },
        source="Formal matrix row",
    )
    if type(expected_row.get("row_id")) is not int:
        raise ArtifactContractError("Formal matrix row_id must be an integer")
    _require_exact_fields(
        seed_payload,
        {"scope": FORMAL_SEED_SCOPE, "dataset": FORMAL_SEED_DATASET},
        source="Validated formal seed payload",
    )
    seed_digest = _require_sha256(seed_payload.get("entries_sha256"), field="Formal seed-table entries_sha256")
    _require_exact_fields(
        launch,
        {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": FORMAL_DATASET,
            "evaluation_policy_seed": SMOKE_EVALUATION_POLICY_SEED,
            "executed_action_horizon": SMOKE_EXECUTED_ACTION_HORIZON,
            "checkpoint_path": SMOKE_CHECKPOINT_PATH,
            "seed_table_scope": FORMAL_SEED_SCOPE,
            "seed_table_dataset": FORMAL_SEED_DATASET,
            "seed_table_entries_sha256": seed_digest,
        },
        source="Formal launch manifest",
    )

    if writer.validate_resume() != "complete":
        raise ArtifactContractError(f"Formal attempt is not complete: {writer.attempt_dir}")
    manifest = _load_json_object(writer.manifest_path)
    result = _load_json_object(writer.result_path)
    initial_conditions = _load_json_object(writer.initial_conditions_path)
    traces = read_jsonl(writer.trace_path)
    if not traces:
        raise ArtifactContractError("Completed formal attempt has no selector trace")

    if set(initial_conditions) != set(SMOKE_INITIAL_CONDITION_HASH_FIELDS):
        raise ArtifactContractError("Formal attempt must record the exact five initial-condition hashes")
    for field in SMOKE_INITIAL_CONDITION_HASH_FIELDS:
        _require_sha256(initial_conditions.get(field), field=f"Initial condition {field}")

    fixed_fields = {
        "dataset": FORMAL_DATASET,
        "max_steps": FORMAL_MAX_STEPS,
        "executed_action_horizon": SMOKE_EXECUTED_ACTION_HORIZON,
        "evaluation_policy_seed": SMOKE_EVALUATION_POLICY_SEED,
        "checkpoint_id": SMOKE_CHECKPOINT_ID,
    }
    _require_exact_fields(manifest, fixed_fields, source="Formal episode manifest")
    _require_exact_fields(result, fixed_fields, source="Formal episode result")
    for source, payload in (("manifest", manifest), ("result", result)):
        for field in (
            "max_steps",
            "executed_action_horizon",
            "evaluation_policy_seed",
            "checkpoint_id",
        ):
            if type(payload.get(field)) is not int:
                raise ArtifactContractError(f"Formal episode {source} {field} must be an integer")
    _require_exact_fields(
        manifest,
        {"protocol_version": PROTOCOL_VERSION, "seed_table_sha256": seed_digest},
        source="Formal episode manifest",
    )
    _require_exact_fields(
        result,
        {
            "task": expected_key.task,
            "episode_id": expected_key.episode_id,
            "selector_arm": expected_key.arm,
        },
        source="Formal episode result",
    )
    if type(result.get("episode_id")) is not int:
        raise ArtifactContractError("Formal result episode_id must be an integer")
    terminal_reason = result.get("terminal_reason")
    if terminal_reason not in FORMAL_TERMINAL_REASONS:
        raise ArtifactContractError("Formal result lacks an official scientific terminal outcome")
    if type(result.get("success")) is not bool:
        raise ArtifactContractError("Formal result success must be a boolean")
    if result["success"] != (terminal_reason == "success"):
        raise ArtifactContractError("Formal result success disagrees with terminal_reason")
    if type(result.get("timeout")) is not bool:
        raise ArtifactContractError("Formal result timeout must be a boolean")
    if result["timeout"] != (terminal_reason == "timeout"):
        raise ArtifactContractError("Formal result timeout disagrees with terminal_reason")
    if terminal_reason == "error" and (
        not result.get("benchmark_error_message") or not result.get("benchmark_exception_type")
    ):
        raise ArtifactContractError("Scientific error outcome lacks benchmark error evidence")
    if type(result.get("collision")) is not bool:
        raise ArtifactContractError("Formal result collision must be a boolean")

    steps = result.get("steps")
    if type(steps) is not int or steps < 1 or steps > FORMAL_MAX_STEPS:
        raise ArtifactContractError("Formal result has an invalid step count")
    expected_policy_calls = math.ceil(steps / SMOKE_EXECUTED_ACTION_HORIZON)
    if not 1 <= expected_policy_calls <= MAX_POLICY_CALLS:
        raise ArtifactContractError("Formal result implies an invalid policy-call count")
    if len(traces) != expected_policy_calls:
        raise ArtifactContractError(
            f"Selector trace count must equal ceil(result.steps / 16): {len(traces)} != {expected_policy_calls}"
        )
    for field in (
        "policy_latency_ms",
        "policy_model_latency_ms",
        "history_lengths_at_policy_calls",
    ):
        values = result.get(field)
        if not isinstance(values, list) or len(values) != expected_policy_calls:
            raise ArtifactContractError(f"Formal result {field} must contain one value per policy call")

    required_trace_hashes = (
        "seed_table_sha256",
        "selected_indices_sha256",
        "mask_sha256",
        "image_tensor_sha256",
        "position_tensor_sha256",
        "state_tensor_sha256",
        "prepared_memory_components_sha256",
        "prepared_memory_input_sha256",
        "final_memory_tensor_sha256",
    )
    latency_fields = (
        "boundary_lookup_latency_ms",
        "selector_decision_latency_ms",
        "selector_bookkeeping_latency_ms",
        "selector_latency_ms",
        "model_latency_ms",
        "end_to_end_request_latency_ms",
    )
    latency_values = {"selector": [], "model": [], "end_to_end": []}
    initial_history_length: int | None = None
    for call_index, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            raise ArtifactContractError("Selector trace contains a non-object record")
        _require_exact_fields(
            trace,
            {
                "schema_version": 1,
                "task": expected_key.task,
                "episode_id": expected_key.episode_id,
                "selector_name": expected_key.arm,
                "seed_table_sha256": seed_digest,
                "seed_table_scope": FORMAL_SEED_SCOPE,
                "seed_table_dataset": FORMAL_SEED_DATASET,
                "policy_call_index": call_index,
                "environment_step": call_index * SMOKE_EXECUTED_ACTION_HORIZON,
            },
            source=f"Formal selector trace call {call_index}",
        )
        for field in ("episode_id", "policy_call_index", "environment_step"):
            if type(trace.get(field)) is not int:
                raise ArtifactContractError(f"Selector trace {field} must be an integer")

        current_history_index = trace.get("current_history_index")
        history_length = trace.get("history_length")
        if type(current_history_index) is not int or current_history_index < 0:
            raise ArtifactContractError("Trace current_history_index must be non-negative")
        if type(history_length) is not int or history_length != current_history_index + 1:
            raise ArtifactContractError("Trace history_length must equal current_history_index + 1")
        if initial_history_length is None:
            initial_history_length = history_length
        if history_length != initial_history_length + trace["environment_step"]:
            raise ArtifactContractError("Trace history cadence differs from the 16-step horizon")

        raw_boundaries = trace.get("visible_boundary_indices")
        if not isinstance(raw_boundaries, list) or any(type(value) is not int for value in raw_boundaries):
            raise ArtifactContractError("Trace visible boundaries must be an integer list")
        visible_boundaries = [int(value) for value in raw_boundaries]
        if (
            visible_boundaries != sorted(set(visible_boundaries))
            or not visible_boundaries
            or visible_boundaries[0] != 0
            or any(value < 0 or value > current_history_index for value in visible_boundaries)
        ):
            raise ArtifactContractError("Trace visible boundaries must be sorted, unique, causal, and include frame 0")

        if "selector_seed" not in trace:
            raise ArtifactContractError("Trace must explicitly record selector_seed")
        expected_seed = None
        if expected_key.arm == SelectorArm.RANDOM.value:
            seed_key = (expected_key.task, expected_key.episode_id, call_index)
            try:
                expected_seed = int(seed_lookup[seed_key])
            except KeyError as exc:
                raise ArtifactContractError(f"Formal seed table lacks preregistered call {seed_key}") from exc
        if trace["selector_seed"] != expected_seed:
            raise ArtifactContractError(f"Selector trace call {call_index} has the wrong preregistered seed")
        expected_selected = select_indices(
            expected_key.arm,
            current_history_index,
            boundary_indices=visible_boundaries,
            random_seed=expected_seed,
        )
        if trace.get("selected_frame_indices") != expected_selected:
            raise ArtifactContractError(f"Selector trace call {call_index} does not match arm {expected_key.arm}")
        selected = validate_selector_output(trace["selected_frame_indices"], current_history_index)
        selected_sha256 = hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode("ascii")).hexdigest()
        if trace.get("selected_indices_sha256") != selected_sha256:
            raise ArtifactContractError("Selected-index trace digest mismatch")
        if (
            trace.get("valid_frame_count") != len(selected)
            or trace.get("padding_frame_count") != 32 - len(selected)
            or trace.get("valid_memory_token_count") != 16 * len(selected)
        ):
            raise ArtifactContractError("Trace frame, padding, or token count mismatch")
        if (
            trace.get("mask_shape") != [512]
            or trace.get("mask_dtype") != "bool"
            or trace.get("mask_valid_prefix_all_true") is not True
            or trace.get("mask_padding_all_false") is not True
        ):
            raise ArtifactContractError("Trace mask is not a true prefix plus false padding")
        if trace.get("prepared_memory_component_shapes") != [list(shape) for shape in SMOKE_PREPARED_COMPONENT_SHAPES]:
            raise ArtifactContractError("Prepared-memory component shapes changed")
        observed_dtypes = (
            trace.get("image_tensor_dtype"),
            trace.get("position_tensor_dtype"),
            trace.get("state_tensor_dtype"),
            trace.get("mask_dtype"),
        )
        if observed_dtypes != released_prepared_component_dtypes(len(selected)):
            raise ArtifactContractError("Prepared-memory component dtypes changed")
        if trace.get("prepared_memory_input_shape") != [512, 2824]:
            raise ArtifactContractError("Prepared-memory input shape must be [512, 2824]")
        for field in required_trace_hashes:
            digest = _require_sha256(trace.get(field), field=f"Trace {field}")
            if field == "seed_table_sha256" and digest != seed_digest:
                raise ArtifactContractError("Trace seed-table digest mismatch")

        latencies = {
            field: _require_nonnegative_finite(trace.get(field), field=f"Trace {field}") for field in latency_fields
        }
        component_latency = (
            latencies["boundary_lookup_latency_ms"]
            + latencies["selector_decision_latency_ms"]
            + latencies["selector_bookkeeping_latency_ms"]
        )
        if not math.isclose(
            latencies["selector_latency_ms"],
            component_latency,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ArtifactContractError("Selector latency differs from its components")
        expected_end_to_end = _require_nonnegative_finite(
            result["policy_latency_ms"][call_index], field="Result policy_latency_ms"
        )
        expected_model = _require_nonnegative_finite(
            result["policy_model_latency_ms"][call_index],
            field="Result policy_model_latency_ms",
        )
        if latencies["end_to_end_request_latency_ms"] != expected_end_to_end:
            raise ArtifactContractError("Trace/request latency differs from result")
        if latencies["model_latency_ms"] != expected_model:
            raise ArtifactContractError("Trace/model latency differs from result")
        if type(result["history_lengths_at_policy_calls"][call_index]) is not int:
            raise ArtifactContractError("Result history lengths must be integers")
        if result["history_lengths_at_policy_calls"][call_index] != history_length:
            raise ArtifactContractError("Trace history length differs from result")
        latency_values["selector"].append(latencies["selector_latency_ms"])
        latency_values["model"].append(latencies["model_latency_ms"])
        latency_values["end_to_end"].append(latencies["end_to_end_request_latency_ms"])

        memory_shape = trace.get("final_memory_tensor_shape")
        if (
            memory_shape != list(SMOKE_FINAL_MEMORY_SHAPE)
            or not isinstance(memory_shape, list)
            or any(type(value) is not int for value in memory_shape)
        ):
            raise ArtifactContractError("Final memory tensor shape must be [1, 512, 1024]")
        if trace.get("final_memory_tensor_dtype") != SMOKE_FINAL_MEMORY_DTYPE:
            raise ArtifactContractError("Final memory tensor dtype changed")
        if trace.get("final_memory_tensor_is_floating") is not True:
            raise ArtifactContractError("Final memory tensor must be floating")
        if trace.get("final_memory_tensor_finite") is not True:
            raise ArtifactContractError("Final memory tensor contains non-finite values")

    report = {
        "strict_formal_contract": True,
        "policy_call_count": expected_policy_calls,
        "selector_arm": expected_key.arm,
        "selector_latency": _latency_summary(latency_values["selector"]),
        "model_latency": _latency_summary(latency_values["model"]),
        "end_to_end_latency": _latency_summary(latency_values["end_to_end"]),
    }
    return report, latency_values, manifest, result


def _scientific_key_from_mapping(raw: Any, *, source: str) -> ScientificKey:
    if not isinstance(raw, Mapping):
        raise ArtifactContractError(f"{source} lacks a scientific key")
    try:
        return ScientificKey(
            str(raw["task"]),
            int(raw["episode_id"]),
            str(raw["arm"]),
            str(raw.get("trajectory_kind", "formal")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactContractError(f"{source} has an invalid scientific key") from exc


def discover_attempts(
    run_root: Path, store: RunArtifactStore
) -> dict[tuple[ScientificKey, int], tuple[Path, dict[str, Any]]]:
    attempts: dict[tuple[ScientificKey, int], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(run_root.glob("trajectories/**/attempt_*")):
        if not path.is_dir() or path.is_symlink():
            raise ArtifactContractError(f"Attempt path is not a real directory: {path}")
        manifest_path = path / "episode_manifest.json"
        if not manifest_path.is_file():
            raise ArtifactContractError(f"Attempt directory lacks a manifest: {path}")
        manifest = _load_json_object(manifest_path)
        key = _scientific_key_from_mapping(manifest.get("scientific_key"), source=f"Attempt manifest {manifest_path}")
        attempt_id = manifest.get("attempt_id")
        if type(attempt_id) is not int or attempt_id not in {0, 1, 2}:
            raise ArtifactContractError(f"Attempt manifest has an invalid attempt ID: {path}")
        if path.resolve() != store.attempt_dir(key, attempt_id).resolve():
            raise ArtifactContractError(f"Attempt directory is non-canonical: {path}")
        identity = (key, attempt_id)
        if identity in attempts:
            raise ArtifactContractError(f"Duplicate formal attempt directory: {identity}")
        attempts[identity] = (path, manifest)
    return attempts


def _failure_key(record: Mapping[str, Any]) -> ScientificKey:
    return _scientific_key_from_mapping(record, source="Failure-ledger record")


def _validate_attempt_submission_binding(
    manifest: Mapping[str, Any],
    *,
    row_id: int,
    authorization: Mapping[str, Any],
) -> None:
    slurm = manifest.get("slurm")
    if not isinstance(slurm, Mapping):
        raise ArtifactContractError("Attempt manifest lacks Slurm provenance")
    expected = {
        "array_job_id": str(authorization["slurm_array_job_id"]),
        "array_task_id": str(authorization["array_task_id"]),
    }
    for field, value in expected.items():
        if slurm.get(field) != value:
            raise ArtifactContractError(f"Attempt manifest Slurm {field} differs from its submission record")
    slurm_row = slurm.get("formal_matrix_row_id")
    if slurm_row is not None and slurm_row != str(row_id):
        raise ArtifactContractError("Attempt manifest records the wrong formal matrix row")
    top_level_row = manifest.get("formal_matrix_row_id")
    if top_level_row is not None and top_level_row != row_id:
        raise ArtifactContractError("Launcher-only manifest records the wrong formal row")


def _repository_provenance(repo_root: Path, *, formal_commit: str) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    try:
        aggregation_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        subprocess.run(
            ["git", "cat-file", "-e", f"{formal_commit}^{{commit}}"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", formal_commit, aggregation_commit],
            cwd=repo_root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtifactContractError("Aggregation checkout must contain the formal commit as an ancestor") from exc
    analysis_path = repo_root / "experiments/keyframe_oracle_sampling/analysis.py"
    aggregator_path = repo_root / "experiments/keyframe_oracle_sampling/aggregate_formal.py"
    for path in (analysis_path, aggregator_path):
        if not path.is_file():
            raise ArtifactContractError(f"Aggregation source file is missing: {path}")
    return {
        "aggregation_commit_sha": aggregation_commit,
        "formal_commit_is_ancestor": True,
        "analysis_source_sha256": sha256_file(analysis_path),
        "aggregator_source_sha256": sha256_file(aggregator_path),
    }


def audit_formal_run(run_root: Path, repo_root: Path) -> tuple[dict[str, Any], dict[ScientificKey, dict[str, Any]]]:
    """Audit the exact completed census and return validated outcome records."""
    run_root = run_root.resolve()
    repo_root = repo_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Formal run root does not exist: {run_root}")
    aggregate_dir = run_root / "aggregate"
    if aggregate_dir.exists() or aggregate_dir.is_symlink():
        raise FileExistsError(f"Refusing to inspect and republish an existing aggregate: {aggregate_dir}")
    try:
        relative_run_root = run_root.relative_to(repo_root)
    except ValueError as exc:
        raise ArtifactContractError("Formal run root must be inside the repository") from exc
    if relative_run_root.parent != Path("runs/keyframe_oracle_sampling"):
        raise ArtifactContractError("Formal run root must be one direct child of runs/keyframe_oracle_sampling")

    (
        launch,
        matrix,
        seed_payload,
        seed_lookup,
        exposure,
        protocol_provenance,
    ) = validate_formal_protocol_bundle(run_root)
    expected_rows = _expected_formal_keys(matrix)
    row_id_by_key = {key: int(row["row_id"]) for key, row in expected_rows.items()}
    key_by_row_id = {row_id: key for key, row_id in row_id_by_key.items()}

    store = RunArtifactStore(run_root)
    completed = store.scan_completed_keys()
    census = completeness_report(set(expected_rows), completed)
    if not census["complete"]:
        raise ArtifactContractError(
            "Formal aggregation requires exactly one completed result for all 3,200 cells: "
            + json.dumps(census, sort_keys=True)
        )
    attempts = discover_attempts(run_root, store)

    failure_records = read_jsonl(store.failures_path) if store.failures_path.exists() else []
    failures: dict[tuple[ScientificKey, int], dict[str, Any]] = {}
    for index, record in enumerate(failure_records):
        if not isinstance(record, Mapping):
            raise ArtifactContractError(f"Failure-ledger record {index} is not an object")
        key = _failure_key(record)
        if key not in expected_rows:
            raise ArtifactContractError(f"Failure ledger contains an unexpected key: {key}")
        attempt_id = record.get("attempt_id")
        if type(attempt_id) is not int or attempt_id not in {0, 1, 2}:
            raise ArtifactContractError("Failure ledger contains an invalid attempt ID")
        identity = (key, attempt_id)
        if identity in failures:
            raise ArtifactContractError(f"Duplicate failure-ledger record: {identity}")
        _require_exact_fields(
            record,
            {
                "task": key.task,
                "episode_id": key.episode_id,
                "arm": key.arm,
                "trajectory_kind": "formal",
                "classification": "infrastructure",
                "retry_allowed": True,
            },
            source=f"Failure-ledger record {index}",
        )
        recorded_row_id = record.get("formal_matrix_row_id")
        if recorded_row_id is not None and recorded_row_id != row_id_by_key[key]:
            raise ArtifactContractError("Failure ledger records the wrong formal row")
        recorded_actions_started = record.get("scientific_actions_started")
        if recorded_actions_started is not None and type(recorded_actions_started) is not bool:
            raise ArtifactContractError("Failure-ledger scientific_actions_started must be boolean when present")
        failures[identity] = dict(record)

    expected_attempts: set[tuple[ScientificKey, int]] = set()
    completed_attempt_by_key: dict[ScientificKey, int] = {}
    for key, result_path in completed.items():
        if key not in expected_rows:
            raise ArtifactContractError(f"Unexpected completed formal key: {key}")
        result = _load_json_object(result_path)
        attempt_id = result.get("attempt_id")
        if type(attempt_id) is not int or attempt_id not in {0, 1, 2}:
            raise ArtifactContractError(f"Completed result has an invalid attempt ID: {result_path}")
        completed_attempt_by_key[key] = attempt_id
        if result_path.parent.resolve() != store.attempt_dir(key, attempt_id).resolve():
            raise ArtifactContractError(f"Completed result path is non-canonical: {result_path}")
        expected_attempts.update((key, prior) for prior in range(attempt_id + 1))
        for prior in range(attempt_id):
            if (key, prior) not in failures:
                raise ArtifactContractError(f"Completed retry {key}/attempt_{attempt_id:02d} skips a failure record")
        if (key, attempt_id) in failures:
            raise ArtifactContractError("A completed attempt also has a failure-ledger record")

    if set(attempts) != expected_attempts:
        missing_attempts = sorted(expected_attempts - set(attempts))
        extra_attempts = sorted(set(attempts) - expected_attempts)
        raise ArtifactContractError(
            "Formal attempt directories do not match the exact retry chains: "
            f"missing={missing_attempts[:3]}, extra={extra_attempts[:3]}"
        )
    expected_failures = {
        (key, attempt_id)
        for key, completed_attempt in completed_attempt_by_key.items()
        for attempt_id in range(completed_attempt)
    }
    if set(failures) != expected_failures:
        raise ArtifactContractError("Failure ledger differs from the exact set of superseded attempts")

    failure_schema_counts: Counter[str] = Counter()
    failed_attempts_with_trace = 0
    for identity, failure in failures.items():
        key, attempt_id = identity
        path, manifest = attempts[identity]
        writer = EpisodeAttemptWriter(path, key, attempt_id)
        if writer.validate_resume() != "incomplete" or writer.result_path.exists():
            raise ArtifactContractError("Failure ledger points at a completed attempt")
        recorded_manifest_sha256 = failure.get("episode_manifest_sha256")
        if recorded_manifest_sha256 is not None and recorded_manifest_sha256 != sha256_file(writer.manifest_path):
            raise ArtifactContractError("Failure ledger manifest digest mismatch")
        recorded_slurm = failure.get("slurm")
        if recorded_slurm is not None and recorded_slurm != manifest.get("slurm"):
            raise ArtifactContractError("Failure ledger and failed manifest Slurm data differ")
        enriched_fields = {
            "formal_matrix_row_id",
            "scientific_actions_started",
            "episode_manifest_sha256",
            "slurm",
        }
        failure_schema_counts["launcher_enriched" if enriched_fields.issubset(failure) else "evaluator_minimal"] += 1
        trace_exists = writer.trace_path.exists()
        failed_attempts_with_trace += int(trace_exists)
        if (
            failure.get("scientific_actions_started") is not None
            and failure["scientific_actions_started"] != trace_exists
        ):
            raise ArtifactContractError("Failure-ledger scientific_actions_started disagrees with trace presence")

    retry_rows_by_attempt: dict[int, list[int]] = {0: list(range(FORMAL_TRAJECTORY_COUNT))}
    for attempt_id in (1, 2):
        retry_rows_by_attempt[attempt_id] = sorted(
            row_id_by_key[key] for key, prior_attempt in failures if prior_attempt == attempt_id - 1
        )
    submission_reports: list[dict[str, Any]] = []
    authorizations: dict[int, dict[int, dict[str, Any]]] = {}
    for attempt_id in (0, 1, 2):
        row_ids = retry_rows_by_attempt[attempt_id]
        plan_path = submission_plan_path(run_root, attempt_id)
        records = submission_paths(run_root, attempt_id)
        if not row_ids:
            if plan_path.exists() or records:
                raise ArtifactContractError(f"Unexpected empty retry submission state for attempt {attempt_id}")
            continue
        report, attempt_authorizations = validate_submission_attempt(
            run_root,
            attempt_id=attempt_id,
            expected_row_ids=row_ids,
            launch=launch,
        )
        submission_reports.append(report)
        authorizations[attempt_id] = attempt_authorizations

    for (key, attempt_id), (_, manifest) in attempts.items():
        row_id = row_id_by_key[key]
        try:
            authorization = authorizations[attempt_id][row_id]
        except KeyError as exc:
            raise ArtifactContractError(
                f"Attempt {key}/attempt_{attempt_id:02d} lacks submission authorization"
            ) from exc
        _validate_attempt_submission_binding(manifest, row_id=row_id, authorization=authorization)

    paired_manifests: list[dict[str, Any]] = []
    validated_records: dict[ScientificKey, dict[str, Any]] = {}
    audit_digest_records = []
    result_digest_records = []
    trace_digest_records = []
    latency_values = {arm: {"selector": [], "model": [], "end_to_end": []} for arm in ALL_ARMS}
    attempt_histogram: Counter[int] = Counter()
    terminal_histogram: dict[str, Counter[str]] = {arm: Counter() for arm in ALL_ARMS}
    collision_counts: Counter[str] = Counter()
    success_counts: Counter[str] = Counter()
    total_policy_calls: Counter[str] = Counter()
    for row_id in range(FORMAL_TRAJECTORY_COUNT):
        key = key_by_row_id[row_id]
        attempt_id = completed_attempt_by_key[key]
        path, _ = attempts[(key, attempt_id)]
        writer = EpisodeAttemptWriter(path, key, attempt_id)
        attempt_report, attempt_latencies, manifest, result = audit_formal_attempt(
            writer,
            expected_key=key,
            expected_row=expected_rows[key],
            launch=launch,
            seed_payload=seed_payload,
            seed_lookup=seed_lookup,
        )
        validated_records[key] = result
        manifest_with_initial_conditions = dict(manifest)
        manifest_with_initial_conditions["initial_condition_hashes"] = _load_json_object(writer.initial_conditions_path)
        paired_manifests.append(manifest_with_initial_conditions)
        attempt_histogram[attempt_id] += 1
        terminal_histogram[key.arm][str(result["terminal_reason"])] += 1
        collision_counts[key.arm] += int(result["collision"])
        success_counts[key.arm] += int(result["success"])
        total_policy_calls[key.arm] += int(attempt_report["policy_call_count"])
        for latency_kind, values in attempt_latencies.items():
            latency_values[key.arm][latency_kind].extend(values)

        result_sha256 = sha256_file(writer.result_path)
        trace_sha256 = sha256_file(writer.trace_path)
        audit_digest_records.append(
            {
                "row_id": row_id,
                "attempt_id": attempt_id,
                "policy_call_count": attempt_report["policy_call_count"],
                "result_sha256": result_sha256,
                "trace_sha256": trace_sha256,
            }
        )
        result_digest_records.append({"row_id": row_id, "attempt_id": attempt_id, "sha256": result_sha256})
        trace_digest_records.append({"row_id": row_id, "attempt_id": attempt_id, "sha256": trace_sha256})

    validate_formal_records(validated_records.values())
    fairness = audit_initial_condition_fairness(paired_manifests)
    manifest_invariants = audit_paired_manifest_invariants(paired_manifests)
    if fairness != {"paired_blocks": 800, "fair": True}:
        raise ArtifactContractError("Formal initial-condition fairness audit is incomplete")
    if manifest_invariants != {"paired_blocks": 800, "invariants_match": True}:
        raise ArtifactContractError("Formal paired-manifest invariant audit is incomplete")

    repository_provenance = _repository_provenance(
        repo_root, formal_commit=protocol_provenance["formal_evaluation_commit_sha"]
    )
    selector_latency_by_arm = {
        arm: {latency_kind: _latency_summary(values) for latency_kind, values in latency_values[arm].items()}
        for arm in ALL_ARMS
    }
    outcome_diagnostics = {
        "success_count_by_arm": {arm: success_counts[arm] for arm in ALL_ARMS},
        "terminal_reason_count_by_arm": {arm: dict(sorted(terminal_histogram[arm].items())) for arm in ALL_ARMS},
        "collision_count_by_arm": {arm: collision_counts[arm] for arm in ALL_ARMS},
        "policy_call_count_by_arm": {arm: total_policy_calls[arm] for arm in ALL_ARMS},
    }
    failure_ledger_sha256 = sha256_file(store.failures_path) if store.failures_path.exists() else None
    report = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "audited_utc": utc_now(),
        "passed": True,
        "formal_result_census_complete": True,
        "strict_selector_trace_audit_complete": True,
        "completeness": census,
        "initial_condition_fairness": fairness,
        "paired_manifest_invariants": manifest_invariants,
        "protocol_provenance": {
            **protocol_provenance,
            **repository_provenance,
        },
        "submission_audits": submission_reports,
        "attempt_count": len(attempts),
        "completed_attempt_histogram": {
            str(attempt_id): attempt_histogram[attempt_id] for attempt_id in sorted(attempt_histogram)
        },
        "infrastructure_failure_count": len(failure_records),
        "failure_record_schema_count": dict(sorted(failure_schema_counts.items())),
        "failed_attempts_with_selector_trace_count": failed_attempts_with_trace,
        "failure_ledger_sha256": failure_ledger_sha256,
        "retry_row_count_by_attempt": {
            str(attempt_id): len(rows) for attempt_id, rows in retry_rows_by_attempt.items() if rows
        },
        "result_set_sha256": hashlib.sha256(canonical_json_bytes(result_digest_records)).hexdigest(),
        "selector_trace_set_sha256": hashlib.sha256(canonical_json_bytes(trace_digest_records)).hexdigest(),
        "strict_attempt_audit_set_sha256": hashlib.sha256(canonical_json_bytes(audit_digest_records)).hexdigest(),
        "selector_latency_by_arm": selector_latency_by_arm,
        "outcome_diagnostics": outcome_diagnostics,
        "prior_exposure_manifest_entry_count": len(exposure.get("entries", [])),
        "protocol_deviations": [],
        "unresolved_risks": [],
    }
    return report, validated_records


def build_per_episode_rows(
    records: Mapping[ScientificKey, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in FORMAL_TASKS:
        for episode_id in FORMAL_EPISODE_IDS:
            by_arm = {arm: records[ScientificKey(task, episode_id, arm, "formal")] for arm in ALL_ARMS}
            row: dict[str, Any] = {"task": task, "episode_id": episode_id}
            for arm in ALL_ARMS:
                result = by_arm[arm]
                row[f"{arm}_success"] = int(result["success"])
                row[f"{arm}_terminal_reason"] = result["terminal_reason"]
                row[f"{arm}_steps"] = result["steps"]
                row[f"{arm}_collision"] = int(result["collision"])
                row[f"{arm}_attempt_id"] = result["attempt_id"]
            row.update(
                {
                    "OC_minus_U": int(by_arm["OC"]["success"]) - int(by_arm["U"]["success"]),
                    "O_minus_U": int(by_arm["O"]["success"]) - int(by_arm["U"]["success"]),
                    "OC_minus_O": int(by_arm["OC"]["success"]) - int(by_arm["O"]["success"]),
                    "R_minus_U": int(by_arm["R"]["success"]) - int(by_arm["U"]["success"]),
                    "OC_minus_R": int(by_arm["OC"]["success"]) - int(by_arm["R"]["success"]),
                }
            )
            rows.append(row)
    if len(rows) != len(FORMAL_TASKS) * len(FORMAL_EPISODE_IDS):
        raise AssertionError("Per-episode table must contain exactly 800 paired rows")
    return rows


def build_per_task_rows(
    records: Mapping[ScientificKey, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for task in FORMAL_TASKS:
        success_counts = {
            arm: sum(
                int(records[ScientificKey(task, episode_id, arm, "formal")]["success"])
                for episode_id in FORMAL_EPISODE_IDS
            )
            for arm in ALL_ARMS
        }
        rates = {arm: success_counts[arm] / len(FORMAL_EPISODE_IDS) for arm in ALL_ARMS}
        row: dict[str, Any] = {
            "task": task,
            "episode_count": len(FORMAL_EPISODE_IDS),
        }
        for arm in ALL_ARMS:
            row[f"{arm}_success_count"] = success_counts[arm]
            row[f"{arm}_success_rate"] = rates[arm]
        row.update(
            {
                "OC_minus_U_pp": 100.0 * (rates["OC"] - rates["U"]),
                "O_minus_U_pp": 100.0 * (rates["O"] - rates["U"]),
                "OC_minus_O_pp": 100.0 * (rates["OC"] - rates["O"]),
                "R_minus_U_pp": 100.0 * (rates["R"] - rates["U"]),
                "OC_minus_R_pp": 100.0 * (rates["OC"] - rates["R"]),
            }
        )
        rows.append(row)
    if len(rows) != len(FORMAL_TASKS):
        raise AssertionError("Per-task table must contain exactly 16 rows")
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ArtifactContractError("Cannot serialize an empty CSV table")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ArtifactContractError("CSV rows do not share one canonical schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _format_percentage(value: Any) -> str:
    return f"{float(value):.3f}"


def render_analysis_markdown(summary: Mapping[str, Any], completeness: Mapping[str, Any]) -> str:
    primary = summary["primary"]
    primary_ci = primary["confidence_interval_95"]
    decision = summary["decision_rule"]
    lines = [
        "# Formal keyframe selector analysis",
        "",
        "## Validity and provenance",
        "",
        (
            "The fail-closed audit passed for exactly 3,200 scientific cells "
            "(16 tasks x 50 episodes x 4 arms). Every selector call was replayed "
            "against the frozen arm definition and preregistered RandomSamp seed table."
        ),
        "",
        (
            f"The run contains {completeness['infrastructure_failure_count']} recorded "
            "infrastructure failures; each was superseded through its explicitly "
            "authorized fresh-reset retry chain. No protocol deviations or unresolved "
            "risks were detected."
        ),
        "",
        "## Primary comparison",
        "",
        "| Comparison | Estimate (pp) | 95% hierarchical bootstrap CI (pp) | Paired randomization p |",
        "|---|---:|---:|---:|",
        (
            f"| {primary['comparison']} | {_format_percentage(primary['estimate_pp'])} | "
            f"[{_format_percentage(primary_ci['lower_pp'])}, "
            f"{_format_percentage(primary_ci['upper_pp'])}] | "
            f"{float(primary['randomization_test']['p_value']):.6g} |"
        ),
        "",
        "The estimate is the equal-task-weighted paired success-rate difference.",
        "",
        "## Prespecified secondary comparisons",
        "",
        "| Comparison | Estimate (pp) | 95% CI (pp) | Raw p | Holm-adjusted p |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison, effect in summary["secondary"].items():
        interval = effect["confidence_interval_95"]
        randomization = effect["randomization_test"]
        lines.append(
            f"| {comparison} | {_format_percentage(effect['estimate_pp'])} | "
            f"[{_format_percentage(interval['lower_pp'])}, "
            f"{_format_percentage(interval['upper_pp'])}] | "
            f"{float(randomization['p_value']):.6g} | "
            f"{float(randomization['holm_adjusted_p_value']):.6g} |"
        )
    lines.extend(
        [
            "",
            "Holm adjustment is applied across exactly the four preregistered secondary comparisons.",
            "",
            "## Preregistered decision",
            "",
            f"Classification: `{summary['decision_classification']}`.",
            "",
            "Satisfied conditions: "
            + ", ".join(f"`{item}`" for item in decision["satisfied_prespecified_conclusions"])
            + ".",
            "",
            (
                f"The practical-effect threshold is "
                f"{decision['practical_effect_threshold_pp']:.1f} percentage points. "
                "The decision uses confidence-interval rules exactly as preregistered; "
                "p-values are not substituted into the GO/NO-GO rule."
            ),
            "",
            "## Frozen prior-exposure sensitivity",
            "",
        ]
    )
    sensitivity = summary.get("prior_exposure_sensitivity")
    if sensitivity is None:
        lines.append("No prior-exposure sensitivity manifest was supplied.")
    else:
        effect = sensitivity["effect"]
        interval = effect["confidence_interval_95"]
        lines.extend(
            [
                (f"Excluded {sensitivity['excluded_formal_block_count']} frozen task/episode blocks."),
                "",
                (
                    f"OC - U sensitivity estimate: "
                    f"{_format_percentage(effect['estimate_pp'])} pp; 95% CI "
                    f"[{_format_percentage(interval['lower_pp'])}, "
                    f"{_format_percentage(interval['upper_pp'])}] pp."
                ),
            ]
        )
    provenance = summary["artifact_provenance"]
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Formal evaluation commit: `{provenance['formal_evaluation_commit_sha']}`",
            f"- Aggregation commit: `{provenance['aggregation_commit_sha']}`",
            f"- Analysis seed: `{summary['analysis_seed']}`",
            f"- Bootstrap replicates: `{summary['bootstrap_replicates']}`",
            f"- Randomization replicates: `{summary['randomization_replicates']}`",
            f"- Result-set SHA-256: `{completeness['result_set_sha256']}`",
            f"- Selector-trace-set SHA-256: `{completeness['selector_trace_set_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_aggregate_payloads(
    completeness: dict[str, Any],
    records: Mapping[ScientificKey, Mapping[str, Any]],
    prior_exposure_manifest: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if completeness.get("passed") is not True:
        raise ArtifactContractError("Cannot aggregate a formal run that failed audit")
    analysis = build_point_analysis(records.values(), prior_exposure_manifest=prior_exposure_manifest)
    if (
        analysis.get("frozen_replicate_contract_met") is not True
        or analysis.get("inferential_analysis_status") != "complete_confirmatory"
        or analysis.get("formal_cell_count") != FORMAL_TRAJECTORY_COUNT
    ):
        raise ArtifactContractError("Frozen confirmatory analysis contract was not met")
    summary = {
        **analysis,
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "generated_utc": completeness["audited_utc"],
        "artifact_provenance": completeness["protocol_provenance"],
        "strict_audit": {
            "passed": True,
            "formal_result_census_complete": True,
            "strict_selector_trace_audit_complete": True,
            "result_set_sha256": completeness["result_set_sha256"],
            "selector_trace_set_sha256": completeness["selector_trace_set_sha256"],
            "infrastructure_failure_count": completeness["infrastructure_failure_count"],
            "protocol_deviations": completeness["protocol_deviations"],
            "unresolved_risks": completeness["unresolved_risks"],
        },
        "selector_latency_by_arm": completeness["selector_latency_by_arm"],
        "outcome_diagnostics": completeness["outcome_diagnostics"],
    }
    per_episode = build_per_episode_rows(records)
    per_task = build_per_task_rows(records)
    payloads = {
        "completeness_report.json": _json_bytes(completeness),
        "per_episode.csv": _csv_bytes(per_episode),
        "per_task.csv": _csv_bytes(per_task),
        "summary.json": _json_bytes(summary),
        "analysis.md": render_analysis_markdown(summary, completeness).encode("utf-8"),
    }
    if tuple(payloads) != AGGREGATE_FILENAMES:
        raise AssertionError("Aggregate payload order differs from the protocol contract")
    return payloads, summary


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_aggregate_directory(run_root: Path, payloads: Mapping[str, bytes]) -> Path:
    """Fsync all five files, then publish the directory once by atomic rename."""
    run_root = run_root.resolve()
    target = run_root / "aggregate"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable aggregate: {target}")
    if tuple(payloads) != AGGREGATE_FILENAMES:
        raise ArtifactContractError("Aggregate publication requires exactly five canonical files")
    with tempfile.TemporaryDirectory(prefix=".aggregate.", dir=run_root) as temporary:
        staging = Path(temporary)
        for filename in AGGREGATE_FILENAMES:
            data = payloads[filename]
            if not isinstance(data, bytes) or not data:
                raise ArtifactContractError(f"Aggregate payload is empty or non-bytes: {filename}")
            path = staging / filename
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(staging)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite immutable aggregate: {target}")
        os.rename(staging, target)
        _fsync_directory(run_root)
    return target


def aggregate_formal_run(run_root: Path, repo_root: Path) -> tuple[Path, dict[str, Any]]:
    completeness, records = audit_formal_run(run_root, repo_root)
    exposure = _load_json_object(run_root.resolve() / "protocol/prior_exposure_manifest.json")
    payloads, summary = build_aggregate_payloads(completeness, records, exposure)
    output = publish_aggregate_directory(run_root, payloads)
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    output, summary = aggregate_formal_run(args.run_root.resolve(), args.repo_root.resolve())
    print(
        json.dumps(
            {
                "aggregate": str(output),
                "decision_classification": summary["decision_classification"],
                "formal_cell_count": summary["formal_cell_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
