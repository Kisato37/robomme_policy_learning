#!/usr/bin/env python3
"""Fail-closed audit and aggregation for the isolated OC3/OC5 extension.

The completed U/O/OC/R experiment is immutable.  This module audits only the
1,600 new OC3/OC5 trajectories, verifies the published OC table by its frozen
digest, and then performs the three extension comparisons defined in the
extension protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import hashlib
import io
from itertools import pairwise
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np

from experiments.keyframe_neighborhood_sampling.analysis import DEFAULT_BOOTSTRAP_REPLICATES
from experiments.keyframe_neighborhood_sampling.analysis import DEFAULT_RANDOMIZATION_REPLICATES
from experiments.keyframe_neighborhood_sampling.analysis import REFERENCE_PER_EPISODE_SHA256
from experiments.keyframe_neighborhood_sampling.analysis import build_extension_analysis
from experiments.keyframe_neighborhood_sampling.analysis import load_published_reference_oc
from experiments.keyframe_neighborhood_sampling.architecture_smoke import validate_architecture_pass_report
from experiments.keyframe_neighborhood_sampling.direct_provenance import runner_backend
from experiments.keyframe_neighborhood_sampling.direct_provenance import validate_direct_attempt
from experiments.keyframe_neighborhood_sampling.direct_provenance import validate_direct_submission
from experiments.keyframe_neighborhood_sampling.formal_artifacts import AGGREGATOR_SOURCE_RELATIVE
from experiments.keyframe_neighborhood_sampling.formal_artifacts import ANALYSIS_SOURCE_RELATIVE
from experiments.keyframe_neighborhood_sampling.formal_artifacts import validate_extension_runtime_location
from experiments.keyframe_neighborhood_sampling.formal_artifacts import validate_frozen_formal_sources
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_DATASET
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_EPISODE_IDS
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_MAX_STEPS
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.formal_matrix import load_formal_matrix
from experiments.keyframe_neighborhood_sampling.record_launcher_failure import validate_extension_failure_record
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
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
from experiments.keyframe_oracle_sampling.artifacts import audit_attempt
from experiments.keyframe_oracle_sampling.artifacts import audit_initial_condition_fairness
from experiments.keyframe_oracle_sampling.artifacts import audit_paired_manifest_invariants
from experiments.keyframe_oracle_sampling.artifacts import canonical_json_bytes
from experiments.keyframe_oracle_sampling.artifacts import completeness_report
from experiments.keyframe_oracle_sampling.artifacts import expected_keys
from experiments.keyframe_oracle_sampling.artifacts import load_seed_table
from experiments.keyframe_oracle_sampling.artifacts import read_jsonl
from experiments.keyframe_oracle_sampling.artifacts import released_prepared_component_dtypes
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from experiments.keyframe_oracle_sampling.artifacts import validate_smoke_formal_seed_disjointness
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_POLICY_CALLS
from mme_vla_suite.shared.keyframe_oracle_sampling import oracle_neighborhood_coverage_decision

AGGREGATE_SCHEMA_VERSION = 1
EXTENSION_PROTOCOL_VERSION = "v1.0"
AGGREGATE_FILENAMES = (
    "completeness_report.json",
    "per_episode.csv",
    "per_task.csv",
    "summary.json",
    "analysis.md",
)
FORMAL_TERMINAL_REASONS = frozenset({"success", "fail", "timeout", "error"})
REFERENCE_RUN_ID = "20260829T231425Z_7b594786_formal_v1"
REFERENCE_RESULTS_RELATIVE = Path("results/keyframe_oracle_sampling/20260829T231425Z_7b594786_formal_v1")
REFERENCE_SUMMARY_SHA256 = "e3f40b7192fc1be9a3b280c30234170b63dfae26d389c9b99ba321e82e3efbe9"
REFERENCE_COMPLETENESS_SHA256 = "9f58713cb8da6762c3a81fcfa1776c25bb52a1be75ed26326325ce0cec655ccc"
SELECTOR_SEED_TABLE_ROLE = "randomsamp_policy_call_rng_audit_only"
REFERENCE_CROSS_RUN_VERIFICATION = {
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
}
_SHA256_HEX = frozenset("0123456789abcdef")


def _load_json_object(path: Path, *, source: str | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Cannot read {source or 'JSON artifact'} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError(f"{source or 'JSON artifact'} must be an object: {path}")
    return payload


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise ArtifactContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    source: str,
) -> None:
    mismatches = {
        field: {"expected": expected_value, "observed": payload.get(field)}
        for field, expected_value in expected.items()
        if field not in payload or payload.get(field) != expected_value
    }
    if mismatches:
        raise ArtifactContractError(
            f"{source} differs from the frozen extension contract: " + json.dumps(mismatches, sort_keys=True)
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


def _integer_distribution_summary(values: Sequence[int], *, field: str) -> dict[str, Any]:
    """Summarize an audited non-negative integer distribution without hiding emptiness."""
    normalized = list(values)
    if any(type(value) is not int or value < 0 for value in normalized):
        raise ArtifactContractError(f"{field} requires non-negative integer values")
    if not normalized:
        return {
            "count": 0,
            "sum": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(normalized, dtype=np.int64)
    return {
        "count": len(normalized),
        "sum": int(array.sum()),
        "mean": float(array.mean()),
        "min": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": int(array.max()),
    }


def _ratio_summary(numerator: int, denominator: int, *, field: str) -> dict[str, Any]:
    if type(numerator) is not int or type(denominator) is not int:
        raise ArtifactContractError(f"{field} counts must be integers")
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ArtifactContractError(f"{field} counts are invalid")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _empty_secondary_diagnostic_accumulator() -> dict[str, Any]:
    return {
        "episode_count": 0,
        "policy_call_count": 0,
        "boundary_recall_numerator": 0,
        "boundary_recall_denominator": 0,
        "requested_candidate_retention_numerator": 0,
        "requested_candidate_retention_denominator": 0,
        "effective_candidate_retention_numerator": 0,
        "effective_candidate_retention_denominator": 0,
        "memory_age_values": [],
        "selected_index_gap_values": [],
        "maximum_temporal_gap_values": [],
        "final_observed_boundary_count_values": [],
    }


def _merge_secondary_diagnostics(
    accumulator: dict[str, Any],
    episode_diagnostics: Mapping[str, Any],
) -> None:
    scalar_fields = (
        "episode_count",
        "policy_call_count",
        "boundary_recall_numerator",
        "boundary_recall_denominator",
        "requested_candidate_retention_numerator",
        "requested_candidate_retention_denominator",
        "effective_candidate_retention_numerator",
        "effective_candidate_retention_denominator",
    )
    list_fields = (
        "memory_age_values",
        "selected_index_gap_values",
        "maximum_temporal_gap_values",
        "final_observed_boundary_count_values",
    )
    for field in scalar_fields:
        value = episode_diagnostics.get(field)
        if type(value) is not int or value < 0:
            raise ArtifactContractError(f"Audited secondary diagnostic {field} is invalid")
        accumulator[field] += value
    for field in list_fields:
        values = episode_diagnostics.get(field)
        if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
            raise ArtifactContractError(f"Audited secondary diagnostic {field} is invalid")
        accumulator[field].extend(values)


def _finalize_secondary_diagnostics(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    """Turn replay-derived sufficient statistics into reportable diagnostics."""
    policy_call_count = accumulator.get("policy_call_count")
    episode_count = accumulator.get("episode_count")
    if type(policy_call_count) is not int or policy_call_count < 0:
        raise ArtifactContractError("Secondary diagnostic policy-call count is invalid")
    if type(episode_count) is not int or episode_count < 0:
        raise ArtifactContractError("Secondary diagnostic episode count is invalid")
    maximum_gaps = accumulator.get("maximum_temporal_gap_values")
    final_boundary_counts = accumulator.get("final_observed_boundary_count_values")
    if not isinstance(maximum_gaps, list) or len(maximum_gaps) != policy_call_count:
        raise ArtifactContractError("Maximum temporal gaps must contain one value per policy call")
    if not isinstance(final_boundary_counts, list) or len(final_boundary_counts) != episode_count:
        raise ArtifactContractError("Final observed boundary counts must contain one value per episode")
    return {
        "episode_count": episode_count,
        "policy_call_count": policy_call_count,
        "boundary_recall": _ratio_summary(
            accumulator["boundary_recall_numerator"],
            accumulator["boundary_recall_denominator"],
            field="boundary recall",
        ),
        "requested_neighborhood_candidate_retention": _ratio_summary(
            accumulator["requested_candidate_retention_numerator"],
            accumulator["requested_candidate_retention_denominator"],
            field="requested neighborhood candidate retention",
        ),
        "effective_neighborhood_candidate_retention": _ratio_summary(
            accumulator["effective_candidate_retention_numerator"],
            accumulator["effective_candidate_retention_denominator"],
            field="effective neighborhood candidate retention",
        ),
        "selected_index_spacing": _integer_distribution_summary(
            accumulator["selected_index_gap_values"], field="selected-index spacing"
        ),
        "memory_age": _integer_distribution_summary(
            accumulator["memory_age_values"], field="memory age"
        ),
        "maximum_temporal_gap_per_policy_call": _integer_distribution_summary(
            maximum_gaps, field="maximum temporal gap"
        ),
        "final_observed_boundary_count_progress_proxy": _integer_distribution_summary(
            final_boundary_counts, field="final observed boundary count"
        ),
    }


def _expected_formal_rows(
    matrix: Mapping[str, Any],
) -> dict[ScientificKey, dict[str, Any]]:
    rows = matrix.get("rows")
    if not isinstance(rows, list):
        raise ArtifactContractError("Extension formal matrix rows must be a list")
    expected: dict[ScientificKey, dict[str, Any]] = {}
    for position, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict) or raw_row.get("row_id") != position:
            raise ArtifactContractError("Extension formal matrix row IDs must be canonical and contiguous")
        key = ScientificKey(
            str(raw_row["task"]),
            int(raw_row["episode_id"]),
            str(raw_row["arm"]),
            str(raw_row["trajectory_kind"]),
        )
        if key in expected:
            raise ArtifactContractError(f"Duplicate extension matrix key: {key}")
        expected[key] = raw_row
    if len(expected) != FORMAL_TRAJECTORY_COUNT:
        raise ArtifactContractError("Extension formal matrix does not contain exactly 1,600 unique cells")
    frozen = expected_keys(
        FORMAL_TASKS,
        FORMAL_EPISODE_IDS,
        arms=EXTENSION_ARMS,
        trajectory_kind="formal",
    )
    if set(expected) != frozen:
        raise ArtifactContractError("Extension matrix is not the exact OC3/OC5 census")
    return expected


def _reference_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    root = repo_root.resolve() / REFERENCE_RESULTS_RELATIVE
    return (
        root / "aggregate/per_episode.csv",
        root / "aggregate/summary.json",
        root / "aggregate/completeness_report.json",
    )


def _validate_reference_alignment(
    repo_root: Path,
    *,
    launch: Mapping[str, Any],
    seed_entries_sha256: str,
) -> dict[str, Any]:
    per_episode_path, summary_path, completeness_path = _reference_paths(repo_root)
    if sha256_file(per_episode_path) != REFERENCE_PER_EPISODE_SHA256:
        raise ArtifactContractError("Published reference OC per_episode.csv digest mismatch")
    if sha256_file(summary_path) != REFERENCE_SUMMARY_SHA256:
        raise ArtifactContractError("Published reference OC summary.json digest mismatch")
    if sha256_file(completeness_path) != REFERENCE_COMPLETENESS_SHA256:
        raise ArtifactContractError("Published reference OC completeness_report.json digest mismatch")
    _require_exact_fields(
        launch,
        {
            "reference_run_id": REFERENCE_RUN_ID,
            "reference_results_relative": str(REFERENCE_RESULTS_RELATIVE),
            "reference_per_episode_sha256": REFERENCE_PER_EPISODE_SHA256,
            "reference_summary_sha256": REFERENCE_SUMMARY_SHA256,
            "reference_completeness_sha256": REFERENCE_COMPLETENESS_SHA256,
            "selector_seed_table_role": SELECTOR_SEED_TABLE_ROLE,
            "reference_cross_run_verification": REFERENCE_CROSS_RUN_VERIFICATION,
        },
        source="Extension reference binding",
    )
    summary = _load_json_object(summary_path, source="published reference summary")
    provenance = summary.get("artifact_provenance")
    if not isinstance(provenance, Mapping):
        raise ArtifactContractError("Published reference summary lacks provenance")
    if summary.get("protocol_version") != "v1.0" or summary.get("formal_cell_count") != 3200:
        raise ArtifactContractError("Published reference summary has the wrong protocol identity or census")
    strict_audit = summary.get("strict_audit")
    if not isinstance(strict_audit, Mapping) or strict_audit.get("passed") is not True:
        raise ArtifactContractError("Published reference summary is not a strict PASS")
    _require_exact_fields(
        provenance,
        {
            "checkpoint_id": SMOKE_CHECKPOINT_ID,
            "checkpoint_path": SMOKE_CHECKPOINT_PATH,
            "checkpoint_unpacked_metadata_sha256": launch.get("checkpoint_unpacked_metadata_sha256"),
            "checkpoint_content_tree_algorithm": launch.get("checkpoint_content_tree_algorithm"),
            "checkpoint_unpacked_content_tree_sha256": launch.get("checkpoint_unpacked_content_tree_sha256"),
            # This table was defined for RandomSamp policy-call RNG. Matching it
            # is reproducibility evidence for that selector contract only; it is
            # not evidence about simulator seeds or initial conditions.
            "formal_seed_table_entries_sha256": seed_entries_sha256,
        },
        source="Extension/reference selector-RNG and checkpoint alignment",
    )
    completeness = _load_json_object(completeness_path, source="published reference completeness report")
    if (
        completeness.get("formal_result_census_complete") is not True
        or completeness.get("strict_selector_trace_audit_complete") is not True
        or completeness.get("initial_condition_fairness") != {"paired_blocks": 800, "fair": True}
        or completeness.get("paired_manifest_invariants")
        != {"paired_blocks": 800, "invariants_match": True}
    ):
        raise ArtifactContractError("Published reference OC completeness evidence is not a strict PASS")
    return {
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_per_episode_sha256": REFERENCE_PER_EPISODE_SHA256,
        "reference_summary_sha256": REFERENCE_SUMMARY_SHA256,
        "reference_completeness_sha256": REFERENCE_COMPLETENESS_SHA256,
        "reference_protocol_identity_verified": True,
        "checkpoint_identity_match": True,
        "selector_seed_table": {
            "role": SELECTOR_SEED_TABLE_ROLE,
            "entries_sha256": seed_entries_sha256,
            "matches_published_reference": True,
            "is_environment_seed_or_initial_state_evidence": False,
        },
        "published_reference_internal_pairing_verified": {
            "paired_blocks": 800,
            "initial_condition_fair": True,
            "manifest_invariants_match": True,
        },
        "cross_run_pairing": {
            "pairing_key": ["task", "episode_id"],
            "status": "not_directly_reauditable_from_published_aggregate",
            "environment_seed_directly_verified": False,
            "difficulty_directly_verified": False,
            "raw_initial_condition_hashes_directly_verified": False,
        },
    }


def validate_extension_protocol_bundle(
    run_root: Path,
    repo_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, int, int], int],
    dict[str, Any],
]:
    """Validate immutable launch inputs without depending on a live formal checkout."""
    protocol_dir = run_root.resolve() / "protocol"
    paths = {
        "manifest": protocol_dir / "launch_manifest.json",
        "protocol": protocol_dir / "protocol_snapshot.md",
        "protocol_sidecar": protocol_dir / "protocol_sha256.txt",
        "seed": protocol_dir / "seed_table.json",
        "development_seed": protocol_dir / "development_seed_audit_table.json",
        "matrix": protocol_dir / "formal_matrix.json",
        "architecture_report": protocol_dir / "architecture_pass_report.json",
        "architecture_submission": protocol_dir / "architecture_submission_record.json",
        "development_smoke": protocol_dir / "development_smoke_audit.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ArtifactContractError(f"Extension protocol bundle is incomplete: {missing}")

    launch = _load_json_object(paths["manifest"], source="extension launch manifest")
    _require_exact_fields(
        launch,
        {
            "protocol_version": EXTENSION_PROTOCOL_VERSION,
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "run_kind": "formal",
            "dataset": FORMAL_DATASET,
            "formal_launch_authorized": True,
            "trajectory_count": FORMAL_TRAJECTORY_COUNT,
            "max_steps": FORMAL_MAX_STEPS,
            "executed_action_horizon": SMOKE_EXECUTED_ACTION_HORIZON,
            "evaluation_policy_seed": SMOKE_EVALUATION_POLICY_SEED,
            "checkpoint_path": SMOKE_CHECKPOINT_PATH,
            "selector_seed_table_role": SELECTOR_SEED_TABLE_ROLE,
            "reference_cross_run_verification": REFERENCE_CROSS_RUN_VERIFICATION,
            "frozen_analysis_source_relative": ANALYSIS_SOURCE_RELATIVE.as_posix(),
            "frozen_aggregator_source_relative": AGGREGATOR_SOURCE_RELATIVE.as_posix(),
        },
        source="Extension launch manifest",
    )
    digest_bindings = {
        "protocol_sha256": paths["protocol"],
        "seed_table_file_sha256": paths["seed"],
        "development_seed_audit_file_sha256": paths["development_seed"],
        "formal_matrix_sha256": paths["matrix"],
        "architecture_report_sha256": paths["architecture_report"],
        "architecture_submission_record_sha256": paths["architecture_submission"],
        "development_smoke_audit_sha256": paths["development_smoke"],
    }
    for field, path in digest_bindings.items():
        if launch.get(field) != sha256_file(path):
            raise ArtifactContractError(f"Extension launch digest mismatch for {path.name}")
    if paths["protocol_sidecar"].read_text().strip() != launch["protocol_sha256"]:
        raise ArtifactContractError("Extension protocol digest sidecar mismatch")

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
        raise ArtifactContractError("Extension seed table is not the exact 16 x 50 x 82 universe")
    development_seed_payload, development_seed_lookup = load_seed_table(paths["development_seed"])
    expected_development_keys = {
        (task, 0, call_index) for task in FORMAL_TASKS for call_index in range(MAX_POLICY_CALLS)
    }
    if set(development_seed_lookup) != expected_development_keys:
        raise ArtifactContractError("Extension development-seed table is not the exact 16 x 1 x 82 universe")
    seed_disjointness = validate_smoke_formal_seed_disjointness(development_seed_payload, seed_payload)
    _require_exact_fields(
        launch,
        {
            "seed_table_scope": seed_payload["scope"],
            "seed_table_dataset": seed_payload["dataset"],
            "seed_table_derivation": seed_payload["derivation"],
            "seed_table_entries_sha256": seed_payload["entries_sha256"],
            "seed_disjointness_audit": seed_disjointness,
        },
        source="Extension seed provenance",
    )

    matrix = load_formal_matrix(paths["matrix"])
    _expected_formal_rows(matrix)
    if launch.get("matrix") != matrix["rows"]:
        raise ArtifactContractError("Extension launch manifest differs from its matrix")

    repository_commit = launch.get("repository", {}).get("commit_sha")
    if (
        not isinstance(repository_commit, str)
        or len(repository_commit) != 40
        or any(character not in _SHA256_HEX for character in repository_commit)
    ):
        raise ArtifactContractError("Extension launch manifest lacks a valid commit SHA")
    environment_digest = _require_sha256(launch.get("environment_lock_sha256"), field="environment_lock_sha256")
    environment_locks = launch.get("environment_locks")
    if not isinstance(environment_locks, Mapping):
        raise ArtifactContractError("Extension launch manifest lacks environment locks")
    if environment_locks.get("policy_uv_lock_sha256") != environment_digest:
        raise ArtifactContractError("Extension policy lock digest binding mismatch")
    for field in (
        "checkpoint_unpacked_metadata_sha256",
        "checkpoint_unpacked_content_tree_sha256",
        "frozen_analysis_source_sha256",
        "frozen_aggregator_source_sha256",
    ):
        _require_sha256(launch.get(field), field=field)
    if not launch.get("checkpoint_content_tree_algorithm"):
        raise ArtifactContractError("Extension checkpoint content-tree algorithm is missing")

    for label, path in (
        ("architecture report", paths["architecture_report"]),
        ("architecture submission", paths["architecture_submission"]),
        ("development-smoke audit", paths["development_smoke"]),
    ):
        payload = _load_json_object(path, source=f"extension {label}")
        if runner_backend(payload) != runner_backend(launch):
            raise ArtifactContractError("Formal launch and its smoke evidence use different runner backends")
        _require_exact_fields(
            payload,
            {
                "protocol_version": EXTENSION_PROTOCOL_VERSION,
                "protocol_family": EXTENSION_PROTOCOL_FAMILY,
                "repository_commit_sha": repository_commit,
            },
            source=f"Extension {label}",
        )
        if label != "architecture submission" and payload.get("passed") is not True:
            raise ArtifactContractError(f"Extension {label} is not a PASS artifact")
        for field in (
            "checkpoint_unpacked_metadata_sha256",
            "checkpoint_content_tree_algorithm",
            "checkpoint_unpacked_content_tree_sha256",
        ):
            observed = payload.get(field)
            if observed is not None and observed != launch.get(field):
                raise ArtifactContractError(f"Extension {label} differs from launch manifest on {field}")

    if runner_backend(launch) == "direct":
        architecture = _load_json_object(paths["architecture_report"])
        validate_architecture_pass_report(
            architecture, run_root=Path(architecture["run_root"]),
            architecture_submission=_load_json_object(paths["architecture_submission"]),
        )
    reference_alignment = _validate_reference_alignment(
        repo_root,
        launch=launch,
        seed_entries_sha256=str(seed_payload["entries_sha256"]),
    )
    provenance = {
        "launch_manifest_sha256": sha256_file(paths["manifest"]),
        "protocol_sha256": launch["protocol_sha256"],
        "formal_matrix_sha256": launch["formal_matrix_sha256"],
        "formal_seed_table_file_sha256": launch["seed_table_file_sha256"],
        "formal_seed_table_entries_sha256": seed_payload["entries_sha256"],
        "architecture_report_sha256": launch["architecture_report_sha256"],
        "development_smoke_audit_sha256": launch["development_smoke_audit_sha256"],
        "formal_evaluation_commit_sha": repository_commit,
        "selector_seed_table_role": SELECTOR_SEED_TABLE_ROLE,
        "frozen_analysis_source_relative": launch["frozen_analysis_source_relative"],
        "frozen_analysis_source_sha256": launch["frozen_analysis_source_sha256"],
        "frozen_aggregator_source_relative": launch["frozen_aggregator_source_relative"],
        "frozen_aggregator_source_sha256": launch["frozen_aggregator_source_sha256"],
        "environment_lock_sha256": environment_digest,
        "checkpoint_id": SMOKE_CHECKPOINT_ID,
        "checkpoint_path": launch["checkpoint_path"],
        "checkpoint_unpacked_metadata_sha256": launch["checkpoint_unpacked_metadata_sha256"],
        "checkpoint_content_tree_algorithm": launch["checkpoint_content_tree_algorithm"],
        "checkpoint_unpacked_content_tree_sha256": launch["checkpoint_unpacked_content_tree_sha256"],
        "seed_disjointness_audit": seed_disjointness,
        "reference_alignment": reference_alignment,
    }
    return launch, matrix, seed_payload, seed_lookup, provenance


def _integer_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ArtifactContractError(f"{field} must be an integer list")
    if len(value) != len(set(value)):
        raise ArtifactContractError(f"{field} contains duplicates")
    return value


def _submission_plan_path(run_root: Path, attempt_id: int) -> Path:
    name = "submission_plan.json" if attempt_id == 0 else f"submission_plan_attempt_{attempt_id:02d}.json"
    return run_root / "protocol" / name


def _submission_path(run_root: Path, attempt_id: int, shard_id: int) -> Path:
    name = (
        f"submission_record_shard_{shard_id:02d}.json"
        if attempt_id == 0
        else f"submission_record_attempt_{attempt_id:02d}_shard_{shard_id:02d}.json"
    )
    return run_root / "protocol" / name


def _submission_paths(run_root: Path, attempt_id: int) -> list[Path]:
    pattern = (
        "submission_record_shard_*.json"
        if attempt_id == 0
        else f"submission_record_attempt_{attempt_id:02d}_shard_*.json"
    )
    return sorted((run_root / "protocol").glob(pattern))


def validate_submission_attempt(
    run_root: Path,
    *,
    attempt_id: int,
    expected_row_ids: Sequence[int],
    launch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Validate one extension submission plan and its exact Slurm shards."""
    plan_path = _submission_plan_path(run_root, attempt_id)
    if not plan_path.is_file():
        raise ArtifactContractError(f"Missing extension submission plan: {plan_path}")
    plan = _load_json_object(plan_path, source="extension submission plan")
    backend = runner_backend(plan)
    if backend != runner_backend(launch):
        raise ArtifactContractError("Formal plan and launch runner backends differ")
    if backend == "direct":
        validate_direct_submission(plan, stage="formal")
    repository_commit = launch["repository"]["commit_sha"]
    _require_exact_fields(
        plan,
        {
            "schema_version": 1,
            "protocol_version": EXTENSION_PROTOCOL_VERSION,
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "attempt_id": attempt_id,
            "repository_commit_sha": repository_commit,
            "run_root": str(run_root.resolve()),
            "formal_launch_authorized": True,
            "max_rows_per_array": 1000,
        },
        source=f"Extension submission plan attempt {attempt_id}",
    )
    shards = plan.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ArtifactContractError("Extension submission plan must contain shards")
    if plan.get("shard_count") != len(shards):
        raise ArtifactContractError("Extension submission shard count mismatch")
    if plan.get("global_max_concurrent") not in {1, 2, 3, 4}:
        raise ArtifactContractError("Extension submission concurrency is invalid")

    planned_rows: list[int] = []
    shard_by_id: dict[int, Mapping[str, Any]] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ArtifactContractError("Extension submission contains a non-object shard")
        shard_id = shard.get("shard_id")
        if type(shard_id) is not int or shard_id in shard_by_id:
            raise ArtifactContractError("Extension shard ID is invalid or duplicated")
        row_ids = _integer_list(shard.get("row_ids"), field="extension shard row_ids")
        array_ids = _integer_list(shard.get("array_task_ids"), field="extension shard array_task_ids")
        if not row_ids or len(row_ids) > 1000:
            raise ArtifactContractError("Extension shard must contain 1..1,000 rows")
        if array_ids != list(range(len(row_ids))):
            raise ArtifactContractError("Extension shard local array IDs are non-canonical")
        if shard.get("shard_count") != len(shards):
            raise ArtifactContractError("Extension shard records the wrong shard count")
        shard_by_id[shard_id] = shard
        planned_rows.extend(row_ids)
    if sorted(shard_by_id) != list(range(len(shards))):
        raise ArtifactContractError("Extension shard IDs must be contiguous from zero")
    if len(planned_rows) != len(set(planned_rows)):
        raise ArtifactContractError("Extension submission assigns a row twice")
    if plan.get("trajectory_count") != len(planned_rows):
        raise ArtifactContractError("Extension submission trajectory count mismatch")
    if sorted(planned_rows) != sorted(expected_row_ids):
        raise ArtifactContractError(f"Extension attempt {attempt_id} does not authorize the exact expected rows")

    record_paths = _submission_paths(run_root, attempt_id)
    if len(record_paths) != len(shards):
        raise ArtifactContractError("Extension submission records do not match its shards")
    plan_sha256 = sha256_file(plan_path)
    row_authorizations: dict[int, dict[str, Any]] = {}
    job_ids: list[str] = []
    record_digests = []
    for record_path in record_paths:
        record = _load_json_object(record_path, source="extension submission record")
        if runner_backend(record) != backend:
            raise ArtifactContractError("Formal plan and shard runner backends differ")
        if backend == "direct":
            validate_direct_submission(record, stage="formal")
            for field in ("gpu_pairs", "runtime_profile"):
                if record.get(field) != plan.get(field):
                    raise ArtifactContractError(f"Direct shard differs from its plan on {field}")
        shard_id = record.get("shard_id")
        if type(shard_id) is not int or shard_id not in shard_by_id:
            raise ArtifactContractError("Extension submission record has unknown shard")
        if record_path != _submission_path(run_root, attempt_id, shard_id):
            raise ArtifactContractError("Extension submission record path is non-canonical")
        planned = shard_by_id[shard_id]
        _require_exact_fields(
            record,
            {
                "schema_version": 1,
                "protocol_version": EXTENSION_PROTOCOL_VERSION,
                "protocol_family": EXTENSION_PROTOCOL_FAMILY,
                "attempt_id": attempt_id,
                "shard_id": shard_id,
                "repository_commit_sha": repository_commit,
                "run_root": str(run_root.resolve()),
                "formal_launch_authorized": True,
                "launch_manifest_sha256": sha256_file(run_root / "protocol/launch_manifest.json"),
                "formal_matrix_sha256": launch["formal_matrix_sha256"],
                "architecture_report_sha256": launch["architecture_report_sha256"],
                "development_smoke_audit_sha256": launch["development_smoke_audit_sha256"],
                "checkpoint_unpacked_metadata_sha256": launch["checkpoint_unpacked_metadata_sha256"],
                "checkpoint_content_tree_algorithm": launch["checkpoint_content_tree_algorithm"],
                "checkpoint_unpacked_content_tree_sha256": launch["checkpoint_unpacked_content_tree_sha256"],
                "submission_plan_sha256": plan_sha256,
                "shard_count": len(shards),
                "global_max_concurrent": plan["global_max_concurrent"],
            },
            source=f"Extension submission record attempt {attempt_id}/{shard_id}",
        )
        for field in ("array_task_ids", "row_ids", "array", "command", "shard_count"):
            if record.get(field) != planned.get(field):
                raise ArtifactContractError(f"Extension submission record differs from plan on {field}")
        row_ids = _integer_list(record.get("row_ids"), field="extension record rows")
        array_ids = _integer_list(record.get("array_task_ids"), field="extension record array IDs")
        if record.get("trajectory_count") != len(row_ids):
            raise ArtifactContractError("Extension submission record count mismatch")
        identity_field = "direct_run_id" if backend == "direct" else "slurm_array_job_id"
        job_id = record.get(identity_field)
        if not isinstance(job_id, str) or not job_id:
            raise ArtifactContractError("Extension submission record lacks a Slurm job ID")
        job_ids.append(job_id)
        record_digests.append({"shard_id": shard_id, "sha256": sha256_file(record_path)})
        for local_id, row_id in zip(array_ids, row_ids, strict=True):
            if row_id in row_authorizations:
                raise ArtifactContractError("Extension row appears in two submission records")
            row_authorizations[row_id] = {
                "attempt_id": attempt_id,
                "shard_id": shard_id,
                "array_task_id": local_id,
                **({"runner_backend": "direct", "direct_run_id": job_id} if backend == "direct" else {"slurm_array_job_id": job_id}),
                "submission_record_sha256": sha256_file(record_path),
            }
    return (
        {
            "attempt_id": attempt_id,
            "trajectory_count": len(planned_rows),
            "shard_count": len(shards),
            **({"runner_backend": "direct", "direct_run_ids": job_ids} if backend == "direct" else {"slurm_array_job_ids": job_ids}),
            "submission_plan_sha256": plan_sha256,
            "submission_record_digests": sorted(record_digests, key=lambda item: item["shard_id"]),
            "row_ids_sha256": hashlib.sha256(canonical_json_bytes(sorted(planned_rows))).hexdigest(),
        },
        row_authorizations,
    )


def _scientific_key(raw: Any, *, source: str) -> ScientificKey:
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
    run_root: Path,
    store: RunArtifactStore,
) -> dict[tuple[ScientificKey, int], tuple[Path, dict[str, Any]]]:
    attempts: dict[tuple[ScientificKey, int], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(run_root.glob("trajectories/**/attempt_*")):
        if not path.is_dir() or path.is_symlink():
            raise ArtifactContractError(f"Extension attempt path is invalid: {path}")
        manifest_path = path / "episode_manifest.json"
        if not manifest_path.is_file():
            raise ArtifactContractError(f"Extension attempt lacks a manifest: {path}")
        manifest = _load_json_object(manifest_path)
        key = _scientific_key(manifest.get("scientific_key"), source=str(manifest_path))
        attempt_id = manifest.get("attempt_id")
        if type(attempt_id) is not int or attempt_id not in {0, 1, 2}:
            raise ArtifactContractError(f"Extension attempt ID is invalid: {path}")
        if path.resolve() != store.attempt_dir(key, attempt_id).resolve():
            raise ArtifactContractError(f"Extension attempt directory is non-canonical: {path}")
        identity = (key, attempt_id)
        if identity in attempts:
            raise ArtifactContractError(f"Duplicate extension attempt: {identity}")
        attempts[identity] = (path, manifest)
    return attempts


def _validate_attempt_submission_binding(
    manifest: Mapping[str, Any],
    *,
    row_id: int,
    authorization: Mapping[str, Any],
    run_root: Path,
    validated_readiness_failure: bool = False,
) -> None:
    if runner_backend(manifest) != runner_backend(authorization):
        raise ArtifactContractError("Formal attempt and submission runner backends differ")
    if runner_backend(authorization) == "direct":
        roles = {"preflight", "policy", "reconcile"}
        if not validated_readiness_failure:
            roles.add("evaluator")
        elif manifest.get("execution_phase") != "policy_server_readiness" or manifest.get("launcher_failure_only") is not True:
            raise ArtifactContractError("Direct readiness lifecycle exemption lacks launcher-only provenance")
        dispatch = validate_direct_attempt(
            manifest, run_root, attempt_id=authorization["attempt_id"], row_id=row_id,
            trajectory_kind="formal", required_roles=roles,
        )
        if (
            dispatch.submission_plan_sha256 != authorization["submission_record_sha256"]
            or dispatch.shard_id != authorization["shard_id"]
        ):
            raise ArtifactContractError("Direct formal attempt differs from its exact submission authorization")
        return
    slurm = manifest.get("slurm")
    if not isinstance(slurm, Mapping):
        raise ArtifactContractError("Extension attempt manifest lacks Slurm provenance")
    expected = {
        "array_job_id": str(authorization["slurm_array_job_id"]),
        "array_task_id": str(authorization["array_task_id"]),
    }
    for field, value in expected.items():
        if slurm.get(field) != value:
            raise ArtifactContractError(f"Extension attempt Slurm {field} differs from its submission")
    if slurm.get("formal_matrix_row_id") not in {None, str(row_id)}:
        raise ArtifactContractError("Extension attempt records the wrong matrix row")
    if manifest.get("formal_matrix_row_id") not in {None, row_id}:
        raise ArtifactContractError("Extension launcher records the wrong matrix row")


def audit_extension_attempt(
    writer: EpisodeAttemptWriter,
    *,
    expected_key: ScientificKey,
    expected_row: Mapping[str, Any],
    launch: Mapping[str, Any],
    seed_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, Any], dict[str, Any]]:
    """Strictly replay one OC3/OC5 selector trace and validate its result."""
    if writer.key != expected_key or expected_key.arm not in EXTENSION_ARMS:
        raise ArtifactContractError("Attempt writer is bound to the wrong extension key")
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
        source="Extension formal matrix row",
    )
    if type(expected_row.get("row_id")) is not int:
        raise ArtifactContractError("Extension matrix row_id must be an integer")
    if writer.validate_resume() != "complete":
        raise ArtifactContractError(f"Extension attempt is not complete: {writer.attempt_dir}")

    generic_report = audit_attempt(writer)
    manifest = _load_json_object(writer.manifest_path)
    if runner_backend(manifest) == "direct" and manifest.get("environment_setup_completed") is not True:
        raise ArtifactContractError("Completed direct formal result lacks confirmed simulator setup")
    result = _load_json_object(writer.result_path)
    initial_conditions = _load_json_object(writer.initial_conditions_path)
    traces = read_jsonl(writer.trace_path)
    seed_digest = _require_sha256(seed_payload.get("entries_sha256"), field="extension seed entries")
    _require_exact_fields(
        launch,
        {
            "protocol_version": EXTENSION_PROTOCOL_VERSION,
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "dataset": FORMAL_DATASET,
            "trajectory_count": FORMAL_TRAJECTORY_COUNT,
            "max_steps": FORMAL_MAX_STEPS,
            "executed_action_horizon": SMOKE_EXECUTED_ACTION_HORIZON,
            "evaluation_policy_seed": SMOKE_EVALUATION_POLICY_SEED,
            "checkpoint_path": SMOKE_CHECKPOINT_PATH,
            "seed_table_scope": FORMAL_SEED_SCOPE,
            "seed_table_dataset": FORMAL_SEED_DATASET,
            "seed_table_entries_sha256": seed_digest,
        },
        source="Validated extension launch manifest",
    )
    if set(initial_conditions) != set(SMOKE_INITIAL_CONDITION_HASH_FIELDS):
        raise ArtifactContractError("Extension attempt must record the exact initial-condition hashes")
    for field in SMOKE_INITIAL_CONDITION_HASH_FIELDS:
        _require_sha256(initial_conditions.get(field), field=f"initial condition {field}")

    fixed = {
        "dataset": FORMAL_DATASET,
        "max_steps": FORMAL_MAX_STEPS,
        "executed_action_horizon": SMOKE_EXECUTED_ACTION_HORIZON,
        "evaluation_policy_seed": SMOKE_EVALUATION_POLICY_SEED,
        "checkpoint_id": SMOKE_CHECKPOINT_ID,
    }
    _require_exact_fields(
        manifest,
        {
            **fixed,
            "protocol_version": EXTENSION_PROTOCOL_VERSION,
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
            "seed_table_sha256": seed_digest,
        },
        source="Extension episode manifest",
    )
    _require_exact_fields(result, fixed, source="Extension episode result")
    _require_exact_fields(
        result,
        {
            "task": expected_key.task,
            "episode_id": expected_key.episode_id,
            "selector_arm": expected_key.arm,
        },
        source="Extension episode result",
    )
    terminal_reason = result.get("terminal_reason")
    if terminal_reason not in FORMAL_TERMINAL_REASONS:
        raise ArtifactContractError("Extension result lacks a valid terminal outcome")
    if type(result.get("success")) is not bool or result["success"] != (terminal_reason == "success"):
        raise ArtifactContractError("Extension success disagrees with terminal reason")
    if type(result.get("timeout")) is not bool or result["timeout"] != (terminal_reason == "timeout"):
        raise ArtifactContractError("Extension timeout disagrees with terminal reason")
    if type(result.get("collision")) is not bool:
        raise ArtifactContractError("Extension collision must be boolean")
    if terminal_reason == "error" and (
        not result.get("benchmark_error_message") or not result.get("benchmark_exception_type")
    ):
        raise ArtifactContractError("Extension error outcome lacks benchmark evidence")

    steps = result.get("steps")
    if type(steps) is not int or not 1 <= steps <= FORMAL_MAX_STEPS:
        raise ArtifactContractError("Extension result has an invalid step count")
    expected_calls = math.ceil(steps / SMOKE_EXECUTED_ACTION_HORIZON)
    if len(traces) != expected_calls:
        raise ArtifactContractError("Extension trace count must equal ceil(result.steps / 16)")
    for field in (
        "policy_latency_ms",
        "policy_model_latency_ms",
        "history_lengths_at_policy_calls",
    ):
        values = result.get(field)
        if not isinstance(values, list) or len(values) != expected_calls:
            raise ArtifactContractError(f"Extension result {field} must have one value per policy call")

    required_hashes = (
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
    latency_values = {"selector": [], "model": [], "end_to_end": []}
    initial_history_length: int | None = None
    fallback_calls = 0
    secondary_thinning_calls = 0
    boundary_core_thinned_calls = 0
    requested_candidate_counts: list[int] = []
    effective_candidate_counts: list[int] = []
    selected_frame_counts: list[int] = []
    boundary_recall_numerator = 0
    boundary_recall_denominator = 0
    requested_candidate_retention_numerator = 0
    effective_candidate_retention_numerator = 0
    memory_age_values: list[int] = []
    selected_index_gap_values: list[int] = []
    maximum_temporal_gap_values: list[int] = []
    final_observed_boundary_count = 0
    requested_frames = 3 if expected_key.arm == "OC3" else 5
    for call_index, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            raise ArtifactContractError("Extension trace contains a non-object")
        _require_exact_fields(
            trace,
            {
                "schema_version": 1,
                "task": expected_key.task,
                "episode_id": expected_key.episode_id,
                "selector_name": expected_key.arm,
                "selector_seed": None,
                "seed_table_sha256": seed_digest,
                "seed_table_scope": FORMAL_SEED_SCOPE,
                "seed_table_dataset": FORMAL_SEED_DATASET,
                "policy_call_index": call_index,
                "environment_step": call_index * SMOKE_EXECUTED_ACTION_HORIZON,
            },
            source=f"Extension selector trace call {call_index}",
        )
        current_index = trace.get("current_history_index")
        history_length = trace.get("history_length")
        if type(current_index) is not int or current_index < 0:
            raise ArtifactContractError("Extension current history index is invalid")
        if type(history_length) is not int or history_length != current_index + 1:
            raise ArtifactContractError("Extension history length/index mismatch")
        if initial_history_length is None:
            initial_history_length = history_length
        if history_length != initial_history_length + trace["environment_step"]:
            raise ArtifactContractError("Extension history cadence changed")

        raw_boundaries = trace.get("visible_boundary_indices")
        if not isinstance(raw_boundaries, list) or any(type(value) is not int for value in raw_boundaries):
            raise ArtifactContractError("Extension visible boundaries must be integers")
        boundaries = list(raw_boundaries)
        if (
            boundaries != sorted(set(boundaries))
            or not boundaries
            or boundaries[0] != 0
            or any(value < 0 or value > current_index for value in boundaries)
        ):
            raise ArtifactContractError("Extension visible boundaries must be sorted, causal, and include zero")
        expected_selected, expected_decision = oracle_neighborhood_coverage_decision(
            current_index,
            boundaries,
            neighborhood_frames=requested_frames,
        )
        if trace.get("selected_frame_indices") != expected_selected:
            raise ArtifactContractError(f"Extension trace call {call_index} does not match {expected_key.arm}")
        for field, expected_value in expected_decision.items():
            if trace.get(field) != expected_value:
                raise ArtifactContractError(f"Extension trace call {call_index} has wrong selector diagnostic {field}")
        expected_ages = [current_index - index for index in expected_selected]
        expected_gaps = [right - left for left, right in pairwise(expected_selected)]
        expected_maximum_gap = max(expected_gaps, default=0)
        selected_set = set(expected_selected)
        expected_selected_boundary_count = len(selected_set.intersection(boundaries))
        expected_boundary_recall = expected_selected_boundary_count / len(boundaries)
        observed_ages = trace.get("age_distribution")
        if (
            not isinstance(observed_ages, list)
            or any(type(value) is not int for value in observed_ages)
            or observed_ages != expected_ages
        ):
            raise ArtifactContractError(
                f"Extension trace call {call_index} age distribution was not replay-derived"
            )
        observed_maximum_gap = trace.get("maximum_temporal_gap")
        if type(observed_maximum_gap) is not int or observed_maximum_gap != expected_maximum_gap:
            raise ArtifactContractError(
                f"Extension trace call {call_index} maximum temporal gap was not replay-derived"
            )
        observed_boundary_recall = trace.get("boundary_recall")
        if (
            isinstance(observed_boundary_recall, bool)
            or not isinstance(observed_boundary_recall, int | float)
            or not math.isfinite(float(observed_boundary_recall))
            or float(observed_boundary_recall) != expected_boundary_recall
        ):
            raise ArtifactContractError(
                f"Extension trace call {call_index} boundary recall was not replay-derived"
            )
        fallback_calls += int(bool(expected_decision["oc5_fell_back_to_oc3"]))
        secondary_thinning_calls += int(bool(expected_decision["oc3_secondary_thinning"]))
        boundary_core_thinned_calls += int(bool(expected_decision["boundary_core_thinned"]))
        requested_candidates = expected_decision["causal_neighborhood_candidates"]
        effective_candidates = expected_decision["effective_neighborhood_candidates"]
        requested_candidate_counts.append(len(requested_candidates))
        effective_candidate_counts.append(len(effective_candidates))
        selected_frame_counts.append(len(expected_selected))
        boundary_recall_numerator += expected_selected_boundary_count
        boundary_recall_denominator += len(boundaries)
        requested_candidate_retention_numerator += len(selected_set.intersection(requested_candidates))
        effective_candidate_retention_numerator += len(selected_set.intersection(effective_candidates))
        memory_age_values.extend(expected_ages)
        selected_index_gap_values.extend(expected_gaps)
        maximum_temporal_gap_values.append(expected_maximum_gap)
        final_observed_boundary_count = len(boundaries)

        if (
            trace.get("mask_shape") != [512]
            or trace.get("mask_dtype") != "bool"
            or trace.get("mask_valid_prefix_all_true") is not True
            or trace.get("mask_padding_all_false") is not True
        ):
            raise ArtifactContractError("Extension memory mask contract changed")
        if trace.get("prepared_memory_component_shapes") != [list(shape) for shape in SMOKE_PREPARED_COMPONENT_SHAPES]:
            raise ArtifactContractError("Extension prepared-memory shapes changed")
        observed_dtypes = (
            trace.get("image_tensor_dtype"),
            trace.get("position_tensor_dtype"),
            trace.get("state_tensor_dtype"),
            trace.get("mask_dtype"),
        )
        if observed_dtypes != released_prepared_component_dtypes(len(expected_selected)):
            raise ArtifactContractError("Extension prepared-memory dtypes changed")
        if trace.get("prepared_memory_input_shape") != [512, 2824]:
            raise ArtifactContractError("Extension memory input shape changed")
        if trace.get("final_memory_tensor_shape") != list(SMOKE_FINAL_MEMORY_SHAPE):
            raise ArtifactContractError("Extension final memory shape changed")
        if trace.get("final_memory_tensor_dtype") != SMOKE_FINAL_MEMORY_DTYPE:
            raise ArtifactContractError("Extension final memory dtype changed")
        if trace.get("final_memory_tensor_is_floating") is not True:
            raise ArtifactContractError("Extension final memory must be floating")
        if trace.get("final_memory_tensor_finite") is not True:
            raise ArtifactContractError("Extension final memory is non-finite")
        for field in required_hashes:
            digest = _require_sha256(trace.get(field), field=f"extension trace {field}")
            if field == "seed_table_sha256" and digest != seed_digest:
                raise ArtifactContractError("Extension trace seed digest mismatch")

        components = {
            field: _require_nonnegative_finite(trace.get(field), field=field)
            for field in (
                "boundary_lookup_latency_ms",
                "selector_decision_latency_ms",
                "selector_bookkeeping_latency_ms",
                "selector_latency_ms",
                "model_latency_ms",
                "end_to_end_request_latency_ms",
            )
        }
        expected_selector_latency = (
            components["boundary_lookup_latency_ms"]
            + components["selector_decision_latency_ms"]
            + components["selector_bookkeeping_latency_ms"]
        )
        if not math.isclose(
            components["selector_latency_ms"],
            expected_selector_latency,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ArtifactContractError("Extension selector latency components disagree")
        end_to_end = _require_nonnegative_finite(result["policy_latency_ms"][call_index], field="result policy latency")
        model = _require_nonnegative_finite(result["policy_model_latency_ms"][call_index], field="result model latency")
        if components["end_to_end_request_latency_ms"] != end_to_end:
            raise ArtifactContractError("Extension trace/result request latency mismatch")
        if components["model_latency_ms"] != model:
            raise ArtifactContractError("Extension trace/result model latency mismatch")
        if result["history_lengths_at_policy_calls"][call_index] != history_length:
            raise ArtifactContractError("Extension trace/result history length mismatch")
        latency_values["selector"].append(components["selector_latency_ms"])
        latency_values["model"].append(components["model_latency_ms"])
        latency_values["end_to_end"].append(components["end_to_end_request_latency_ms"])

    report = {
        "strict_extension_contract": True,
        "generic_attempt_audit": generic_report,
        "policy_call_count": expected_calls,
        "selector_arm": expected_key.arm,
        "selector_latency": _latency_summary(latency_values["selector"]),
        "model_latency": _latency_summary(latency_values["model"]),
        "end_to_end_latency": _latency_summary(latency_values["end_to_end"]),
        "selector_diagnostics": {
            "oc5_fallback_calls": fallback_calls,
            "oc3_secondary_thinning_calls": secondary_thinning_calls,
            "boundary_core_thinned_calls": boundary_core_thinned_calls,
            "requested_candidate_count_sum": sum(requested_candidate_counts),
            "effective_candidate_count_sum": sum(effective_candidate_counts),
            "selected_frame_count_sum": sum(selected_frame_counts),
        },
        "secondary_diagnostics": {
            "episode_count": 1,
            "policy_call_count": expected_calls,
            "boundary_recall_numerator": boundary_recall_numerator,
            "boundary_recall_denominator": boundary_recall_denominator,
            "requested_candidate_retention_numerator": requested_candidate_retention_numerator,
            "requested_candidate_retention_denominator": sum(requested_candidate_counts),
            "effective_candidate_retention_numerator": effective_candidate_retention_numerator,
            "effective_candidate_retention_denominator": sum(effective_candidate_counts),
            "memory_age_values": memory_age_values,
            "selected_index_gap_values": selected_index_gap_values,
            "maximum_temporal_gap_values": maximum_temporal_gap_values,
            "final_observed_boundary_count_values": [final_observed_boundary_count],
        },
    }
    return report, latency_values, manifest, result


def _repository_provenance(
    repo_root: Path,
    *,
    formal_commit: str,
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    """Require analysis from the exact clean checkout frozen before formal launch."""
    try:
        aggregation_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        status = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--",
                ".",
                ":(exclude)runs/keyframe_neighborhood_sampling",
                ":(exclude)runs/keyframe_oracle_sampling",
                f":(exclude){SMOKE_CHECKPOINT_PATH}",
            ],
            cwd=repo_root,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtifactContractError("Cannot inspect the extension aggregation checkout") from exc
    if status:
        raise ArtifactContractError("Extension aggregation requires a clean committed worktree")
    if aggregation_commit != formal_commit:
        raise ArtifactContractError(
            "Extension aggregation HEAD must exactly equal the frozen formal evaluation commit"
        )
    source_digests = validate_frozen_formal_sources(
        launch,
        repo_root,
        source="Extension aggregation",
    )
    return {
        "aggregation_commit_sha": aggregation_commit,
        "formal_commit_exact_match": True,
        "aggregation_worktree_clean": True,
        "frozen_source_hashes_match": True,
        "analysis_source_sha256": source_digests["frozen_analysis_source_sha256"],
        "aggregator_source_sha256": source_digests["frozen_aggregator_source_sha256"],
    }


def audit_extension_run(
    run_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[ScientificKey, dict[str, Any]]]:
    """Audit the exact 1,600-cell extension census and all retry chains."""
    run_root = run_root.resolve()
    repo_root = repo_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Extension formal root does not exist: {run_root}")
    if (run_root / "aggregate").exists() or (run_root / "aggregate").is_symlink():
        raise FileExistsError("Refusing to inspect and republish an existing extension aggregate")
    validate_extension_runtime_location(run_root, repo_root, source="Extension formal aggregation")

    launch, matrix, seed_payload, _, protocol_provenance = validate_extension_protocol_bundle(run_root, repo_root)
    repository_provenance = _repository_provenance(
        repo_root,
        formal_commit=protocol_provenance["formal_evaluation_commit_sha"],
        launch=launch,
    )
    expected_rows = _expected_formal_rows(matrix)
    row_id_by_key = {key: int(row["row_id"]) for key, row in expected_rows.items()}
    key_by_row = {row_id: key for key, row_id in row_id_by_key.items()}
    store = RunArtifactStore(run_root)
    completed = store.scan_completed_keys()
    census = completeness_report(set(expected_rows), completed)
    if not census["complete"]:
        raise ArtifactContractError(
            "Extension aggregation requires exactly 1,600 OC3/OC5 results: " + json.dumps(census, sort_keys=True)
        )
    attempts = discover_attempts(run_root, store)

    failure_records = read_jsonl(store.failures_path) if store.failures_path.exists() else []
    failures: dict[tuple[ScientificKey, int], dict[str, Any]] = {}
    for index, record in enumerate(failure_records):
        if not isinstance(record, Mapping):
            raise ArtifactContractError(f"Extension failure record {index} is invalid")
        key = _scientific_key(record, source=f"failure record {index}")
        if key not in expected_rows:
            raise ArtifactContractError(f"Unexpected extension failure key: {key}")
        attempt_id = record.get("attempt_id")
        if type(attempt_id) is not int or attempt_id not in {0, 1, 2}:
            raise ArtifactContractError("Extension failure has an invalid attempt ID")
        identity = (key, attempt_id)
        if identity in failures:
            raise ArtifactContractError(f"Duplicate extension failure record: {identity}")
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
            source=f"Extension failure record {index}",
        )
        validate_extension_failure_record(
            run_root,
            record,
            expected_row_id=row_id_by_key[key],
        )
        failures[identity] = dict(record)

    expected_attempts: set[tuple[ScientificKey, int]] = set()
    completed_attempt_by_key: dict[ScientificKey, int] = {}
    for key, result_path in completed.items():
        result = _load_json_object(result_path)
        attempt_id = result.get("attempt_id")
        if type(attempt_id) is not int or attempt_id not in {0, 1, 2}:
            raise ArtifactContractError("Completed extension result has invalid attempt ID")
        completed_attempt_by_key[key] = attempt_id
        if result_path.parent.resolve() != store.attempt_dir(key, attempt_id).resolve():
            raise ArtifactContractError("Completed extension result path is non-canonical")
        expected_attempts.update((key, prior) for prior in range(attempt_id + 1))
        for prior in range(attempt_id):
            if (key, prior) not in failures:
                raise ArtifactContractError("Extension retry chain skips a failure record")
        if (key, attempt_id) in failures:
            raise ArtifactContractError("Completed extension attempt is also a failure")
    if set(attempts) != expected_attempts:
        raise ArtifactContractError("Extension attempt directories do not match exact retry chains")
    expected_failures = {
        (key, prior)
        for key, completed_attempt in completed_attempt_by_key.items()
        for prior in range(completed_attempt)
    }
    if set(failures) != expected_failures:
        raise ArtifactContractError("Extension failure ledger differs from retry chains")

    failed_attempts_with_trace = 0
    for identity, failure in failures.items():
        key, attempt_id = identity
        path, manifest = attempts[identity]
        writer = EpisodeAttemptWriter(path, key, attempt_id)
        if writer.validate_resume() != "incomplete" or writer.result_path.exists():
            raise ArtifactContractError("Extension failure ledger points at a completed attempt")
        recorded_manifest_sha = failure.get("episode_manifest_sha256")
        if recorded_manifest_sha is not None and recorded_manifest_sha != sha256_file(writer.manifest_path):
            raise ArtifactContractError("Extension failure manifest digest mismatch")
        if failure.get("slurm") is not None and failure["slurm"] != manifest.get("slurm"):
            raise ArtifactContractError("Extension failure/manifest Slurm data mismatch")
        trace_exists = writer.trace_path.exists()
        failed_attempts_with_trace += int(trace_exists)
        if (
            failure.get("scientific_actions_started") is not None
            and failure["scientific_actions_started"] != trace_exists
        ):
            raise ArtifactContractError("Extension failure trace presence contradicts scientific_actions_started")

    retry_rows_by_attempt = {0: list(range(FORMAL_TRAJECTORY_COUNT))}
    for attempt_id in (1, 2):
        retry_rows_by_attempt[attempt_id] = sorted(
            row_id_by_key[key] for key, prior_attempt in failures if prior_attempt == attempt_id - 1
        )
    submission_reports = []
    authorizations: dict[int, dict[int, dict[str, Any]]] = {}
    for attempt_id in (0, 1, 2):
        rows = retry_rows_by_attempt[attempt_id]
        if not rows:
            if _submission_plan_path(run_root, attempt_id).exists() or _submission_paths(run_root, attempt_id):
                raise ArtifactContractError("Unexpected empty extension retry submission")
            continue
        report, row_authorizations = validate_submission_attempt(
            run_root,
            attempt_id=attempt_id,
            expected_row_ids=rows,
            launch=launch,
        )
        submission_reports.append(report)
        authorizations[attempt_id] = row_authorizations
    for (key, attempt_id), (_, manifest) in attempts.items():
        row_id = row_id_by_key[key]
        try:
            authorization = authorizations[attempt_id][row_id]
        except KeyError as exc:
            raise ArtifactContractError("Extension attempt lacks submission authorization") from exc
        _validate_attempt_submission_binding(
            manifest, row_id=row_id, authorization=authorization, run_root=run_root,
            validated_readiness_failure=(
                (key, attempt_id) in failures
                and failures[(key, attempt_id)].get("failure_phase") == "policy_server_readiness"
            ),
        )

    paired_manifests = []
    validated_records: dict[ScientificKey, dict[str, Any]] = {}
    result_digests = []
    trace_digests = []
    strict_audit_digests = []
    latency_values = {arm: {"selector": [], "model": [], "end_to_end": []} for arm in EXTENSION_ARMS}
    attempt_histogram: Counter[int] = Counter()
    terminal_histogram = {arm: Counter() for arm in EXTENSION_ARMS}
    success_counts: Counter[str] = Counter()
    collision_counts: Counter[str] = Counter()
    policy_calls: Counter[str] = Counter()
    selector_diagnostics = {
        arm: Counter(
            {
                "oc5_fallback_calls": 0,
                "oc3_secondary_thinning_calls": 0,
                "boundary_core_thinned_calls": 0,
                "requested_candidate_count_sum": 0,
                "effective_candidate_count_sum": 0,
                "selected_frame_count_sum": 0,
            }
        )
        for arm in EXTENSION_ARMS
    }
    secondary_diagnostics_by_arm = {
        arm: _empty_secondary_diagnostic_accumulator() for arm in EXTENSION_ARMS
    }
    secondary_diagnostics_by_task = {
        task: {
            arm: _empty_secondary_diagnostic_accumulator() for arm in EXTENSION_ARMS
        }
        for task in FORMAL_TASKS
    }
    for row_id in range(FORMAL_TRAJECTORY_COUNT):
        key = key_by_row[row_id]
        attempt_id = completed_attempt_by_key[key]
        path, _ = attempts[(key, attempt_id)]
        writer = EpisodeAttemptWriter(path, key, attempt_id)
        attempt_report, attempt_latencies, manifest, result = audit_extension_attempt(
            writer,
            expected_key=key,
            expected_row=expected_rows[key],
            launch=launch,
            seed_payload=seed_payload,
        )
        validated_records[key] = result
        paired_manifest = dict(manifest)
        paired_manifest["initial_condition_hashes"] = _load_json_object(writer.initial_conditions_path)
        paired_manifests.append(paired_manifest)
        attempt_histogram[attempt_id] += 1
        terminal_histogram[key.arm][str(result["terminal_reason"])] += 1
        success_counts[key.arm] += int(result["success"])
        collision_counts[key.arm] += int(result["collision"])
        policy_calls[key.arm] += int(attempt_report["policy_call_count"])
        selector_diagnostics[key.arm].update(attempt_report["selector_diagnostics"])
        _merge_secondary_diagnostics(
            secondary_diagnostics_by_arm[key.arm], attempt_report["secondary_diagnostics"]
        )
        _merge_secondary_diagnostics(
            secondary_diagnostics_by_task[key.task][key.arm],
            attempt_report["secondary_diagnostics"],
        )
        for latency_kind, values in attempt_latencies.items():
            latency_values[key.arm][latency_kind].extend(values)
        result_sha = sha256_file(writer.result_path)
        trace_sha = sha256_file(writer.trace_path)
        result_digests.append({"row_id": row_id, "attempt_id": attempt_id, "sha256": result_sha})
        trace_digests.append({"row_id": row_id, "attempt_id": attempt_id, "sha256": trace_sha})
        strict_audit_digests.append(
            {
                "row_id": row_id,
                "attempt_id": attempt_id,
                "policy_call_count": attempt_report["policy_call_count"],
                "result_sha256": result_sha,
                "trace_sha256": trace_sha,
            }
        )

    fairness = audit_initial_condition_fairness(paired_manifests, required_arms=EXTENSION_ARMS)
    invariants = audit_paired_manifest_invariants(paired_manifests, required_arms=EXTENSION_ARMS)
    if fairness != {"paired_blocks": 800, "fair": True}:
        raise ArtifactContractError("Extension initial-condition pairing is incomplete")
    if invariants != {"paired_blocks": 800, "invariants_match": True}:
        raise ArtifactContractError("Extension manifest pairing is incomplete")

    diagnostics_by_arm = {}
    for arm in EXTENSION_ARMS:
        calls = policy_calls[arm]
        diagnostics = dict(selector_diagnostics[arm])
        diagnostics_by_arm[arm] = {
            **diagnostics,
            "policy_call_count": calls,
            "oc5_fallback_frequency": (diagnostics["oc5_fallback_calls"] / calls if calls else 0.0),
            "oc3_secondary_thinning_frequency": (diagnostics["oc3_secondary_thinning_calls"] / calls if calls else 0.0),
            "boundary_core_thinned_frequency": (diagnostics["boundary_core_thinned_calls"] / calls if calls else 0.0),
        }
    finalized_secondary_diagnostics = {
        "scope": "exploratory_secondary_diagnostics",
        "definitions": {
            "boundary_recall": (
                "Micro-aggregated selected visible-boundary count divided by visible-boundary count across policy calls."
            ),
            "requested_neighborhood_candidate_retention": (
                "Micro-aggregated fraction of causal candidates from the arm's requested neighborhood present "
                "in final memory selection. For OC5, the denominator still includes five-frame candidates on "
                "calls that wholesale-fall back to OC3."
            ),
            "effective_neighborhood_candidate_retention": (
                "Micro-aggregated fraction of candidates from the post-fallback effective neighborhood present "
                "in final memory selection."
            ),
            "selected_index_spacing": (
                "Consecutive differences between strictly increasing selected history-frame indices, pooled "
                "across policy calls."
            ),
            "memory_age": (
                "Current history-frame index minus selected history-frame index, pooled across policy calls."
            ),
            "maximum_temporal_gap_per_policy_call": (
                "Maximum consecutive selected-index spacing at each policy call; zero when fewer than two "
                "frames are selected."
            ),
            "final_observed_boundary_count_progress_proxy": (
                "Number of causal boundary indices visible at the final policy call, including the required "
                "initial index zero."
            ),
        },
        "limitations": [
            "The runtime records no task-independent ground-truth stage-completion metric. Final observed "
            "boundary count is reported only as an exploratory within-task progress proxy; it is not true "
            "stage completion and should not be compared as an absolute progress scale across tasks."
        ],
        "by_arm": {
            arm: _finalize_secondary_diagnostics(secondary_diagnostics_by_arm[arm])
            for arm in EXTENSION_ARMS
        },
        "by_task": {
            task: {
                arm: _finalize_secondary_diagnostics(secondary_diagnostics_by_task[task][arm])
                for arm in EXTENSION_ARMS
            }
            for task in FORMAL_TASKS
        },
    }
    failure_ledger_sha = sha256_file(store.failures_path) if store.failures_path.exists() else None
    report = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "protocol_version": EXTENSION_PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "audited_utc": utc_now(),
        "passed": True,
        "formal_result_census_complete": True,
        "strict_selector_trace_audit_complete": True,
        "completeness": census,
        "initial_condition_fairness": fairness,
        "paired_manifest_invariants": invariants,
        "reference_oc_pairing": protocol_provenance["reference_alignment"],
        "protocol_provenance": {**protocol_provenance, **repository_provenance},
        "submission_audits": submission_reports,
        "attempt_count": len(attempts),
        "completed_attempt_histogram": {
            str(attempt): attempt_histogram[attempt] for attempt in sorted(attempt_histogram)
        },
        "infrastructure_failure_count": len(failure_records),
        "failed_attempts_with_selector_trace_count": failed_attempts_with_trace,
        "failure_ledger_sha256": failure_ledger_sha,
        "retry_row_count_by_attempt": {
            str(attempt): len(rows) for attempt, rows in retry_rows_by_attempt.items() if rows
        },
        "result_set_sha256": hashlib.sha256(canonical_json_bytes(result_digests)).hexdigest(),
        "selector_trace_set_sha256": hashlib.sha256(canonical_json_bytes(trace_digests)).hexdigest(),
        "strict_attempt_audit_set_sha256": hashlib.sha256(canonical_json_bytes(strict_audit_digests)).hexdigest(),
        "latency_by_arm": {
            arm: {kind: _latency_summary(values) for kind, values in latency_values[arm].items()}
            for arm in EXTENSION_ARMS
        },
        "selector_diagnostics_by_arm": diagnostics_by_arm,
        "secondary_diagnostics": finalized_secondary_diagnostics,
        "outcome_diagnostics": {
            "success_count_by_arm": {arm: success_counts[arm] for arm in EXTENSION_ARMS},
            "terminal_reason_count_by_arm": {
                arm: dict(sorted(terminal_histogram[arm].items())) for arm in EXTENSION_ARMS
            },
            "collision_count_by_arm": {arm: collision_counts[arm] for arm in EXTENSION_ARMS},
            "policy_call_count_by_arm": {arm: policy_calls[arm] for arm in EXTENSION_ARMS},
        },
        "protocol_deviations": [],
        "unresolved_risks": [
            "OC3/OC5 initial conditions are hash-verified against one another within the new run, but the "
            "published OC aggregate exposes neither environment seeds, difficulty values, nor raw initial-condition "
            "hashes. Therefore OC-to-extension matching is aligned by the frozen task/episode design and cannot be "
            "directly re-audited for those fields. The verified RandomSamp selector-RNG seed table is not evidence "
            "of simulator-seed or initial-state equality."
        ],
    }
    return report, validated_records


def _reference_success_lookup(
    reference_oc_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], bool]:
    lookup: dict[tuple[str, int], bool] = {}
    for record in reference_oc_records:
        key = (str(record["task"]), int(record["episode_id"]))
        if key in lookup:
            raise ArtifactContractError(f"Duplicate reference OC outcome: {key}")
        if type(record.get("success")) is not bool:
            raise ArtifactContractError(f"Reference OC outcome is not binary: {key}")
        lookup[key] = bool(record["success"])
    expected = {(task, episode_id) for task in FORMAL_TASKS for episode_id in FORMAL_EPISODE_IDS}
    if set(lookup) != expected:
        raise ArtifactContractError("Reference OC lookup is not the exact 800-cell census")
    return lookup


def build_per_episode_rows(
    records: Mapping[ScientificKey, Mapping[str, Any]],
    reference_oc_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reference = _reference_success_lookup(reference_oc_records)
    rows = []
    for task in FORMAL_TASKS:
        for episode_id in FORMAL_EPISODE_IDS:
            by_arm = {arm: records[ScientificKey(task, episode_id, arm, "formal")] for arm in EXTENSION_ARMS}
            oc_success = int(reference[(task, episode_id)])
            oc3_success = int(by_arm["OC3"]["success"])
            oc5_success = int(by_arm["OC5"]["success"])
            row: dict[str, Any] = {
                "task": task,
                "episode_id": episode_id,
                "OC_success": oc_success,
            }
            for arm in EXTENSION_ARMS:
                result = by_arm[arm]
                row[f"{arm}_success"] = int(result["success"])
                row[f"{arm}_terminal_reason"] = result["terminal_reason"]
                row[f"{arm}_steps"] = result["steps"]
                row[f"{arm}_collision"] = int(result["collision"])
                row[f"{arm}_attempt_id"] = result["attempt_id"]
            row.update(
                {
                    "OC3_minus_OC": oc3_success - oc_success,
                    "OC5_minus_OC": oc5_success - oc_success,
                    "OC5_minus_OC3": oc5_success - oc3_success,
                }
            )
            rows.append(row)
    if len(rows) != 800:
        raise AssertionError("Extension per-episode table must contain 800 paired rows")
    return rows


def build_per_task_rows(
    records: Mapping[ScientificKey, Mapping[str, Any]],
    reference_oc_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reference = _reference_success_lookup(reference_oc_records)
    rows = []
    for task in FORMAL_TASKS:
        counts = {
            "OC": sum(int(reference[(task, episode_id)]) for episode_id in FORMAL_EPISODE_IDS),
            **{
                arm: sum(
                    int(records[ScientificKey(task, episode_id, arm, "formal")]["success"])
                    for episode_id in FORMAL_EPISODE_IDS
                )
                for arm in EXTENSION_ARMS
            },
        }
        rates = {arm: count / len(FORMAL_EPISODE_IDS) for arm, count in counts.items()}
        rows.append(
            {
                "task": task,
                "episode_count": len(FORMAL_EPISODE_IDS),
                "OC_success_count": counts["OC"],
                "OC_success_rate": rates["OC"],
                "OC3_success_count": counts["OC3"],
                "OC3_success_rate": rates["OC3"],
                "OC5_success_count": counts["OC5"],
                "OC5_success_rate": rates["OC5"],
                "OC3_minus_OC_pp": 100.0 * (rates["OC3"] - rates["OC"]),
                "OC5_minus_OC_pp": 100.0 * (rates["OC5"] - rates["OC"]),
                "OC5_minus_OC3_pp": 100.0 * (rates["OC5"] - rates["OC3"]),
            }
        )
    if len(rows) != len(FORMAL_TASKS):
        raise AssertionError("Extension per-task table must contain 16 rows")
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ArtifactContractError("Cannot serialize an empty extension table")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ArtifactContractError("Extension CSV rows do not share one schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _markdown_ratio(metric: Mapping[str, Any]) -> str:
    numerator = int(metric["numerator"])
    denominator = int(metric["denominator"])
    rate = metric["rate"]
    if rate is None:
        return f"{numerator}/{denominator} (n/a)"
    return f"{numerator}/{denominator} ({100.0 * float(rate):.3f}%)"


def _markdown_stat(metric: Mapping[str, Any], field: str) -> str:
    value = metric[field]
    return "n/a" if value is None else f"{float(value):.3f}"


def render_analysis_markdown(summary: Mapping[str, Any], completeness: Mapping[str, Any]) -> str:
    lines = [
        "# OC3/OC5 keyframe-neighborhood extension",
        "",
        "This is a paired post-hoc follow-up: the completed OC outcomes were inspected "
        "before this extension was designed, while the extension analysis plan was frozen "
        "before any OC3/OC5 trajectory was launched.",
        "",
        "The strict audit passed for exactly 1,600 new OC3/OC5 cells. The completed OC "
        "outcomes were read from the immutable published table after SHA-256 verification.",
        "",
        "| Arm | Successes | Episodes | Pooled rate | Equal-task-weighted rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("OC", *EXTENSION_ARMS):
        outcome = summary["arm_outcomes"][arm]
        lines.append(
            f"| {arm} | {outcome['pooled_success_count']} | "
            f"{outcome['pooled_episode_count']} | "
            f"{100.0 * outcome['pooled_success_rate']:.3f}% | "
            f"{100.0 * outcome['equal_task_weighted_success_rate']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Estimate (pp) | 95% CI (pp) | Raw p | Holm-adjusted p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, comparison in summary["comparisons"].items():
        interval = comparison["confidence_interval_95"]
        randomization = comparison["randomization_test"]
        lines.append(
            f"| {name} | {comparison['estimate_pp']:.3f} | "
            f"[{interval['lower_pp']:.3f}, {interval['upper_pp']:.3f}] | "
            f"{randomization['p_value']:.6g} | "
            f"{randomization['holm_adjusted_p_value']:.6g} |"
        )
    secondary = summary["secondary_diagnostics"]
    lines.extend(
        [
            "",
            "## Exploratory selector and progress diagnostics",
            "",
            "All values below are recomputed from audited selected indices, current history indices, "
            "and visible causal boundaries; logged derived diagnostic values are validated against this replay.",
            "",
            "| Arm | Boundary recall | Requested-neighborhood retention | "
            "Effective-neighborhood retention | Mean spacing | P95 spacing | Mean age | "
            "Mean final boundary count |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in EXTENSION_ARMS:
        diagnostics = secondary["by_arm"][arm]
        lines.append(
            f"| {arm} | {_markdown_ratio(diagnostics['boundary_recall'])} | "
            f"{_markdown_ratio(diagnostics['requested_neighborhood_candidate_retention'])} | "
            f"{_markdown_ratio(diagnostics['effective_neighborhood_candidate_retention'])} | "
            f"{_markdown_stat(diagnostics['selected_index_spacing'], 'mean')} | "
            f"{_markdown_stat(diagnostics['selected_index_spacing'], 'p95')} | "
            f"{_markdown_stat(diagnostics['memory_age'], 'mean')} | "
            f"{_markdown_stat(diagnostics['final_observed_boundary_count_progress_proxy'], 'mean')} |"
        )
    lines.extend(
        [
            "",
            "Per-task exploratory diagnostics:",
            "",
            "| Task | Arm | Boundary recall | Requested retention | Effective retention | "
            "Mean spacing | Mean age | Mean final boundary count |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task in FORMAL_TASKS:
        for arm in EXTENSION_ARMS:
            diagnostics = secondary["by_task"][task][arm]
            lines.append(
                f"| {task} | {arm} | {_markdown_ratio(diagnostics['boundary_recall'])} | "
                f"{_markdown_ratio(diagnostics['requested_neighborhood_candidate_retention'])} | "
                f"{_markdown_ratio(diagnostics['effective_neighborhood_candidate_retention'])} | "
                f"{_markdown_stat(diagnostics['selected_index_spacing'], 'mean')} | "
                f"{_markdown_stat(diagnostics['memory_age'], 'mean')} | "
                f"{_markdown_stat(diagnostics['final_observed_boundary_count_progress_proxy'], 'mean')} |"
            )
    lines.extend(
        [
            "",
            f"Limitation: {secondary['limitations'][0]}",
            "",
            "## Audit notes",
            "",
            f"- New scientific cells: {summary['extension_cell_count']}",
            f"- Immutable OC reference cells: {summary['reference_oc_cell_count']}",
            f"- Infrastructure failures: {completeness['infrastructure_failure_count']}",
            f"- Analysis seed: {summary['analysis_seed']}",
            f"- Bootstrap replicates: {summary['bootstrap_replicates']}",
            f"- Randomization replicates: {summary['randomization_replicates']}",
            f"- RNG implementation: {summary['resampling_reproducibility']['rng_implementation']}",
            "- Bootstrap implementation: "
            f"chunk size {summary['resampling_reproducibility']['bootstrap']['implementation_chunk_size']}, "
            f"quantiles {summary['resampling_reproducibility']['bootstrap']['percentile_bounds']} "
            f"with {summary['resampling_reproducibility']['bootstrap']['quantile_method']}",
            "- Bootstrap common-schedule SHA-256: "
            f"{summary['resampling_reproducibility']['bootstrap']['common_schedule_sha256']}",
            "- Randomization implementation: "
            f"chunk size {summary['resampling_reproducibility']['randomization']['implementation_chunk_size']}",
            "- Randomization common-schedule SHA-256: "
            f"{summary['resampling_reproducibility']['randomization']['common_schedule_sha256']}",
            "- OC3 and OC5 initial conditions are hash-verified against each other. The published OC aggregate "
            "contains no environment seed, difficulty, or raw initial-condition hashes, so cross-run OC matching "
            "is aligned by task/episode design rather than directly re-audited for those fields.",
            "- The matching frozen seed table covers RandomSamp policy-call RNG only; it does not prove simulator "
            "seed or initial-state equality. Checkpoint identity and the published OC protocol identity are verified.",
            "",
        ]
    )
    return "\n".join(lines)


def build_aggregate_payloads(
    completeness: Mapping[str, Any],
    records: Mapping[ScientificKey, Mapping[str, Any]],
    reference_oc_records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    randomization_replicates: int = DEFAULT_RANDOMIZATION_REPLICATES,
    allow_nonconfirmatory_replicate_override: bool = False,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if completeness.get("passed") is not True:
        raise ArtifactContractError("Cannot aggregate an extension run that failed audit")
    analysis = build_extension_analysis(
        records.values(),
        reference_oc_records,
        bootstrap_replicates=bootstrap_replicates,
        randomization_replicates=randomization_replicates,
        allow_nonconfirmatory_replicate_override=allow_nonconfirmatory_replicate_override,
    )
    expected_status = (
        "complete_nonconfirmatory_test_override"
        if allow_nonconfirmatory_replicate_override
        and (
            bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES
            or randomization_replicates != DEFAULT_RANDOMIZATION_REPLICATES
        )
        else "complete_frozen_extension"
    )
    if analysis.get("analysis_status") != expected_status:
        raise ArtifactContractError("Extension analysis replicate contract was not met")
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
        "latency_by_arm": completeness["latency_by_arm"],
        "selector_diagnostics_by_arm": completeness["selector_diagnostics_by_arm"],
        "secondary_diagnostics": completeness["secondary_diagnostics"],
        "outcome_diagnostics": completeness["outcome_diagnostics"],
    }
    payloads = {
        "completeness_report.json": _json_bytes(completeness),
        "per_episode.csv": _csv_bytes(build_per_episode_rows(records, reference_oc_records)),
        "per_task.csv": _csv_bytes(build_per_task_rows(records, reference_oc_records)),
        "summary.json": _json_bytes(summary),
        "analysis.md": render_analysis_markdown(summary, completeness).encode("utf-8"),
    }
    if tuple(payloads) != AGGREGATE_FILENAMES:
        raise AssertionError("Extension aggregate file order changed")
    return payloads, summary


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_aggregate_directory(run_root: Path, payloads: Mapping[str, bytes]) -> Path:
    """Write all five products once, then atomically publish the directory."""
    run_root = run_root.resolve()
    target = run_root / "aggregate"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite extension aggregate: {target}")
    if tuple(payloads) != AGGREGATE_FILENAMES:
        raise ArtifactContractError("Extension publication requires exactly five canonical artifacts")
    with tempfile.TemporaryDirectory(prefix=".aggregate.", dir=run_root) as temporary:
        staging = Path(temporary)
        for filename in AGGREGATE_FILENAMES:
            data = payloads[filename]
            if not isinstance(data, bytes) or not data:
                raise ArtifactContractError(f"Extension payload is empty: {filename}")
            path = staging / filename
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(staging)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite extension aggregate: {target}")
        os.rename(staging, target)
        _fsync_directory(run_root)
    return target


def aggregate_formal_run(
    run_root: Path,
    repo_root: Path,
) -> tuple[Path, dict[str, Any]]:
    completeness, records = audit_extension_run(run_root, repo_root)
    reference_path, _, _ = _reference_paths(repo_root)
    reference_records = load_published_reference_oc(reference_path)
    payloads, summary = build_aggregate_payloads(completeness, records, reference_records)
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
                "extension_cell_count": summary["extension_cell_count"],
                "analysis_status": summary["analysis_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
