"""Frozen paired analysis for the OC3/OC5 post-hoc extension."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
import dataclasses
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_EPISODE_IDS
from experiments.keyframe_oracle_sampling.analysis import holm_adjust
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import sha256_file

ANALYSIS_SEED = 2026082502
DEFAULT_BOOTSTRAP_REPLICATES = 100_000
DEFAULT_RANDOMIZATION_REPLICATES = 100_000
BOOTSTRAP_CHUNK_SIZE = 512
RANDOMIZATION_CHUNK_SIZE = 2_048
REFERENCE_ARM = "OC"
ANALYSIS_ARMS = (REFERENCE_ARM, *EXTENSION_ARMS)
COMPARISONS = (("OC3", "OC"), ("OC5", "OC"), ("OC5", "OC3"))
REFERENCE_PER_EPISODE_SHA256 = (
    "e1ec5d1273f3ba97b14d252a1a1668c00635a71477acca687f3e47d7e9f3a478"
)
RNG_IMPLEMENTATION = "numpy.random.Generator(PCG64)"
PERCENTILE_BOUNDS = (2.5, 97.5)
QUANTILE_METHOD = "numpy-linear"
BOOTSTRAP_SCHEDULE_VERSION = "paired-hierarchical-bootstrap-schedule-v1"
RANDOMIZATION_SCHEDULE_VERSION = "paired-label-swap-schedule-v1"


class AnalysisProtocolError(RuntimeError):
    """Raised rather than emitting an analysis that departs from the protocol."""


@dataclasses.dataclass(frozen=True)
class CombinedOutcomeTable:
    """Exact 16-task x 50-episode outcomes for OC, OC3, and OC5."""

    successes: tuple[tuple[tuple[bool, ...], ...], ...]

    def success(self, task: str, episode_id: int, arm: str) -> bool:
        try:
            task_index = FORMAL_TASKS.index(task)
            episode_index = FORMAL_EPISODE_IDS.index(int(episode_id))
            arm_index = ANALYSIS_ARMS.index(arm)
        except ValueError as exc:
            raise ArtifactContractError(
                f"Outcome lookup is outside the extension analysis matrix: "
                f"{task}/{episode_id}/{arm}"
            ) from exc
        return self.successes[task_index][episode_index][arm_index]


def _coalesce(
    record: Mapping[str, Any],
    scientific_key: Mapping[str, Any] | None,
    field: str,
) -> Any:
    values = []
    if field in record:
        values.append(record[field])
    if scientific_key is not None and field in scientific_key:
        values.append(scientific_key[field])
    if not values:
        raise ArtifactContractError(f"Extension analysis record lacks {field!r}")
    if any(value != values[0] for value in values[1:]):
        raise ArtifactContractError(f"Conflicting flat/nested values for {field!r}")
    return values[0]


def _normalize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    allowed_arms: tuple[str, ...],
    require_formal_metadata: bool,
) -> dict[tuple[str, int, str], bool]:
    observed: dict[tuple[str, int, str], bool] = {}
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ArtifactContractError(f"Analysis record {record_index} is not a mapping")
        raw_key = record.get("scientific_key")
        if raw_key is not None and not isinstance(raw_key, Mapping):
            raise ArtifactContractError(
                f"Analysis record {record_index} has an invalid scientific_key"
            )
        task = _coalesce(record, raw_key, "task")
        episode_id = _coalesce(record, raw_key, "episode_id")
        arm = _coalesce(record, raw_key, "arm")
        if task not in FORMAL_TASKS:
            raise ArtifactContractError(f"Unexpected extension task: {task!r}")
        if (
            isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
            or episode_id not in FORMAL_EPISODE_IDS
        ):
            raise ArtifactContractError(
                f"Unexpected extension episode for {task}: {episode_id!r}"
            )
        if arm not in allowed_arms:
            raise ArtifactContractError(f"Unexpected analysis arm: {arm!r}")
        has_trajectory_kind = "trajectory_kind" in record or (
            raw_key is not None and "trajectory_kind" in raw_key
        )
        if require_formal_metadata and not has_trajectory_kind:
            raise ArtifactContractError(
                "Extension analysis record lacks required 'trajectory_kind'"
            )
        if has_trajectory_kind and _coalesce(
            record, raw_key, "trajectory_kind"
        ) != "formal":
            raise ArtifactContractError("Extension analysis accepts only formal records")
        has_dataset = "dataset" in record or (
            raw_key is not None and "dataset" in raw_key
        )
        if require_formal_metadata and not has_dataset:
            raise ArtifactContractError(
                "Extension analysis record lacks required 'dataset'"
            )
        if has_dataset and _coalesce(record, raw_key, "dataset") != "test":
            raise ArtifactContractError("Extension analysis accepts only test records")
        success = record.get("success")
        if type(success) is not bool:
            raise ArtifactContractError(
                f"Result {task}/{episode_id}/{arm} lacks a binary bool success"
            )
        key = (task, int(episode_id), str(arm))
        if key in observed:
            raise ArtifactContractError(f"Duplicate analysis cell: {key}")
        observed[key] = success
    return observed


def validate_combined_records(
    extension_records: Iterable[Mapping[str, Any]],
    reference_oc_records: Iterable[Mapping[str, Any]],
) -> CombinedOutcomeTable:
    """Require 1,600 extension cells and the exact 800-cell reference OC census."""
    extension = _normalize_records(
        extension_records,
        allowed_arms=EXTENSION_ARMS,
        require_formal_metadata=True,
    )
    # The immutable published OC CSV predates the extension adapter and does
    # not carry trajectory_kind/dataset columns.  Its exact byte digest and
    # exact 800-cell census are validated separately.
    reference = _normalize_records(
        reference_oc_records,
        allowed_arms=(REFERENCE_ARM,),
        require_formal_metadata=False,
    )
    expected_extension = {
        (task, episode_id, arm)
        for task in FORMAL_TASKS
        for episode_id in FORMAL_EPISODE_IDS
        for arm in EXTENSION_ARMS
    }
    expected_reference = {
        (task, episode_id, REFERENCE_ARM)
        for task in FORMAL_TASKS
        for episode_id in FORMAL_EPISODE_IDS
    }
    if set(extension) != expected_extension:
        missing = sorted(expected_extension - set(extension))
        unexpected = sorted(set(extension) - expected_extension)
        raise ArtifactContractError(
            "Extension outcomes are not the exact 1,600-cell census: "
            f"observed={len(extension)}, missing_examples={missing[:3]}, "
            f"unexpected_examples={unexpected[:3]}"
        )
    if set(reference) != expected_reference:
        missing = sorted(expected_reference - set(reference))
        unexpected = sorted(set(reference) - expected_reference)
        raise ArtifactContractError(
            "Reference OC outcomes are not the exact 800-cell census: "
            f"observed={len(reference)}, missing_examples={missing[:3]}, "
            f"unexpected_examples={unexpected[:3]}"
        )
    combined = {**reference, **extension}
    successes = tuple(
        tuple(
            tuple(combined[(task, episode_id, arm)] for arm in ANALYSIS_ARMS)
            for episode_id in FORMAL_EPISODE_IDS
        )
        for task in FORMAL_TASKS
    )
    return CombinedOutcomeTable(successes)


def load_published_reference_oc(
    path: str | Path,
    *,
    expected_sha256: str = REFERENCE_PER_EPISODE_SHA256,
) -> list[dict[str, Any]]:
    """Load only OC outcomes from the immutable completed per-episode artifact."""
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ArtifactContractError("Published reference per_episode.csv digest mismatch")
    records = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            raw_success = row.get("OC_success")
            if raw_success not in {"0", "1"}:
                raise ArtifactContractError("Reference OC_success must be 0 or 1")
            records.append(
                {
                    "task": row.get("task"),
                    "episode_id": int(row["episode_id"]),
                    "arm": "OC",
                    "success": raw_success == "1",
                }
            )
    # Apply the exact-census validator to the reference side without inventing
    # dummy extension records.
    expected = {
        (task, episode_id, "OC")
        for task in FORMAL_TASKS
        for episode_id in FORMAL_EPISODE_IDS
    }
    normalized = _normalize_records(
        records,
        allowed_arms=("OC",),
        require_formal_metadata=False,
    )
    if set(normalized) != expected:
        raise ArtifactContractError("Published reference OC CSV is not the exact 800-cell census")
    return records


def _difference_arrays(table: CombinedOutcomeTable) -> dict[str, np.ndarray]:
    arrays = {}
    for treatment, control in COMPARISONS:
        arrays[f"{treatment} - {control}"] = np.asarray(
            [
                [
                    int(table.success(task, episode_id, treatment))
                    - int(table.success(task, episode_id, control))
                    for episode_id in FORMAL_EPISODE_IDS
                ]
                for task in FORMAL_TASKS
            ],
            dtype=np.float64,
        )
    return arrays


def _point_effect(
    table: CombinedOutcomeTable,
    treatment: str,
    control: str,
) -> dict[str, Any]:
    per_task = []
    for task in FORMAL_TASKS:
        treatment_values = np.asarray(
            [int(table.success(task, episode_id, treatment)) for episode_id in FORMAL_EPISODE_IDS]
        )
        control_values = np.asarray(
            [int(table.success(task, episode_id, control)) for episode_id in FORMAL_EPISODE_IDS]
        )
        per_task.append(
            {
                "task": task,
                "paired_episode_count": len(FORMAL_EPISODE_IDS),
                "treatment_success_rate": float(treatment_values.mean()),
                "control_success_rate": float(control_values.mean()),
                "paired_difference_pp": float(
                    100.0 * (treatment_values - control_values).mean()
                ),
            }
        )
    treatment_rate = math.fsum(
        row["treatment_success_rate"] for row in per_task
    ) / len(per_task)
    control_rate = math.fsum(row["control_success_rate"] for row in per_task) / len(
        per_task
    )
    estimate = treatment_rate - control_rate
    return {
        "comparison": f"{treatment} - {control}",
        "treatment": treatment,
        "control": control,
        "task_count": len(FORMAL_TASKS),
        "paired_episode_count": len(FORMAL_TASKS) * len(FORMAL_EPISODE_IDS),
        "treatment_equal_task_weighted_success_rate": treatment_rate,
        "control_equal_task_weighted_success_rate": control_rate,
        "estimate": estimate,
        "estimate_pp": 100.0 * estimate,
        "per_task": per_task,
    }


def _arm_outcomes(table: CombinedOutcomeTable) -> dict[str, dict[str, Any]]:
    """Report explicit pooled counts plus equal-task-weighted rates per arm."""
    denominator = len(FORMAL_TASKS) * len(FORMAL_EPISODE_IDS)
    outcomes: dict[str, dict[str, Any]] = {}
    for arm in ANALYSIS_ARMS:
        per_task = []
        for task in FORMAL_TASKS:
            success_count = sum(
                int(table.success(task, episode_id, arm))
                for episode_id in FORMAL_EPISODE_IDS
            )
            per_task.append(
                {
                    "task": task,
                    "success_count": success_count,
                    "episode_count": len(FORMAL_EPISODE_IDS),
                    "success_rate": success_count / len(FORMAL_EPISODE_IDS),
                }
            )
        pooled_success_count = sum(row["success_count"] for row in per_task)
        outcomes[arm] = {
            "pooled_success_count": pooled_success_count,
            "pooled_episode_count": denominator,
            "pooled_success_rate": pooled_success_count / denominator,
            "equal_task_weighted_success_rate": math.fsum(
                row["success_rate"] for row in per_task
            )
            / len(per_task),
            "per_task": per_task,
        }
    return outcomes


def _schedule_hasher(
    *,
    version: str,
    seed: int,
    replicates: int,
    task_count: int,
    episode_count: int,
    chunk_size: int,
) -> Any:
    hasher = hashlib.sha256()
    header = (
        f"{version}|seed={seed}|replicates={replicates}|tasks={task_count}|"
        f"episodes={episode_count}|chunk_size={chunk_size}\n"
    )
    hasher.update(header.encode("ascii"))
    return hasher


def _update_schedule_hash(
    hasher: Any,
    *,
    label: str,
    values: np.ndarray,
) -> None:
    normalized = np.ascontiguousarray(values)
    hasher.update(
        f"{label}|dtype={normalized.dtype.str}|shape={normalized.shape}\n".encode(
            "ascii"
        )
    )
    hasher.update(normalized.tobytes(order="C"))


def _common_bootstrap(
    differences: Mapping[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(seed))
    names = tuple(differences)
    values = np.stack([differences[name] for name in names])
    estimates = np.empty((len(names), replicates), dtype=np.float64)
    task_count, episode_count = values.shape[1:]
    schedule_hasher = _schedule_hasher(
        version=BOOTSTRAP_SCHEDULE_VERSION,
        seed=seed,
        replicates=replicates,
        task_count=task_count,
        episode_count=episode_count,
        chunk_size=BOOTSTRAP_CHUNK_SIZE,
    )
    for start in range(0, replicates, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, replicates)
        size = stop - start
        sampled_tasks = rng.integers(
            0,
            task_count,
            size=(size, task_count),
            dtype=np.int64,
        )
        sampled_episodes = rng.integers(
            0,
            episode_count,
            size=(size, task_count, episode_count),
            dtype=np.int64,
        )
        _update_schedule_hash(
            schedule_hasher,
            label=f"task_indices[{start}:{stop}]",
            values=sampled_tasks,
        )
        _update_schedule_hash(
            schedule_hasher,
            label=f"episode_indices[{start}:{stop}]",
            values=sampled_episodes,
        )
        for comparison_index in range(len(names)):
            sampled = values[comparison_index][
                sampled_tasks[:, :, None], sampled_episodes
            ]
            estimates[comparison_index, start:stop] = sampled.mean(axis=(1, 2))
    results = {}
    for comparison_index, name in enumerate(names):
        lower, upper = np.percentile(
            estimates[comparison_index],
            PERCENTILE_BOUNDS,
            method="linear",
        )
        results[name] = {
            "method": "paired hierarchical percentile bootstrap",
            "confidence_level": 0.95,
            "percentile_bounds": list(PERCENTILE_BOUNDS),
            "quantile_method": QUANTILE_METHOD,
            "seed": seed,
            "rng_implementation": RNG_IMPLEMENTATION,
            "replicates": replicates,
            "implementation_chunk_size": BOOTSTRAP_CHUNK_SIZE,
            "resampling_schedule_version": BOOTSTRAP_SCHEDULE_VERSION,
            "common_resampling_schedule_sha256": schedule_hasher.hexdigest(),
            "common_resampling_schedule_across_comparisons": True,
            "lower": float(lower),
            "upper": float(upper),
            "lower_pp": 100.0 * float(lower),
            "upper_pp": 100.0 * float(upper),
        }
    return results


def _common_randomization(
    differences: Mapping[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(seed))
    names = tuple(differences)
    values = np.stack([differences[name] for name in names])
    observed = values.mean(axis=(1, 2))
    thresholds = np.asarray(
        [math.nextafter(abs(float(value)), -math.inf) for value in observed]
    )
    extreme = np.zeros(len(names), dtype=np.int64)
    task_count, episode_count = values.shape[1:]
    schedule_hasher = _schedule_hasher(
        version=RANDOMIZATION_SCHEDULE_VERSION,
        seed=seed,
        replicates=replicates,
        task_count=task_count,
        episode_count=episode_count,
        chunk_size=RANDOMIZATION_CHUNK_SIZE,
    )
    for start in range(0, replicates, RANDOMIZATION_CHUNK_SIZE):
        size = min(RANDOMIZATION_CHUNK_SIZE, replicates - start)
        signs = rng.integers(
            0,
            2,
            size=(size, task_count, episode_count),
            dtype=np.int8,
        )
        signs = signs * 2 - 1
        _update_schedule_hash(
            schedule_hasher,
            label=f"signs[{start}:{start + size}]",
            values=signs,
        )
        permuted = (values[:, None, :, :] * signs[None, :, :, :]).mean(axis=(2, 3))
        extreme += np.count_nonzero(np.abs(permuted) >= thresholds[:, None], axis=1)
    return {
        name: {
            "method": "paired two-sided label-swap randomization test",
            "tail": "two-sided",
            "seed": seed,
            "rng_implementation": RNG_IMPLEMENTATION,
            "replicates": replicates,
            "implementation_chunk_size": RANDOMIZATION_CHUNK_SIZE,
            "randomization_schedule_version": RANDOMIZATION_SCHEDULE_VERSION,
            "common_randomization_schedule_sha256": schedule_hasher.hexdigest(),
            "common_randomization_schedule_across_comparisons": True,
            "observed_estimate": float(observed[index]),
            "observed_estimate_pp": 100.0 * float(observed[index]),
            "extreme_permutation_count": int(extreme[index]),
            "finite_permutation_correction": "(1 + extreme_count) / (replicates + 1)",
            "p_value": float((1 + extreme[index]) / (replicates + 1)),
        }
        for index, name in enumerate(names)
    }


def _discordant_counts(
    table: CombinedOutcomeTable,
    treatment: str,
    control: str,
) -> dict[str, Any]:
    per_task = []
    for task in FORMAL_TASKS:
        counts = {
            "treatment_success_control_fail": 0,
            "treatment_fail_control_success": 0,
            "both_success": 0,
            "both_fail": 0,
        }
        for episode_id in FORMAL_EPISODE_IDS:
            treatment_success = table.success(task, episode_id, treatment)
            control_success = table.success(task, episode_id, control)
            if treatment_success and not control_success:
                counts["treatment_success_control_fail"] += 1
            elif not treatment_success and control_success:
                counts["treatment_fail_control_success"] += 1
            elif treatment_success:
                counts["both_success"] += 1
            else:
                counts["both_fail"] += 1
        per_task.append(
            {"task": task, "denominator": len(FORMAL_EPISODE_IDS), **counts}
        )
    pooled = {
        field: sum(row[field] for row in per_task)
        for field in (
            "treatment_success_control_fail",
            "treatment_fail_control_success",
            "both_success",
            "both_fail",
        )
    }
    pooled["denominator"] = len(FORMAL_TASKS) * len(FORMAL_EPISODE_IDS)
    return {
        "comparison": f"{treatment} - {control}",
        "pooled": pooled,
        "per_task": per_task,
    }


def build_extension_analysis(
    extension_records: Iterable[Mapping[str, Any]],
    reference_oc_records: Iterable[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    randomization_replicates: int = DEFAULT_RANDOMIZATION_REPLICATES,
    allow_nonconfirmatory_replicate_override: bool = False,
) -> dict[str, Any]:
    """Build the frozen three-comparison extension analysis."""
    for field, value in (
        ("bootstrap_replicates", bootstrap_replicates),
        ("randomization_replicates", randomization_replicates),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ArtifactContractError(f"{field} must be a positive integer")
    frozen_replicates = (
        bootstrap_replicates == DEFAULT_BOOTSTRAP_REPLICATES
        and randomization_replicates == DEFAULT_RANDOMIZATION_REPLICATES
    )
    if not frozen_replicates and not allow_nonconfirmatory_replicate_override:
        raise AnalysisProtocolError(
            "Formal extension analysis requires exactly 100,000 bootstrap and "
            "100,000 randomization replicates"
        )
    table = validate_combined_records(extension_records, reference_oc_records)
    arm_outcomes = _arm_outcomes(table)
    differences = _difference_arrays(table)
    bootstrap = _common_bootstrap(
        differences,
        replicates=bootstrap_replicates,
        seed=ANALYSIS_SEED,
    )
    randomization = _common_randomization(
        differences,
        replicates=randomization_replicates,
        seed=ANALYSIS_SEED,
    )
    holm = holm_adjust(
        {name: result["p_value"] for name, result in randomization.items()}
    )
    bootstrap_schedule_digests = {
        result["common_resampling_schedule_sha256"]
        for result in bootstrap.values()
    }
    randomization_schedule_digests = {
        result["common_randomization_schedule_sha256"]
        for result in randomization.values()
    }
    if len(bootstrap_schedule_digests) != 1:
        raise AssertionError("Bootstrap comparisons did not share one schedule")
    if len(randomization_schedule_digests) != 1:
        raise AssertionError("Randomization comparisons did not share one schedule")
    comparisons = {}
    for treatment, control in COMPARISONS:
        name = f"{treatment} - {control}"
        comparisons[name] = {
            **_point_effect(table, treatment, control),
            "confidence_interval_95": bootstrap[name],
            "randomization_test": {
                **randomization[name],
                "holm_family": "three frozen extension comparisons",
                "holm_rank": holm[name]["holm_rank"],
                "holm_adjusted_p_value": holm[name]["holm_adjusted_p_value"],
            },
            "discordant_counts": _discordant_counts(table, treatment, control),
        }
    return {
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "experiment_type": "paired_post_hoc_follow_up",
        "post_hoc_follow_up": True,
        "analysis_plan_frozen_before_new_trajectories": True,
        "extension_cell_count": 1600,
        "reference_oc_cell_count": 800,
        "analysis_seed": ANALYSIS_SEED,
        "bootstrap_replicates": bootstrap_replicates,
        "randomization_replicates": randomization_replicates,
        "frozen_replicate_contract_met": frozen_replicates,
        "analysis_status": (
            "complete_frozen_extension"
            if frozen_replicates
            else "complete_nonconfirmatory_test_override"
        ),
        "resampling_reproducibility": {
            "rng_implementation": RNG_IMPLEMENTATION,
            "bootstrap": {
                "schedule_version": BOOTSTRAP_SCHEDULE_VERSION,
                "implementation_chunk_size": BOOTSTRAP_CHUNK_SIZE,
                "percentile_bounds": list(PERCENTILE_BOUNDS),
                "quantile_method": QUANTILE_METHOD,
                "common_schedule_sha256": next(iter(bootstrap_schedule_digests)),
            },
            "randomization": {
                "schedule_version": RANDOMIZATION_SCHEDULE_VERSION,
                "implementation_chunk_size": RANDOMIZATION_CHUNK_SIZE,
                "common_schedule_sha256": next(
                    iter(randomization_schedule_digests)
                ),
            },
        },
        "arm_outcomes": arm_outcomes,
        "comparisons": comparisons,
        "holm_family": {
            "method": "Holm step-down adjustment",
            "family_size": len(COMPARISONS),
            "comparisons": [f"{treatment} - {control}" for treatment, control in COMPARISONS],
            "results": holm,
        },
    }
