"""Protocol-safe confirmatory analysis for the frozen formal paired matrix.

The implementation intentionally starts by validating the exact 3200-cell
formal census.  Inferential procedures then preserve pairing within
task/episode blocks and give every task equal weight, including when the frozen
prior-exposure sensitivity set leaves different episode counts across tasks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import math
from typing import Any

import numpy as np

from experiments.keyframe_oracle_sampling.artifacts import ALL_ARMS
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError

ANALYSIS_SEED = 2026082502
DEFAULT_BOOTSTRAP_REPLICATES = 100_000
DEFAULT_RANDOMIZATION_REPLICATES = 100_000
FORMAL_EPISODE_IDS = tuple(range(50))
PRIMARY_COMPARISON = ("OC", "U")
SECONDARY_COMPARISONS = (
    ("O", "U"),
    ("OC", "O"),
    ("R", "U"),
    ("OC", "R"),
)
PRACTICAL_EFFECT_THRESHOLD_PP = 3.0

PROTOCOL_ANALYSIS_BLOCKERS: tuple[str, ...] = ()
CONFIDENCE_LEVEL = 0.95
PERCENTILE_BOUNDS = (2.5, 97.5)
RNG_IMPLEMENTATION = "numpy.random.Generator(PCG64)"
BOOTSTRAP_CHUNK_SIZE = 2_048
RANDOMIZATION_CHUNK_SIZE = 4_096


class AnalysisProtocolError(RuntimeError):
    """Raised rather than silently departing from the registered analysis."""


@dataclasses.dataclass(frozen=True)
class FormalOutcomeTable:
    """Immutable 16-task x 50-episode x 4-arm binary outcome table."""

    successes: tuple[tuple[tuple[bool, ...], ...], ...]

    def success(self, task: str, episode_id: int, arm: str) -> bool:
        try:
            task_index = FORMAL_TASKS.index(task)
            arm_index = ALL_ARMS.index(arm)
            episode_index = FORMAL_EPISODE_IDS.index(int(episode_id))
        except ValueError as exc:
            raise ArtifactContractError(
                f"Outcome lookup is outside the frozen formal matrix: "
                f"{task}/{episode_id}/{arm}"
            ) from exc
        return self.successes[task_index][episode_index][arm_index]

    @property
    def cell_count(self) -> int:
        return len(FORMAL_TASKS) * len(FORMAL_EPISODE_IDS) * len(ALL_ARMS)


@dataclasses.dataclass(frozen=True)
class ExposureManifest:
    entries: tuple[tuple[str, str, int], ...]
    excluded_formal_blocks: frozenset[tuple[str, int]]


def _coalesce_key_field(
    record: Mapping[str, Any], scientific_key: Mapping[str, Any] | None, field: str
) -> Any:
    values = []
    if field in record:
        values.append(record[field])
    if scientific_key is not None and field in scientific_key:
        values.append(scientific_key[field])
    if not values:
        raise ArtifactContractError(f"Formal analysis record lacks {field!r}")
    if any(value != values[0] for value in values[1:]):
        raise ArtifactContractError(f"Conflicting flat/nested values for {field!r}")
    return values[0]


def validate_formal_records(records: Iterable[Mapping[str, Any]]) -> FormalOutcomeTable:
    """Require exactly one completed binary result for every frozen formal cell."""
    observed: dict[tuple[str, int, str], bool] = {}
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ArtifactContractError(
                f"Formal analysis record {record_index} is not a mapping"
            )
        raw_key = record.get("scientific_key")
        if raw_key is not None and not isinstance(raw_key, Mapping):
            raise ArtifactContractError(
                f"Formal analysis record {record_index} has an invalid scientific_key"
            )
        task = _coalesce_key_field(record, raw_key, "task")
        episode_id = _coalesce_key_field(record, raw_key, "episode_id")
        arm = _coalesce_key_field(record, raw_key, "arm")
        trajectory_kind = _coalesce_key_field(
            record, raw_key, "trajectory_kind"
        )
        if not isinstance(task, str) or task not in FORMAL_TASKS:
            raise ArtifactContractError(f"Unexpected formal task: {task!r}")
        if (
            isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
            or episode_id not in FORMAL_EPISODE_IDS
        ):
            raise ArtifactContractError(
                f"Unexpected formal episode for {task}: {episode_id!r}"
            )
        if not isinstance(arm, str) or arm not in ALL_ARMS:
            raise ArtifactContractError(f"Unexpected formal arm: {arm!r}")
        if trajectory_kind != "formal":
            raise ArtifactContractError(
                f"Formal analysis received trajectory_kind={trajectory_kind!r}"
            )
        if record.get("dataset") != "test":
            raise ArtifactContractError(
                f"Formal analysis received dataset={record.get('dataset')!r}"
            )
        success = record.get("success")
        if type(success) is not bool:
            raise ArtifactContractError(
                f"Formal result {task}/{episode_id}/{arm} lacks a binary bool success"
            )
        key = (task, episode_id, arm)
        if key in observed:
            raise ArtifactContractError(f"Duplicate formal scientific cell: {key}")
        observed[key] = success

    expected = {
        (task, episode_id, arm)
        for task in FORMAL_TASKS
        for episode_id in FORMAL_EPISODE_IDS
        for arm in ALL_ARMS
    }
    observed_keys = set(observed)
    missing = sorted(expected - observed_keys)
    unexpected = sorted(observed_keys - expected)
    if missing or unexpected:
        raise ArtifactContractError(
            "Formal result matrix is not the exact frozen 3200-cell census: "
            f"observed={len(observed_keys)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}, missing_examples={missing[:3]}, "
            f"unexpected_examples={unexpected[:3]}"
        )

    successes = tuple(
        tuple(
            tuple(observed[(task, episode_id, arm)] for arm in ALL_ARMS)
            for episode_id in FORMAL_EPISODE_IDS
        )
        for task in FORMAL_TASKS
    )
    table = FormalOutcomeTable(successes)
    if table.cell_count != 3200:
        raise AssertionError("Frozen formal table must contain exactly 3200 cells")
    return table


def _validate_comparison(treatment: str, control: str) -> None:
    if treatment not in ALL_ARMS or control not in ALL_ARMS:
        raise ArtifactContractError(
            f"Comparison uses an unknown arm: {treatment} - {control}"
        )
    if treatment == control:
        raise ArtifactContractError("Paired comparison requires two distinct arms")


def paired_effect(
    table: FormalOutcomeTable,
    treatment: str,
    control: str,
    *,
    excluded_blocks: frozenset[tuple[str, int]] = frozenset(),
) -> dict[str, Any]:
    """Compute the section 12 equal-task-weighted paired binary estimand."""
    _validate_comparison(treatment, control)
    per_task = []
    for task in FORMAL_TASKS:
        episodes = [
            episode_id
            for episode_id in FORMAL_EPISODE_IDS
            if (task, episode_id) not in excluded_blocks
        ]
        if not episodes:
            raise ArtifactContractError(
                f"Exposure sensitivity leaves no paired episodes for task {task}"
            )
        treatment_values = [
            int(table.success(task, episode_id, treatment)) for episode_id in episodes
        ]
        control_values = [
            int(table.success(task, episode_id, control)) for episode_id in episodes
        ]
        treatment_rate = math.fsum(treatment_values) / len(episodes)
        control_rate = math.fsum(control_values) / len(episodes)
        per_task.append(
            {
                "task": task,
                "paired_episode_count": len(episodes),
                "treatment_success_rate": treatment_rate,
                "control_success_rate": control_rate,
                "paired_difference": treatment_rate - control_rate,
                "paired_difference_pp": 100.0 * (treatment_rate - control_rate),
            }
        )
    treatment_macro = math.fsum(
        row["treatment_success_rate"] for row in per_task
    ) / len(per_task)
    control_macro = math.fsum(row["control_success_rate"] for row in per_task) / len(
        per_task
    )
    estimate = math.fsum(row["paired_difference"] for row in per_task) / len(
        per_task
    )
    return {
        "comparison": f"{treatment} - {control}",
        "treatment": treatment,
        "control": control,
        "task_count": len(per_task),
        "paired_episode_count": sum(
            row["paired_episode_count"] for row in per_task
        ),
        "treatment_equal_task_weighted_success_rate": treatment_macro,
        "control_equal_task_weighted_success_rate": control_macro,
        "estimate": estimate,
        "estimate_pp": 100.0 * estimate,
        "per_task": per_task,
    }


def _validate_replicates(replicates: int, *, field: str) -> int:
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise ArtifactContractError(f"{field} must be an integer")
    if replicates <= 0:
        raise ArtifactContractError(f"{field} must be positive")
    return replicates


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ArtifactContractError("Analysis seed must be a non-negative integer")
    return seed


def _paired_differences_by_task(
    table: FormalOutcomeTable,
    treatment: str,
    control: str,
    *,
    excluded_blocks: frozenset[tuple[str, int]] = frozenset(),
) -> tuple[np.ndarray, ...]:
    """Return ordered paired binary differences, preserving every task."""
    _validate_comparison(treatment, control)
    differences: list[np.ndarray] = []
    for task in FORMAL_TASKS:
        values = [
            int(table.success(task, episode_id, treatment))
            - int(table.success(task, episode_id, control))
            for episode_id in FORMAL_EPISODE_IDS
            if (task, episode_id) not in excluded_blocks
        ]
        if not values:
            raise ArtifactContractError(
                f"Exposure sensitivity leaves no paired episodes for task {task}"
            )
        differences.append(np.asarray(values, dtype=np.float64))
    return tuple(differences)


def hierarchical_percentile_bootstrap_ci(
    table: FormalOutcomeTable,
    treatment: str,
    control: str,
    *,
    excluded_blocks: frozenset[tuple[str, int]] = frozenset(),
    seed: int = ANALYSIS_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Paired equal-task hierarchical percentile-bootstrap 95% CI.

    Each replicate first samples 16 task identities with replacement.  For each
    sampled task occurrence it then independently samples paired episodes from
    that task with replacement, using that task's observed episode count.  The
    replicate estimate is the mean of the 16 sampled task-level means.

    ``replicates`` is injectable so unit tests can stay small.  Confirmatory
    report construction rejects non-frozen replicate counts unless its explicit
    test-only override is enabled.
    """
    replicates = _validate_replicates(
        replicates, field="hierarchical bootstrap replicates"
    )
    seed = _validate_seed(seed)
    differences = _paired_differences_by_task(
        table,
        treatment,
        control,
        excluded_blocks=excluded_blocks,
    )
    task_count = len(differences)
    episode_counts = np.asarray([len(values) for values in differences], dtype=np.int64)
    max_episode_count = int(episode_counts.max())
    padded = np.zeros((task_count, max_episode_count), dtype=np.float64)
    for task_index, values in enumerate(differences):
        padded[task_index, : len(values)] = values

    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap_estimates = np.empty(replicates, dtype=np.float64)
    episode_positions = np.arange(max_episode_count, dtype=np.int64)
    for start in range(0, replicates, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, replicates)
        chunk_size = stop - start
        sampled_tasks = rng.integers(
            0,
            task_count,
            size=(chunk_size, task_count),
            dtype=np.int64,
        )
        sampled_task_means = np.empty((chunk_size, task_count), dtype=np.float64)
        # Repeated task identities are separate cluster occurrences and receive
        # independent within-task episode resamples, as required by a two-level
        # hierarchical bootstrap.
        for sampled_position in range(task_count):
            task_indices = sampled_tasks[:, sampled_position]
            row_counts = episode_counts[task_indices]
            uniforms = rng.random((chunk_size, max_episode_count))
            sampled_episode_indices = np.floor(
                uniforms * row_counts[:, None]
            ).astype(np.int64)
            sampled_values = padded[task_indices[:, None], sampled_episode_indices]
            active = episode_positions[None, :] < row_counts[:, None]
            sampled_task_means[:, sampled_position] = (
                np.sum(sampled_values * active, axis=1) / row_counts
            )
        bootstrap_estimates[start:stop] = np.mean(sampled_task_means, axis=1)

    lower, upper = np.percentile(
        bootstrap_estimates,
        PERCENTILE_BOUNDS,
        method="linear",
    )
    return {
        "method": "paired hierarchical percentile bootstrap",
        "confidence_level": CONFIDENCE_LEVEL,
        "percentile_bounds": list(PERCENTILE_BOUNDS),
        "quantile_method": "numpy-linear",
        "seed": seed,
        "rng_implementation": RNG_IMPLEMENTATION,
        "replicates": replicates,
        "implementation_chunk_size": BOOTSTRAP_CHUNK_SIZE,
        "task_resampling": "tasks sampled with replacement with equal probability",
        "episode_resampling": (
            "paired episodes sampled with replacement independently within each "
            "sampled task occurrence"
        ),
        "task_count": task_count,
        "per_task_episode_counts": {
            task: int(count)
            for task, count in zip(FORMAL_TASKS, episode_counts, strict=True)
        },
        "lower": float(lower),
        "upper": float(upper),
        "lower_pp": 100.0 * float(lower),
        "upper_pp": 100.0 * float(upper),
    }


def paired_randomization_test(
    table: FormalOutcomeTable,
    treatment: str,
    control: str,
    *,
    excluded_blocks: frozenset[tuple[str, int]] = frozenset(),
    seed: int = ANALYSIS_SEED,
    replicates: int = DEFAULT_RANDOMIZATION_REPLICATES,
) -> dict[str, Any]:
    """Two-sided paired label-swap randomization test with +1 correction."""
    replicates = _validate_replicates(
        replicates, field="paired randomization replicates"
    )
    seed = _validate_seed(seed)
    differences = _paired_differences_by_task(
        table,
        treatment,
        control,
        excluded_blocks=excluded_blocks,
    )
    task_count = len(differences)
    coefficients = np.concatenate(
        [values / (task_count * len(values)) for values in differences]
    )
    observed = float(math.fsum(float(value) for value in coefficients))
    # nextafter makes mathematically equal values count as ties even if a BLAS
    # accumulation differs from the scalar observed sum by one floating ULP.
    threshold = math.nextafter(abs(observed), -math.inf)
    rng = np.random.Generator(np.random.PCG64(seed))
    extreme_count = 0
    for start in range(0, replicates, RANDOMIZATION_CHUNK_SIZE):
        chunk_size = min(RANDOMIZATION_CHUNK_SIZE, replicates - start)
        signs = rng.integers(
            0,
            2,
            size=(chunk_size, len(coefficients)),
            dtype=np.int8,
        )
        signs = signs * 2 - 1
        permuted = signs @ coefficients
        extreme_count += int(np.count_nonzero(np.abs(permuted) >= threshold))
    p_value = (1.0 + extreme_count) / (replicates + 1.0)
    return {
        "method": "paired two-sided label-swap randomization test",
        "tail": "two-sided",
        "seed": seed,
        "rng_implementation": RNG_IMPLEMENTATION,
        "replicates": replicates,
        "implementation_chunk_size": RANDOMIZATION_CHUNK_SIZE,
        "equal_task_weighting": True,
        "paired_block_count": int(sum(len(values) for values in differences)),
        "per_task_episode_counts": {
            task: len(values)
            for task, values in zip(FORMAL_TASKS, differences, strict=True)
        },
        "observed_estimate": observed,
        "observed_estimate_pp": 100.0 * observed,
        "extreme_permutation_count": extreme_count,
        "finite_permutation_correction": "(1 + extreme_count) / (replicates + 1)",
        "p_value": p_value,
    }


def holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """Apply deterministic Holm step-down adjustment to one named family."""
    if not raw_p_values:
        raise ArtifactContractError("Holm adjustment requires at least one p-value")
    items = list(raw_p_values.items())
    for comparison, p_value in items:
        if (
            isinstance(p_value, bool)
            or not isinstance(p_value, (int, float))
            or not math.isfinite(float(p_value))
            or not 0.0 <= float(p_value) <= 1.0
        ):
            raise ArtifactContractError(
                f"Invalid raw p-value for Holm adjustment: {comparison}={p_value!r}"
            )
    order = sorted(range(len(items)), key=lambda index: (float(items[index][1]), index))
    adjusted: dict[str, dict[str, Any]] = {}
    running_max = 0.0
    family_size = len(items)
    for zero_based_rank, item_index in enumerate(order):
        comparison, raw_p = items[item_index]
        candidate = (family_size - zero_based_rank) * float(raw_p)
        running_max = min(1.0, max(running_max, candidate))
        adjusted[comparison] = {
            "raw_p_value": float(raw_p),
            "holm_rank": zero_based_rank + 1,
            "holm_adjusted_p_value": running_max,
        }
    return adjusted


def _inferential_effect(
    table: FormalOutcomeTable,
    treatment: str,
    control: str,
    *,
    excluded_blocks: frozenset[tuple[str, int]] = frozenset(),
    bootstrap_replicates: int,
    randomization_replicates: int | None,
) -> dict[str, Any]:
    result = paired_effect(
        table,
        treatment,
        control,
        excluded_blocks=excluded_blocks,
    )
    result["confidence_interval_95"] = hierarchical_percentile_bootstrap_ci(
        table,
        treatment,
        control,
        excluded_blocks=excluded_blocks,
        seed=ANALYSIS_SEED,
        replicates=bootstrap_replicates,
    )
    if randomization_replicates is not None:
        result["randomization_test"] = paired_randomization_test(
            table,
            treatment,
            control,
            excluded_blocks=excluded_blocks,
            seed=ANALYSIS_SEED,
            replicates=randomization_replicates,
        )
    return result


def _ci_beats_zero(effect: Mapping[str, Any]) -> bool:
    interval = effect.get("confidence_interval_95")
    if not isinstance(interval, Mapping):
        raise ArtifactContractError("Decision input lacks a 95% confidence interval")
    lower = interval.get("lower_pp")
    if not isinstance(lower, (int, float)) or not math.isfinite(float(lower)):
        raise ArtifactContractError("Decision input has an invalid CI lower bound")
    return float(lower) > 0.0


def classify_decision(
    primary: Mapping[str, Any],
    oc_vs_r: Mapping[str, Any],
    oc_vs_o: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered GO/NO-GO rule without p-value substitution."""
    primary_interval = primary.get("confidence_interval_95")
    oc_vs_r_interval = oc_vs_r.get("confidence_interval_95")
    if not isinstance(primary_interval, Mapping) or not isinstance(
        oc_vs_r_interval, Mapping
    ):
        raise ArtifactContractError("Decision classification requires both CIs")
    estimate_pp = float(primary["estimate_pp"])
    primary_lower = float(primary_interval["lower_pp"])
    primary_upper = float(primary_interval["upper_pp"])
    oc_vs_r_lower = float(oc_vs_r_interval["lower_pp"])
    oc_vs_r_upper = float(oc_vs_r_interval["upper_pp"])
    for value in (
        estimate_pp,
        primary_lower,
        primary_upper,
        oc_vs_r_lower,
        oc_vs_r_upper,
    ):
        if not math.isfinite(value):
            raise ArtifactContractError("Decision classification received non-finite input")

    oc_beats_u = _ci_beats_zero(primary)
    oc_beats_r = _ci_beats_zero(oc_vs_r)
    oc_beats_o = _ci_beats_zero(oc_vs_o)
    oc_not_distinguished_from_r = oc_vs_r_lower <= 0.0 <= oc_vs_r_upper
    semantic_selection_go = (
        estimate_pp >= PRACTICAL_EFFECT_THRESHOLD_PP
        and oc_beats_u
        and oc_beats_r
    )
    decisive_no_go = primary_upper < PRACTICAL_EFFECT_THRESHOLD_PP
    sampling_sensitivity_only = oc_beats_u and oc_not_distinguished_from_r

    satisfied_conclusions = []
    if semantic_selection_go:
        satisfied_conclusions.append("semantic_selection_go")
    if decisive_no_go:
        satisfied_conclusions.append("decisive_test_time_no_go")
    if sampling_sensitivity_only:
        satisfied_conclusions.append("sampling_sensitivity_only")
    if not satisfied_conclusions:
        satisfied_conclusions.append("inconclusive")
    classification = (
        satisfied_conclusions[0]
        if len(satisfied_conclusions) == 1
        else "overlapping_prespecified_conclusions"
    )
    return {
        "classification": classification,
        "practical_effect_threshold_pp": PRACTICAL_EFFECT_THRESHOLD_PP,
        "beats_definition": "the corresponding 95% CI lower bound is greater than 0",
        "not_distinguished_definition": (
            "the corresponding 95% CI contains 0 (lower <= 0 <= upper)"
        ),
        "oc_beats_u": oc_beats_u,
        "oc_beats_r": oc_beats_r,
        "oc_not_distinguished_from_r": oc_not_distinguished_from_r,
        "semantic_selection_go": semantic_selection_go,
        "decisive_test_time_no_go": decisive_no_go,
        "sampling_sensitivity_only": sampling_sensitivity_only,
        "coverage_complementarity_evidence": oc_beats_o,
        "satisfied_prespecified_conclusions": satisfied_conclusions,
        "overlap_handling": (
            "report every satisfied preregistered condition; no unregistered "
            "priority is imposed"
        ),
    }


def discordant_counts(
    table: FormalOutcomeTable,
    treatment: str = "OC",
    control: str = "U",
    *,
    excluded_blocks: frozenset[tuple[str, int]] = frozenset(),
) -> dict[str, Any]:
    """Report both discordant directions without dropping ties or denominators."""
    _validate_comparison(treatment, control)
    per_task = []
    for task in FORMAL_TASKS:
        counts = {
            "treatment_success_control_fail": 0,
            "treatment_fail_control_success": 0,
            "both_success": 0,
            "both_fail": 0,
        }
        denominator = 0
        for episode_id in FORMAL_EPISODE_IDS:
            if (task, episode_id) in excluded_blocks:
                continue
            denominator += 1
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
        if denominator == 0:
            raise ArtifactContractError(
                f"Exposure sensitivity leaves no discordant denominator for {task}"
            )
        per_task.append({"task": task, "denominator": denominator, **counts})
    pooled = {
        key: sum(row[key] for row in per_task)
        for key in (
            "treatment_success_control_fail",
            "treatment_fail_control_success",
            "both_success",
            "both_fail",
        )
    }
    pooled["denominator"] = sum(row["denominator"] for row in per_task)
    return {
        "comparison": f"{treatment} - {control}",
        "pooled": pooled,
        "per_task": per_task,
    }


def normalize_prior_exposure_manifest(
    manifest: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> ExposureManifest:
    """Normalize task/split/episode entries without inspecting formal outcomes."""
    if isinstance(manifest, Mapping):
        entries = manifest.get("entries")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ArtifactContractError(
                "Prior-exposure manifest mapping must contain an entries sequence"
            )
    elif isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)):
        entries = manifest
    else:
        raise ArtifactContractError("Prior-exposure manifest has an invalid shape")

    normalized = []
    seen = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ArtifactContractError(f"Exposure entry {index} is not a mapping")
        task = entry.get("task")
        episode_id = entry.get("episode_id")
        split_values = [entry[field] for field in ("split", "dataset") if field in entry]
        if len(split_values) != 1:
            raise ArtifactContractError(
                f"Exposure entry {index} must define exactly one of split or dataset"
            )
        split = split_values[0]
        if not isinstance(task, str) or not task:
            raise ArtifactContractError(f"Exposure entry {index} has invalid task")
        if not isinstance(split, str) or not split:
            raise ArtifactContractError(f"Exposure entry {index} has invalid split")
        if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id < 0:
            raise ArtifactContractError(f"Exposure entry {index} has invalid episode_id")
        key = (task, split, episode_id)
        if key in seen:
            raise ArtifactContractError(f"Duplicate prior-exposure entry: {key}")
        seen.add(key)
        normalized.append(key)
    excluded = frozenset(
        (task, episode_id)
        for task, split, episode_id in normalized
        if split == "test"
        and task in FORMAL_TASKS
        and episode_id in FORMAL_EPISODE_IDS
    )
    return ExposureManifest(tuple(normalized), excluded)


def build_point_analysis(
    records: Iterable[Mapping[str, Any]],
    *,
    prior_exposure_manifest: Mapping[str, Any]
    | Sequence[Mapping[str, Any]]
    | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    randomization_replicates: int = DEFAULT_RANDOMIZATION_REPLICATES,
    allow_nonconfirmatory_replicate_override: bool = False,
) -> dict[str, Any]:
    """Build the complete preregistered formal report.

    Production calls use the frozen 100,000 replicates for both procedures.  A
    smaller count is accepted only behind an explicit nonconfirmatory flag so
    unit tests cannot accidentally emit a report that looks confirmatory.
    """
    bootstrap_replicates = _validate_replicates(
        bootstrap_replicates, field="hierarchical bootstrap replicates"
    )
    randomization_replicates = _validate_replicates(
        randomization_replicates, field="paired randomization replicates"
    )
    frozen_replicate_contract_met = (
        bootstrap_replicates == DEFAULT_BOOTSTRAP_REPLICATES
        and randomization_replicates == DEFAULT_RANDOMIZATION_REPLICATES
    )
    if not frozen_replicate_contract_met and not allow_nonconfirmatory_replicate_override:
        raise AnalysisProtocolError(
            "Confirmatory analysis requires exactly 100,000 bootstrap and 100,000 "
            "randomization replicates; a smaller count is test-only and requires "
            "allow_nonconfirmatory_replicate_override=True"
        )
    table = validate_formal_records(records)
    primary = _inferential_effect(
        table,
        *PRIMARY_COMPARISON,
        bootstrap_replicates=bootstrap_replicates,
        randomization_replicates=randomization_replicates,
    )
    secondary: dict[str, dict[str, Any]] = {}
    for treatment, control in SECONDARY_COMPARISONS:
        comparison = f"{treatment} - {control}"
        secondary[comparison] = _inferential_effect(
            table,
            treatment,
            control,
            bootstrap_replicates=bootstrap_replicates,
            randomization_replicates=randomization_replicates,
        )
    holm = holm_adjust(
        {
            comparison: effect["randomization_test"]["p_value"]
            for comparison, effect in secondary.items()
        }
    )
    for comparison, effect in secondary.items():
        effect["randomization_test"]["holm_family"] = (
            "four prespecified secondary paired comparisons"
        )
        effect["randomization_test"]["holm_rank"] = holm[comparison]["holm_rank"]
        effect["randomization_test"]["holm_adjusted_p_value"] = holm[comparison][
            "holm_adjusted_p_value"
        ]

    sensitivity = None
    if prior_exposure_manifest is not None:
        exposure = normalize_prior_exposure_manifest(prior_exposure_manifest)
        sensitivity = {
            "label": "frozen prior-exposure sensitivity analysis",
            "manifest_entry_count": len(exposure.entries),
            "excluded_formal_block_count": len(exposure.excluded_formal_blocks),
            "effect": _inferential_effect(
                table,
                *PRIMARY_COMPARISON,
                excluded_blocks=exposure.excluded_formal_blocks,
                bootstrap_replicates=bootstrap_replicates,
                randomization_replicates=None,
            ),
            "discordant_counts": discordant_counts(
                table,
                *PRIMARY_COMPARISON,
                excluded_blocks=exposure.excluded_formal_blocks,
            ),
        }
    decision = classify_decision(
        primary,
        secondary["OC - R"],
        secondary["OC - O"],
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "formal_cell_count": table.cell_count,
        "analysis_seed": ANALYSIS_SEED,
        "rng_implementation": RNG_IMPLEMENTATION,
        "rng_stream_policy": (
            "initialize a fresh PCG64 stream from analysis_seed for each "
            "comparison and inferential procedure"
        ),
        "numpy_version": np.__version__,
        "bootstrap_replicates": bootstrap_replicates,
        "randomization_replicates": randomization_replicates,
        "frozen_bootstrap_replicates": DEFAULT_BOOTSTRAP_REPLICATES,
        "frozen_randomization_replicates": DEFAULT_RANDOMIZATION_REPLICATES,
        "frozen_replicate_contract_met": frozen_replicate_contract_met,
        "primary": primary,
        "primary_discordant_counts": discordant_counts(table),
        "secondary": secondary,
        # Kept as an explicit point-estimate view for downstream table writers.
        "secondary_unadjusted_point_estimates": {
            comparison: {
                key: value
                for key, value in effect.items()
                if key not in {"confidence_interval_95", "randomization_test"}
            }
            for comparison, effect in secondary.items()
        },
        "secondary_holm_family": {
            "method": "Holm step-down adjustment",
            "family_size": len(SECONDARY_COMPARISONS),
            "comparisons": [
                f"{treatment} - {control}"
                for treatment, control in SECONDARY_COMPARISONS
            ],
            "results": holm,
        },
        "prior_exposure_sensitivity": sensitivity,
        "inferential_analysis_status": (
            "complete_confirmatory"
            if frozen_replicate_contract_met
            else "complete_nonconfirmatory_test_override"
        ),
        "protocol_analysis_blockers": [],
        "decision_rule": decision,
        "decision_classification": (
            decision["classification"] if frozen_replicate_contract_met else None
        ),
        "nonconfirmatory_decision_preview": (
            None if frozen_replicate_contract_met else decision["classification"]
        ),
    }


def require_resolved_inferential_protocol() -> None:
    """Return successfully now that the versioned amendment resolves inference."""
