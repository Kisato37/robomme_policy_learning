"""Pure paired-outcome analysis for same-run Lighthouse U/UK48/UN48.

No artifacts are loaded or written here.  Upstream aggregation must supply the
first valid scientific outcome per cell, not infrastructure/invariant failures.
The run attestation binds the caller's raw-provenance and three-arm initial-input
audit to these exact outcomes; this function cannot authenticate that audit by itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from experiments.keyframe_oracle_sampling import analysis as parent
from experiments.uniform_keyframe_expansion.contract import FORMAL_TASKS
from experiments.uniform_keyframe_expansion.contract import PROTOCOL_FAMILY
from experiments.uniform_keyframe_expansion.contract import canonical_sha256

ANALYSIS_SEED = parent.ANALYSIS_SEED
DEFAULT_BOOTSTRAP_REPLICATES = parent.DEFAULT_BOOTSTRAP_REPLICATES
DEFAULT_RANDOMIZATION_REPLICATES = parent.DEFAULT_RANDOMIZATION_REPLICATES
ARMS = ("U", "UK48", "UN48")
PRIMARY_COMPARISON = ("UK48", "U")
AUXILIARY_COMPARISONS = (("UN48", "U"), ("UK48", "UN48"))
PRACTICAL_EFFECT_THRESHOLD_PP = 3.0
_TO_PARENT = {"U": "U", "UK48": "OC", "UN48": "R"}
_FROM_PARENT = {value: key for key, value in _TO_PARENT.items()}


class ExpansionAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class OutcomeTable:
    # Canonical immutable (task, episode, U-success, UK-success, UN-success) rows.
    blocks: tuple[tuple[str, int, bool, bool, bool], ...]

    @property
    def cell_count(self) -> int:
        return 3 * len(self.blocks)

    @property
    def excluded_blocks(self) -> frozenset[tuple[str, int]]:
        present = {(row[0], row[1]) for row in self.blocks}
        return frozenset((task, episode) for task in FORMAL_TASKS for episode in range(50)
                         if (task, episode) not in present)

    def records(self) -> list[dict[str, Any]]:
        return [{"task": block[0], "episode_id": block[1], "arm": arm,
                 "dataset": "test", "trajectory_kind": "formal", "success": block[2 + index]}
                for block in self.blocks for index, arm in enumerate(ARMS)]


class _ParentTableAdapter:
    """Read-only arm aliases reuse the exact existing statistical algorithms.

    These are internal array addresses, not relabelled experiment records or
    copied outcomes. No global parent constants or validators are modified.
    """
    def __init__(self, table: OutcomeTable):
        self.lookup = {(block[0], block[1], arm): block[2 + index]
                       for block in table.blocks for index, arm in enumerate(ARMS)}

    def success(self, task: str, episode_id: int, arm: str) -> bool:
        return self.lookup[(task, episode_id, _FROM_PARENT[arm])]


def _field(record: Mapping[str, Any], field: str) -> Any:
    nested = record.get("scientific_key")
    if nested is not None and not isinstance(nested, Mapping):
        raise ExpansionAnalysisError("scientific_key must be a mapping")
    values = [source[field] for source in (record, nested) if source is not None and field in source]
    if not values:
        raise ExpansionAnalysisError(f"Outcome lacks {field}")
    if any(type(value) is not type(values[0]) or value != values[0] for value in values[1:]):
        raise ExpansionAnalysisError(f"Conflicting outcome field {field}")
    return values[0]


def validate_outcome_records(
    records: Iterable[Mapping[str, Any]] | Mapping[str, Any], *, allow_fixture: bool = False,
) -> OutcomeTable:
    """Exact 800 paired blocks by default; fixtures still retain all 16 tasks."""
    if type(allow_fixture) is not bool:
        raise ExpansionAnalysisError("allow_fixture must be a bool")
    if isinstance(records, Mapping):
        records = records.get("records")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ExpansionAnalysisError("Outcome mapping must contain a records sequence")
    if not isinstance(records, Iterable) or isinstance(records, (str, bytes)):
        raise ExpansionAnalysisError("Outcomes must be an iterable of record mappings")
    observed: dict[tuple[str, int, str], bool] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ExpansionAnalysisError("Every outcome must be a mapping")
        task, episode, arm = (_field(record, name) for name in ("task", "episode_id", "arm"))
        if type(task) is not str or task not in FORMAL_TASKS:
            raise ExpansionAnalysisError("Unknown task")
        if type(episode) is not int or episode not in range(50):
            raise ExpansionAnalysisError("Episode must be an integer in 0..49")
        if type(arm) is not str or arm not in ARMS:
            raise ExpansionAnalysisError("Unknown arm")
        if _field(record, "dataset") != "test" or _field(record, "trajectory_kind") != "formal":
            raise ExpansionAnalysisError("Only formal test outcomes may enter analysis")
        success = record.get("success")
        if type(success) is not bool:
            raise ExpansionAnalysisError("success must be an exact boolean")
        # Optional status must not smuggle a retryable failure into the binary
        # table. The adapter does not guess a missing outcome from a log.
        if "terminal_status" in record:
            status = record["terminal_status"]
            if status not in ("success", "fail", "timeout", "error"):
                raise ExpansionAnalysisError("Non-scientific terminal status")
            if success != (status == "success"):
                raise ExpansionAnalysisError("terminal_status contradicts success")
        if record.get("infrastructure_failure", False) is not False or record.get("protocol_invariant_failure", False) is not False:
            raise ExpansionAnalysisError("Infrastructure/invariant failures are not scientific outcomes")
        key = (task, episode, arm)
        if key in observed:
            raise ExpansionAnalysisError(f"Duplicate scientific cell: {key}")
        observed[key] = success
    if not allow_fixture:
        expected = {(task, episode, arm) for task in FORMAL_TASKS for episode in range(50) for arm in ARMS}
        if set(observed) != expected:
            raise ExpansionAnalysisError(f"Analysis requires the exact 2400-cell census; got {len(observed)}")
    present = {(task, episode) for task, episode, _ in observed}
    if {task for task, _ in present} != set(FORMAL_TASKS):
        raise ExpansionAnalysisError("Even fixtures must contain at least one complete block per canonical task")
    if any((task, episode, arm) not in observed for task, episode in present for arm in ARMS):
        raise ExpansionAnalysisError("Every paired block requires U, UK48 and UN48")
    blocks = tuple((task, episode, *(observed[(task, episode, arm)] for arm in ARMS))
                   for task in FORMAL_TASKS for episode in range(50) if (task, episode) in present)
    return OutcomeTable(blocks)


def outcome_binding_hashes(table: OutcomeTable) -> dict[str, str]:
    records = table.records()
    return {"all_outcomes_sha256": canonical_sha256(records),
            "u_outcomes_sha256": canonical_sha256([record for record in records if record["arm"] == "U"])}


def _validate_run_attestation(attestation: Any, table: OutcomeTable) -> dict[str, Any]:
    if not isinstance(attestation, Mapping):
        raise ExpansionAnalysisError("Formal analysis needs a verified same-run three-arm attestation, not a bool")
    required_true = ("all_2400_outcomes_audited", "initial_inputs_states_text_matched",
                     "seed_difficulty_mapping_matched", "checkpoint_control_pipeline_matched",
                     "execution_provenance_reviewed", "same_run_three_arm_pairing_verified")
    if attestation.get("status") != "verified" or any(attestation.get(key) is not True for key in required_true):
        raise ExpansionAnalysisError("Same-run three-arm/initial-input audit is not verified")
    for key in ("source_run_id", "comparison_audit_id"):
        if not isinstance(attestation.get(key), str) or not attestation[key].strip():
            raise ExpansionAnalysisError(f"Run attestation needs {key}")
    for key in ("raw_provenance_manifest_sha256", "comparison_audit_sha256"):
        digest = attestation.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ExpansionAnalysisError(f"Run attestation needs a valid {key}")
    for key, digest in outcome_binding_hashes(table).items():
        if attestation.get(key) != digest:
            raise ExpansionAnalysisError(f"Run attestation does not bind these exact outcomes: {key}")
    return dict(attestation)


def classify_primary_result(primary: Mapping[str, Any]) -> dict[str, Any]:
    """Primary success never depends on beating UN48 or on a p-value threshold."""
    if primary.get("comparison") != "UK48 - U":
        raise ExpansionAnalysisError("Primary decision requires UK48 - U")
    interval = primary.get("confidence_interval_95")
    if not isinstance(interval, Mapping):
        raise ExpansionAnalysisError("Primary decision needs its 95% CI")
    raw_values = (primary.get("estimate_pp"), interval.get("lower_pp"), interval.get("upper_pp"))
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in raw_values):
        raise ExpansionAnalysisError("Primary effect and CI must be finite numbers")
    estimate, lower, upper = (float(value) for value in raw_values)
    if lower > upper:
        raise ExpansionAnalysisError("CI lower exceeds upper")
    statistically_positive = lower > 0.0
    return {
        "practical_effect_threshold_pp": PRACTICAL_EFFECT_THRESHOLD_PP,
        "statistically_positive_primary": statistically_positive,
        "practically_positive_primary": statistically_positive and estimate >= PRACTICAL_EFFECT_THRESHOLD_PP,
        "practical_benefit_ruled_out": upper < PRACTICAL_EFFECT_THRESHOLD_PP,
        "ci_crosses_or_touches_zero": lower <= 0.0 <= upper,
        "requires_beating_random_control": False,
        "conditions_are_separate_not_exclusive": True,
    }


def _effect(table: OutcomeTable, treatment: str, control: str, bootstrap: int, permutations: int) -> dict[str, Any]:
    adapter = _ParentTableAdapter(table)
    aliases = (_TO_PARENT[treatment], _TO_PARENT[control])
    kwargs = {"excluded_blocks": table.excluded_blocks}
    result = parent.paired_effect(adapter, *aliases, **kwargs)
    result.update({"comparison": f"{treatment} - {control}", "treatment": treatment, "control": control})
    result["confidence_interval_95"] = parent.hierarchical_percentile_bootstrap_ci(
        adapter, *aliases, **kwargs, seed=ANALYSIS_SEED, replicates=bootstrap)
    result["randomization_test"] = parent.paired_randomization_test(
        adapter, *aliases, **kwargs, seed=ANALYSIS_SEED, replicates=permutations)
    discordant = parent.discordant_counts(adapter, *aliases, **kwargs)
    discordant["comparison"] = result["comparison"]
    result["discordant_counts"] = discordant
    result["confidence_interval_multiplicity_adjusted"] = False
    return result


def analyze_matched_outcomes(
    records: Iterable[Mapping[str, Any]] | Mapping[str, Any], *,
    run_attestation: Mapping[str, Any] | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    randomization_replicates: int = DEFAULT_RANDOMIZATION_REPLICATES,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Validate before inference; explicit fixtures can never look like formal reports."""
    for name, count in (("bootstrap", bootstrap_replicates), ("randomization", randomization_replicates)):
        if type(count) is not int or count <= 0:
            raise ExpansionAnalysisError(f"{name} replicate count must be a positive integer")
    if not allow_fixture and (bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES or randomization_replicates != DEFAULT_RANDOMIZATION_REPLICATES):
        raise ExpansionAnalysisError("Formal analysis requires 100000 bootstrap and 100000 randomization replicates")
    table = validate_outcome_records(records, allow_fixture=allow_fixture)
    attestation = None if allow_fixture else _validate_run_attestation(run_attestation, table)
    primary = _effect(table, *PRIMARY_COMPARISON, bootstrap_replicates, randomization_replicates)
    auxiliary = {f"{treatment} - {control}": _effect(table, treatment, control, bootstrap_replicates, randomization_replicates)
                 for treatment, control in AUXILIARY_COMPARISONS}
    holm = parent.holm_adjust({key: effect["randomization_test"]["p_value"] for key, effect in auxiliary.items()})
    for key, effect in auxiliary.items():
        effect["randomization_test"].update(holm[key])
        effect["randomization_test"]["holm_family_size"] = 2
    decisions = classify_primary_result(primary)
    records_by_arm = {arm: [record for record in table.records() if record["arm"] == arm] for arm in ARMS}
    return {
        "protocol_family": PROTOCOL_FAMILY, "analysis_protocol_version": "v1.0",
        "analysis_status": "nonformal_fixture_only" if allow_fixture else "complete_under_verified_same_run_attestation",
        "analysis_scope": "outcome-statistics; raw-launch-provenance-external-required",
        "formal_analysis_contract_met": not allow_fixture,
        "raw_artifacts_reverified_by_this_function": False,
        "run_attestation": attestation, **outcome_binding_hashes(table),
        "cell_count": table.cell_count, "paired_block_count": len(table.blocks), "task_count": len(FORMAL_TASKS),
        "analysis_seed": ANALYSIS_SEED, "numpy_version": np.__version__,
        "analysis_implementation": "expansion arm adapter; unchanged parent paired_effect/hierarchical_percentile_bootstrap_ci/paired_randomization_test/holm_adjust",
        "rng_stream_policy": "fresh PCG64 from shared seed for each comparison and procedure; same task/episode schedule",
        "bootstrap_replicates": bootstrap_replicates, "randomization_replicates": randomization_replicates,
        "arm_counts": {arm: {"success": sum(record["success"] for record in values), "total": len(values)} for arm, values in records_by_arm.items()},
        "primary": primary, "primary_p_value_adjustment": "none; sole prespecified primary",
        "auxiliary": auxiliary,
        "auxiliary_holm_family": {"family_size": 2, "comparisons": list(auxiliary), "results": holm},
        "decision": None if allow_fixture else decisions,
        "nonformal_decision_preview": decisions if allow_fixture else None,
        "exposure_based_exclusions": [],
        "limitations": ["Entire prior-viewed census retained; no unexposed-subset sensitivity.",
                        "Primary tests the whole expansion scheme, including unchanged positional effects.",
                        "Caller must authenticate supplied run audit and all raw scientific outcomes."],
    }
