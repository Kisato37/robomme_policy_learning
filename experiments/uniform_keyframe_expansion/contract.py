"""CPU-only matrices, seed tables and fail-closed preparation contracts.

This module does not submit, launch, contact a server, or authorize a run. Its
readiness checker validates a supplied evidence manifest's structure and binding;
the future launcher must also verify the actual referenced evidence and approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from mme_vla_suite.shared.uniform_keyframe_expansion import derive_expansion_seed

PROTOCOL_FAMILY = "uniform_keyframe_expansion-v1"
EXPANSION_ARMS = ("UK48", "UN48")
FORMAL_EPISODE_IDS = tuple(range(50))
FORMAL_DATASET = "test"
SMOKE_DATASET = "val"
SMOKE_EPISODE_ID = 0
BASE_FRAME_CAPACITY = 32
BASE_TOKEN_CAPACITY = 512
FRAME_CAPACITY = 48
TOKEN_CAPACITY = 768
TOKENS_PER_FRAME = 16
POLICY_SEED = 7
ACTION_HORIZON = 20
EXECUTED_ACTION_HORIZON = 16
MAX_STEPS = 1300
SHORT_MAX_STEPS = 64
MAX_POLICY_CALLS = 82
CHECKPOINT_ID = 79999
CHECKPOINT_MODEL = "Yinpei/perceptual-framesamp-modul"
CHECKPOINT_ARCHIVE_SHA256 = "2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"
MASTER_SELECTOR_SEED = 2026091001
FORMAL_TRAJECTORY_COUNT = len(FORMAL_TASKS) * len(FORMAL_EPISODE_IDS) * len(EXPANSION_ARMS)
SMOKE_TRAJECTORY_COUNT = len(FORMAL_TASKS) * (len(EXPANSION_ARMS) + 1)


class ExpansionContractError(ValueError):
    """A matrix, configuration or readiness claim violates the new protocol."""


def canonical_json(value: Any) -> str:
    """Unambiguous encoding: importantly, JSON booleans are not integer IDs."""
    try:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExpansionContractError("Expected finite, JSON-serializable contract data") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if canonical_json(actual) != canonical_json(expected):
        raise ExpansionContractError(f"{label} differs from the frozen expansion contract")


def frozen_inference_contract() -> dict[str, Any]:
    """A fresh mapping on every call; modifying a caller copy changes no defaults."""
    return {
        "protocol_family": PROTOCOL_FAMILY,
        "base_frame_capacity": BASE_FRAME_CAPACITY,
        "base_token_capacity": BASE_TOKEN_CAPACITY,
        "frame_capacity": FRAME_CAPACITY,
        "token_capacity": TOKEN_CAPACITY,
        "tokens_per_frame": TOKENS_PER_FRAME,
        "policy_seed": POLICY_SEED,
        "action_horizon": ACTION_HORIZON,
        "executed_action_horizon": EXECUTED_ACTION_HORIZON,
        "max_steps": MAX_STEPS,
        "max_policy_calls": MAX_POLICY_CALLS,
        "checkpoint_model": CHECKPOINT_MODEL,
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_archive_sha256": CHECKPOINT_ARCHIVE_SHA256,
        "training_enabled": False,
        "state_memory_enabled": False,
        "symbolic_memory_enabled": False,
        "extra_uniform_fill_enabled": False,
        "padding_values": "zero",
        "padding_mask": False,
        "weights_strict_load_required": True,
    }


def validate_inference_contract(payload: Mapping[str, Any]) -> None:
    _require_equal(dict(payload), frozen_inference_contract(), "Inference configuration")


def _row(rows: list[dict[str, Any]], *, task: str, episode: int, arm: str,
         kind: str, max_steps: int, dataset: str) -> None:
    rows.append({
        "row_id": len(rows), "task": task, "episode_id": episode, "arm": arm,
        "trajectory_kind": kind, "max_steps": max_steps, "dataset": dataset,
    })


def build_formal_matrix() -> dict[str, Any]:
    """The two NEW arms only: 16 canonical tasks x 50 test episodes x 2."""
    rows: list[dict[str, Any]] = []
    for task in FORMAL_TASKS:
        for episode in FORMAL_EPISODE_IDS:
            for arm in EXPANSION_ARMS:
                _row(rows, task=task, episode=episode, arm=arm, kind="formal",
                     max_steps=MAX_STEPS, dataset=FORMAL_DATASET)
    return {
        "schema_version": 1, "protocol_family": PROTOCOL_FAMILY,
        "order": ["task", "episode_id", "arm"], "tasks": list(FORMAL_TASKS),
        "episode_ids": list(FORMAL_EPISODE_IDS), "arms": list(EXPANSION_ARMS),
        "dataset": FORMAL_DATASET, "max_steps": MAX_STEPS,
        "trajectory_kind": "formal", "trajectory_count": FORMAL_TRAJECTORY_COUNT,
        "rows": rows,
    }


def build_smoke_matrix() -> dict[str, Any]:
    """32 short (64 steps or official terminal), plus 16 full terminal paths."""
    rows: list[dict[str, Any]] = []
    terminal_arms: dict[str, str] = {}
    for index, task in enumerate(FORMAL_TASKS):
        for arm in EXPANSION_ARMS:
            _row(rows, task=task, episode=SMOKE_EPISODE_ID, arm=arm, kind="short",
                 max_steps=SHORT_MAX_STEPS, dataset=SMOKE_DATASET)
        terminal_arm = EXPANSION_ARMS[index % len(EXPANSION_ARMS)]
        terminal_arms[task] = terminal_arm
        _row(rows, task=task, episode=SMOKE_EPISODE_ID, arm=terminal_arm, kind="terminal",
             max_steps=MAX_STEPS, dataset=SMOKE_DATASET)
    return {
        "schema_version": 1, "protocol_family": PROTOCOL_FAMILY,
        "order": ["task", "trajectory_kind", "arm"], "tasks": list(FORMAL_TASKS),
        "episode_id": SMOKE_EPISODE_ID, "arms": list(EXPANSION_ARMS),
        "terminal_arms": terminal_arms, "dataset": SMOKE_DATASET,
        "short_max_steps": SHORT_MAX_STEPS, "terminal_max_steps": MAX_STEPS,
        "short_stop_on_official_terminal": True,
        "trajectory_count": SMOKE_TRAJECTORY_COUNT, "rows": rows,
    }


def _matrix(stage: str) -> dict[str, Any]:
    if stage == "formal":
        return build_formal_matrix()
    if stage == "smoke":
        return build_smoke_matrix()
    raise ExpansionContractError("Stage must be exactly 'formal' or 'smoke'")


def validate_matrix(payload: Mapping[str, Any], stage: str) -> None:
    """Require exact matrix semantics, including order, row fields and ID types."""
    _require_equal(dict(payload), _matrix(stage), f"{stage} matrix")


def validate_row(row: Mapping[str, Any], stage: str | None = None) -> dict[str, Any]:
    """Reject extra/missing fields, noncanonical tasks, wrong splits and IDs."""
    if not isinstance(row, Mapping):
        raise ExpansionContractError("Matrix row must be a mapping")
    if stage is None:
        kind = row.get("trajectory_kind")
        if kind not in ("formal", "short", "terminal"):
            raise ExpansionContractError("Unknown trajectory kind")
        stage = "formal" if kind == "formal" else "smoke"
    row_id = row.get("row_id")
    count = FORMAL_TRAJECTORY_COUNT if stage == "formal" else SMOKE_TRAJECTORY_COUNT
    if stage not in ("formal", "smoke"):
        raise ExpansionContractError("Stage must be exactly 'formal' or 'smoke'")
    if type(row_id) is not int or not 0 <= row_id < count:
        raise ExpansionContractError(f"Invalid {stage} row_id")
    # Validate a single row without constructing all 1,600 rows for every seed
    # table.  Tests compare this arithmetic binding to every generated row.
    if stage == "formal":
        context_index, arm_index = divmod(row_id, len(EXPANSION_ARMS))
        task_index, episode = divmod(context_index, len(FORMAL_EPISODE_IDS))
        arm, kind, max_steps, dataset = EXPANSION_ARMS[arm_index], "formal", MAX_STEPS, FORMAL_DATASET
    else:
        task_index, offset = divmod(row_id, len(EXPANSION_ARMS) + 1)
        episode, dataset = SMOKE_EPISODE_ID, SMOKE_DATASET
        if offset < len(EXPANSION_ARMS):
            arm, kind, max_steps = EXPANSION_ARMS[offset], "short", SHORT_MAX_STEPS
        else:
            arm, kind, max_steps = EXPANSION_ARMS[task_index % len(EXPANSION_ARMS)], "terminal", MAX_STEPS
    expected = {
        "row_id": row_id, "task": FORMAL_TASKS[task_index], "episode_id": episode,
        "arm": arm, "trajectory_kind": kind, "max_steps": max_steps, "dataset": dataset,
    }
    _require_equal(dict(row), expected, f"{stage} row {row_id}")
    return expected


def build_selector_config(row: Mapping[str, Any]) -> dict[str, Any]:
    """Derive an independent UN seed table for both arms; UK never consumes it.

    Smoke short and terminal passes intentionally reuse the same (val, task,
    episode, call) selector seed. No val seed is shared with test. No terminal
    result, wall-clock timestamp, task order, host or global RNG is consulted.
    """
    exact = validate_row(row)
    seeds = [derive_expansion_seed(exact["dataset"], exact["task"], exact["episode_id"], call)
             for call in range(MAX_POLICY_CALLS)]
    return {
        "arm": exact["arm"], "task": exact["task"], "episode_id": exact["episode_id"],
        "split": exact["dataset"], "random_seeds": seeds,
        "seed_table_sha256": canonical_sha256(seeds),
    }


def validate_selector_config(config: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    _require_equal(dict(config), build_selector_config(row), "Selector runtime configuration")


def build_seed_manifest(stage: str) -> dict[str, Any]:
    """Unique trajectory contexts with 82 future *random draws*, not future data."""
    matrix = _matrix(stage)
    seen: set[tuple[str, str, int]] = set()
    records = []
    for row in matrix["rows"]:
        context = (row["dataset"], row["task"], row["episode_id"])
        if context in seen:
            continue
        seen.add(context)
        config = build_selector_config(row)
        records.append({key: value for key, value in config.items() if key != "arm"})
    payload = {
        "schema_version": 1, "protocol_family": PROTOCOL_FAMILY, "stage": stage,
        "master_selector_seed": MASTER_SELECTOR_SEED,
        "random_selector_label": "UN48", "policy_call_indices": list(range(MAX_POLICY_CALLS)),
        "context_count": len(records), "records": records,
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def build_readiness_template() -> dict[str, Any]:
    """Unresolved by default; constructing a matrix never implies permission."""
    return {
        "protocol_family": PROTOCOL_FAMILY,
        "stage": "formal",
        "readiness_scope": "planning-manifest-only-not-launch-authorization",
        "explicit_formal_user_approval": False,
        "protocol_frozen": False,
        "git_clean": False,
        "code_commit": None,
        "protocol_sha256": None,
        "environment_manifest_sha256": None,
        "inference_contract": frozen_inference_contract(),
        "formal_matrix_sha256": canonical_sha256(build_formal_matrix()),
        "formal_seed_manifest_sha256": build_seed_manifest("formal")["manifest_sha256"],
        "cpu_gate": {"status": "pending", "report_sha256": None},
        "gpu_smoke_gate": {"status": "pending", "report_sha256": None},
        "u_baseline": {
            "status": "unresolved", "source_run_id": None,
            "artifact_manifest_sha256": None,
            "matching_initial_conditions": False,
            "matching_execution_provenance": False,
            "matching_seed_mapping": False,
            "checkpoint_matches": False,
        },
    }


def _sha(value: Any, digits: int, label: str) -> None:
    if not isinstance(value, str) or len(value) != digits or any(c not in "0123456789abcdef" for c in value):
        raise ExpansionContractError(f"{label} must be a lowercase {digits}-digit hexadecimal digest")


def validate_formal_readiness(evidence: Mapping[str, Any]) -> None:
    """Fail closed unless all independent formal gates have bound attestations.

    This is a validation helper, NOT a scheduler or an authorization interface.
    A future launcher must verify hashes against actual artifacts and the user's
    stage-specific approval. Passing fabricated strings is never sufficient.
    """
    if not isinstance(evidence, Mapping):
        raise ExpansionContractError("Readiness evidence must be a mapping")
    template = build_readiness_template()
    if set(evidence) != set(template):
        raise ExpansionContractError("Readiness evidence fields differ from the contract")
    for key in ("protocol_family", "stage", "readiness_scope", "formal_matrix_sha256", "formal_seed_manifest_sha256"):
        _require_equal(evidence[key], template[key], f"Readiness {key}")
    for key in ("explicit_formal_user_approval", "protocol_frozen", "git_clean"):
        if evidence[key] is not True:
            raise ExpansionContractError(f"Formal launch remains blocked: {key}")
    _sha(evidence["code_commit"], 40, "code_commit")
    for key in ("protocol_sha256", "environment_manifest_sha256"):
        _sha(evidence[key], 64, key)
    validate_inference_contract(evidence["inference_contract"])
    for name in ("cpu_gate", "gpu_smoke_gate"):
        gate = evidence[name]
        if not isinstance(gate, Mapping) or set(gate) != {"status", "report_sha256"} or gate["status"] != "passed":
            raise ExpansionContractError(f"Formal launch remains blocked: {name}")
        _sha(gate["report_sha256"], 64, f"{name} report_sha256")
    baseline = evidence["u_baseline"]
    if not isinstance(baseline, Mapping) or set(baseline) != set(template["u_baseline"]):
        raise ExpansionContractError("Primary U baseline evidence fields are invalid")
    if baseline["status"] != "verified":
        raise ExpansionContractError("Primary UK48 minus U comparison requires a verified U baseline")
    if not isinstance(baseline["source_run_id"], str) or not baseline["source_run_id"].strip():
        raise ExpansionContractError("U baseline source_run_id is required")
    _sha(baseline["artifact_manifest_sha256"], 64, "U artifact_manifest_sha256")
    for key in ("matching_initial_conditions", "matching_execution_provenance", "matching_seed_mapping", "checkpoint_matches"):
        if baseline[key] is not True:
            raise ExpansionContractError(f"Primary U baseline is not verified: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print CPU-only expansion contract metadata; never launches jobs")
    parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--matrix", action="store_true", help="Print the matrix rather than its compact summary")
    args = parser.parse_args()
    matrix = _matrix(args.stage)
    if args.matrix:
        print(json.dumps(matrix, indent=2, ensure_ascii=True))
        return
    seeds = build_seed_manifest(args.stage)
    print(json.dumps({
        "protocol_family": PROTOCOL_FAMILY, "stage": args.stage,
        "trajectory_count": matrix["trajectory_count"],
        "matrix_sha256": canonical_sha256(matrix),
        "seed_manifest_sha256": seeds["manifest_sha256"],
        "seed_context_count": seeds["context_count"],
        "policy_calls_per_seed_table": MAX_POLICY_CALLS,
        "inference_contract": frozen_inference_contract(),
        "u_baseline_status": "unresolved", "launch_authorized": False,
    }, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
