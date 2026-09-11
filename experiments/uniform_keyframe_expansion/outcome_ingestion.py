"""Read-only ingestion of the audited same-run U/UK48/UN48 formal artifacts.

The returned ``records`` contain the exact 2,400-cell census. Initial conditions
are compared within every task/episode triple before any statistics may run.
Internal artifact checks still cannot authenticate external checkpoint, software,
or environment provenance claims beyond the launch evidence bound to the run.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from experiments.keyframe_oracle_sampling.artifacts import SMOKE_INITIAL_CONDITION_HASH_FIELDS
from experiments.uniform_keyframe_expansion import contract as c
from experiments.uniform_keyframe_expansion.artifacts import ExpansionRunStore


class ExpansionIngestionError(ValueError):
    pass


_ENVIRONMENT_FIELDS = ("dataset", "resolved_environment_seed", "difficulty")
_RESET_FIELDS = (
    "policy_seed", "memory_cleared", "policy_rng_reset",
    "reset_prefix_frame_count", "reset_prefix_stage_count",
    "reset_prefix_frames_sha256", "reset_prefix_stages_sha256",
)


def _read_bytes(path: Path) -> bytes:
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ExpansionIngestionError(f"Missing or symlinked source: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ExpansionIngestionError(f"Cannot read source: {path}") from exc


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = _read_bytes(path)
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON mapping")
        c.canonical_json(value)
    except (ValueError, TypeError) as exc:
        raise ExpansionIngestionError(f"Malformed JSON source: {path}") from exc
    return value, hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(_read_bytes(path)).hexdigest()


def _equal(actual: Any, expected: Any, label: str) -> None:
    if c.canonical_json(actual) != c.canonical_json(expected):
        raise ExpansionIngestionError(f"{label} mismatch")


def _digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ExpansionIngestionError(f"Invalid SHA-256: {label}")


def _pair_signature(initial: Mapping[str, Any]) -> dict[str, Any]:
    """Only actual initial conditions/reset fields, never closed-loop outcomes."""
    hashes = initial.get("hashes")
    if not isinstance(hashes, Mapping) or set(hashes) != set(SMOKE_INITIAL_CONDITION_HASH_FIELDS):
        raise ExpansionIngestionError("Exactly five initial-condition hashes are required")
    for name, digest in hashes.items():
        _digest(digest, name)
    env, reset = initial.get("environment_provenance"), initial.get("reset_evidence")
    if not isinstance(env, Mapping) or not isinstance(reset, Mapping):
        raise ExpansionIngestionError("Initial environment/reset evidence is missing")
    if any(name not in env for name in _ENVIRONMENT_FIELDS) or any(name not in reset for name in _RESET_FIELDS):
        raise ExpansionIngestionError("Initial environment/reset fields are incomplete")
    if env["dataset"] != "test" or type(env["resolved_environment_seed"]) is not int or env["resolved_environment_seed"] < 0:
        raise ExpansionIngestionError("Formal actual seed/split evidence is invalid")
    if type(env["difficulty"]) not in (str, int) or env["difficulty"] == "":
        raise ExpansionIngestionError("Actual resolved difficulty is missing")
    for name, expected in (("policy_seed", c.POLICY_SEED), ("memory_cleared", True), ("policy_rng_reset", True)):
        _equal(reset[name], expected, f"initial reset {name}")
    count = reset["reset_prefix_frame_count"]
    if type(count) is not int or count < 1:
        raise ExpansionIngestionError("Initial reset-prefix count is invalid")
    _equal(reset["reset_prefix_stage_count"], count, "initial reset-prefix alignment")
    for name in ("reset_prefix_frames_sha256", "reset_prefix_stages_sha256"):
        _digest(reset[name], name)
    _equal(reset["reset_prefix_frames_sha256"], hashes["front_observations_sha256"], "initial front/prefix digest")
    boundaries = initial.get("reset_prefix_boundary_indices")
    if (not isinstance(boundaries, list) or not boundaries or boundaries[0] != 0
            or any(type(index) is not int or not 0 <= index < count for index in boundaries)
            or boundaries != sorted(set(boundaries))):
        raise ExpansionIngestionError("Initial reset-prefix boundary evidence is invalid")
    environment = {name: env[name] for name in _ENVIRONMENT_FIELDS}
    # Do not conflate a missing optional field with an explicit null or value.
    if "resolved_difficulty_hint" in env:
        environment["resolved_difficulty_hint"] = env["resolved_difficulty_hint"]
    return {
        "initial_condition_hashes": dict(hashes),
        "environment_provenance": environment,
        "reset_evidence": {name: reset[name] for name in _RESET_FIELDS},
        "reset_prefix_boundary_indices": boundaries,
    }


def _normalize_result(result: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    _equal(result.get("row"), row, "audited result row")
    _equal(result.get("protocol_family"), c.PROTOCOL_FAMILY, "audited result family")
    if result.get("terminal_reason") not in ("success", "fail", "timeout", "error"):
        raise ExpansionIngestionError("Only official scientific terminal outcomes can be ingested")
    _equal(result.get("success"), result["terminal_reason"] == "success", "terminal success")
    _equal(result.get("official_terminal"), True, "official terminal")
    _equal(result.get("reset_verified"), True, "result reset")
    for name in ("infrastructure_failure", "protocol_invariant_failure"):
        if result.get(name, False) is not False:
            raise ExpansionIngestionError("Infrastructure/invariant failures are not scientific outcomes")
    return {
        "task": row["task"], "episode_id": row["episode_id"], "arm": row["arm"],
        "dataset": row["dataset"], "trajectory_kind": row["trajectory_kind"],
        "success": result["success"], "terminal_status": result["terminal_reason"],
    }


def ingest_formal_run(run_root: str | Path) -> dict[str, Any]:
    """Audit the entire frozen three-arm census, then return records and provenance.

    No partial/fixture mode, exclusions, generated defaults, or file writes are
    supported. ``completed_rows`` enforces immutable attempts and
    first scientific completion, including prior infrastructure retry evidence.
    Each selected attempt is re-audited before ingestion, not merely trusted
    because a completion ledger or a result file happens to exist.
    """
    store = ExpansionRunStore.open(run_root)
    if store.stage != "formal":
        raise ExpansionIngestionError("Smoke/development artifacts cannot enter formal outcome ingestion")
    manifest_path = store.run_root / "run_manifest.json"
    manifest, manifest_sha = _read_json(manifest_path)
    _equal(manifest, store.manifest, "opened run manifest")
    rows = c.build_formal_matrix()["rows"]
    completed = store.completed_rows()  # Full original store audit; exceptions propagate.
    expected_ids = {row["row_id"] for row in rows}
    if (not isinstance(completed, Mapping) or any(type(key) is not int for key in completed)
            or set(completed) != expected_ids):
        observed = len(completed) if isinstance(completed, Mapping) else "invalid"
        raise ExpansionIngestionError(f"Exact 2400-cell three-arm census required; completed={observed}")
    if len({str(path) for path in completed.values()}) != len(rows):
        raise ExpansionIngestionError("Duplicate source result paths")
    ledger_path = store.run_root / "completions/completion_ledger.jsonl"
    ledger_sha = _sha256(ledger_path)
    records, sources, signatures = [], [], {}
    for row in rows:
        result_path = Path(completed[row["row_id"]])
        result, result_sha = _read_json(result_path)
        attempt = result.get("attempt_id")
        if type(attempt) is not int or not 0 <= attempt <= 2:
            raise ExpansionIngestionError("Invalid completed attempt id")
        attempt_dir = store.attempt_dir(row, attempt)
        if result_path != attempt_dir / "episode_result.json":
            raise ExpansionIngestionError("Completed result source path does not match its frozen row/attempt")
        audit = store.audit_attempt(row, attempt)
        if audit.get("status") != "complete" or audit.get("retry_allowed") is not False:
            raise ExpansionIngestionError("Only an audited immutable scientific completion may be ingested")
        _equal(audit.get("result"), result, "re-audited result")
        _equal(_sha256(result_path), result_sha, "result bytes changed during audit")
        initial_path = attempt_dir / "initial_condition_hashes.json"
        initial, initial_sha = _read_json(initial_path)
        episode_path = attempt_dir / "episode_manifest.json"
        episode, episode_sha = _read_json(episode_path)
        bound_hashes = result.get("artifact_sha256")
        if not isinstance(bound_hashes, Mapping):
            raise ExpansionIngestionError("Result lacks artifact checksum bindings")
        for name, digest in (("initial_condition_hashes.json", initial_sha), ("episode_manifest.json", episode_sha)):
            _equal(bound_hashes.get(name), digest, f"ingested {name} checksum")
        _equal(initial.get("episode_manifest_sha256"), episode_sha, "initial/episode manifest binding")
        _equal(episode.get("run_manifest_sha256"), manifest_sha, "episode/run manifest binding")
        signature = _pair_signature(initial)
        key = (row["task"], row["episode_id"], row["arm"])
        if key in signatures:
            raise ExpansionIngestionError(f"Duplicate scientific cell: {key}")
        signatures[key] = signature
        records.append(_normalize_result(result, row))
        sources.append({
            "row_id": row["row_id"], "task": row["task"], "episode_id": row["episode_id"], "arm": row["arm"],
            "attempt_id": attempt, "result_path": str(result_path), "result_sha256": result_sha,
            "initial_conditions_path": str(initial_path), "initial_conditions_sha256": initial_sha,
            "episode_manifest_path": str(episode_path), "episode_manifest_sha256": episode_sha,
            "artifact_sha256": dict(bound_hashes),
        })
    pair_records = []
    for index in range(0, len(rows), 3):
        u, uk, un = rows[index:index + 3]
        reference = signatures[(u["task"], u["episode_id"], "U")]
        # Compare initial evidence only. Later histories, boundaries, counts and
        # terminal outcomes can legitimately diverge in closed-loop execution.
        for arm in ("UK48", "UN48"):
            observed = signatures[(u["task"], u["episode_id"], arm)]
            for field in reference:
                _equal(reference[field], observed[field],
                       f"U/{arm} {u['task']} episode {u['episode_id']} {field}")
        pair_records.append({
            "task": u["task"], "episode_id": u["episode_id"],
            "source_row_ids": [u["row_id"], uk["row_id"], un["row_id"]],
            "arms": list(c.FORMAL_ARMS), "status": "matched",
            "matched_initial_evidence": reference,
        })
    # Abort rather than publish a mixed-time snapshot if run evidence changes.
    _equal(_sha256(manifest_path), manifest_sha, "run manifest changed during ingestion")
    _equal(_sha256(ledger_path), ledger_sha, "completion ledger changed during ingestion")
    for relative, digest in store.manifest["snapshot_sha256"].items():
        _equal(_sha256(store.run_root / relative), digest, f"run snapshot changed: {relative}")
    pairing = {
        "scope": "same_run_U_UK48_UN48_initial_pairing", "paired_block_count": len(pair_records),
        "all_three_arm_blocks_matched": True, "records": pair_records,
    }
    return {
        "schema_version": 1, "protocol_family": c.PROTOCOL_FAMILY,
        "stage": "formal", "source_run_id": store.manifest["run_id"],
        "source_run_root": str(store.run_root),
        "run_manifest_path": str(manifest_path), "run_manifest_sha256": manifest_sha,
        "run_provenance": store.manifest["provenance"],
        "completion_ledger_sha256": ledger_sha, "run_snapshot_sha256": store.manifest["snapshot_sha256"],
        "study_cell_count": len(records), "records": records, "sources": sources,
        "normalized_records_sha256": c.canonical_sha256(records),
        "sources_sha256": c.canonical_sha256(sources),
        "pairing_audit": pairing, "pairing_audit_sha256": c.canonical_sha256(pairing),
        "comparison_scope": "same-run Lighthouse U/UK48/UN48 with matched initial conditions",
        "formal_statistics_executed": False,
    }
