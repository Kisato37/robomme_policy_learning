"""Strict 1,600-cell aggregation across a sealed parent and authorized recovery."""
# ruff: noqa: SLF001
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.keyframe_neighborhood_sampling import aggregate_formal as frozen
from experiments.keyframe_neighborhood_sampling import recovery
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS, FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.record_launcher_failure import validate_extension_failure_record
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError, FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import audit_initial_condition_fairness, audit_paired_manifest_invariants
from experiments.keyframe_oracle_sampling.artifacts import sha256_file, utc_now


def audit_recovery_attempts(root: Path, plan: dict, rows: dict, context: dict) -> None:
    """Require all local retry chains, exact submission sets and original caps."""
    if set(rows) != set(plan["row_ids"]):
        raise ArtifactContractError("Recovery is incomplete: no partial final aggregate")
    failures = {}
    by_key = {item["key"]: row for row, item in rows.items()}
    for failure in context["failures"]:
        report = validate_extension_failure_record(root, failure)
        key, attempt = report["key"], report["attempt_id"]
        if (key, attempt) in failures or key not in by_key:
            raise ArtifactContractError("Duplicate/out-of-scope recovery failure")
        if failure["classification"] != "infrastructure" or failure["retry_allowed"] is not True:
            raise ArtifactContractError("Recovery hard stop requires additional review")
        failures[(key, attempt)] = failure
    expected_attempts, expected_failures = set(), set()
    retry_rows = {0: plan["row_ids"], 1: [], 2: []}
    for row, item in rows.items():
        key, final_attempt = item["key"], item["writer"].attempt_id
        recovery.validate_retry_budget(root, [row], final_attempt)
        for attempt in range(final_attempt + 1):
            expected_attempts.add((key, attempt))
            if attempt < final_attempt:
                expected_failures.add((key, attempt))
                retry_rows[attempt + 1].append(row)
    if set(context["attempts"]) != expected_attempts or set(failures) != expected_failures:
        raise ArtifactContractError("Recovery attempt/failure inventories differ from exact retry chains")
    dispatches = [recovery.load(p) for p in (root / "direct").glob("attempt_*/row_*/dispatch.json")]
    expected_dispatches = {(by_key[k], a) for k, a in expected_attempts}
    if (len(dispatches) != len(expected_dispatches)
            or {(d["row_id"], d["attempt_id"]) for d in dispatches} != expected_dispatches):
        raise ArtifactContractError("Recovery has unmatched/unfinished direct dispatches")
    plan_digest = sha256_file(root / "protocol/recovery_plan.json")
    for attempt, row_ids in retry_rows.items():
        paths = frozen._submission_paths(root, attempt)
        if not row_ids:
            if paths or frozen._submission_plan_path(root, attempt).exists():
                raise ArtifactContractError("Unexpected recovery retry submission")
            continue
        _, authorizations = frozen.validate_submission_attempt(root, attempt_id=attempt,
                                                                expected_row_ids=sorted(row_ids), launch=context["launch"])
        for path in [frozen._submission_plan_path(root, attempt), *paths]:
            if recovery.load(path).get("recovery_plan_sha256") != plan_digest:
                raise ArtifactContractError("Recovery submission lost its immutable parent binding")
        for (key, a), (_, manifest) in context["attempts"].items():
            if a == attempt:
                failure = failures.get((key, a), {})
                frozen._validate_attempt_submission_binding(
                    manifest, row_id=by_key[key], authorization=authorizations[by_key[key]], run_root=root,
                    validated_readiness_failure=failure.get("failure_phase") == "policy_server_readiness")


def summarize(rows: dict, *, plan: dict, provenance: dict, child_context: dict) -> tuple[dict, dict]:
    if sorted(rows) != list(range(FORMAL_TRAJECTORY_COUNT)):
        raise ArtifactContractError("Combined recovery must have exactly 1,600 unique rows")
    paired = [{**item["manifest"], "initial_condition_hashes": item["initial"]} for item in rows.values()]
    fairness = audit_initial_condition_fairness(paired, required_arms=EXTENSION_ARMS)
    invariants = audit_paired_manifest_invariants(paired, required_arms=EXTENSION_ARMS)
    if fairness != {"paired_blocks": 800, "fair": True} or invariants != {"paired_blocks": 800, "invariants_match": True}:
        raise ArtifactContractError("Combined initial-state/configuration pairing is incomplete")
    latency = {arm: {k: [] for k in ("selector", "model", "end_to_end")} for arm in EXTENSION_ARMS}
    diagnostics = {arm: Counter() for arm in EXTENSION_ARMS}
    secondary = {arm: frozen._empty_secondary_diagnostic_accumulator() for arm in EXTENSION_ARMS}
    by_task = {t: {a: frozen._empty_secondary_diagnostic_accumulator() for a in EXTENSION_ARMS} for t in FORMAL_TASKS}
    calls, successes, collisions, attempts = Counter(), Counter(), Counter(), Counter()
    terminal = {arm: Counter() for arm in EXTENSION_ARMS}
    sources, records, trace_digests, result_digests = [], {}, [], []
    for row, item in sorted(rows.items()):
        writer, key, result = item["writer"], item["key"], item["result"]
        arm, report = key.arm, item["report"]
        recovered = row in plan["row_ids"]
        global_attempt = writer.attempt_id + (plan["global_retry_offsets"][str(row)] if recovered else 0)
        attempts[global_attempt] += 1
        calls[arm] += report["policy_call_count"]
        successes[arm] += int(result["success"])
        collisions[arm] += int(result["collision"])
        terminal[arm][result["terminal_reason"]] += 1
        diagnostics[arm].update(report["selector_diagnostics"])
        frozen._merge_secondary_diagnostics(secondary[arm], report["secondary_diagnostics"])
        frozen._merge_secondary_diagnostics(by_task[key.task][arm], report["secondary_diagnostics"])
        for kind, values in item["latencies"].items():
            latency[arm][kind].extend(values)
        source = {"row_id": row, "source": "recovery" if recovered else "parent",
                  "attempt_path": str(writer.attempt_dir), "local_attempt_id": writer.attempt_id,
                  "global_attempt_id": global_attempt, "result_sha256": sha256_file(writer.result_path),
                  "trace_sha256": sha256_file(writer.trace_path)}
        sources.append(source)
        result_digests.append({"row_id": row, "sha256": source["result_sha256"]})
        trace_digests.append({"row_id": row, "sha256": source["trace_sha256"]})
        records[key] = result
    risks = [
        "Operational recovery combines two recorded commits with unchanged scientific code and inputs; "
        "it is not a single-commit execution. Shared-GPU load/device allocation and wall-clock latency may differ.",
        "Published OC lacks raw initial-state hashes/environment seeds/difficulty; OC-to-extension pairing "
        "is by frozen task/episode design, not direct raw-state re-audit. The selector RNG table is not an environment seed table.",
    ]
    report = {
        "schema_version": frozen.AGGREGATE_SCHEMA_VERSION, "protocol_version": frozen.EXTENSION_PROTOCOL_VERSION,
        "protocol_family": frozen.EXTENSION_PROTOCOL_FAMILY, "audited_utc": utc_now(), "passed": True,
        "formal_result_census_complete": True, "strict_selector_trace_audit_complete": True,
        "completeness": {"complete": True, "expected": 1600, "observed": 1600, "missing": [], "unexpected": []},
        "initial_condition_fairness": fairness, "paired_manifest_invariants": invariants,
        "protocol_provenance": provenance, "per_row_provenance": sources,
        "completed_attempt_histogram": {str(k): v for k, v in sorted(attempts.items())},
        "infrastructure_failure_count": 3 + len(child_context["failures"]),
        "parent_ledger_failure_count": 2, "parent_pre_evaluator_interruption_count": 1,
        "recovery_ledger_failure_count": len(child_context["failures"]),
        "result_set_sha256": recovery.digest(result_digests), "selector_trace_set_sha256": recovery.digest(trace_digests),
        "strict_attempt_audit_set_sha256": recovery.digest(sources),
        "latency_by_arm": {a: {k: frozen._latency_summary(v) for k, v in latency[a].items()} for a in EXTENSION_ARMS},
        "selector_diagnostics_by_arm": {a: {**dict(diagnostics[a]), "policy_call_count": calls[a],
            "oc5_fallback_frequency": diagnostics[a]["oc5_fallback_calls"] / calls[a],
            "oc3_secondary_thinning_frequency": diagnostics[a]["oc3_secondary_thinning_calls"] / calls[a],
            "boundary_core_thinned_frequency": diagnostics[a]["boundary_core_thinned_calls"] / calls[a]} for a in EXTENSION_ARMS},
        "secondary_diagnostics": {
            "scope": "exploratory_secondary_diagnostics",
            "limitations": ["Boundary count is an exploratory within-task progress proxy, not ground-truth stage completion."],
            "by_arm": {a: frozen._finalize_secondary_diagnostics(secondary[a]) for a in EXTENSION_ARMS},
            "by_task": {t: {a: frozen._finalize_secondary_diagnostics(by_task[t][a]) for a in EXTENSION_ARMS} for t in FORMAL_TASKS},
        },
        "outcome_diagnostics": {"success_count_by_arm": dict(successes), "collision_count_by_arm": dict(collisions),
                                "terminal_reason_count_by_arm": {a: dict(v) for a, v in terminal.items()},
                                "policy_call_count_by_arm": dict(calls)},
        "protocol_deviations": ["Authorized operational recovery described in RECOVERY_2026-09-08.md; no scientific changes."],
        "unresolved_risks": risks,
    }
    return report, records


def aggregate(root: Path, repo: Path):
    root, repo = root.resolve(), repo.resolve()
    if (root / "aggregate").exists():
        raise FileExistsError("Never overwrite a published recovery aggregate")
    plan = recovery.validate_recovery_plan(root, verify_parent=True)
    parent_rows, parent_context, parent_report = recovery.audit_parent(Path(plan["parent_root"]), repo, progress=True)
    if parent_report != plan["parent_audit"]:
        raise ArtifactContractError("Parent no longer passes its exact sealed audit")
    recovery.audit_closed_sessions(root)
    before = recovery.inventory(root)
    child_rows, child_context = recovery.audited_results(root, repo, authorized_rows=plan["row_ids"], progress=True)
    audit_recovery_attempts(root, plan, child_rows, child_context)
    source = recovery.source_contract(parent_context["launch"], child_context["launch"], repo)
    if source != plan["repair_source"] or set(parent_rows) & set(child_rows):
        raise ArtifactContractError("Source drift or duplicate scientific result across recovery roots")
    provenance = {**child_context["provenance"], "recovery": {
        **source, "parent_root": plan["parent_root"], "recovery_root": str(root),
        "recovery_plan_sha256": sha256_file(root / "protocol/recovery_plan.json"),
        "parent_inventory_sha256": parent_report["inventory_sha256"],
        "parent_launch_sha256": sha256_file(Path(plan["parent_root"]) / "protocol/launch_manifest.json"),
        "incident": parent_report["incident"],
    }}
    report, records = summarize({**parent_rows, **child_rows}, plan=plan, provenance=provenance, child_context=child_context)
    if recovery.inventory(root) != before:
        raise ArtifactContractError("Recovery evidence changed during audit")
    reference_path, _, _ = frozen._reference_paths(repo)
    reference = frozen.load_published_reference_oc(reference_path)
    payloads, summary = frozen.build_aggregate_payloads(report, records, reference)
    # Keep the frozen statistical/rendering implementation; add explicit lineage
    # disclosure instead of silently presenting this as one uninterrupted run.
    payloads["analysis.md"] += ("\n## Authorized recovery provenance\n\n"
        + f"Retained 1,396 parent results from `{recovery.PARENT_COMMIT}`; completed 204 missing rows with `{source['recovery_commit']}`. "
        + "Original records were not rewritten. All 1,600 cells and 800 paired blocks passed strict audit. "
        + "See completeness_report.json for per-row source/attempt provenance and the reviewed OOM override. "
        + "Shared-GPU wall-clock latency is descriptive, not a controlled performance comparison.\n").encode()
    return frozen.publish_aggregate_directory(root, payloads), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    output, _ = aggregate(args.run_root, args.repo_root)
    print(json.dumps({"aggregate": str(output), "passed": True, "cells": 1600}), flush=True)


if __name__ == "__main__":
    main()
