"""Read-only local U evidence inventory; never grants baseline/launch approval.

No simulator, model, network, subprocess, or file-write operations are used.
Cached remote checks are reported as historical assertions, not re-executed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


TASKS = (
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes", "ButtonUnmask",
    "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap", "PickHighlight",
    "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "MoveCube",
    "InsertPeg", "PatternLock", "RouteStick",
)
RUN_ID = "20260829T231425Z_7b594786_formal_v1"
PAIR_FIELDS = (
    "dataset", "max_steps", "executed_action_horizon", "evaluation_policy_seed",
    "checkpoint_id", "seed_table_sha256", "resolved_environment_seed",
    "resolved_difficulty_hint", "difficulty",
)
INITIAL_FIELDS = (
    "front_observations_sha256", "wrist_observations_sha256",
    "robot_states_sha256", "task_state_sha256", "task_instruction_sha256",
)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest_value(value) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True, allow_nan=False).encode("utf-8"))


def audit(repo: Path, migration: Path) -> dict:
    result_dir = repo / "results/keyframe_oracle_sampling" / RUN_ID
    paths = {
        "results_manifest": result_dir / "RESULTS_MANIFEST.json",
        "completeness": result_dir / "aggregate/completeness_report.json",
        "summary": result_dir / "aggregate/summary.json",
        "athena_extraction": migration / "failure_audit_20260908/athena_records.json",
        "extraction_source": migration / "failure_audit_20260908/collect_existing.py",
        "oc_initial_inventory": migration / "initial_image_audit_20260908_r1/athena_inventory.json",
        "cross_host_pairs": migration / "initial_image_audit_20260908_r1/paired_initial_records.json",
        "athena_source_check": migration / "initial_image_audit_20260908_r1/athena_source_check.json",
        "lighthouse_checkpoint_check": migration / "checkpoint_verified_20260907.json",
    }
    source_evidence = {}
    loaded = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        source_evidence[name] = {"path": str(path), "size_bytes": len(raw),
                                 "sha256": digest_bytes(raw)}
        if path.suffix == ".json":
            loaded[name] = json.loads(raw)

    aggregate_file_checks = {}
    for relative, declared in loaded["results_manifest"]["files"].items():
        raw = (result_dir / relative).read_bytes()
        aggregate_file_checks[relative] = {
            "size_matches": len(raw) == declared["size_bytes"],
            "sha256_matches": digest_bytes(raw) == declared["sha256"],
        }

    records = loaded["athena_extraction"]["records"]
    uniform = [record for record in records if record["scientific_key"]["arm"] == "U"]
    old_oc = {(record["scientific_key"]["task"], record["scientific_key"]["episode_id"]): record
              for record in records if record["scientific_key"]["arm"] == "OC"}
    keys = [(record["scientific_key"]["task"], record["scientific_key"]["episode_id"])
            for record in uniform]
    expected = {(task, episode) for task in TASKS for episode in range(50)}
    lookup = dict(zip(keys, uniform))
    paired_field_difference_counts = {field: 0 for field in PAIR_FIELDS}
    for key, record in lookup.items():
        for field in PAIR_FIELDS:
            paired_field_difference_counts[field] += record["manifest"].get(field) != old_oc[key]["manifest"].get(field)

    cached_hash_check_counts = {name: sum(record["hash_checks"].get(name) is True for record in uniform)
                               for name in ("episode_manifest.json", "selector_trace.jsonl", "initial_condition_hashes.json")}
    terminal = dict(Counter(record["terminal_reason"] for record in uniform))
    calls = sum(len(record["calls"]) for record in uniform)
    initial_lengths = [record["calls"][0]["history_length"] for record in uniform]
    summary_diag = loaded["summary"]["outcome_diagnostics"]
    seed_records = [{"task": task, "episode_id": episode,
                     "resolved_environment_seed": lookup[(task, episode)]["manifest"]["resolved_environment_seed"],
                     "difficulty": lookup[(task, episode)]["manifest"]["difficulty"]}
                    for task in TASKS for episode in range(50)]
    inventory = loaded["oc_initial_inventory"]
    initial_records = inventory["records"]
    pair_summaries = []
    for left, right in (("OC", "OC3"), ("OC", "OC5"), ("OC3", "OC5")):
        pairs = [row for row in loaded["cross_host_pairs"] if (row["left"], row["right"]) == (left, right)]
        pair_summaries.append({
            "left": left, "right": right, "blocks": len(pairs),
            "field_mismatch_counts": {field: sum(row["matches"][field] is False for row in pairs)
                                       for field in INITIAL_FIELDS},
            "either_image_mismatch_count": sum(not row["matches"][INITIAL_FIELDS[0]] or
                                               not row["matches"][INITIAL_FIELDS[1]] for row in pairs),
        })

    return {
        "kind": "local-cached-baseline-evidence-inventory-not-comparison-attestation",
        "baseline_status": "unresolved", "launch_authorized": False,
        "network_used": False, "source_run_id": RUN_ID, "source_evidence": source_evidence,
        "published_aggregate_byte_checks": aggregate_file_checks,
        "uniform_local_evidence": {
            "count": len(uniform), "unique_blocks": len(set(keys)),
            "exact_800_block_census": len(keys) == len(set(keys)) == 800 and set(keys) == expected,
            "terminal_counts": terminal, "attempt_counts": dict(Counter(record["attempt_id"] for record in uniform)),
            "success_count": sum(record["success"] is True for record in uniform),
            "success_terminal_consistent": all(record["success"] == (record["terminal_reason"] == "success") for record in uniform),
            "policy_call_count": calls,
            "matches_published_summary": terminal == summary_diag["terminal_reason_count_by_arm"]["U"] and
                calls == summary_diag["policy_call_count_by_arm"]["U"] and
                sum(record["success"] is True for record in uniform) == summary_diag["success_count_by_arm"]["U"],
            "cached_remote_hash_check_true_counts_not_rechecked_raw_files": cached_hash_check_counts,
            "cached_remote_hash_error_count": len(loaded["athena_extraction"]["hash_errors"]),
            "direct_initial_hash_value_records": sum(any(field in record.get("initial_hashes", {}) or field in record for field in INITIAL_FIELDS) for record in uniform),
            "policy_seeds": sorted({record["evaluation_policy_seed"] for record in uniform}),
            "max_steps": sorted({record["max_steps"] for record in uniform}),
            "executed_horizons": sorted({record["executed_action_horizon"] for record in uniform}),
            "checkpoint_ids": sorted({record["manifest"]["checkpoint_id"] for record in uniform}),
            "datasets": sorted({record["manifest"]["dataset"] for record in uniform}),
            "difficulty_counts": dict(Counter(record["manifest"]["difficulty"] for record in uniform)),
            "initial_history_length_min": min(initial_lengths), "initial_history_length_max": max(initial_lengths),
            "initial_history_multi_frame_episodes": sum(length > 1 for length in initial_lengths),
            "task_goal_log_present_count": sum(any(line.startswith("task_goal:") for log in record["logs"] for line in log["relevant_lines"]) for record in uniform),
            "actual_seed_mapping_sha256": digest_value(seed_records),
            "seed_nonzero_last_two_digits_count": sum(row["resolved_environment_seed"] % 100 != 0 for row in seed_records),
            "old_U_vs_OC_manifest_field_difference_counts": paired_field_difference_counts,
        },
        "initial_evidence_limits": {
            "cached_initial_inventory_arms": sorted({row["key"]["arm"] for row in initial_records}),
            "cached_initial_inventory_count": len(initial_records),
            "cached_inventory_errors": inventory["errors"],
            "cached_possible_lossless_files": inventory["possible_lossless_files"],
            "prior_aggregate_claims_initial_fairness": loaded["completeness"]["initial_condition_fairness"],
            "direct_U_initial_hash_values_absent_from_inspected_caches":
                not any(any(field in record.get("initial_hashes", {}) or field in record for field in INITIAL_FIELDS)
                        for record in uniform) and
                not any(row["key"]["arm"] == "U" and row.get("initial_hashes") for row in initial_records),
            "cannot_substitute_OC_inventory_for_direct_U_evidence": True,
        },
        "historical_cross_host_OC_comparisons_not_U_or_new_arms": pair_summaries,
        "historically_reported_protocol_provenance": loaded["completeness"]["protocol_provenance"],
        "cached_migration_checkpoint_content_matches_old_report":
            loaded["lighthouse_checkpoint_check"]["content_tree_identity"]["content_tree_sha256"] ==
            loaded["completeness"]["protocol_provenance"]["checkpoint_unpacked_content_tree_sha256"],
        "not_established": [
            "Direct re-audit of U raw result, manifest, full trace and initial-condition files",
            "Historic U raw initial image/state arrays and exact per-frame stage/text provenance",
            "Complete historical U execution software/driver/hardware identity against the chosen new environment",
            "Any UK48/UN48 initial-condition or execution-provenance alignment",
            "Real-checkpoint 768-token GPU smoke or authorization for any rerun",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--migration", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    print(json.dumps(audit(repo, args.migration or repo.parent / "lighthouse_migration"), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
