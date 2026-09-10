"""Offline capacity planning from existing causal selector traces, not a rollout.

Uses no terminal outcomes to filter or choose capacity. For U, uses the recorded
literal U indices; on other recorded histories, computes the known U32 rule only
as a storage-requirement diagnostic, never as a counterfactual policy outcome.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(rows):
    unions, boundaries, extra_counts, peaks = [], [], [], []
    task_max = defaultdict(int)
    shortage_calls = 0
    for row in rows:
        arm = row["scientific_key"]["arm"]
        episode_counts = []
        for call in row["calls"]:
            t = call["current_history_index"]
            u32 = np.linspace(0, t, min(32, t + 1), dtype=np.int32).tolist()
            if arm == "U":
                assert call["selected_frame_indices"] == u32
                u = set(call["selected_frame_indices"])
            else:
                u = set(u32)
            b = set(call["visible_boundary_indices"])
            assert all(0 <= i <= t for i in b)
            n = len(u | b)
            k = len(b - u)
            candidates = t + 1 - n
            shortage_calls += candidates < k
            unions.append(n)
            boundaries.append(len(b))
            extra_counts.append(k)
            episode_counts.append(n)
            task = row["scientific_key"]["task"]
            task_max[task] = max(task_max[task], n)
        peaks.append(max(episode_counts))
    return {
        "episodes": len(rows), "policy_calls": len(unions),
        "max_visible_boundaries": max(boundaries),
        "max_new_boundaries_not_in_U": max(extra_counts),
        "union_size_quantiles": dict(zip(
            ["min", "p50", "p90", "p95", "p99", "max"],
            [float(x) for x in np.quantile(unions, [0, .5, .9, .95, .99, 1])],
        )),
        "episode_peak_quantiles": dict(zip(
            ["p50", "p90", "p95", "p99", "max"],
            [float(x) for x in np.quantile(peaks, [.5, .9, .95, .99, 1])],
        )),
        "capacity_overflows": {str(k): {
            "calls": sum(n > k for n in unions),
            "episodes": sum(n > k for n in peaks),
        } for k in [40, 48, 56, 64]},
        "random_nonkey_candidate_shortage_calls": shortage_calls,
        "max_union_by_task": dict(sorted(task_max.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a capacity evidence snapshot")
    records, hashes = [], {}
    for name in ["athena_records.json", "lighthouse_records.json"]:
        path = args.audit_dir / name
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        assert not data["hash_errors"]
        assert all(all(r["hash_checks"].values()) for r in data["records"])
        records.extend(data["records"])
    arms = ["U", "OC", "OC3", "OC5"]
    assert len(records) == 3200
    keys = [(r["scientific_key"]["arm"], r["scientific_key"]["task"],
             r["scientific_key"]["episode_id"]) for r in records]
    assert len(set(keys)) == 3200
    results = {arm: summarize([r for r in records if r["scientific_key"]["arm"] == arm])
               for arm in arms}
    assert all(d["episodes"] == 800 for d in results.values())
    output = {
        "schema_version": 1, "date": "2026-09-09",
        "purpose": "Post-hoc planning of fixed capacity; no success-rate optimization or new policy evaluation",
        "population_filter": "All recorded episodes in U, OC, OC3, OC5; no success/failure selection",
        "source_sha256": hashes,
        "total_policy_calls": sum(d["policy_calls"] for d in results.values()),
        "proposed_fixed_frame_slots": 48, "base_U_frame_slots": 32,
        "tokens_per_frame": 16, "proposed_memory_token_slots": 768,
        "results": results,
        "limits": [
            "These are previously observed histories, not trajectories of either proposed new arm.",
            "No assurance that new rollouts cannot overflow or lack random candidates.",
            "The original outcomes have been inspected previously; this is an explicitly post-hoc follow-up.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({"calls": output["total_policy_calls"], "U": results["U"],
                      "capacity": 48, "tokens": 768}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
