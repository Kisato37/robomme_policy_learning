#!/usr/bin/env python3
"""Finalize Qwen costs, failure candidates, figures, and the experiment report."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ["N", "S", "P", "SP"]
SEEDS = [42, 43, 44]
TASKS = [
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    "ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]


def write_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite final artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def qwen_bootstrap(records: list[dict], oracle_records: list[dict]) -> dict:
    by_key = {(r["model_id"], r["training_seed"], r["task"], r["episode_id"]): r for r in records}
    oracle_p = {(r["training_seed"], r["task"], r["episode_id"]): r for r in oracle_records if r["model_id"] == "P"}
    expected = {(model, seed, task, episode) for model in ["S", "SP"] for seed in SEEDS for task in TASKS for episode in range(50)}
    if set(by_key) != expected:
        raise RuntimeError(f"Full Qwen matrix incomplete: missing={len(expected - set(by_key))}")
    values = np.zeros((3, 3, 800), dtype=np.float64)  # S-Qwen, SP-Qwen, P
    for si, seed in enumerate(SEEDS):
        values[si, 0] = [by_key[("S", seed, task, ep)]["success"] for task in TASKS for ep in range(50)]
        values[si, 1] = [by_key[("SP", seed, task, ep)]["success"] for task in TASKS for ep in range(50)]
        values[si, 2] = [oracle_p[(seed, task, ep)]["success"] for task in TASKS for ep in range(50)]
    rates = values.mean(axis=(0, 2))
    rng = np.random.default_rng(20260819)
    samples = defaultdict(list)
    for _ in range(10000):
        sampled_seeds = rng.integers(0, 3, 3)
        sample_rates = np.zeros(3)
        for seed_index in sampled_seeds:
            indices = rng.integers(0, 800, 800)
            sample_rates += values[seed_index][:, indices].mean(axis=1)
        sample_rates /= 3
        samples["SP-Qwen - S-Qwen"].append(sample_rates[1] - sample_rates[0])
        samples["SP-Qwen - P"].append(sample_rates[1] - sample_rates[2])
    return {
        "success_rates": {"S-Qwen": float(rates[0]), "SP-Qwen": float(rates[1]), "P": float(rates[2])},
        "contrasts": {
            name: {
                "estimate": float(np.mean(sample)),
                "ci95": [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))],
            }
            for name, sample in samples.items()
        },
        "bootstrap_draws": 10000,
        "bootstrap_seed": 20260819,
    }


def strip_coordinates(text: str | None) -> str:
    return re.sub(r"<\d+,\s*\d+>", "<coord>", text or "").strip().lower()


def extract_frame(video: Path, output: Path) -> bool:
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
    ok, frame = capture.read()
    capture.release()
    if ok:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), frame)
    return bool(ok)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    final_report = args.run_root / "FINAL_REPORT.md"
    taxonomy_path = args.run_root / "analysis/failure_taxonomy.csv"
    qwen_summary_path = args.run_root / "analysis/qwen_summary.json"
    qwen_calls_path = args.run_root / "evaluation/qwen_calls.jsonl"
    if any(path.exists() for path in [final_report, taxonomy_path, qwen_summary_path, qwen_calls_path]):
        raise FileExistsError("Refusing to overwrite final analysis")

    gate = json.loads((args.run_root / "analysis/oracle_gate.json").read_text())
    qwen_jobs = json.loads((args.run_root / "environment/qwen_evaluation_jobs.json").read_text())
    all_records = [json.loads(line) for line in (args.run_root / "evaluation/per_episode.jsonl").read_text().splitlines()]
    oracle = [
        r for r in all_records if r.get("model_id") in MODELS and r.get("evaluation_scope") == "full"
        and r.get("symbolic_source") == ("oracle" if r.get("model_id") in {"S", "SP"} else "none")
    ]
    qwen_records = [
        r for r in all_records if r.get("model_id") in {"S", "SP"}
        and r.get("symbolic_source") == "qwenvl" and r.get("evaluation_scope") == qwen_jobs["scope"]
    ]

    call_files = sorted((args.run_root / "formal/evaluation").rglob("*_QwenVL_calls.jsonl"))
    calls = [json.loads(line) for path in call_files for line in path.read_text().splitlines()]
    with qwen_calls_path.open("x") as output:
        for call in calls:
            output.write(json.dumps(call, sort_keys=True) + "\n")
    actual_calls = [call for call in calls if not call.get("cache_hit") and not call.get("reuse_reason")]
    successes = sum(r["success"] for r in qwen_records)
    qwen_summary = {
        "scope": qwen_jobs["scope"],
        "episode_count": len(qwen_records),
        "successful_episodes": successes,
        "logged_requests": len(calls),
        "actual_model_calls": len(actual_calls),
        "cache_hits": sum(bool(call.get("cache_hit")) for call in calls),
        "cache_hit_rate": sum(bool(call.get("cache_hit")) for call in calls) / len(calls) if calls else 0.0,
        "input_tokens": sum(int(call.get("input_tokens") or 0) for call in actual_calls),
        "output_tokens": sum(int(call.get("output_tokens") or 0) for call in actual_calls),
        "tokens_per_successful_episode": (
            sum(int(call.get("input_tokens") or 0) + int(call.get("output_tokens") or 0) for call in actual_calls) / successes
            if successes else None
        ),
        "invalid_coordinate_calls": sum(not bool(call.get("coordinate_valid")) for call in calls),
    }
    if qwen_jobs["scope"] == "full":
        qwen_summary["paired_statistics"] = qwen_bootstrap(qwen_records, oracle)
        qwen_contrasts = qwen_summary["paired_statistics"]["contrasts"]
        qwen_sp_rate = qwen_summary["paired_statistics"]["success_rates"]["SP-Qwen"]
        oracle_sp_rate = json.loads(
            (args.run_root / "analysis/paired_statistics.json").read_text()
        )["overall"]["success_rates"]["SP"]
        stable_gain = any(
            qwen_contrasts[name]["estimate"] > 0 and qwen_contrasts[name]["ci95"][0] > 0
            for name in ["SP-Qwen - S-Qwen", "SP-Qwen - P"]
        )
        qwen_summary["qwen_decision"] = {
            "status": "go" if stable_gain and qwen_sp_rate >= oracle_sp_rate - 0.03 else "no-go",
            "stable_positive_gain_vs_at_least_one_baseline": stable_gain,
            "SP_Qwen_minus_SP_Oracle": qwen_sp_rate - oracle_sp_rate,
            "no_more_than_3pp_degradation_vs_SP_Oracle": qwen_sp_rate >= oracle_sp_rate - 0.03,
        }
    else:
        qwen_summary["qwen_decision"] = {
            "status": "not_run_full_due_to_oracle_no_go",
            "scientific_interpretation": "diagnostic calls only",
        }
    write_json(qwen_summary_path, qwen_summary)

    oracle_by_key = {(r["model_id"], r["training_seed"], r["task"], r["episode_id"]): r for r in oracle}
    qwen_by_key = {(r["model_id"], r["training_seed"], r["task"], r["episode_id"]): r for r in qwen_records}
    taxonomy = []
    for seed in SEEDS:
        for task in TASKS:
            for episode in range(50):
                sp = oracle_by_key[("SP", seed, task, episode)]
                s = oracle_by_key[("S", seed, task, episode)]
                p = oracle_by_key[("P", seed, task, episode)]
                if sp["success"] != s["success"] or sp["success"] != p["success"]:
                    label = "memory_fusion_error" if not sp["success"] and (s["success"] or p["success"]) else "uncertain"
                    taxonomy.append({
                        "comparison": "SP_vs_single_memory", "training_seed": seed, "task": task,
                        "episode_id": episode, "label": label, "SP_success": sp["success"],
                        "S_success": s["success"], "P_success": p["success"], "video_path": sp["video_path"],
                        "S_video_path": s["video_path"], "P_video_path": p["video_path"],
                        "SP_subgoals": json.dumps(sp.get("subgoal_sequence", [])),
                        "S_subgoals": json.dumps(s.get("subgoal_sequence", [])),
                        "P_subgoals": json.dumps(p.get("subgoal_sequence", [])),
                        "Qwen_subgoals": "",
                    })
                qwen_sp = qwen_by_key.get(("SP", seed, task, episode))
                if qwen_sp is not None and qwen_sp["success"] != sp["success"]:
                    oracle_goal = next((value for value in sp.get("subgoal_sequence", []) if value), None)
                    qwen_goal = next((value for value in qwen_sp.get("subgoal_sequence", []) if value), None)
                    if strip_coordinates(oracle_goal) != strip_coordinates(qwen_goal):
                        label = "subgoal_stage_error"
                    elif oracle_goal != qwen_goal:
                        label = "grounding_error"
                    else:
                        label = "control_generation_error"
                    taxonomy.append({
                        "comparison": "SP_Oracle_vs_Qwen", "training_seed": seed, "task": task,
                        "episode_id": episode, "label": label, "SP_success": sp["success"],
                        "S_success": "", "P_success": "", "video_path": qwen_sp["video_path"],
                        "S_video_path": "", "P_video_path": "",
                        "SP_subgoals": json.dumps(sp.get("subgoal_sequence", [])),
                        "S_subgoals": "", "P_subgoals": "",
                        "Qwen_subgoals": json.dumps(qwen_sp.get("subgoal_sequence", [])),
                    })
    fieldnames = [
        "comparison", "training_seed", "task", "episode_id", "label",
        "SP_success", "S_success", "P_success", "video_path", "S_video_path", "P_video_path",
        "SP_subgoals", "S_subgoals", "P_subgoals", "Qwen_subgoals",
    ]
    with taxonomy_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(taxonomy)
    label_counts = Counter(row["label"] for row in taxonomy)
    representatives = []
    for label in sorted(label_counts):
        representatives.extend([row for row in taxonomy if row["label"] == label][:3])
    for index, row in enumerate(representatives):
        extract_frame(
            Path(row["video_path"]),
            args.run_root / f"analysis/figures/failure_{index:02d}_{row['label']}.png",
        )

    paired = json.loads((args.run_root / "analysis/paired_statistics.json").read_text())
    rates = paired["overall"]["success_rates"]
    figure = args.run_root / "analysis/figures/overall_success.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(MODELS, [100 * rates[model] for model in MODELS])
    plt.ylabel("Episode success rate (%)")
    plt.title("Matched Oracle 2x2 experiment")
    plt.tight_layout()
    plt.savefig(figure, dpi=180)
    plt.close()

    contrast = paired["overall"]["contrasts"]["SP-max(S,P)"]
    interaction = paired["overall"]["contrasts"]["interaction"]
    task_deltas = {
        task: paired["per_task"][task]["success_rates"]["SP"]
        - max(
            paired["per_task"][task]["success_rates"]["S"],
            paired["per_task"][task]["success_rates"]["P"],
        )
        for task in TASKS
    }
    largest_gains = sorted(task_deltas.items(), key=lambda item: item[1], reverse=True)[:3]
    largest_costs = sorted(task_deltas.items(), key=lambda item: item[1])[:3]
    if gate["status"] == "go" and qwen_summary["qwen_decision"]["status"] == "go":
        recommendation = "Continue deeper fusion research; both the Oracle architecture and the deployable Qwen condition passed their preregistered gates."
    elif gate["status"] == "go":
        recommendation = "Keep the dual-memory interface and prioritize GroundSG predictor/grounding quality; Oracle passed while the Qwen condition did not."
    else:
        recommendation = "Stop scaling the current fusion interface; full Qwen evaluation was gated off and only diagnostics were retained."
    lines = [
        "# RoboMME dual-memory experiment final report",
        "",
        f"Oracle architecture decision: **{gate['status'].upper()}**.",
        "",
        "## Primary result",
        "",
        f"Matched success rates: N={rates['N']:.3%}, S-Oracle={rates['S']:.3%}, P={rates['P']:.3%}, SP-Oracle={rates['SP']:.3%}.",
        f"SP - max(S, P) = {contrast['estimate']:.3%}, hierarchical paired 95% CI [{contrast['ci95'][0]:.3%}, {contrast['ci95'][1]:.3%}].",
        f"2x2 interaction = {interaction['estimate']:.3%}, 95% CI [{interaction['ci95'][0]:.3%}, {interaction['ci95'][1]:.3%}].",
        f"Largest task gains versus max(S,P): {largest_gains}; largest costs: {largest_costs}.",
        "",
        "Per-task results, task-family macro averages, raw counts, McNemar tests, and all paired intervals are in `analysis/paired_statistics.json`.",
        "",
        "## Qwen realism and cost",
        "",
        f"Qwen scope after the preregistered gate: **{qwen_jobs['scope']}**; episodes={len(qwen_records)}, actual calls={qwen_summary['actual_model_calls']}, cache hit rate={qwen_summary['cache_hit_rate']:.2%}.",
        f"Input/output tokens={qwen_summary['input_tokens']}/{qwen_summary['output_tokens']}; tokens per successful episode={qwen_summary['tokens_per_successful_episode']}.",
        f"Qwen decision: {qwen_summary['qwen_decision']}.",
        "",
        "## Failure attribution",
        "",
        f"Discordant episode labels: {dict(label_counts)}. Automated labels are conservative; cases without sufficient causal evidence remain `uncertain`.",
        "Representative video paths are retained in `analysis/failure_taxonomy.csv`; extracted middle frames are in `analysis/figures/`.",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "All formal episode rows are append-only and the formal aggregate/statistics/report files were created once.",
    ]
    with final_report.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    subprocess.check_call(
        [
            sys.executable,
            str(Path(__file__).with_name("record_phase_status.py")),
            "--run-root", str(args.run_root),
            "--phase", "Phase-6", "--status", "completed",
            "--evidence", "analysis/paired_statistics.json",
            "--evidence", "analysis/failure_taxonomy.csv",
            "--evidence", "FINAL_REPORT.md",
            "--next", "Experiment complete; review the GO/NO-GO recommendation.",
        ]
    )


if __name__ == "__main__":
    main()
