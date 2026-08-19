#!/usr/bin/env python3
"""Write-once paired hierarchical analysis for the complete Oracle matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


MODELS = ["N", "S", "P", "SP"]
SEEDS = [42, 43, 44]
TASKS = [
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    "ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]
TASK_FAMILIES = {
    "dynamic_repetition": TASKS[0:4],
    "unmasking": TASKS[4:8],
    "visual_regrasp_and_placement": TASKS[8:12],
    "spatial_long_horizon": TASKS[12:16],
}


def interval(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def statistics(array: np.ndarray, rng: np.random.Generator, draws: int = 10000) -> dict:
    # array shape: seed, model, paired_episode
    observed_rates = array.mean(axis=(0, 2))
    samples = defaultdict(list)
    seed_count, _, episode_count = array.shape
    for _ in range(draws):
        sampled_seeds = rng.integers(0, seed_count, seed_count)
        rates = np.zeros(len(MODELS), dtype=np.float64)
        for seed_index in sampled_seeds:
            episode_indices = rng.integers(0, episode_count, episode_count)
            rates += array[seed_index][:, episode_indices].mean(axis=1)
        rates /= seed_count
        n, s, p, sp = rates
        samples["SP-S"].append(sp - s)
        samples["SP-P"].append(sp - p)
        samples["SP-max(S,P)"].append(sp - max(s, p))
        samples["interaction"].append(sp - s - p + n)
    n, s, p, sp = observed_rates
    estimates = {
        "SP-S": float(sp - s),
        "SP-P": float(sp - p),
        "SP-max(S,P)": float(sp - max(s, p)),
        "interaction": float(sp - s - p + n),
    }
    return {
        "success_rates": {model: float(observed_rates[index]) for index, model in enumerate(MODELS)},
        "contrasts": {
            name: {"estimate": estimates[name], "ci95": interval(values)}
            for name, values in samples.items()
        },
        "bootstrap_draws": draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    stats_output = args.run_root / "analysis/paired_statistics.json"
    aggregate_output = args.run_root / "evaluation/aggregate.json"
    csv_output = args.run_root / "evaluation/per_episode.csv"
    gate_output = args.run_root / "analysis/oracle_gate.json"
    if any(path.exists() for path in [stats_output, aggregate_output, csv_output, gate_output]):
        raise FileExistsError("Refusing to overwrite formal Oracle results")

    all_records = [
        json.loads(line)
        for line in (args.run_root / "evaluation/per_episode.jsonl").read_text().splitlines()
    ]
    records = [
        record for record in all_records
        if record.get("model_id") in MODELS
        and record.get("training_seed") in SEEDS
        and record.get("evaluation_scope") == "full"
        and record.get("symbolic_source") == ("oracle" if record.get("model_id") in {"S", "SP"} else "none")
    ]
    expected = {(model, seed, task, episode) for model in MODELS for seed in SEEDS for task in TASKS for episode in range(50)}
    by_key = {(r["model_id"], r["training_seed"], r["task"], r["episode_id"]): r for r in records}
    if set(by_key) != expected:
        raise RuntimeError(
            f"Formal Oracle matrix is incomplete: missing={len(expected - set(by_key))}, extra={len(set(by_key) - expected)}"
        )

    array = np.zeros((len(SEEDS), len(MODELS), len(TASKS) * 50), dtype=np.float64)
    for si, seed in enumerate(SEEDS):
        for mi, model in enumerate(MODELS):
            array[si, mi] = [
                by_key[(model, seed, task, episode)]["success"]
                for task in TASKS for episode in range(50)
            ]
    rng = np.random.default_rng(20260818)
    overall = statistics(array, rng)
    per_task = {}
    for task_index, task in enumerate(TASKS):
        start = task_index * 50
        per_task[task] = statistics(array[:, :, start:start + 50], rng, draws=2000)
    family_macro = {
        family: {
            model: float(np.mean([per_task[task]["success_rates"][model] for task in tasks]))
            for model in MODELS
        }
        for family, tasks in TASK_FAMILIES.items()
    }

    mcnemar = {}
    for other in ["S", "P"]:
        sp_values = array[:, MODELS.index("SP")].astype(bool).ravel()
        other_values = array[:, MODELS.index(other)].astype(bool).ravel()
        sp_only = int(np.sum(sp_values & ~other_values))
        other_only = int(np.sum(~sp_values & other_values))
        pvalue = float(binomtest(sp_only, sp_only + other_only, 0.5).pvalue) if sp_only + other_only else 1.0
        mcnemar[f"SP_vs_{other}"] = {"SP_only_success": sp_only, f"{other}_only_success": other_only, "exact_pvalue": pvalue}

    counts = defaultdict(lambda: defaultdict(dict))
    for model in MODELS:
        for seed in SEEDS:
            for task in TASKS:
                values = [by_key[(model, seed, task, episode)] for episode in range(50)]
                counts[model][str(seed)][task] = {
                    "success": sum(record["success"] for record in values),
                    "total": 50,
                    "timeout": sum(record["timeout"] for record in values),
                    "collision": sum(record["collision"] for record in values),
                    "mean_steps": float(np.mean([record["steps"] for record in values])),
                }
    contrast = overall["contrasts"]["SP-max(S,P)"]
    oracle_go = contrast["estimate"] >= 0.03 and contrast["ci95"][0] > 0
    stats_result = {
        "method": "hierarchical paired bootstrap: resample training seeds, then paired task/episode IDs within seed",
        "bootstrap_seed": 20260818,
        "overall": overall,
        "per_task": per_task,
        "task_family_macro_success_rates": family_macro,
        "mcnemar": mcnemar,
        "success_counts": counts,
    }
    aggregate = {
        "record_count": len(records),
        "models": MODELS,
        "training_seeds": SEEDS,
        "task_count": len(TASKS),
        "episodes_per_task": 50,
        "overall_success_rates": overall["success_rates"],
        "terminal_reason_counts": Counter(record["terminal_reason"] for record in records),
    }
    gate = {
        "status": "go" if oracle_go else "no-go",
        "threshold_absolute_gain": 0.03,
        "observed_SP_minus_max": contrast["estimate"],
        "ci95": contrast["ci95"],
        "qwen_action": "full_S_and_SP" if oracle_go else "diagnostic_only",
    }
    for path, value in [(stats_output, stats_result), (aggregate_output, aggregate), (gate_output, gate)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    fieldnames = sorted({key for record in records for key in record})
    with csv_output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda r: (r["model_id"], r["training_seed"], r["task"], r["episode_id"])):
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in record.items()})


if __name__ == "__main__":
    main()
