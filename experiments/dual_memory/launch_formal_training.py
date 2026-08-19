#!/usr/bin/env python3
"""Freeze the final manifest, estimate cost from smoke, and submit 12 formal runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from experiments.dual_memory.run_training import build_config, write_config_snapshot


MODELS = ["N", "S", "P", "SP"]
SEEDS = [42, 43, 44]


def smoke_rate(run_root: Path, model: str) -> float:
    path = run_root / f"training/checkpoints/mme_vla_suite/smoke100_{model}_seed42/training_metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) < 2 or records[-1]["step"] <= records[0]["step"]:
        raise RuntimeError(f"Insufficient throughput records: {path}")
    return (records[-1]["wall_time_unix"] - records[0]["wall_time_unix"]) / (
        records[-1]["step"] - records[0]["step"]
    )


def submit(repo: Path, run_root: Path, model: str, seed: int, partition: str, dependency: str | None) -> str:
    command = [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        f"--job-name=dual-{model.lower()}-{seed}",
        f"--output={run_root}/training/logs/formal-{model}-{seed}-%j.out",
        f"--error={run_root}/training/logs/formal-{model}-{seed}-%j.err",
    ]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command += ["experiments/dual_memory/run_formal_training.sbatch", model, str(seed)]
    return subprocess.check_output(command, cwd=repo, text=True).strip().split(";")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_root / "environment/formal_jobs.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite formal job record: {output}")
    pilot = json.loads((args.run_root / "audits/pilot_report.json").read_text())
    if pilot["status"] != "pass":
        raise RuntimeError("Formal training is blocked by the pilot gate")
    for model in MODELS:
        reload_report = json.loads(
            (args.run_root / f"audits/checkpoint_reload_smoke_{model}_seed42.json").read_text()
        )
        if reload_report["status"] != "pass":
            raise RuntimeError(f"Formal training is blocked by {model} checkpoint reload audit")

    rates = {model: smoke_rate(args.run_root, model) for model in MODELS}
    estimated_hours = {model: rates[model] * 80000 / 3600 for model in MODELS}
    for model in MODELS:
        for seed in SEEDS:
            config_args = SimpleNamespace(
                model=model,
                seed=seed,
                mode="formal",
                num_steps=80000,
                dataset=args.repo / "data/robomme_preprocessed_data",
                assets_base_dir=args.repo / "runs/assets",
                run_root=args.run_root,
            )
            write_config_snapshot(build_config(config_args), config_args)

    assignment = {
        "athena": [("N", 42), ("S", 43), ("P", 44), ("SP", 42)],
        "athena-mini": [("N", 43), ("S", 44), ("P", 42), ("SP", 43)],
        "athena-small": [("N", 44), ("S", 42), ("P", 43), ("SP", 44)],
    }
    jobs = {}
    for partition, runs in assignment.items():
        previous = None
        for model, seed in runs:
            job_id = submit(args.repo, args.run_root, model, seed, partition, previous)
            jobs[f"{model}_{seed}"] = {"job_id": job_id, "partition": partition, "afterok": previous}
            previous = job_id
    evaluation_dependency = "afterok:" + ":".join(item["job_id"] for item in jobs.values())
    oracle_launch_job = subprocess.check_output(
        [
            "sbatch", "--parsable", f"--dependency={evaluation_dependency}",
            f"--output={args.run_root}/training/logs/oracle-launch-%j.out",
            f"--error={args.run_root}/training/logs/oracle-launch-%j.err",
            "experiments/dual_memory/run_oracle_launch_gate.sbatch",
        ],
        cwd=args.repo,
        text=True,
    ).strip().split(";")[0]
    record = {
        "jobs": jobs,
        "seconds_per_step_from_smoke": rates,
        "estimated_wall_hours_per_80k_run": estimated_hours,
        "estimated_total_gpu_hours": sum(estimated_hours[model] * 8 * 3 for model in MODELS),
        "parallel_schedule": "three 8-GPU nodes, four sequential formal runs assigned per node",
        "time_limit_per_run_days": 14,
        "oracle_evaluation_launch_job": oracle_launch_job,
        "oracle_evaluation_dependency": evaluation_dependency,
    }
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")

    prereg = json.loads((args.run_root / "manifest.preregistered.json").read_text())
    environment = json.loads((args.run_root / "environment/environment_summary.json").read_text())
    manifest = {
        **prereg,
        "status": "formal_training_launched_after_all_engineering_gates",
        "environment": environment,
        "asset_manifests": {
            "data": "environment/data_assets.json",
            "models": "environment/model_assets.json",
            "qwen": "environment/qwen_setup.json",
        },
        "formal_schedule": record,
        "documented_deviations": [
            "Qwen uses the RoboMME-documented SDPA backend because Athena has no nvcc for optional flash-attn.",
            "Weights & Biases is disabled; append-only local training_metrics.jsonl is authoritative.",
            "Official 4xA40-40GB hardware is replaced by an audited 8xRTX-A5000-24GB FSDP layout.",
        ],
    }
    manifest_path = args.run_root / "manifest.json"
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
