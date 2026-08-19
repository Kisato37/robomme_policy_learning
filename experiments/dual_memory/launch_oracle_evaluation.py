#!/usr/bin/env python3
"""Submit the complete matched Oracle matrix after all formal checkpoints exist."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def submit(repo: Path, run_root: Path, model: str, seed: int, partition: str) -> str:
    source = "oracle" if model in {"S", "SP"} else "none"
    return subprocess.check_output(
        [
            "sbatch", "--parsable", f"--partition={partition}",
            f"--job-name=dual-eval-{model.lower()}-{seed}",
            f"--output={run_root}/training/logs/oracle-{model}-{seed}-%j.out",
            f"--error={run_root}/training/logs/oracle-{model}-{seed}-%j.err",
            "experiments/dual_memory/run_formal_evaluation.sbatch",
            model, str(seed), source, "full",
        ],
        cwd=repo,
        text=True,
    ).strip().split(";")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_root / "environment/oracle_evaluation_jobs.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Oracle evaluation job record: {output}")
    assignments = {
        "athena": [("N", 42), ("S", 42), ("P", 42), ("SP", 42)],
        "athena-mini": [("N", 43), ("S", 43), ("P", 43), ("SP", 43)],
        "athena-small": [("N", 44), ("S", 44), ("P", 44), ("SP", 44)],
    }
    jobs = {}
    for partition, runs in assignments.items():
        for model, seed in runs:
            jobs[f"{model}_{seed}"] = {
                "job_id": submit(args.repo, args.run_root, model, seed, partition),
                "partition": partition,
            }
    dependency = "afterok:" + ":".join(item["job_id"] for item in jobs.values())
    analysis_job = subprocess.check_output(
        [
            "sbatch", "--parsable", "--partition=athena", "--time=08:00:00",
            "--cpus-per-task=8", "--mem=64G", f"--dependency={dependency}",
            f"--output={args.run_root}/training/logs/oracle-analysis-%j.out",
            f"--error={args.run_root}/training/logs/oracle-analysis-%j.err",
            "--wrap",
            f"cd {args.repo} && .venv/bin/python experiments/dual_memory/analyze_oracle.py --run-root {args.run_root}",
        ],
        text=True,
    ).strip().split(";")[0]
    qwen_launch = subprocess.check_output(
        [
            "sbatch", "--parsable", "--partition=athena", "--time=01:00:00",
            "--cpus-per-task=2", "--mem=8G", f"--dependency=afterok:{analysis_job}",
            f"--output={args.run_root}/training/logs/qwen-launch-%j.out",
            f"--error={args.run_root}/training/logs/qwen-launch-%j.err",
            "--wrap",
            f"cd {args.repo} && .venv/bin/python experiments/dual_memory/launch_qwen_evaluation.py --repo {args.repo} --run-root {args.run_root}",
        ],
        text=True,
    ).strip().split(";")[0]
    with output.open("x") as stream:
        json.dump(
            {"jobs": jobs, "analysis_job": analysis_job, "qwen_launch_job": qwen_launch, "analysis_dependency": dependency},
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
