#!/usr/bin/env python3
"""Apply the preregistered Oracle gate before spending Qwen tokens."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def submit(repo: Path, run_root: Path, model: str, seed: int, partition: str, scope: str) -> str:
    return subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            f"--partition={partition}",
            f"--job-name=dual-qwen-{model.lower()}-{seed}",
            f"--output={run_root}/training/logs/qwen-{scope}-{model}-{seed}-%j.out",
            f"--error={run_root}/training/logs/qwen-{scope}-{model}-{seed}-%j.err",
            "experiments/dual_memory/run_formal_evaluation.sbatch",
            model,
            str(seed),
            "qwenvl",
            scope,
        ],
        cwd=repo,
        text=True,
    ).strip().split(";")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_root / "environment/qwen_evaluation_jobs.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Qwen evaluation job record: {output}")
    gate = json.loads((args.run_root / "analysis/oracle_gate.json").read_text())
    jobs = {}
    if gate["status"] == "go":
        scope = "full"
        assignments = [
            ("S", 42, "athena"), ("SP", 42, "athena"),
            ("S", 43, "athena-mini"), ("SP", 43, "athena-mini"),
            ("S", 44, "athena-small"), ("SP", 44, "athena-small"),
        ]
    else:
        scope = "diagnostic"
        assignments = [("S", 42, "athena-mini"), ("SP", 42, "athena-small")]
    for model, seed, partition in assignments:
        jobs[f"{model}_{seed}"] = {
            "job_id": submit(args.repo, args.run_root, model, seed, partition, scope),
            "partition": partition,
        }
    record = {"oracle_gate": gate, "scope": scope, "jobs": jobs}
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")

    dependency = "afterok:" + ":".join(item["job_id"] for item in jobs.values())
    final_job = subprocess.check_output(
        [
            "sbatch", "--parsable", "--partition=athena", "--time=08:00:00",
            "--cpus-per-task=8", "--mem=64G", f"--dependency={dependency}",
            f"--output={args.run_root}/training/logs/final-analysis-%j.out",
            f"--error={args.run_root}/training/logs/final-analysis-%j.err",
            "--wrap",
            f"cd {args.repo} && .venv/bin/python experiments/dual_memory/finalize_analysis.py --run-root {args.run_root}",
        ],
        text=True,
    ).strip().split(";")[0]
    final_record = args.run_root / "environment/final_analysis_job.json"
    with final_record.open("x") as stream:
        json.dump({"final_analysis": final_job, "dependency": dependency}, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
