#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO = Path("/home/jp673/robomme_dual_memory/robomme_policy_learning")
RUN_ROOT = REPO / "outputs/dual_memory/20260818-214308_ecf086c"


def submit_rollout(model: str, partition: str, dependency: str) -> str:
    return subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            f"--partition={partition}",
            f"--job-name=dual-pilot-{model.lower()}",
            f"--dependency={dependency}",
            f"--output={RUN_ROOT}/training/logs/pilot-{model}-%j.out",
            f"--error={RUN_ROOT}/training/logs/pilot-{model}-%j.err",
            "experiments/dual_memory/run_pilot_rollout.sbatch",
            model,
        ],
        cwd=REPO,
        text=True,
    ).strip().split(";")[0]


def main() -> None:
    output = RUN_ROOT / "environment/pilot_jobs.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite job record: {output}")
    matched = json.loads((RUN_ROOT / "environment/matched_smoke_jobs.json").read_text())["jobs"]
    sp_smoke = json.loads((RUN_ROOT / "environment/smoke_jobs.json").read_text())["sp_training_smoke"]
    rollout_smoke = json.loads((RUN_ROOT / "environment/smoke_rollout_jobs.json").read_text())["sp_rollout_smoke"]
    train_jobs = {**matched, "SP": sp_smoke}
    partitions = {"N": "athena", "S": "athena-mini", "P": "athena-small", "SP": "athena"}
    rollouts = {
        model: submit_rollout(
            model,
            partitions[model],
            f"afterok:{train_jobs[model]}:{rollout_smoke}",
        )
        for model in ["N", "S", "P", "SP"]
    }
    analysis_dependency = "afterok:" + ":".join(rollouts.values())
    analysis_job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=athena",
            "--time=01:00:00",
            "--cpus-per-task=2",
            "--mem=8G",
            f"--dependency={analysis_dependency}",
            f"--output={RUN_ROOT}/training/logs/pilot-report-%j.out",
            f"--error={RUN_ROOT}/training/logs/pilot-report-%j.err",
            "--wrap",
            f"cd {REPO} && .venv/bin/python experiments/dual_memory/build_pilot_report.py --run-root {RUN_ROOT}",
        ],
        text=True,
    ).strip().split(";")[0]
    with output.open("x") as stream:
        json.dump(
            {"rollouts": rollouts, "pilot_report": analysis_job, "analysis_dependency": analysis_dependency},
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(json.dumps({"rollouts": rollouts, "pilot_report": analysis_job}, sort_keys=True))


if __name__ == "__main__":
    main()
