#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO = Path("/home/jp673/robomme_dual_memory/robomme_policy_learning")
RUN_ROOT = REPO / "outputs/dual_memory/20260818-214308_ecf086c"


def main() -> None:
    output = RUN_ROOT / "environment/smoke_jobs.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite job record: {output}")
    gpu_jobs = json.loads((RUN_ROOT / "environment/gpu_audit_jobs_retry1.json").read_text())
    dependency = f"afterok:{gpu_jobs['sp_gpu_architecture_audit']}"
    command = [
        "sbatch",
        "--parsable",
        f"--dependency={dependency}",
        f"--output={RUN_ROOT}/training/logs/sp-training-smoke-%j.out",
        f"--error={RUN_ROOT}/training/logs/sp-training-smoke-%j.err",
        "experiments/dual_memory/run_sp_training_smoke.sbatch",
    ]
    job_id = subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]
    with output.open("x") as stream:
        json.dump({"sp_training_smoke": job_id, "dependency": dependency}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(job_id)


if __name__ == "__main__":
    main()
