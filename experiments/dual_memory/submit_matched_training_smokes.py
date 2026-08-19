#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO = Path("/home/jp673/robomme_dual_memory/robomme_policy_learning")
RUN_ROOT = REPO / "outputs/dual_memory/20260818-214308_ecf086c"


def submit(model: str, partition: str, dependency: str) -> str:
    command = [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        f"--job-name=dual-{model.lower()}-smoke",
        f"--dependency={dependency}",
        f"--output={RUN_ROOT}/training/logs/{model}-training-smoke-%j.out",
        f"--error={RUN_ROOT}/training/logs/{model}-training-smoke-%j.err",
        "experiments/dual_memory/run_matched_training_smoke.sbatch",
        model,
    ]
    return subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]


def main() -> None:
    output = RUN_ROOT / "environment/matched_smoke_jobs.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite job record: {output}")
    audit = json.loads((RUN_ROOT / "environment/gpu_audit_jobs_retry1.json").read_text())
    dependency = f"afterok:{audit['sp_gpu_architecture_audit']}"
    jobs = {
        "N": submit("N", "athena", dependency),
        "S": submit("S", "athena-mini", dependency),
        "P": submit("P", "athena-small", dependency),
    }
    with output.open("x") as stream:
        json.dump({"jobs": jobs, "dependency": dependency}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(jobs, sort_keys=True))


if __name__ == "__main__":
    main()
