#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO = Path("/home/jp673/robomme_dual_memory/robomme_policy_learning")
RUN_ROOT = REPO / "outputs/dual_memory/20260818-214308_ecf086c"


def main() -> None:
    output = RUN_ROOT / "environment/qwen_setup_job.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite job record: {output}")
    command = [
        "sbatch",
        "--parsable",
        f"--output={RUN_ROOT}/training/logs/qwen-setup-%j.out",
        f"--error={RUN_ROOT}/training/logs/qwen-setup-%j.err",
        "experiments/dual_memory/run_setup_qwen.sbatch",
    ]
    job_id = subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]
    with output.open("x") as stream:
        json.dump({"qwen_setup": job_id}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(job_id)


if __name__ == "__main__":
    main()
