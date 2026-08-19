#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO = Path("/home/jp673/robomme_dual_memory/robomme_policy_learning")
RUN_ROOT = REPO / "outputs/dual_memory/20260818-214308_ecf086c"


def main() -> None:
    output = RUN_ROOT / "environment/official_test_job.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite job record: {output}")
    job_id = subprocess.check_output(
        [
            "sbatch", "--parsable",
            f"--output={RUN_ROOT}/training/logs/official-tests-%j.out",
            f"--error={RUN_ROOT}/training/logs/official-tests-%j.err",
            "experiments/dual_memory/run_official_tests.sbatch",
        ],
        cwd=REPO,
        text=True,
    ).strip().split(";")[0]
    with output.open("x") as stream:
        json.dump({"official_tests": job_id}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(job_id)


if __name__ == "__main__":
    main()
