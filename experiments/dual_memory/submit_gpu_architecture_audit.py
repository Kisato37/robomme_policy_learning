#!/usr/bin/env python3
"""Submit the SP GPU gate after both data and model assets are ready."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO = Path("/home/jp673/robomme_dual_memory/robomme_policy_learning")
RUN_ROOT = REPO / "outputs/dual_memory/20260818-214308_ecf086c"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry", type=int, default=0)
    parser.add_argument("--supersedes", default=None)
    args = parser.parse_args()
    suffix = "" if args.retry == 0 else f"_retry{args.retry}"
    output = RUN_ROOT / f"environment/gpu_audit_jobs{suffix}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite job record: {output}")
    asset_jobs = json.loads((RUN_ROOT / "environment/asset_jobs.json").read_text())
    data_jobs = json.loads((RUN_ROOT / "environment/audit_jobs.json").read_text())
    dependency = f"afterok:{asset_jobs['unzip_models']}:{data_jobs['data_audit']}"
    command = [
        "sbatch",
        "--parsable",
        f"--dependency={dependency}",
        f"--output={RUN_ROOT}/training/logs/gpu-architecture-%j.out",
        f"--error={RUN_ROOT}/training/logs/gpu-architecture-%j.err",
        "experiments/dual_memory/run_gpu_architecture_audit.sbatch",
    ]
    job_id = subprocess.check_output(command, cwd=REPO, text=True).strip()
    with output.open("x") as stream:
        json.dump(
            {
                "sp_gpu_architecture_audit": job_id,
                "dependency": dependency,
                "supersedes": args.supersedes,
                "retry_reason": "corrected Slurm log parent path" if args.retry else None,
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(job_id)


if __name__ == "__main__":
    main()
