#!/usr/bin/env python3
"""Create the write-once run root and submit pinned asset jobs."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


REPO = Path("/home/jp673/robomme_dual_memory/robomme_policy_learning")
RUN_ID = "20260818-214308_ecf086c"
RUN_ROOT = REPO / "outputs" / "dual_memory" / RUN_ID


def submit(script: str, *, dependency: str | None = None) -> str:
    command = [
        "sbatch",
        "--parsable",
        f"--output={RUN_ROOT / 'environment' / 'slurm'}_{Path(script).stem}_%j.log",
    ]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(REPO / "experiments" / "dual_memory" / script))
    return subprocess.check_output(command, text=True).strip().split(";")[0]


def main() -> None:
    for relative in [
        "environment",
        "configs",
        "audits/coordinate_examples",
        "tests",
        "training/logs",
        "training/curves",
        "training/checkpoints",
        "evaluation/videos",
        "analysis/tables",
        "analysis/figures",
    ]:
        (RUN_ROOT / relative).mkdir(parents=True, exist_ok=True)

    target_manifest = RUN_ROOT / "manifest.preregistered.json"
    if target_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {target_manifest}")
    shutil.copy2(REPO / "experiments/dual_memory/preregistered_manifest.json", target_manifest)

    data_job = submit("run_download_data.sbatch")
    model_job = submit("run_download_models.sbatch")
    jobs = {
        "download_data": data_job,
        "download_models": model_job,
        "unzip_data": submit("run_unzip_data.sbatch", dependency=data_job),
        "unzip_models": submit("run_unzip_models.sbatch", dependency=model_job),
    }
    with (RUN_ROOT / "environment" / "asset_jobs.json").open("x") as stream:
        json.dump(jobs, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(jobs, sort_keys=True))


if __name__ == "__main__":
    main()
