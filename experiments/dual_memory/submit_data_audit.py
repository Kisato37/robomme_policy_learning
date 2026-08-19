#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


REPO = Path("/home/jp673/robomme_dual_memory/robomme_policy_learning")
RUN_ROOT = REPO / "outputs/dual_memory/20260818-214308_ecf086c"


def main() -> None:
    jobs = json.loads((RUN_ROOT / "environment/asset_jobs.json").read_text())
    command = [
        "sbatch",
        "--parsable",
        f"--dependency=afterok:{jobs['unzip_data']}",
        f"--output={RUN_ROOT / 'audits' / 'data_audit_%j.log'}",
        str(REPO / "experiments/dual_memory/run_data_audit.sbatch"),
    ]
    job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
    with (RUN_ROOT / "environment/audit_jobs.json").open("x") as stream:
        json.dump({"data_audit": job_id}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(job_id)


if __name__ == "__main__":
    main()
