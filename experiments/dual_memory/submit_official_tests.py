#!/usr/bin/env python3
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
    report_name = "official_test_report.txt" if args.retry == 0 else f"official_test_report_retry{args.retry}.txt"
    output = RUN_ROOT / f"environment/official_test_job{suffix}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite job record: {output}")
    job_id = subprocess.check_output(
        [
            "sbatch", "--parsable",
            f"--export=ALL,OFFICIAL_TEST_REPORT={report_name}",
            f"--output={RUN_ROOT}/training/logs/official-tests-%j.out",
            f"--error={RUN_ROOT}/training/logs/official-tests-%j.err",
            "experiments/dual_memory/run_official_tests.sbatch",
        ],
        cwd=REPO,
        text=True,
    ).strip().split(";")[0]
    with output.open("x") as stream:
        json.dump(
            {
                "official_tests": job_id,
                "report": f"tests/{report_name}",
                "supersedes": args.supersedes,
                "retry_reason": (
                    "force single-CPU JAX; exclude optional real LeRobot dataset test"
                    if args.retry else None
                ),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(job_id)


if __name__ == "__main__":
    main()
