#!/usr/bin/env python3
"""Run the closest non-manual official regression suite without materializing a 2B model on CPU."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


TESTS = [
    "packages/openpi-client/src/openpi_client/image_tools_test.py",
    "packages/openpi-client/src/openpi_client/msgpack_numpy_test.py",
    "src/openpi/shared/image_tools_test.py",
    "src/openpi/shared/normalize_test.py",
    "src/openpi/transforms_test.py",
    "src/openpi/training/data_loader_test.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite official test evidence: {args.output}")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            str(args.repo / ".venv/bin/python"), "-m", "pytest", "-q",
            *TESTS,
            "-k", "not test_with_real_dataset",
        ],
        cwd=args.repo,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "Official non-manual regression subset (single CPU JAX backend; optional real LeRobot dataset test excluded):\n"
        + "\n".join(TESTS)
        + f"\n\nExit code: {result.returncode}\n\n{result.stdout}"
    )
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
