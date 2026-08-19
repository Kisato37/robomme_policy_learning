#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite test evidence: {args.output}")
    result = subprocess.run(
        ["bash", "scripts/test_dual_memory.sh"],
        cwd=args.repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    content = (
        "Dual-memory test command: bash scripts/test_dual_memory.sh\n"
        f"Exit code: {result.returncode}\n\n{result.stdout}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
