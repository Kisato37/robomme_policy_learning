#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def command(args, cwd=None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite environment evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    environment = args.run_root / "environment"
    policy_commit = command(["git", "rev-parse", "HEAD"], args.repo).strip()
    benchmark = args.repo / "third_party/robomme_benchmark"
    benchmark_commit = command(["git", "rev-parse", "HEAD"], benchmark).strip()
    # Runtime data/checkpoint/result roots are deliberately untracked.  Freeze
    # requires every tracked source file to match the experiment commit.
    status = command(["git", "status", "--short", "--untracked-files=no"], args.repo)
    if status.strip():
        raise RuntimeError(f"Formal launch requires a clean committed worktree:\n{status}")
    write_once(
        environment / "git_commit.txt",
        f"policy_experiment_commit={policy_commit}\nbenchmark_commit={benchmark_commit}\nbranch="
        + command(["git", "branch", "--show-current"], args.repo).strip()
        + f"\nworktree_status_at_freeze=\n{status}",
    )
    package_text = "\n".join(
        [
            f"python={sys.version}",
            f"platform={platform.platform()}",
            f"uv_lock_sha256={sha256(args.repo / 'uv.lock')}",
            f"benchmark_uv_lock_sha256={sha256(benchmark / 'uv.lock')}",
            "",
            "[policy pip freeze]",
            command([str(args.repo / ".venv/bin/python"), "-m", "pip", "freeze", "--all"]),
            "[benchmark pip freeze]",
            command([str(benchmark / ".venv/bin/python"), "-m", "pip", "freeze", "--all"]),
            "[qwen benchmark pip freeze]",
            command([str(benchmark / ".venv_qwen/bin/python"), "-m", "pip", "freeze", "--all"]),
        ]
    )
    write_once(environment / "package_lock.txt", package_text)
    cuda = command(["nvidia-smi", "-q"]) + "\n" + command([str(args.repo / ".venv/bin/python"), "-c", "import jax; print(jax.__version__); print(jax.devices())"])
    write_once(environment / "cuda_driver.txt", cuda)
    scheduler = command(["scontrol", "show", "partition"]) + "\n" + command(["sinfo", "-N", "-o", "%P|%N|%t|%G|%c|%m"])
    write_once(environment / "scheduler.txt", scheduler)
    summary = {
        "policy_experiment_commit": policy_commit,
        "benchmark_commit": benchmark_commit,
        "worktree_clean": not status.strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "uv_lock_sha256": sha256(args.repo / "uv.lock"),
        "benchmark_uv_lock_sha256": sha256(benchmark / "uv.lock"),
    }
    write_once(environment / "environment_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
