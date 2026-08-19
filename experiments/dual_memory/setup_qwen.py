#!/usr/bin/env python3
"""Download and hash the pinned Qwen base model after dependencies are installed."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

from experiments.dual_memory.download_assets import inventory


QWEN_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--qwen-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite Qwen setup manifest: {args.output}")
    target = args.repo / "runs/ckpts/vlm_subgoal_predictor/qwenvl/Qwen3-VL-4B-Instruct"
    snapshot_download(
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        revision=QWEN_REVISION,
        local_dir=target,
        max_workers=4,
    )
    packages = subprocess.check_output(
        [str(args.qwen_python), "-m", "pip", "freeze", "--all"], text=True
    ).splitlines()
    result = {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": QWEN_REVISION,
        "attention_backend": "sdpa",
        "attention_backend_reason": "official documented fallback; cluster has no nvcc for optional flash-attn",
        "environment_python": str(args.qwen_python.resolve()),
        "packages": packages,
        **inventory(target, hash_files=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
