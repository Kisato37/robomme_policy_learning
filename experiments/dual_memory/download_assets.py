#!/usr/bin/env python3
"""Download only the pinned public assets required by the dual-memory study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REVISIONS = {
    "data": "ddf0baf55b633cc6657dcd53ac0e089a273de612",
    "mme_vla_suite": "5db4d53ddb98c7f80cab08792dd53d985d712ab1",
    "pi05_baseline": "80caa59a4933804b6521e72981b107ae0051d32a",
    "vlm_subgoal_predictor": "a243356b18a33a8b812be06952e49d239e8e614d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, *, hash_files: bool) -> dict:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache/huggingface" in path.as_posix():
            continue
        item = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
        if hash_files:
            item["sha256"] = sha256_file(path)
        files.append(item)
    return {"root": str(root.resolve()), "files": files, "total_bytes": sum(i["bytes"] for i in files)}


def download_data(repo: Path) -> dict:
    target = repo / "data" / "robomme_preprocessed_data"
    snapshot_download(
        repo_id="Yinpei/robomme_preprocessed_data",
        repo_type="dataset",
        revision=REVISIONS["data"],
        local_dir=target,
        max_workers=16,
    )
    return {
        "repo_id": "Yinpei/robomme_preprocessed_data",
        "revision": REVISIONS["data"],
        **inventory(target, hash_files=True),
    }


def download_models(repo: Path) -> dict:
    targets = {}
    mme_target = repo / "runs" / "ckpts" / "mme_vla_suite"
    snapshot_download(
        repo_id="Yinpei/mme_vla_suite",
        revision=REVISIONS["mme_vla_suite"],
        local_dir=mme_target,
        allow_patterns=[
            "README.md",
            "symbolic-grounded-subgoal/**",
            "perceptual-framesamp-modul/**",
        ],
        max_workers=8,
    )
    targets["mme_vla_suite"] = {
        "repo_id": "Yinpei/mme_vla_suite",
        "revision": REVISIONS["mme_vla_suite"],
        **inventory(mme_target, hash_files=True),
    }

    baseline_target = repo / "runs" / "ckpts" / "pi05_baseline"
    snapshot_download(
        repo_id="Yinpei/pi05_baseline",
        revision=REVISIONS["pi05_baseline"],
        local_dir=baseline_target,
        max_workers=4,
    )
    targets["pi05_baseline"] = {
        "repo_id": "Yinpei/pi05_baseline",
        "revision": REVISIONS["pi05_baseline"],
        **inventory(baseline_target, hash_files=True),
    }

    qwen_target = repo / "runs" / "ckpts" / "vlm_subgoal_predictor"
    snapshot_download(
        repo_id="Yinpei/vlm_subgoal_predictor",
        revision=REVISIONS["vlm_subgoal_predictor"],
        local_dir=qwen_target,
        allow_patterns=[
            "README.md",
            "qwenvl/grounded_subgoal/checkpoint-1200.zip",
            "qwenvl/grounded_subgoal/logging.jsonl",
        ],
        max_workers=4,
    )
    targets["vlm_subgoal_predictor"] = {
        "repo_id": "Yinpei/vlm_subgoal_predictor",
        "revision": REVISIONS["vlm_subgoal_predictor"],
        **inventory(qwen_target, hash_files=True),
    }

    from openpi.shared import download

    base_path = Path(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"))
    targets["pi05_base"] = inventory(base_path, hash_files=True)
    targets["pi05_base"]["source"] = "gs://openpi-assets/checkpoints/pi05_base"
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["data", "models"])
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite asset manifest: {args.output}")
    result = download_data(args.repo) if args.group == "data" else download_models(args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
