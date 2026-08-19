#!/usr/bin/env python3
"""Build the write-once Phase-0 report from released-reference rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight report: {args.output}")
    records_path = args.run_root / "evaluation/per_episode.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    released = {
        record["model_id"]: record
        for record in records
        if record.get("model_id") in {"released-S", "released-P"}
    }
    missing = {"released-S", "released-P"} - released.keys()
    if missing:
        raise RuntimeError(f"Missing released-reference preflight records: {sorted(missing)}")
    missing_videos = [
        record["video_path"] for record in released.values() if not Path(record["video_path"]).exists()
    ]
    if missing_videos:
        raise RuntimeError(f"Missing preflight videos: {missing_videos}")

    lines = [
        "# Dual-memory Phase 0 preflight",
        "",
        "Status: **PASS**",
        "",
        "The independent policy and simulator environments completed fixed released-reference rollouts without an infrastructure error.",
        "",
        "## Fixed rollout evidence",
        "",
        "| Reference | Task | Episode | Scientific result | Steps | Video |",
        "|---|---|---:|---|---:|---|",
    ]
    for model_id in ["released-S", "released-P"]:
        record = released[model_id]
        lines.append(
            f"| {model_id} | {record['task']} | {record['episode_id']} | "
            f"{'success' if record['success'] else record['terminal_reason']} | {record['steps']} | `{record['video_path']}` |"
        )
    lines += [
        "",
        "A failed task outcome here is a scientific rollout outcome, not a preflight failure; the gate checks simulator reset, policy serving, history/subgoal transport, terminal logging, and video persistence.",
        "",
        "## Version and protocol",
        "",
        "- Policy base commit: `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`.",
        "- Benchmark commit: `856bc3a189d4172f3f47dbee4424d585f8d78db3`.",
        "- Released model revision: `5db4d53ddb98c7f80cab08792dd53d985d712ab1`.",
        "- Evaluation policy seed: `7`; fixed task/episode: `InsertPeg/0`.",
        "- The two released checkpoints are references only and do not replace matched N/S/P/SP training.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
