#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite Qwen smoke audit: {args.output}")
    call_files = sorted(args.eval_root.rglob("*_QwenVL_calls.jsonl"))
    calls = [
        {**json.loads(line), "call_log": str(path)}
        for path in call_files
        for line in path.read_text().splitlines()
    ]
    videos = sorted(str(path) for path in args.eval_root.rglob("*.mp4"))
    problems = []
    if not calls:
        problems.append("no Qwen calls recorded")
    for index, call in enumerate(calls):
        if not call.get("raw_response"):
            problems.append(f"call {index}: missing raw response")
        if not call.get("parsed_groundsg"):
            problems.append(f"call {index}: missing parsed GroundSG")
        if not call.get("coordinate_valid"):
            problems.append(f"call {index}: missing or out-of-range GroundSG coordinate")
    if not videos:
        problems.append("no rollout video with drawn GroundSG coordinates")
    report = {
        "status": "pass" if not problems else "fail",
        "call_count": len(calls),
        "cache_hits": sum(bool(call.get("cache_hit")) for call in calls),
        "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls if not call.get("cache_hit")),
        "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls if not call.get("cache_hit")),
        "all_coordinates_valid": bool(calls) and all(call.get("coordinate_valid") for call in calls),
        "call_logs": [str(path) for path in call_files],
        "videos_with_drawn_coordinates": videos,
        "calls": calls,
        "problems": problems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if problems:
        raise RuntimeError(f"Qwen smoke audit failed; see {args.output}")


if __name__ == "__main__":
    main()
