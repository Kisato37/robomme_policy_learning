#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


MODELS = ["pilot-N", "pilot-S", "pilot-P", "pilot-SP"]
TASKS = ["VideoUnmask", "VideoRepick", "InsertPeg", "VideoPlaceOrder", "RouteStick"]
EPISODES = list(range(5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    json_output = args.run_root / "audits/pilot_report.json"
    markdown_output = args.run_root / "audits/pilot_report.md"
    if json_output.exists() or markdown_output.exists():
        raise FileExistsError("Refusing to overwrite pilot evidence")
    records = [
        json.loads(line)
        for line in (args.run_root / "evaluation/per_episode.jsonl").read_text().splitlines()
    ]
    pilot = [record for record in records if record.get("model_id") in MODELS]
    failures_path = args.run_root / "evaluation/infra_failures.jsonl"
    failures = (
        [json.loads(line) for line in failures_path.read_text().splitlines()]
        if failures_path.exists()
        else []
    )
    pilot_failures = [failure for failure in failures if failure.get("model_id") in MODELS]
    problems = []
    expected = {(model, task, episode) for model in MODELS for task in TASKS for episode in EPISODES}
    observed = {(record["model_id"], record["task"], record["episode_id"]) for record in pilot}
    if expected != observed:
        problems.append(
            f"paired episode mismatch: missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )
    if pilot_failures:
        problems.append(f"documented infrastructure failures: {len(pilot_failures)}")
    for record in pilot:
        lengths = record.get("history_lengths_at_policy_calls", [])
        if record["model_id"] in {"pilot-P", "pilot-SP"}:
            if not lengths or max(lengths) <= 1 or any(b < a for a, b in zip(lengths, lengths[1:])):
                problems.append(
                    f"{record['model_id']}/{record['task']}/{record['episode_id']}: history is not cumulative"
                )
        if record["model_id"] in {"pilot-S", "pilot-SP"}:
            subgoals = record.get("subgoal_sequence", [])
            if not subgoals or any(not value for value in subgoals):
                problems.append(
                    f"{record['model_id']}/{record['task']}/{record['episode_id']}: Oracle GroundSG missing"
                )
        if not Path(record["video_path"]).exists():
            problems.append(f"missing video: {record['video_path']}")

    success = Counter(record["model_id"] for record in pilot if record["success"])
    totals = Counter(record["model_id"] for record in pilot)
    report = {
        "status": "pass" if not problems else "fail",
        "purpose": "engineering gate only; not a scientific conclusion",
        "training_prefix_steps": 100,
        "training_seed": 42,
        "evaluation_policy_seed": 7,
        "tasks": TASKS,
        "episode_ids": EPISODES,
        "record_count": len(pilot),
        "success_counts_diagnostic_only": {
            model: {"success": success[model], "total": totals[model]} for model in MODELS
        },
        "infra_failures": pilot_failures,
        "problems": problems,
    }
    with json_output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    lines = [
        "# Dual-memory pilot engineering gate",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        "This 100-step, one-seed pilot verifies closed-loop wiring only; its success counts are not used as scientific evidence.",
        "",
        "| Model | Success / 25 (diagnostic only) |",
        "|---|---:|",
        *[f"| {model} | {success[model]} / {totals[model]} |" for model in MODELS],
        "",
        f"Problems: {problems if problems else 'none'}",
    ]
    with markdown_output.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    if problems:
        raise RuntimeError(f"Pilot gate failed; see {json_output}")
    subprocess.check_call(
        [
            sys.executable,
            str(Path(__file__).with_name("record_phase_status.py")),
            "--run-root", str(args.run_root),
            "--phase", "Phase-4", "--status", "completed",
            "--evidence", "audits/pilot_report.json",
            "--evidence", "audits/pilot_report.md",
            "--next", "Freeze the environment and launch formal training.",
        ]
    )


if __name__ == "__main__":
    main()
