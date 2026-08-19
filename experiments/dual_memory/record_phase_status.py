#!/usr/bin/env python3
"""Append an experiment phase event and refresh the human-readable run status."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--status", choices=["in_progress", "completed", "failed"], required=True)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--next", default="")
    args = parser.parse_args()
    event = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": args.phase,
        "status": args.status,
        "evidence": args.evidence,
        "next": args.next,
    }
    events_path = args.run_root / "status_events.jsonl"
    args.run_root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(events_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, (json.dumps(event, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    latest = {}
    for item in events:
        latest[item["phase"]] = item
    lines = [
        "# Dual-memory experiment run status",
        "",
        f"Run root: `{args.run_root}`",
        "",
        "| Phase | Status | Last update (UTC) | Evidence |",
        "|---|---|---|---|",
    ]
    for phase in sorted(latest):
        item = latest[phase]
        evidence = "<br>".join(f"`{value}`" for value in item["evidence"]) or "—"
        lines.append(f"| {phase} | {item['status']} | {item['timestamp_utc']} | {evidence} |")
    if event["next"]:
        lines += ["", f"Next: {event['next']}"]
    temporary = args.run_root / "RUN_STATUS.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(args.run_root / "RUN_STATUS.md")


if __name__ == "__main__":
    main()
