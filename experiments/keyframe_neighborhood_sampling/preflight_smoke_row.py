#!/usr/bin/env python3
"""Fail-closed runtime binding for one OC3/OC5 smoke-array row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.keyframe_neighborhood_sampling.formal_artifacts import validate_prepared_smoke_root
from experiments.keyframe_neighborhood_sampling.smoke_matrix import validate_runtime_row_binding
from experiments.keyframe_oracle_sampling.environment_contract import validate_environment_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed-table", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--attempt-id", type=int, required=True)
    parser.add_argument("--row-id", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--trajectory-kind", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    manifest = validate_prepared_smoke_root(
        args.run_root,
        args.seed_table,
        args.repo_root,
        attempt_id=args.attempt_id,
    )
    validate_environment_contract(manifest)
    row = validate_runtime_row_binding(
        args.run_root,
        attempt_id=args.attempt_id,
        row_id=args.row_id,
        task=args.task,
        episode_id=args.episode_id,
        arm=args.arm,
        trajectory_kind=args.trajectory_kind,
        max_steps=args.max_steps,
        dataset=args.dataset,
    )
    print(json.dumps({"validated": True, "row_id": args.row_id, "row": row}, sort_keys=True))


if __name__ == "__main__":
    main()
