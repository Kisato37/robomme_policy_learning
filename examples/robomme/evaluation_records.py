from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path


def append_jsonl_locked(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class EpisodeResultWriter:
    def __init__(self, run_root: str | Path):
        self.evaluation_dir = Path(run_root) / "evaluation"
        self.results_path = self.evaluation_dir / "per_episode.jsonl"
        self.infra_path = self.evaluation_dir / "infra_failures.jsonl"
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(record: dict) -> tuple:
        return (
            record["model_id"],
            int(record["training_seed"]),
            record["symbolic_source"],
            record["task"],
            int(record["episode_id"]),
        )

    def _existing_keys(self) -> set[tuple]:
        if not self.results_path.exists():
            return set()
        return {
            self.key(json.loads(line))
            for line in self.results_path.read_text().splitlines()
            if line.strip()
        }

    def append_episode(self, record: dict) -> None:
        key = self.key(record)
        if key in self._existing_keys():
            raise RuntimeError(f"Refusing to overwrite or duplicate episode result: {key}")
        append_jsonl_locked(self.results_path, record)

    def append_infra_failure(self, record: dict) -> None:
        append_jsonl_locked(self.infra_path, record)
