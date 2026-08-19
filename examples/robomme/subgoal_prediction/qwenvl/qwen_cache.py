from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        digest.update(str(file_path.stat().st_size).encode())
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class ContentAddressedQwenCache:
    """Exact-input cache; it never performs similarity or trajectory-level reuse."""

    def __init__(
        self,
        root: str | Path,
        predictor_checkpoint: str | Path,
        base_model_checkpoint: str | Path | None = None,
    ):
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.predictor_checkpoint = str(Path(predictor_checkpoint).resolve())
        self.predictor_checkpoint_hash = hash_path(Path(predictor_checkpoint))
        self.base_model_checkpoint = (
            str(Path(base_model_checkpoint).resolve())
            if base_model_checkpoint is not None
            else None
        )
        self.base_model_checkpoint_hash = (
            hash_path(Path(base_model_checkpoint))
            if base_model_checkpoint is not None
            else None
        )

    def key(
        self,
        *,
        current_image: np.ndarray,
        task_instruction: str,
        subgoal_history: list[str],
        prompt_version: str,
        generation: dict,
        video_hash: str | None,
    ) -> tuple[str, dict]:
        payload = {
            "current_front_image_sha256": hash_array(current_image),
            "task_instruction": task_instruction,
            "subgoal_history": subgoal_history,
            "prompt_version": prompt_version,
            "predictor_checkpoint": self.predictor_checkpoint,
            "predictor_checkpoint_sha256": self.predictor_checkpoint_hash,
            "base_model_checkpoint": self.base_model_checkpoint,
            "base_model_checkpoint_sha256": self.base_model_checkpoint_hash,
            "generation": generation,
            "video_sha256": video_hash,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256_bytes(encoded), payload

    def get(self, key: str) -> dict | None:
        path = self.objects / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, key: str, value: dict) -> None:
        path = self.objects / f"{key}.json"
        content = json.dumps(value, indent=2, sort_keys=True) + "\n"
        try:
            with path.open("x") as stream:
                stream.write(content)
        except FileExistsError:
            if path.read_text() != content:
                raise RuntimeError(f"Cache-key collision or non-deterministic response: {key}")


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, line.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
