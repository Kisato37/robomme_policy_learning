"""Write-once episode artifacts for the independent 48-slot experiment.

The store reserves attempts before environment setup, never resumes an episode
mid-state, and never turns scientific failures into retryable infrastructure
failures.  It records evidence and validates its internal hashes; hashes alone
cannot establish the truth of external checkpoint/environment attestations.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from experiments.keyframe_oracle_sampling.artifacts import (
    SMOKE_INITIAL_CONDITION_HASH_FIELDS, atomic_write_bytes, atomic_write_json,
    append_jsonl_locked, sha256_file, utc_now,
)
from experiments.uniform_keyframe_expansion import contract as c
from experiments.uniform_keyframe_expansion.trace_validation import validate_selector_trace, validate_trace_sequence


class ExpansionArtifactError(ValueError):
    pass


def _equal(actual: Any, expected: Any, label: str) -> None:
    if c.canonical_json(actual) != c.canonical_json(expected):
        raise ExpansionArtifactError(f"{label} differs from its frozen/bound value")


def _digest(value: Any, field: str, digits: int = 64) -> str:
    if not isinstance(value, str) or len(value) != digits or any(ch not in "0123456789abcdef" for ch in value):
        raise ExpansionArtifactError(f"{field} must be a lowercase {digits}-digit digest")
    return value


def _int(value: Any, field: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ExpansionArtifactError(f"Invalid nonnegative integer {field}")
    return value


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExpansionArtifactError(f"Missing or symlinked artifact: {path}")
    try:
        payload = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise ExpansionArtifactError(f"Unreadable artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ExpansionArtifactError(f"Artifact must be a mapping: {path}")
    c.canonical_json(payload)
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    # Reject nonfinite values before the parent's generic NumPy log conversion.
    c.canonical_json(dict(payload))
    atomic_write_json(path, dict(payload))


def _provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExpansionArtifactError("Run provenance must be a mapping")
    value = dict(value)
    c.canonical_json(value)
    for key in ("code_commit", "benchmark_commit"):
        _digest(value.get(key), key, 40)
    for key in ("protocol_sha256", "environment_manifest_sha256"):
        _digest(value.get(key), key)
    _equal(value.get("checkpoint_archive_sha256"), c.CHECKPOINT_ARCHIVE_SHA256, "checkpoint archive")
    if not isinstance(value.get("host"), str) or not value["host"].strip():
        raise ExpansionArtifactError("Run host is required")
    if not isinstance(value.get("hardware"), Mapping) or not value["hardware"]:
        raise ExpansionArtifactError("Run hardware evidence is required")
    command = value.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(part, str) or not part for part in command):
        raise ExpansionArtifactError("Run command must be a nonempty string list")
    return value


def _safe_root(path: str | Path) -> Path:
    root = Path(path).absolute()
    if root.parent.name != "uniform_keyframe_expansion" or root.name in ("", ".", ".."):
        raise ExpansionArtifactError("Run root must be <parent>/uniform_keyframe_expansion/<unique_run_id>")
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise ExpansionArtifactError("Run root cannot traverse a symlink")
    return root


class ExpansionRunStore:
    """One run family/stage, with independently locked append-only row attempts."""

    def __init__(self, root: Path, manifest: dict[str, Any]):
        self.run_root = root
        self.manifest = manifest
        self.stage = manifest["stage"]

    @classmethod
    def create(cls, run_root: str | Path, *, stage: str, run_manifest: Mapping[str, Any]):
        root = _safe_root(run_root)
        provenance = _provenance(run_manifest)
        matrix = c.build_formal_matrix() if stage == "formal" else c.build_smoke_matrix() if stage == "smoke" else None
        if matrix is None:
            raise ExpansionArtifactError("Stage must be formal or smoke")
        root.mkdir(parents=True, exist_ok=False)
        (root / "locks").mkdir()
        (root / "trajectories").mkdir()
        snapshots = {
            "protocol/matrix.json": matrix,
            "protocol/seed_manifest.json": c.build_seed_manifest(stage),
            "protocol/inference_contract.json": c.frozen_inference_contract(),
        }
        hashes = {}
        for name, value in snapshots.items():
            _write(root / name, value)
            hashes[name] = sha256_file(root / name)
        manifest = {"schema_version": 1, "protocol_family": c.PROTOCOL_FAMILY,
                    "stage": stage, "run_id": root.name, "created_utc": utc_now(),
                    "provenance": provenance, "snapshot_sha256": hashes,
                    "creation_scope": "artifact_preparation_not_launch_authorization"}
        _write(root / "run_manifest.json", manifest)
        return cls.open(root)

    @classmethod
    def open(cls, run_root: str | Path):
        root = _safe_root(run_root)
        manifest = _json(root / "run_manifest.json")
        _equal(manifest.get("protocol_family"), c.PROTOCOL_FAMILY, "run family")
        _equal(manifest.get("schema_version"), 1, "run schema")
        _equal(manifest.get("run_id"), root.name, "run identity")
        stage = manifest.get("stage")
        if stage not in ("smoke", "formal"):
            raise ExpansionArtifactError("Invalid recorded stage")
        _provenance(manifest.get("provenance"))
        expected_paths = {"protocol/matrix.json", "protocol/seed_manifest.json", "protocol/inference_contract.json"}
        hashes = manifest.get("snapshot_sha256")
        if not isinstance(hashes, Mapping) or set(hashes) != expected_paths:
            raise ExpansionArtifactError("Run snapshots are incomplete")
        for name, digest in hashes.items():
            _digest(digest, name)
            _json(root / name)
            _equal(sha256_file(root / name), digest, f"snapshot {name}")
        c.validate_matrix(_json(root / "protocol/matrix.json"), stage)
        c.validate_inference_contract(_json(root / "protocol/inference_contract.json"))
        _equal(_json(root / "protocol/seed_manifest.json"), c.build_seed_manifest(stage), "seed manifest")
        return cls(root, manifest)

    def _row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return c.validate_row(row, self.stage)

    def attempt_dir(self, row: Mapping[str, Any], attempt_id: int) -> Path:
        row = self._row(row)
        _int(attempt_id, "attempt_id", 2)
        return (self.run_root / "trajectories" / f"row_{row['row_id']:04d}" /
                f"attempt_{attempt_id:02d}")

    @contextmanager
    def _lock(self, row: Mapping[str, Any]):
        row = self._row(row)
        lock = self.run_root / "locks" / f"row_{row['row_id']:04d}.lock"
        if lock.is_symlink():
            raise ExpansionArtifactError("Row lock cannot be a symlink")
        with lock.open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def new_attempt(self, row: Mapping[str, Any], attempt_id: int,
                    episode_provenance: Mapping[str, Any] | None = None):
        """Reserve BEFORE setup; failures before initial observation stay recordable."""
        row = self._row(row)
        path = self.attempt_dir(row, attempt_id)
        planned = {} if episode_provenance is None else dict(episode_provenance)
        c.canonical_json(planned)
        with self._lock(row):
            # Re-open immutable run evidence, including source configuration.
            self.open(self.run_root)
            self._reject_run_hard_stop()
            for prior in range(attempt_id):
                audit = self.audit_attempt(row, prior)
                if audit["status"] != "infrastructure_failure" or audit["retry_allowed"] is not True:
                    raise ExpansionArtifactError("Retry requires every earlier attempt to be a retryable infrastructure failure")
            row_dir = path.parent
            if row_dir.exists():
                existing = sorted(item.name for item in row_dir.iterdir())
                expected = [f"attempt_{number:02d}" for number in range(attempt_id)]
                if existing != expected:
                    raise ExpansionArtifactError("Cannot reuse, skip, or replace an existing row attempt")
            path.mkdir(parents=True, exist_ok=False)
            (path / "traces").mkdir()
            (path / "attachments").mkdir()
            (path / ".video_staging").mkdir()
            manifest = {"schema_version": 1, "protocol_family": c.PROTOCOL_FAMILY,
                        "run_id": self.run_root.name, "row": row, "attempt_id": attempt_id,
                        "run_manifest_sha256": sha256_file(self.run_root / "run_manifest.json"),
                        "selector_config": c.build_selector_config(row),
                        "inference_contract": c.frozen_inference_contract(),
                        "episode_provenance": planned, "created_utc": utc_now(),
                        "fresh_reset_required": True, "mid_episode_resume_allowed": False}
            _write(path / "episode_manifest.json", manifest)
        return ExpansionEpisodeWriter(self, row, attempt_id, owns_active_attempt=True)

    def _reject_run_hard_stop(self) -> None:
        """A protocol hard stop blocks new rows across the whole run."""
        for path in self.run_root.glob("trajectories/row_*/attempt_*/failure.json"):
            payload = _json(path)
            row = self._row(payload.get("row"))
            attempt_id = _int(payload.get("attempt_id"), "attempt_id", 2)
            if self.attempt_dir(row, attempt_id) / "failure.json" != path:
                raise ExpansionArtifactError("Failure closure path does not match its row/attempt")
            writer = ExpansionEpisodeWriter(self, row, attempt_id)
            # Verify closure and ledger binding before interpreting the kind.
            writer._manifest()
            writer._validate_ledger("failures/failure_ledger.jsonl", "failure_path", "failure_sha256", path)
            if payload.get("kind") == "hard_stop":
                raise ExpansionArtifactError("Run-wide protocol hard stop blocks every new attempt")

    def audit_attempt(self, row: Mapping[str, Any], attempt_id: int) -> dict[str, Any]:
        self.open(self.run_root)
        return ExpansionEpisodeWriter(self, self._row(row), attempt_id)._audit()

    def completed_rows(self) -> dict[int, Path]:
        completed = {}
        matrix = _json(self.run_root / "protocol/matrix.json")
        for row in matrix["rows"]:
            row_dir = self.attempt_dir(row, 0).parent
            if not row_dir.exists():
                continue
            names = sorted(item.name for item in row_dir.iterdir())
            if names != [f"attempt_{i:02d}" for i in range(len(names))] or len(names) > 3:
                raise ExpansionArtifactError("Unexpected/noncontiguous attempt paths")
            for attempt_id in range(len(names)):
                audit = self.audit_attempt(row, attempt_id)
                if audit["status"] == "complete":
                    if row["row_id"] in completed or attempt_id != len(names) - 1:
                        raise ExpansionArtifactError("Duplicate/retried completed scientific outcome")
                    completed[row["row_id"]] = self.attempt_dir(row, attempt_id) / "episode_result.json"
                elif attempt_id < len(names) - 1 and not audit.get("retry_allowed", False):
                    raise ExpansionArtifactError("Attempt was retried without an eligible infrastructure failure")
        expected_dirs = {f"row_{row['row_id']:04d}" for row in matrix["rows"]}
        unexpected = {item.name for item in (self.run_root / "trajectories").iterdir()} - expected_dirs
        if unexpected:
            raise ExpansionArtifactError("Unexpected trajectory row directories")
        return completed

    def completeness(self) -> dict[str, Any]:
        completed = self.completed_rows()
        rows = _json(self.run_root / "protocol/matrix.json")["rows"]
        missing = [row["row_id"] for row in rows if row["row_id"] not in completed]
        return {"protocol_family": c.PROTOCOL_FAMILY, "stage": self.stage,
                "expected_count": len(rows), "completed_count": len(completed),
                "missing_row_ids": missing, "complete": not missing}


class ExpansionEpisodeWriter:
    def __init__(self, store: ExpansionRunStore, row: Mapping[str, Any], attempt_id: int, *, owns_active_attempt: bool = False):
        self.store, self.row, self.attempt_id = store, store._row(row), _int(attempt_id, "attempt_id", 2)
        self.attempt_dir = store.attempt_dir(row, attempt_id)
        self.manifest_path = self.attempt_dir / "episode_manifest.json"
        self.initial_conditions_path = self.attempt_dir / "initial_condition_hashes.json"
        self.trace_path = self.attempt_dir / "selector_trace.jsonl"
        self.result_path = self.attempt_dir / "episode_result.json"
        self.failure_path = self.attempt_dir / "failure.json"
        self._owns_active_attempt = owns_active_attempt
        self._live_initial_cache: dict[str, Any] | None = None
        self._live_trace_cache: list[dict[str, Any]] = []

    def _manifest(self) -> dict[str, Any]:
        manifest = _json(self.manifest_path)
        for field, value in (("schema_version", 1), ("protocol_family", c.PROTOCOL_FAMILY),
                             ("run_id", self.store.run_root.name), ("row", self.row),
                             ("attempt_id", self.attempt_id), ("fresh_reset_required", True),
                             ("mid_episode_resume_allowed", False)):
            _equal(manifest.get(field), value, f"attempt {field}")
        _equal(manifest.get("run_manifest_sha256"), sha256_file(self.store.run_root / "run_manifest.json"), "run manifest hash")
        c.validate_selector_config(manifest.get("selector_config"), self.row)
        c.validate_inference_contract(manifest.get("inference_contract"))
        return manifest

    def _mutable(self) -> None:
        if not self._owns_active_attempt:
            raise ExpansionArtifactError("Only the fresh attempt owner may write; mid-episode resume is forbidden")
        self._manifest()
        if self.result_path.exists() or self.failure_path.exists():
            raise ExpansionArtifactError("Closed attempts are immutable")

    def _initial(self) -> dict[str, Any]:
        payload = _json(self.initial_conditions_path)
        _validate_initial_payload(self, payload)
        return payload

    def write_attachment(self, name: str, data: bytes) -> str:
        """Publish raw evidence once; no path traversal, implicit text conversion or replacement."""
        if (not isinstance(name, str) or not name or name in (".", "..") or name.startswith(".")
                or "/" in name or "\\" in name or Path(name).name != name):
            raise ExpansionArtifactError("Attachment name must be one plain filename")
        if not isinstance(data, bytes):
            raise ExpansionArtifactError("Attachment content must be bytes")
        with self.store._lock(self.row):
            self._mutable()
            directory = self.attempt_dir / "attachments"
            if directory.is_symlink():
                raise ExpansionArtifactError("Attachment directory cannot be a symlink")
            path = directory / name
            atomic_write_bytes(path, data)
            return sha256_file(path)

    def publish_video(self, staged_path: str | Path) -> str:
        """Publish this attempt's own staged video atomically, without copying it.

        Only the exact reserved temporary source below may be unlinked after
        successful publication. It is not an API for moving arbitrary user files.
        """
        expected = self.attempt_dir / ".video_staging" / "rollout.mp4"
        source = Path(staged_path).absolute()
        if source != expected or source.is_symlink() or source.parent.is_symlink() or not source.is_file():
            raise ExpansionArtifactError("Video source must be this attempt's exact private staging file")
        with self.store._lock(self.row):
            self._mutable()
            with source.open("rb") as stream:
                os.fsync(stream.fileno())
            target = self.attempt_dir / "rollout.mp4"
            os.link(source, target)  # Atomic EEXIST, never replace an existing video.
            descriptor = os.open(self.attempt_dir, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            source.unlink()  # Remove only our successfully published temporary link.
            descriptor = os.open(source.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return sha256_file(target)

    def record_initial_conditions(self, hashes: Mapping[str, str], *,
                                  environment_provenance: Mapping[str, Any],
                                  reset_evidence: Mapping[str, Any]) -> None:
        with self.store._lock(self.row):
            self._mutable()
            if any((self.attempt_dir / "traces").iterdir()):
                raise ExpansionArtifactError("Initial conditions must precede inference traces")
            payload = {"hashes": dict(hashes), "environment_provenance": dict(environment_provenance),
                       "reset_evidence": dict(reset_evidence),
                       "raw_initial_attachment_sha256": self._initial_attachment_hashes(),
                       "episode_manifest_sha256": sha256_file(self.manifest_path), "recorded_utc": utc_now()}
            # Validate without publishing an invalid write-once artifact.
            self._validate_initial_payload(payload)
            _write(self.initial_conditions_path, payload)
            self._live_initial_cache = json.loads(c.canonical_json(payload))

    def _validate_initial_payload(self, payload: dict[str, Any]) -> None:
        # Same validator as reads, but the file has not yet been published.
        _validate_initial_payload(self, payload, populate_derived=True)

    def _initial_attachment_hashes(self) -> dict[str, str]:
        result = {}
        for name in ("initial_observations.npz", "initial_task_state.json", "initial_task_instruction.json"):
            path = self.attempt_dir / "attachments" / name
            if path.is_symlink() or not path.is_file():
                raise ExpansionArtifactError(f"Missing raw initial evidence: {name}")
            result[name] = sha256_file(path)
        return result

    def _all_files(self) -> list[Path]:
        paths = sorted(self.attempt_dir.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise ExpansionArtifactError("Attempt artifacts cannot contain symlinks")
        return [path for path in paths if path.is_file()]

    def _traces(self, initial_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        directory = self.attempt_dir / "traces"
        paths = sorted(directory.iterdir())
        if [path.name for path in paths] != [f"call_{index:03d}.json" for index in range(len(paths))]:
            raise ExpansionArtifactError("Trace calls are missing, extra or noncontiguous")
        traces = [_json(path) for path in paths]
        if traces:
            validate_trace_sequence(traces, require_final_memory=True, expected_call_count=len(traces))
            evidence = initial_payload if initial_payload is not None else self._initial()
            prefix_count = evidence["reset_evidence"]["reset_prefix_frame_count"]
            _equal(traces[0]["visible_boundary_indices"], evidence["reset_prefix_boundary_indices"], "raw reset-prefix stage boundaries")
            for call, trace in enumerate(traces):
                for key, expected in (("arm", self.row["arm"]), ("task", self.row["task"]),
                                      ("episode_id", self.row["episode_id"]), ("split", self.row["dataset"]),
                                      ("environment_step", 16 * call), ("history_length", prefix_count + 16 * call)):
                    _equal(trace.get(key), expected, f"episode trace {key}")
        return traces

    def append_trace(self, trace: Mapping[str, Any]) -> None:
        with self.store._lock(self.row):
            self._mutable()
            if self._live_initial_cache is None:
                raise ExpansionArtifactError("The live owner must record verified initial conditions before inference")
            prefix_count = self._live_initial_cache["reset_evidence"]["reset_prefix_frame_count"]
            previous = self._live_trace_cache
            new = dict(trace)
            validate_selector_trace(new, require_final_memory=True)
            _equal(new.get("policy_call_index"), len(previous), "contiguous policy call")
            for key, value in (("arm", self.row["arm"]), ("task", self.row["task"]),
                               ("episode_id", self.row["episode_id"]), ("split", self.row["dataset"]),
                               ("environment_step", 16 * len(previous)),
                               ("history_length", prefix_count + 16 * len(previous))):
                _equal(new.get(key), value, f"new trace {key}")
            if previous:
                old = previous[-1]
                old_boundaries = [index for index in new["visible_boundary_indices"] if index <= old["step_idx"]]
                _equal(old_boundaries, old["visible_boundary_indices"], "immutable historical boundary prefix")
            else:
                _equal(new["visible_boundary_indices"], self._live_initial_cache["reset_prefix_boundary_indices"], "raw reset-prefix stage boundaries")
            _write(self.attempt_dir / "traces" / f"call_{len(previous):03d}.json", new)
            self._live_trace_cache.append(json.loads(c.canonical_json(new)))

    def _validate_result(self, result: Mapping[str, Any], traces: list[dict[str, Any]]) -> None:
        reason = result.get("terminal_reason")
        if reason not in ("success", "fail", "timeout", "error", "short_limit"):
            raise ExpansionArtifactError("Invalid terminal reason")
        _equal(result.get("success"), reason == "success", "binary scientific success")
        steps = _int(result.get("environment_steps"), "environment_steps", self.row["max_steps"])
        calls = _int(result.get("policy_call_count"), "policy_call_count", c.MAX_POLICY_CALLS)
        if calls != len(traces) or calls != math.ceil(steps / 16):
            raise ExpansionArtifactError("Result count/step evidence disagrees with complete action-chunk traces")
        _equal(result.get("official_terminal"), reason != "short_limit", "official terminal evidence")
        if reason == "short_limit" and (self.row["trajectory_kind"] != "short" or steps != 64):
            raise ExpansionArtifactError("Only a 64-step development-short trajectory may stop at short_limit")
        metadata = result.get("terminal_metadata")
        if reason == "timeout" and steps != self.row["max_steps"] and (
                not isinstance(metadata, Mapping) or metadata.get("official_stop_flag") is not True):
            raise ExpansionArtifactError("Early timeout needs actual official-stop evidence")
        if reason == "error" and (not result.get("benchmark_error_message") or not result.get("benchmark_exception_type")):
            raise ExpansionArtifactError("Scientific benchmark error requires wrapper exception evidence")
        if not isinstance(result.get("terminal_metadata"), Mapping) or not result["terminal_metadata"]:
            raise ExpansionArtifactError("Actual terminal metadata must be preserved")
        _equal(result.get("reset_verified"), True, "result reset evidence")
        if "video_filename" in result or "video_sha256" in result:
            _equal(result.get("video_filename"), "rollout.mp4", "video filename")
            video = self.attempt_dir / "rollout.mp4"
            if video.is_symlink() or not video.is_file():
                raise ExpansionArtifactError("Episode recording is absent or symlinked")
            _equal(result.get("video_sha256"), sha256_file(video), "video checksum")
        c.canonical_json(dict(result))

    def finalize_scientific(self, result: Mapping[str, Any]) -> dict[str, Any]:
        with self.store._lock(self.row):
            self._mutable()
            initial_payload = self._initial()  # Full bytes/archive revalidation at closure.
            traces = self._traces(initial_payload)
            self._validate_result(result, traces)
            reserved = {"schema_version", "protocol_family", "row", "attempt_id", "artifact_sha256", "completed_utc"}
            if reserved & set(result):
                raise ExpansionArtifactError("Result may not override immutable identity or hash fields")
            trace_bytes = b"".join(c.canonical_json(trace).encode("utf-8") + b"\n" for trace in traces)
            atomic_write_bytes(self.trace_path, trace_bytes)
            files = self._all_files()
            payload = {**dict(result), "schema_version": 1, "protocol_family": c.PROTOCOL_FAMILY,
                       "row": self.row, "attempt_id": self.attempt_id,
                       "artifact_sha256": {str(path.relative_to(self.attempt_dir)): sha256_file(path) for path in files},
                       "completed_utc": utc_now()}
            _write(self.result_path, payload)
            append_jsonl_locked(self.store.run_root / "completions" / "completion_ledger.jsonl",
                                {"row_id": self.row["row_id"], "attempt_id": self.attempt_id,
                                 "result_path": str(self.result_path.relative_to(self.store.run_root)),
                                 "result_sha256": sha256_file(self.result_path)})
            return payload

    def record_failure(self, kind: str, code: str, message: str,
                       evidence: Mapping[str, Any]) -> dict[str, Any]:
        with self.store._lock(self.row):
            self._mutable()
            if kind not in ("infrastructure", "hard_stop"):
                raise ExpansionArtifactError("Failure kind must be infrastructure or hard_stop; benchmark failures finalize scientifically")
            if not isinstance(code, str) or not code or not isinstance(message, str) or not message:
                raise ExpansionArtifactError("Failure code and message are required")
            if kind == "infrastructure" and code not in ("transport", "scheduler", "filesystem", "node", "gpu"):
                raise ExpansionArtifactError("Unrecognized retryable infrastructure class")
            if not isinstance(evidence, Mapping) or not evidence:
                raise ExpansionArtifactError("Documented failure evidence is required")
            files = self._all_files()
            payload = {"schema_version": 1, "protocol_family": c.PROTOCOL_FAMILY, "row": self.row,
                       "attempt_id": self.attempt_id, "kind": kind, "code": code, "message": message,
                       "evidence": dict(evidence), "retry_allowed": kind == "infrastructure" and self.attempt_id < 2,
                       "artifact_sha256": {str(path.relative_to(self.attempt_dir)): sha256_file(path) for path in files},
                       "recorded_utc": utc_now()}
            _write(self.failure_path, payload)
            append_jsonl_locked(self.store.run_root / "failures" / "failure_ledger.jsonl",
                                {"row_id": self.row["row_id"], "attempt_id": self.attempt_id,
                                 "failure_path": str(self.failure_path.relative_to(self.store.run_root)),
                                 "failure_sha256": sha256_file(self.failure_path),
                                 "retry_allowed": payload["retry_allowed"]})
            return payload

    def _audit(self) -> dict[str, Any]:
        self._manifest()
        if self.result_path.exists() and self.failure_path.exists():
            raise ExpansionArtifactError("Attempt cannot have both scientific result and failure closure")
        closure_path = self.result_path if self.result_path.exists() else self.failure_path if self.failure_path.exists() else None
        if closure_path is None:
            if self.initial_conditions_path.exists():
                initial_payload = self._initial()
                self._traces(initial_payload)
            return {"status": "incomplete", "retry_allowed": False, "mid_episode_resume_allowed": False}
        closure = _json(closure_path)
        for key, expected in (("protocol_family", c.PROTOCOL_FAMILY), ("row", self.row), ("attempt_id", self.attempt_id)):
            _equal(closure.get(key), expected, f"closure {key}")
        actual_files = {str(path.relative_to(self.attempt_dir)) for path in self._all_files() if path != closure_path}
        hashes = closure.get("artifact_sha256")
        if not isinstance(hashes, Mapping) or set(hashes) != actual_files:
            raise ExpansionArtifactError("Closed attempt artifact census differs from its checksums")
        for name, digest in hashes.items():
            _digest(digest, name)
            path = self.attempt_dir / name
            if path.is_symlink():
                raise ExpansionArtifactError("Closed artifacts must not be symlinks")
            _equal(sha256_file(path), digest, f"closed artifact {name}")
        if closure_path == self.result_path:
            self._validate_ledger("completions/completion_ledger.jsonl", "result_path", "result_sha256", closure_path)
            initial_payload = self._initial()
            traces = self._traces(initial_payload)
            self._validate_result(closure, traces)
            expected_bytes = b"".join(c.canonical_json(trace).encode("utf-8") + b"\n" for trace in traces)
            if self.trace_path.read_bytes() != expected_bytes:
                raise ExpansionArtifactError("Final trace JSONL differs from per-call evidence")
            return {"status": "complete", "retry_allowed": False, "result": closure,
                    "smoke_readiness_pass": closure["terminal_reason"] != "error"}
        kind, code = closure.get("kind"), closure.get("code")
        self._validate_ledger("failures/failure_ledger.jsonl", "failure_path", "failure_sha256", closure_path)
        if kind not in ("infrastructure", "hard_stop"):
            raise ExpansionArtifactError("Invalid failure closure")
        if kind == "infrastructure" and code not in ("transport", "scheduler", "filesystem", "node", "gpu"):
            raise ExpansionArtifactError("Invalid infrastructure failure class")
        _equal(closure.get("retry_allowed"), kind == "infrastructure" and self.attempt_id < 2, "failure retry rule")
        return {"status": "infrastructure_failure" if kind == "infrastructure" else "hard_stop",
                "retry_allowed": closure["retry_allowed"], "failure": closure}

    def _validate_ledger(self, relative: str, path_key: str, sha_key: str, closure_path: Path) -> None:
        ledger = self.store.run_root / relative
        if ledger.is_symlink() or not ledger.is_file():
            raise ExpansionArtifactError("Closed artifact lacks its append-only ledger binding")
        try:
            entries = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        except ValueError as exc:
            raise ExpansionArtifactError("Malformed closure ledger") from exc
        if any(not isinstance(entry, Mapping) for entry in entries):
            raise ExpansionArtifactError("Malformed closure ledger entry")
        matching = [entry for entry in entries if entry.get("row_id") == self.row["row_id"] and entry.get("attempt_id") == self.attempt_id]
        if len(matching) != 1:
            raise ExpansionArtifactError("Closure ledger must contain exactly one binding per attempt")
        match = matching[0]
        _equal(match.get("row_id"), self.row["row_id"], "ledger row")
        _equal(match.get("attempt_id"), self.attempt_id, "ledger attempt")
        _equal(match.get(path_key), str(closure_path.relative_to(self.store.run_root)), "ledger closure path")
        _equal(match.get(sha_key), sha256_file(closure_path), "ledger closure checksum")


def _validate_initial_payload(writer: ExpansionEpisodeWriter, payload: dict[str, Any], *, populate_derived: bool = False) -> None:
    """Validate in-memory pre-publication evidence without creating temp artifacts."""
    # Reuse the same checks by factoring a payload-only instance view, rather
    # than publishing first and discovering a permanently invalid artifact.
    hashes = payload.get("hashes")
    if not isinstance(hashes, Mapping) or set(hashes) != set(SMOKE_INITIAL_CONDITION_HASH_FIELDS):
        raise ExpansionArtifactError("Exactly five inherited initial hash fields are required")
    for key, value in hashes.items():
        _digest(value, key)
    env, reset = payload.get("environment_provenance"), payload.get("reset_evidence")
    if not isinstance(env, Mapping) or not isinstance(reset, Mapping):
        raise ExpansionArtifactError("Initial conditions lack actual environment/reset provenance")
    _int(env.get("resolved_environment_seed"), "resolved_environment_seed")
    if type(env.get("difficulty")) not in (str, int) or env["difficulty"] == "":
        raise ExpansionArtifactError("Resolved difficulty must be recorded")
    _equal(env.get("dataset"), writer.row["dataset"], "resolved split")
    for key, expected in (("policy_seed", 7), ("memory_cleared", True), ("policy_rng_reset", True)):
        _equal(reset.get(key), expected, f"reset {key}")
    count = _int(reset.get("reset_prefix_frame_count"), "reset_prefix_frame_count")
    if count < 1:
        raise ExpansionArtifactError("Reset prefix cannot be empty")
    _equal(reset.get("reset_prefix_stage_count"), count, "reset prefix alignment")
    for key in ("reset_prefix_frames_sha256", "reset_prefix_stages_sha256"):
        _digest(reset.get(key), key)
    _equal(reset["reset_prefix_frames_sha256"], hashes["front_observations_sha256"], "initial front/reset prefix hash")
    _equal(payload.get("episode_manifest_sha256"), sha256_file(writer.manifest_path), "initial manifest binding")
    _equal(payload.get("raw_initial_attachment_sha256"), writer._initial_attachment_hashes(), "raw initial observation attachments")
    try:
        with np.load(writer.attempt_dir / "attachments/initial_observations.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"front", "wrist", "robot_state", "current_task_index"}:
                raise ExpansionArtifactError("Initial NPZ fields differ from the lossless observation archive")
            for camera in ("front", "wrist"):
                array = archive[camera]
                if array.ndim != 4 or array.shape[0] != count or array.shape[-1] != 3 or array.dtype != np.uint8:
                    raise ExpansionArtifactError("Initial RGB archive is malformed or misses reset-prefix frames")
            states, stages = archive["robot_state"], archive["current_task_index"]
            if states.shape != (count, 8) or states.dtype.kind not in "uif" or not np.isfinite(states).all():
                raise ExpansionArtifactError("Initial robot-state archive is malformed")
            if stages.shape != (count,) or stages.dtype.kind not in "iu":
                raise ExpansionArtifactError("Initial stage archive is missing/alignment-invalid")
            raw_boundaries = [0, *(int(index) + 1 for index in np.flatnonzero(stages[1:] != stages[:-1]))]
            if populate_derived:
                payload["reset_prefix_boundary_indices"] = raw_boundaries
            else:
                _equal(payload.get("reset_prefix_boundary_indices"), raw_boundaries, "archived reset-prefix boundary indices")
        state = json.loads((writer.attempt_dir / "attachments/initial_task_state.json").read_text())
        instruction = json.loads((writer.attempt_dir / "attachments/initial_task_instruction.json").read_text())
        if not isinstance(state, Mapping) or "type" not in state or not isinstance(instruction, str) or not instruction:
            raise ExpansionArtifactError("Initial task-state/instruction raw archive is malformed")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ExpansionArtifactError):
            raise
        raise ExpansionArtifactError("Initial raw evidence could not be decoded without pickle") from exc
    if writer.attempt_id > 0:
        prior = ExpansionEpisodeWriter(writer.store, writer.row, writer.attempt_id - 1)
        if prior.initial_conditions_path.exists():
            old = prior._initial()
            for key in ("resolved_environment_seed", "difficulty", "dataset", "resolved_difficulty_hint"):
                _equal(env.get(key), old["environment_provenance"].get(key), f"retry environment {key}")
            _equal(hashes, old["hashes"], "fresh-reset retry initial conditions")
            for key in ("reset_prefix_frame_count", "reset_prefix_stage_count", "reset_prefix_frames_sha256", "reset_prefix_stages_sha256"):
                _equal(reset.get(key), old["reset_evidence"].get(key), f"retry {key}")
    c.canonical_json(payload)
