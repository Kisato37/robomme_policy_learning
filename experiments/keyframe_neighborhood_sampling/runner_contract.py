"""Direct-process evidence foundation; this module grants no launch authority.

A dispatch binds execution identity, not scientific validity or user permission.
The existing Slurm schemas and protocol version remain unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid

EXECUTION_SCHEMA = "keyframe-neighborhood-direct-process-v1"
PROTOCOL_FAMILY = "keyframe_neighborhood_sampling_v1"
_GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")


class RunnerContractError(ValueError):
    """Execution evidence is missing, inconsistent, or ambiguous."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def record_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_uuid(value: str, *, field: str) -> None:
    try:
        valid = isinstance(value, str) and str(uuid.UUID(value)) == value
    except ValueError:
        valid = False
    if not valid:
        raise RunnerContractError(f"{field} must be a canonical UUID")


def require_gpu_uuid(value: str) -> None:
    if not isinstance(value, str) or _GPU_UUID.fullmatch(value) is None:
        raise RunnerContractError("GPU allocation requires a full physical GPU UUID, not an index or MIG alias")


def _require_hex(value: str, length: int, *, field: str) -> None:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise RunnerContractError(f"{field} must contain exactly {length} lowercase hexadecimal characters")


@dataclass(frozen=True)
class DirectDispatch:
    """One process group allocation; row semantics remain the matrix's responsibility.

    gpu_uuids is ordered: policy first, simulator second. Architecture inference
    has one policy GPU and no row, shard, or policy-server port. The source
    digests refer to separately reviewed artifacts; copying a digest or UUID
    does not constitute external authorization.
    """

    execution_id: str
    run_root: str
    repository_commit_sha: str
    stage: str
    launch_manifest_sha256: str
    submission_plan_sha256: str
    matrix_sha256: str
    attempt_id: int
    row_id: int | None
    shard_id: int | None
    gpu_uuids: tuple[str, ...]
    host_name: str
    host_boot_id: str
    policy_port: int | None
    gpu_layout: str = "separate"
    _legacy_implicit_layout: bool = dataclass_field(default=False, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        require_uuid(self.execution_id, field="execution_id")
        require_uuid(self.host_boot_id, field="host_boot_id")
        if not isinstance(self.run_root, str):
            raise RunnerContractError("run_root must be an absolute path string")
        root = Path(self.run_root)
        if (
            not root.is_absolute()
            or ".." in root.parts
            or str(root) != self.run_root
            or root.parent.name != "keyframe_neighborhood_sampling"
            or root.parent.parent.name != "runs"
        ):
            raise RunnerContractError(
                "run_root must be an absolute direct child of runs/keyframe_neighborhood_sampling"
            )
        _require_hex(self.repository_commit_sha, 40, field="repository_commit_sha")
        for field in ("launch_manifest_sha256", "submission_plan_sha256", "matrix_sha256"):
            _require_hex(getattr(self, field), 64, field=field)
        if not isinstance(self.host_name, str) or not self.host_name or any(c.isspace() for c in self.host_name):
            raise RunnerContractError("host_name must be a nonempty host identifier")
        if type(self.attempt_id) is not int or self.attempt_id not in {0, 1, 2}:
            raise RunnerContractError("attempt_id must be 0, 1, or 2")
        if not isinstance(self.gpu_uuids, tuple):
            raise RunnerContractError("gpu_uuids must be an ordered immutable tuple")
        for gpu in self.gpu_uuids:
            require_gpu_uuid(gpu)
        if not isinstance(self.gpu_layout, str) or self.gpu_layout not in {"separate", "colocated"}:
            raise RunnerContractError("GPU layout must be separate or colocated")
        if self.stage == "architecture_smoke":
            if len(self.gpu_uuids) != 1 or self.attempt_id != 0:
                raise RunnerContractError("Architecture dispatch requires one GPU and attempt zero")
            if self.row_id is not None or self.shard_id is not None or self.policy_port is not None:
                raise RunnerContractError("Architecture dispatch cannot carry a trajectory row, shard, or server port")
        elif self.stage in {"development_smoke", "formal"}:
            if len(self.gpu_uuids) != 2 or type(self.row_id) is not int or self.row_id < 0:
                raise RunnerContractError(
                    "Trajectory dispatch requires two GPU role identities and a nonnegative row ID"
                )
            distinct_count = 1 if self.gpu_layout == "colocated" else 2
            if len(set(self.gpu_uuids)) != distinct_count:
                raise RunnerContractError("Policy and simulator GPU identities differ from the explicit GPU layout")
            if type(self.policy_port) is not int or not 1024 <= self.policy_port <= 65535:
                raise RunnerContractError("Trajectory dispatch requires a nonprivileged policy port")
            if self.stage == "formal":
                if type(self.shard_id) is not int or self.shard_id < 0:
                    raise RunnerContractError("Formal dispatch requires an explicit nonnegative shard ID")
            elif self.shard_id is not None:
                raise RunnerContractError("Development-smoke dispatch does not use formal shards")
        else:
            raise RunnerContractError("Unknown execution stage")

    def as_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("_legacy_implicit_layout")
        if self._legacy_implicit_layout:
            # Old direct evidence keeps its original digest and implicit separate layout.
            payload.pop("gpu_layout")
        return {
            "execution_schema": EXECUTION_SCHEMA,
            "backend": "direct",
            "protocol_family": PROTOCOL_FAMILY,
            "grants_launch_authority": False,
            **payload,
            "gpu_uuids": list(self.gpu_uuids),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> DirectDispatch:
        fields = {
            "execution_schema": EXECUTION_SCHEMA,
            "backend": "direct",
            "protocol_family": PROTOCOL_FAMILY,
            "grants_launch_authority": False,
        }
        serialized_fields = {name for name, field in cls.__dataclass_fields__.items() if field.init}
        expected_fields = serialized_fields | set(fields)
        if set(record) not in (expected_fields, expected_fields - {"gpu_layout"}):
            raise RunnerContractError("Dispatch has missing or unknown fields; Slurm fields are not direct evidence")
        if any(record.get(field) != expected for field, expected in fields.items()):
            raise RunnerContractError("Dispatch backend/schema/authority identity mismatch")
        if record["grants_launch_authority"] is not False:
            raise RunnerContractError("Execution evidence cannot grant launch authority")
        if not isinstance(record["gpu_uuids"], list):
            raise RunnerContractError("Serialized gpu_uuids must be an ordered list")
        payload = {field: record[field] for field in serialized_fields if field in record}
        payload["gpu_uuids"] = tuple(record["gpu_uuids"])
        try:
            dispatch = cls(**payload)
            if "gpu_layout" not in record:
                object.__setattr__(dispatch, "_legacy_implicit_layout", True)
            return dispatch
        except (TypeError, AttributeError) as exc:
            raise RunnerContractError("Dispatch field types are invalid") from exc


def validate_runtime_binding(
    record: Mapping[str, Any],
    *,
    expected_sha256: str,
    observed: DirectDispatch,
) -> DirectDispatch:
    """Check complete expected-vs-observed identity without reading or inventing Slurm variables."""
    _require_hex(expected_sha256, 64, field="expected_sha256")
    dispatch = DirectDispatch.from_record(record)
    if record_digest(record) != expected_sha256:
        raise RunnerContractError("Dispatch changed after it was bound")
    if dispatch != observed:
        raise RunnerContractError("Live execution identity differs from the recorded dispatch")
    return dispatch


def write_once_record(path: Path, record: Mapping[str, Any]) -> str:
    """Publish complete canonical bytes without replacing an existing path.

    Parent directories must already exist. Hard-link publication keeps the
    destination exclusive under concurrent writers; unlike rename it cannot
    replace an earlier record. Temporary files are removed after publication.
    """
    data = canonical_bytes(record)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)
    return hashlib.sha256(data).hexdigest()


def process_start_record(
    dispatch: DirectDispatch,
    *,
    process_identity: Mapping[str, Any],
    command: list[str],
    working_directory: str,
) -> dict[str, Any]:
    """Bind observed process identity and argv to the dispatch, after owned-child validation."""
    integer_fields = {"pid", "parent_pid", "process_group", "session", "start_ticks", "uid"}
    if set(process_identity) != integer_fields | {"boot_id"}:
        raise RunnerContractError("Start evidence must include the complete Linux process identity")
    if process_identity["boot_id"] != dispatch.host_boot_id:
        raise RunnerContractError("Start evidence belongs to a different host boot")
    if any(type(process_identity[field]) is not int or process_identity[field] < 0 for field in integer_fields):
        raise RunnerContractError("Invalid numeric process identity")
    pid = process_identity["pid"]
    if (
        pid <= 1
        or process_identity["parent_pid"] <= 1
        or process_identity["start_ticks"] <= 0
        or process_identity["process_group"] != pid
        or process_identity["session"] != pid
    ):
        raise RunnerContractError("Start evidence requires an owned process in a dedicated session")
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) or "\0" in arg for arg in command):
        raise RunnerContractError("Command must be an explicit argv list")
    if not command[0] or not isinstance(working_directory, str) or not Path(working_directory).is_absolute():
        raise RunnerContractError("Start evidence requires an executable and absolute working directory")
    return {
        "execution_schema": EXECUTION_SCHEMA,
        "backend": "direct",
        "record_kind": "process_start",
        "execution_id": dispatch.execution_id,
        "dispatch_sha256": record_digest(dispatch.as_record()),
        "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "process_identity": dict(process_identity),
        "command": list(command),
        "working_directory": working_directory,
        "grants_launch_authority": False,
    }


def process_exit_record(
    dispatch: DirectDispatch,
    *,
    started_record_sha256: str,
    returncode: int,
    wall_clock_limit_reached: bool,
) -> dict[str, Any]:
    """Record subprocess exit facts without classifying an episode or permitting a retry."""
    _require_hex(started_record_sha256, 64, field="started_record_sha256")
    if type(returncode) is not int or not -64 <= returncode <= 255:
        raise RunnerContractError("returncode must be a subprocess exit code or negative signal")
    if type(wall_clock_limit_reached) is not bool:
        raise RunnerContractError("wall_clock_limit_reached must be a boolean")
    return {
        "execution_schema": EXECUTION_SCHEMA,
        "backend": "direct",
        "record_kind": "process_exit",
        "execution_id": dispatch.execution_id,
        "dispatch_sha256": record_digest(dispatch.as_record()),
        "started_record_sha256": started_record_sha256,
        "finished_utc": dt.datetime.now(dt.UTC).isoformat(),
        "returncode": returncode,
        "termination_signal": -returncode if returncode < 0 else None,
        "wall_clock_limit_reached": wall_clock_limit_reached,
        "scientific_result_validated": False,
        "retry_authorized": False,
    }
