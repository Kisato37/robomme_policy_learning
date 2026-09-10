"""Read-only worker ancestry proof, before importing a model or simulator.

This checks the controller's existing start record against live Linux identities;
it is not a new launcher, authorization source, process killer, or security token.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
import uuid


class WorkerOwnershipError(RuntimeError):
    pass


_IDENTITY_FIELDS = {"boot_id", "pid", "parent_pid", "process_group", "session", "start_ticks", "uid"}
_MAX_RECORD_BYTES = 1024 * 1024
_PROC_ROOT = Path("/proc")


def _record(path: Path) -> tuple[dict, str]:
    if not path.is_absolute() or any(part.is_symlink() for part in (path, *path.parents)):
        raise WorkerOwnershipError("Worker start evidence must not traverse symlinks")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise WorkerOwnershipError("Worker start evidence must be a regular file owned by this UID")
        data = stream.read(_MAX_RECORD_BYTES + 1)
    if len(data) > _MAX_RECORD_BYTES:
        raise WorkerOwnershipError("Worker start evidence is too large")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise WorkerOwnershipError("Worker start evidence must be a mapping")
    return value, hashlib.sha256(data).hexdigest()


def _boot_id() -> str:
    value = (_PROC_ROOT / "sys/kernel/random/boot_id").read_text().strip()
    if str(uuid.UUID(value)) != value:
        raise WorkerOwnershipError("Invalid live Linux boot identity")
    return value


def _read_own_process(pid: int, boot_id: str) -> dict:
    if type(pid) is not int or pid <= 1:
        raise WorkerOwnershipError("Invalid process identity")
    directory = _PROC_ROOT / str(pid)
    uid = directory.stat().st_uid
    if uid != os.getuid():
        raise WorkerOwnershipError("Ownership proof cannot inspect another user's process")
    text = (directory / "stat").read_text()
    recorded_pid, rest = text.split(" (", 1)
    _, tail = rest.rsplit(") ", 1)
    fields = tail.split()
    if int(recorded_pid) != pid or len(fields) < 20 or fields[0] in {"Z", "X"}:
        raise WorkerOwnershipError("Process has exited or its identity is malformed")
    return {"boot_id": boot_id, "pid": pid, "parent_pid": int(fields[1]),
            "process_group": int(fields[2]), "session": int(fields[3]),
            "start_ticks": int(fields[19]), "uid": uid}


def _identity(value, boot: str) -> dict:
    if not isinstance(value, dict) or set(value) != _IDENTITY_FIELDS:
        raise WorkerOwnershipError("Complete immutable process identity is required")
    if value["boot_id"] != boot or value["uid"] != os.getuid():
        raise WorkerOwnershipError("Start evidence belongs to another boot or UID")
    for field in _IDENTITY_FIELDS - {"boot_id"}:
        minimum = 0 if field == "uid" else 1
        if type(value[field]) is not int or value[field] < minimum:
            raise WorkerOwnershipError("Process fields must have their exact integer types")
    if value["pid"] <= 1:
        raise WorkerOwnershipError("Cannot bind an init/system process")
    return value


def _match_live(expected: dict, boot: str) -> dict:
    observed = _read_own_process(expected["pid"], boot)
    if observed != expected:
        raise WorkerOwnershipError("Recorded process exited, changed identity, or its PID was reused")
    return observed


def verify_worker_ownership(plan, role: str, row_id: int | None = None) -> dict:
    """Prove this worker descends from its recorded live owned watchdog.

    All lookups are limited to this UID and a maximum 64-link parent chain.
    Missing/racing process evidence fails closed; no process is signaled.
    """
    from experiments.uniform_keyframe_expansion.launch_contract import ValidatedExecutionPlan

    if type(plan) is not ValidatedExecutionPlan:
        raise WorkerOwnershipError("An actual validated execution plan is required")
    plan.require_runtime_stage()
    if sys.platform != "linux":
        raise WorkerOwnershipError("Worker entry requires authorized Linux execution")
    if role == "architecture":
        if plan.stage != "architecture_smoke" or plan.rows != [] or row_id is not None:
            raise WorkerOwnershipError("Architecture worker requires its own empty-row architecture plan")
        name, recorded_role = "architecture", "policy"
    elif plan.stage not in {"end_to_end_smoke", "formal"}:
        raise WorkerOwnershipError("Trajectory workers cannot use architecture authorization")
    elif role == "policy":
        if row_id is not None:
            raise WorkerOwnershipError("Resident policy cannot impersonate a row")
        name, recorded_role = "policy", "policy"
    elif role == "simulator":
        if type(row_id) is not int or row_id not in {row["row_id"] for row in plan.rows}:
            raise WorkerOwnershipError("Simulator row is outside the authorized plan")
        name, recorded_role = f"row_{row_id:04d}", "simulator"
    else:
        raise WorkerOwnershipError("Unknown worker role")
    identity = plan.execution_identity
    path = Path(plan.store_root) / "executions" / identity["execution_id"] / f"{name}_start.json"
    record, checksum = _record(path)
    if record.get("execution_identity") != identity or record.get("role") != recorded_role:
        raise WorkerOwnershipError("Worker record is from a different execution or role")
    if record.get("cwd") != str(plan.policy_root):
        raise WorkerOwnershipError("Worker source checkout differs from its start record")
    deadline = record.get("deadline_monotonic")
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.monotonic():
        raise WorkerOwnershipError("Worker startup is outside its recorded lifetime")
    boot = _boot_id()
    watchdog = _identity(record.get("process_identity"), boot)
    controller = _identity(record.get("controller_identity"), boot)
    if (watchdog["process_group"] != watchdog["pid"] or watchdog["session"] != watchdog["pid"]
            or watchdog["parent_pid"] != controller["pid"]):
        raise WorkerOwnershipError("Watchdog is not the recorded controller's dedicated process group")
    _match_live(controller, boot)
    _match_live(watchdog, boot)
    worker_pid = os.getpid()
    if worker_pid in {watchdog["pid"], controller["pid"]}:
        raise WorkerOwnershipError("The controller/watchdog cannot masquerade as its payload")
    worker = _read_own_process(worker_pid, boot)
    observed = worker
    chain = []
    ancestors = []
    for _ in range(64):
        if observed["pid"] in chain:
            raise WorkerOwnershipError("Cyclic process ancestry")
        chain.append(observed["pid"])
        ancestors.append(observed)
        if (observed["uid"] != watchdog["uid"] or observed["boot_id"] != boot
                or observed["process_group"] != watchdog["pid"] or observed["session"] != watchdog["pid"]):
            raise WorkerOwnershipError("Worker or wrapper escaped the owned process group/session")
        if observed["pid"] == watchdog["pid"]:
            if observed != watchdog:
                raise WorkerOwnershipError("Watchdog changed while proving ancestry")
            break
        observed = _read_own_process(observed["parent_pid"], boot)
    else:
        raise WorkerOwnershipError("Worker ancestry is unexpectedly deep")
    # Recheck after traversing to reject a parent exit/reparenting during proof.
    _match_live(controller, boot)
    for ancestor in ancestors:
        _match_live(ancestor, boot)
    if _boot_id() != boot:
        raise WorkerOwnershipError("Linux boot changed during the ownership check")
    return {"worker_ownership_verified": True, "execution_identity": identity,
            "role": role, "row_id": row_id, "worker_identity": worker,
            "watchdog_identity": watchdog, "controller_identity": controller,
            "ancestor_pid_chain": chain, "start_record_path": str(path),
            "start_record_sha256": checksum}
