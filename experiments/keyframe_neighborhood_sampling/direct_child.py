"""Linux-only, stdlib process-group watchdog for an explicitly authorized child.

The controller must start this trampoline in a new session, retain an unreaped
Popen, and pass a pipe read FD. Only b"1" releases the payload barrier, after
the controller has recorded ownership. No model or simulator is imported here.
Payload descendants must remain in this session/process group; daemonization
and setsid are unsupported. This is not a cgroup or a security sandbox.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import select
import signal
import stat
import subprocess
import sys
import time
import uuid

POLL_SECONDS = 0.1
BOOT_PATH = Path("/proc/sys/kernel/random/boot_id")


@dataclass(frozen=True)
class ProcessSample:
    pid: int
    parent_pid: int
    process_group: int
    session: int
    start_ticks: int
    uid: int
    state: str
    resident_bytes: int


def read_process(pid: int) -> ProcessSample | None:
    directory = Path("/proc") / str(pid)
    try:
        owner = directory.stat().st_uid
        recorded_pid, rest = (directory / "stat").read_text().split(" (", 1)
        _, tail = rest.rsplit(") ", 1)
        fields = tail.split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if int(recorded_pid) != pid or len(fields) < 22:
        raise RuntimeError("Malformed Linux process identity")
    return ProcessSample(
        pid=pid,
        parent_pid=int(fields[1]),
        process_group=int(fields[2]),
        session=int(fields[3]),
        start_ticks=int(fields[19]),
        uid=owner,
        state=fields[0],
        resident_bytes=max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE"),
    )


def parent_alive(parent_pid: int, parent_start_ticks: int, boot_id: str) -> bool:
    """Never treat a reused PID, reparented child, or unverified /proc read as ownership."""
    try:
        parent = read_process(parent_pid)
        return (
            os.getppid() == parent_pid
            and BOOT_PATH.read_text().strip() == boot_id
            and parent is not None
            and parent.start_ticks == parent_start_ticks
            and parent.uid == os.getuid()
            and parent.state not in {"Z", "X"}
        )
    except (OSError, ValueError, RuntimeError):
        return False


def require_own_group() -> int:
    process_id = os.getpid()
    if process_id <= 1 or os.getpgrp() != process_id or os.getsid(0) != process_id:
        raise RuntimeError("Watchdog must be its own dedicated session and process-group leader")
    return process_id


def group_members() -> list[ProcessSample]:
    """Read only our UID's process metadata; return live members of our own session/group."""
    group = require_own_group()
    members = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal() or int(directory.name) <= 1:
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        member = read_process(int(directory.name))
        if member is not None and member.process_group == group and member.state not in {"Z", "X"}:
            if member.session != group or member.uid != os.getuid():
                raise RuntimeError("Process-group member has unexpected session or UID")
            members.append(member)
    return members


def configure_affinity(cpu_ids: str | None) -> None:
    if cpu_ids is None:
        return
    fields = cpu_ids.split(",")
    if not fields or any(not field.isdecimal() for field in fields):
        raise ValueError("cpu_ids must be a comma-separated list of nonnegative CPU IDs")
    requested = {int(field) for field in fields}
    if len(requested) != len(fields) or not requested.issubset(os.sched_getaffinity(0)):
        raise ValueError("cpu_ids must be unique and within the inherited CPU affinity")
    os.sched_setaffinity(0, requested)


def stop_group(child: subprocess.Popen, grace_seconds: float) -> None:
    """TERM our pinned group; KILL it if live descendants cannot be proven gone.

    The watchdog handles its own TERM and stays alive to pin the group ID until
    cleanup is confirmed. The final KILL includes this watchdog intentionally.
    """
    group = require_own_group()
    os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    empty_observations = 0
    while True:
        child.poll()  # Reap the immediate payload, not this watchdog's group leader.
        try:
            survivors = [member for member in group_members() if member.pid != group]
        except (OSError, ValueError, RuntimeError):
            # Missing proof must never be reported as successful cleanup.
            survivors = [None]
        empty_observations = empty_observations + 1 if not survivors else 0
        if empty_observations >= 2:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            os.killpg(require_own_group(), signal.SIGKILL)
            raise RuntimeError("SIGKILL unexpectedly returned without terminating the watchdog")
        time.sleep(min(POLL_SECONDS, remaining))


def run(args: argparse.Namespace) -> int:
    if sys.platform != "linux" or not BOOT_PATH.is_file():
        raise RuntimeError("Direct watchdog requires Linux /proc")
    require_own_group()
    if args.parent_pid <= 1 or args.parent_start_ticks <= 0:
        raise ValueError("Invalid parent process identity")
    if str(uuid.UUID(args.boot_id)) != args.boot_id:
        raise ValueError("boot_id must be a canonical UUID")
    if not math.isfinite(args.deadline_monotonic) or args.deadline_monotonic <= 0:
        raise ValueError("deadline_monotonic must be a finite absolute monotonic timestamp")
    if not math.isfinite(args.grace_seconds) or not 0 <= args.grace_seconds <= 30:
        raise ValueError("grace_seconds must be finite and between zero and 30")
    if args.memory_limit_bytes is not None and args.memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be positive")
    descriptors = [args.start_fd, *args.pass_fd]
    if any(fd < 3 for fd in descriptors) or len(set(descriptors)) != len(descriptors):
        raise ValueError("start_fd and pass_fd must be distinct descriptors >= 3")
    for descriptor in descriptors:
        os.fstat(descriptor)
    if not stat.S_ISFIFO(os.fstat(args.start_fd).st_mode):
        raise ValueError("start_fd must be an inherited pipe")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not command[0]:
        raise ValueError("An explicit payload command is required")
    configure_affinity(args.cpu_ids)
    received_signal = 0

    def interrupted(signum, _frame):
        nonlocal received_signal
        if not received_signal:
            received_signal = signum

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)

    def stop_reason() -> tuple[int, str] | None:
        if received_signal:
            return 128 + received_signal, f"signal {received_signal}"
        if not parent_alive(args.parent_pid, args.parent_start_ticks, args.boot_id):
            return 125, "controller ownership lost"
        if time.monotonic() >= args.deadline_monotonic:
            return 124, "absolute runtime deadline reached"
        return None

    child = None
    cleaned = False
    try:
        # Parent crash, rejected ownership, and any malformed/closed barrier all
        # return before Popen can import the model or allocate any GPU memory.
        while True:
            if reason := stop_reason():
                print(f"direct-child: {reason[1]} before payload authorization", file=sys.stderr, flush=True)
                return reason[0]
            if select.select([args.start_fd], [], [], POLL_SECONDS)[0]:
                if os.read(args.start_fd, 1) != b"1":
                    print("direct-child: payload authorization pipe closed or invalid", file=sys.stderr, flush=True)
                    return 125
                break
        os.close(args.start_fd)
        args.start_fd = -1
        if reason := stop_reason():
            print(f"direct-child: {reason[1]} before payload startup", file=sys.stderr, flush=True)
            return reason[0]
        child = subprocess.Popen(command, pass_fds=tuple(args.pass_fd))
        next_memory_check = 0.0
        while True:
            if reason := stop_reason():
                result, message = reason
                print(f"direct-child: {message}", file=sys.stderr, flush=True)
                break
            status = child.poll()
            if status is not None:
                result = 128 - status if status < 0 else status
                break
            if args.memory_limit_bytes is not None and time.monotonic() >= next_memory_check:
                next_memory_check = time.monotonic() + 1.0
                if sum(member.resident_bytes for member in group_members()) > args.memory_limit_bytes:
                    print("direct-child: process-group RSS limit reached", file=sys.stderr, flush=True)
                    result = 137
                    break
            time.sleep(POLL_SECONDS)
        stop_group(child, args.grace_seconds)
        cleaned = True
        return result
    finally:
        if args.start_fd >= 0:
            os.close(args.start_fd)
        if child is not None and not cleaned:
            stop_group(child, args.grace_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-fd", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-start-ticks", type=int, required=True)
    parser.add_argument("--boot-id", required=True)
    parser.add_argument("--deadline-monotonic", type=float, required=True)
    parser.add_argument("--grace-seconds", type=float, default=10)
    parser.add_argument("--pass-fd", type=int, action="append", default=[])
    parser.add_argument("--cpu-ids")
    parser.add_argument("--memory-limit-bytes", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    try:
        return run(parser.parse_args())
    except Exception as error:
        print(f"direct-child: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 125


if __name__ == "__main__":
    sys.exit(main())
