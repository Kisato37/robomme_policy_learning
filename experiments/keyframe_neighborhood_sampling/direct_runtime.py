"""CPU-only ownership primitives for a future Linux direct runner.

No imports initialize CUDA, allocate device memory, start a policy, or authorize
an experiment. Linux process control accepts only an unreaped child of this
supervisor in its own session. Cooperating children must not daemonize/setsid;
production integration still needs supervision and reconciliation of crashes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import fcntl
import math
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import time
from typing import Protocol

from experiments.keyframe_neighborhood_sampling.runner_contract import RunnerContractError
from experiments.keyframe_neighborhood_sampling.runner_contract import require_gpu_uuid
from experiments.keyframe_neighborhood_sampling.runner_contract import require_uuid


class ResourceBusyError(RuntimeError):
    """A requested resource is owned; callers must not steal or silently substitute it."""


class OwnershipError(RuntimeError):
    """Process ownership cannot be proved, or cleanup did not finish."""


class GpuLease:
    """Nonblocking advisory locks on declared physical GPUs, with no CUDA work.

    Every cooperating runner must use the same local host-wide lock directory.
    Persistent lockfiles are never unlinked: replacing an inode defeats flock.
    pass_fds must be inherited by *all* GPU children so a supervisor crash cannot
    release a live child's GPU. Closing our descriptors does not unlock theirs.
    This does not claim to exclude unrelated CUDA clients or other users.
    """

    def __init__(self, lock_directory: Path, gpu_uuids: Sequence[str]):
        if not gpu_uuids or len(gpu_uuids) != len(set(gpu_uuids)):
            raise RunnerContractError("GPU lease requires a nonempty non-overlapping allocation")
        for gpu in gpu_uuids:
            require_gpu_uuid(gpu)
        if not lock_directory.is_dir() or lock_directory.is_symlink():
            raise OwnershipError("Use an existing, stable, nonsymlink host lock directory")
        self.gpu_uuids = tuple(gpu_uuids)
        self._descriptors: list[int] = []
        try:
            for gpu in sorted(gpu_uuids):
                descriptor = os.open(
                    lock_directory / f"{gpu}.lock",
                    os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o600,
                )
                self._descriptors.append(descriptor)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise OwnershipError("GPU lock must be a regular file")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ResourceBusyError(f"GPU already leased: {gpu}") from exc
        except BaseException:
            self.close()
            raise

    @property
    def pass_fds(self) -> tuple[int, ...]:
        if not self._descriptors:
            raise OwnershipError("GPU lease is closed")
        return tuple(self._descriptors)

    def close(self) -> None:
        while self._descriptors:
            # Do not LOCK_UN: inherited children share this open-file description.
            os.close(self._descriptors.pop())

    def __enter__(self) -> GpuLease:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PortReservation:
    """A held loopback listening socket; hand off its FD without a close/rebind gap.

    The existing policy server does not yet accept this descriptor. Integration
    must add FD adoption before this can serve as its production reservation.
    A bare TCP connection is not proof of the policy server's identity.
    """

    def __init__(self, port: int = 0):
        if type(port) is not int or (port != 0 and not 1024 <= port <= 65535):
            raise RunnerContractError("Port must be zero (OS allocation) or 1024..65535")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.bind(("127.0.0.1", port))
            self.socket.listen(1)
        except BaseException:
            self.socket.close()
            raise
        self.port = int(self.socket.getsockname()[1])

    @property
    def pass_fds(self) -> tuple[int]:
        descriptor = self.socket.fileno()
        if descriptor < 0:
            raise OwnershipError("Port reservation is closed")
        return (descriptor,)

    def close(self) -> None:
        self.socket.close()

    def __enter__(self) -> PortReservation:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class ProcessIdentity:
    boot_id: str
    pid: int
    parent_pid: int
    process_group: int
    session: int
    start_ticks: int
    uid: int


class ProcessOperations(Protocol):
    """Injectable OS boundary, used by deterministic CPU-only lifecycle tests."""

    def identity(self, pid: int) -> ProcessIdentity | None: ...

    def live_group_members(self, process_group: int) -> Sequence[ProcessIdentity]: ...

    def signal_group(self, process_group: int, signal_number: int) -> None: ...


class LinuxProcessOperations:
    def __init__(self, proc_root: Path = Path("/proc")):
        self.proc_root = proc_root

    def _read(self, pid: int) -> tuple[ProcessIdentity, str] | None:
        if type(pid) is not int or pid <= 1:
            raise OwnershipError("Process identity requires a PID above one")
        try:
            boot_id = (self.proc_root / "sys/kernel/random/boot_id").read_text().strip()
        except FileNotFoundError as exc:
            raise OwnershipError("Linux /proc boot identity is required; no PID-only fallback") from exc
        require_uuid(boot_id, field="boot_id")
        try:
            directory = self.proc_root / str(pid)
            text = (directory / "stat").read_text()
            uid = directory.stat().st_uid
        except (FileNotFoundError, ProcessLookupError):
            return None
        # The parenthesized comm field can contain spaces and ')'.
        try:
            recorded_pid, rest = text.split(" (", 1)
            _, tail = rest.rsplit(") ", 1)
            fields = tail.split()
            if int(recorded_pid) != pid or len(fields) < 20:
                raise ValueError("Wrong PID or truncated stat")
            identity = ProcessIdentity(
                boot_id, pid, int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19]), uid
            )
        except (IndexError, ValueError) as exc:
            raise OwnershipError("Malformed Linux process identity") from exc
        return identity, fields[0]

    def identity(self, pid: int) -> ProcessIdentity | None:
        record = self._read(pid)
        return None if record is None else record[0]

    def live_group_members(self, process_group: int) -> Sequence[ProcessIdentity]:
        members = []
        for directory in self.proc_root.iterdir():
            if not directory.name.isdecimal() or int(directory.name) <= 1:
                continue
            record = self._read(int(directory.name))
            if record is not None and record[0].process_group == process_group and record[1] not in {"Z", "X"}:
                members.append(record[0])
        return members

    def signal_group(self, process_group: int, signal_number: int) -> None:
        if self.proc_root != Path("/proc"):
            raise OwnershipError("Synthetic process tables cannot signal the operating system")
        os.killpg(process_group, signal_number)


class ChildProcess(Protocol):
    pid: int

    def wait(self, timeout: float | None = None) -> int: ...


class OwnedProcessGroup:
    """Track an owned new-session child until its complete process group exits.

    The caller MUST NOT poll/wait/reap the child elsewhere. Keeping the group
    leader unreaped pins its PID even after it exits, preventing group-ID reuse
    during descendant cleanup. OS signals that auto-reap children are unsupported.
    This class installs no global signal handlers and performs no automatic retry.
    """

    def __init__(
        self,
        child: ChildProcess,
        operations: ProcessOperations,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        identity = operations.identity(child.pid)
        if (
            identity is None
            or identity.parent_pid != os.getpid()
            or identity.uid != os.getuid()
            or identity.process_group != child.pid
            or identity.session != child.pid
        ):
            raise OwnershipError("Process must be this supervisor's unreaped child in a dedicated session")
        self.child = child
        self.identity = identity
        self.operations = operations
        self.monotonic = monotonic
        self.sleep = sleep
        self.returncode: int | None = None

    def _require_owner(self) -> None:
        if self.returncode is not None or self.operations.identity(self.identity.pid) != self.identity:
            raise OwnershipError("Process exited/reaped, changed identity, or belongs to a different boot")

    def send_signal(self, signal_number: int) -> None:
        if signal_number not in {signal.SIGINT, signal.SIGTERM, signal.SIGKILL}:
            raise OwnershipError("Only explicit INT, TERM, and KILL lifecycle signals are supported")
        self._require_owner()
        try:
            self.operations.signal_group(self.identity.process_group, signal_number)
        except ProcessLookupError:
            # The leader still pins the identity; concurrent natural exit is fine.
            self._require_owner()

    def _wait_empty(self, duration: float) -> bool:
        deadline = self.monotonic() + duration
        while True:
            self._require_owner()
            if not self.operations.live_group_members(self.identity.process_group):
                return True
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                return False
            self.sleep(min(0.05, remaining))

    def reap_if_finished(self) -> int | None:
        """Nonblocking completion observation; an exited leader with live descendants is still pending."""
        if self.returncode is not None:
            return self.returncode
        self._require_owner()
        if self.operations.live_group_members(self.identity.process_group):
            return None
        try:
            self.returncode = self.child.wait(timeout=0)
        except subprocess.TimeoutExpired:
            return None
        return self.returncode

    def stop(self, *, grace_seconds: float, kill_wait_seconds: float) -> int:
        """TERM, bounded wait, KILL if needed, then reap only after no live members remain."""
        for duration in (grace_seconds, kill_wait_seconds):
            if isinstance(duration, bool) or not math.isfinite(duration) or duration < 0:
                raise ValueError("Cleanup deadlines must be finite and nonnegative")
        if self.returncode is not None:
            return self.returncode
        self.send_signal(signal.SIGTERM)
        if not self._wait_empty(grace_seconds):
            self.send_signal(signal.SIGKILL)
            if not self._wait_empty(kill_wait_seconds):
                raise OwnershipError(
                    "Owned processes remain alive; keep resources leased and preserve pending evidence"
                )
        try:
            self.returncode = self.child.wait(timeout=kill_wait_seconds)
        except subprocess.TimeoutExpired as exc:
            raise OwnershipError("Child could not be reaped; completion is unconfirmed") from exc
        return self.returncode
