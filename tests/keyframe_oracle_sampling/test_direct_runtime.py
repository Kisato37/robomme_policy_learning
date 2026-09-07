from __future__ import annotations

from dataclasses import replace
import errno
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import pytest

from experiments.keyframe_neighborhood_sampling.direct_runtime import GpuLease
from experiments.keyframe_neighborhood_sampling.direct_runtime import LinuxProcessOperations
from experiments.keyframe_neighborhood_sampling.direct_runtime import OwnedProcessGroup
from experiments.keyframe_neighborhood_sampling.direct_runtime import OwnershipError
from experiments.keyframe_neighborhood_sampling.direct_runtime import PortReservation
from experiments.keyframe_neighborhood_sampling.direct_runtime import ProcessIdentity
from experiments.keyframe_neighborhood_sampling.direct_runtime import ResourceBusyError
from experiments.keyframe_neighborhood_sampling.runner_contract import RunnerContractError

GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
GPU_C = "GPU-cccccccc-cccc-cccc-cccc-cccccccccccc"
BOOT = "12345678-1234-4234-8234-123456789def"


def test_overlapping_pair_fails_and_releases_partial_acquisition(tmp_path):
    with GpuLease(tmp_path, (GPU_B, GPU_C)):
        with pytest.raises(ResourceBusyError, match="already leased"), GpuLease(tmp_path, (GPU_A, GPU_B)):
            pytest.fail("Overlapping pair was leased")
        with GpuLease(tmp_path, (GPU_A,)):
            pass
    inodes = {path.name: path.stat().st_ino for path in tmp_path.iterdir()}
    with GpuLease(tmp_path, (GPU_A, GPU_B)):
        assert {path.name: path.stat().st_ino for path in tmp_path.iterdir()} == inodes


@pytest.mark.parametrize("allocation", [(GPU_A, GPU_A), ("0", "1"), ()])
def test_gpu_lease_rejects_alias_or_duplicate_allocations(tmp_path, allocation):
    with pytest.raises(RunnerContractError):
        GpuLease(tmp_path, allocation)
    assert not list(tmp_path.iterdir())


def test_lock_symlink_cannot_redirect_resource_ownership(tmp_path):
    target = tmp_path / "unrelated"
    target.write_text("preserve")
    (tmp_path / f"{GPU_A}.lock").symlink_to(target)
    with pytest.raises(OSError, match=r"Too many levels|symbolic link"):
        GpuLease(tmp_path, (GPU_A,))
    assert target.read_text() == "preserve"


def test_inherited_gpu_lease_survives_supervisor_close_until_child_exits(tmp_path):
    lease = GpuLease(tmp_path, (GPU_A,))
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.read()"],
        pass_fds=lease.pass_fds,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        lease.close()
        with pytest.raises(ResourceBusyError):
            GpuLease(tmp_path, (GPU_A,))
    finally:
        lease.close()
        child.communicate(timeout=5)
    with GpuLease(tmp_path, (GPU_A,)):
        pass


def test_live_socket_reservation_excludes_duplicate_listener_and_hands_off_fd():
    with PortReservation() as reservation:
        port = reservation.port
        with socket.socket() as collision, pytest.raises(OSError, match="Address already in use"):
            collision.bind(("127.0.0.1", port))
        inherited = socket.socket(fileno=os.dup(reservation.pass_fds[0]))
        try:
            reservation.close()
            assert inherited.getsockname() == ("127.0.0.1", port)
            with socket.socket() as collision, pytest.raises(OSError, match="Address already in use"):
                collision.bind(("127.0.0.1", port))
        finally:
            inherited.close()
    with PortReservation(port) as reused:
        assert reused.port == port


def test_linux_identity_parser_handles_parentheses_and_refuses_pid_only_fallback(tmp_path):
    reader = LinuxProcessOperations(tmp_path)
    with pytest.raises(OwnershipError, match="no PID-only fallback"):
        reader.identity(999)
    boot_path = tmp_path / "sys/kernel/random/boot_id"
    boot_path.parent.mkdir(parents=True)
    boot_path.write_text(BOOT)
    directory = tmp_path / "999"
    directory.mkdir()
    fields = ["S", str(os.getpid()), "999", "999", *(["0"] * 15), "123456"]
    (directory / "stat").write_text("999 (name with ) and spaces) " + " ".join(fields))
    identity = reader.identity(999)
    assert identity.start_ticks == 123456
    assert identity.parent_pid == os.getpid()
    assert reader.live_group_members(999) == [identity]
    fields[0] = "Z"
    (directory / "stat").write_text("999 (name) " + " ".join(fields))
    assert reader.identity(999) == identity
    assert reader.live_group_members(999) == []
    with pytest.raises(OwnershipError, match="Synthetic"):
        reader.signal_group(999, signal.SIGTERM)


def test_linux_process_disappearing_during_stat_read_is_not_a_cleanup_error(tmp_path, monkeypatch):
    boot_path = tmp_path / "sys/kernel/random/boot_id"
    boot_path.parent.mkdir(parents=True)
    boot_path.write_text(BOOT)
    process_path = tmp_path / "999"
    process_path.mkdir()
    original_read_text = Path.read_text

    def disappearing_read(path, *args, **kwargs):
        if path == process_path / "stat":
            raise ProcessLookupError(errno.ESRCH, "Process disappeared")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", disappearing_read)
    reader = LinuxProcessOperations(tmp_path)
    assert reader.identity(999) is None
    assert reader.live_group_members(999) == []


class FakeOperations:
    def __init__(self, *, ignore_kill=False):
        self.leader = ProcessIdentity(BOOT, 999, os.getpid(), 999, 999, 100, os.getuid())
        self.members = [self.leader, replace(self.leader, pid=1000, parent_pid=999)]
        self.signals = []
        self.ignore_kill = ignore_kill

    def identity(self, pid):
        assert pid == 999
        return self.leader

    def live_group_members(self, process_group):
        assert process_group == 999
        return self.members

    def signal_group(self, process_group, signal_number):
        self.signals.append((process_group, signal_number))
        if signal_number == signal.SIGTERM:
            # Leader exits, but an actual descendant still occupies resources.
            self.members = [member for member in self.members if member.pid != 999]
        elif signal_number == signal.SIGKILL and not self.ignore_kill:
            self.members = []


class FakeChild:
    pid = 999

    def __init__(self, operations):
        self.operations = operations
        self.waited = False

    def wait(self, timeout=None):
        assert not self.operations.members, "Reaped the leader while a descendant was alive"
        self.waited = True
        self.operations.leader = None
        return -15


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def test_cleanup_waits_for_descendants_escalates_and_reaps_only_owned_group():
    operations = FakeOperations()
    child = FakeChild(operations)
    clock = FakeClock()
    owner = OwnedProcessGroup(child, operations, monotonic=clock.monotonic, sleep=clock.sleep)
    assert owner.reap_if_finished() is None
    assert owner.stop(grace_seconds=0.1, kill_wait_seconds=0.1) == -15
    assert operations.signals == [(999, signal.SIGTERM), (999, signal.SIGKILL)]
    assert child.waited
    assert owner.stop(grace_seconds=0, kill_wait_seconds=0) == -15


def test_exited_leader_is_not_completion_while_descendants_live():
    operations = FakeOperations()
    child = FakeChild(operations)
    owner = OwnedProcessGroup(child, operations)
    operations.members = operations.members[1:]
    assert owner.reap_if_finished() is None
    assert not child.waited
    operations.members = []
    assert owner.reap_if_finished() == -15
    assert not operations.signals


@pytest.mark.parametrize("changed", ["pid_reuse", "reboot", "reaped"])
def test_identity_change_prevents_signaling_any_group(changed):
    operations = FakeOperations()
    owner = OwnedProcessGroup(FakeChild(operations), operations)
    if changed == "pid_reuse":
        operations.leader = replace(operations.leader, start_ticks=101)
    elif changed == "reboot":
        operations.leader = replace(operations.leader, boot_id="99999999-1234-4234-8234-123456789def")
    else:
        operations.leader = None
    with pytest.raises(OwnershipError, match="changed identity"):
        owner.send_signal(signal.SIGTERM)
    assert not operations.signals


@pytest.mark.parametrize(
    "changed", [{"parent_pid": 998}, {"uid": os.getuid() + 1}, {"session": 998}, {"process_group": 998}]
)
def test_cannot_adopt_an_unrelated_process(changed):
    operations = FakeOperations()
    operations.leader = replace(operations.leader, **changed)
    with pytest.raises(OwnershipError, match="this supervisor"):
        OwnedProcessGroup(FakeChild(operations), operations)
    assert not operations.signals


def test_unkillable_descendant_stays_pending_and_is_never_reported_complete():
    operations = FakeOperations(ignore_kill=True)
    child = FakeChild(operations)
    clock = FakeClock()
    owner = OwnedProcessGroup(child, operations, monotonic=clock.monotonic, sleep=clock.sleep)
    with pytest.raises(OwnershipError, match="keep resources leased"):
        owner.stop(grace_seconds=0.1, kill_wait_seconds=0.2)
    assert not child.waited
    assert owner.returncode is None
    assert clock.now == pytest.approx(0.3)
