"""CPU-only Linux tests of the launch barrier and independent group watchdog."""

from contextlib import contextmanager
from contextlib import suppress
import ctypes
import json
import os
import platform
import select
import signal
import subprocess
import sys
import time

import pytest

from experiments.keyframe_neighborhood_sampling import direct_child

MODULE = "experiments.keyframe_neighborhood_sampling.direct_child"
LINUX = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc and process-group integration")


def checked_pidfd_result(result):
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_libc():
    if sys.platform != "linux":
        raise RuntimeError("Exact-process pidfd cleanup requires Linux")
    return ctypes.CDLL(None, use_errno=True)


def open_pidfd(pid):
    """Managed Python may omit pidfd bindings although the Linux kernel supports them."""
    if native := getattr(os, "pidfd_open", None):
        return native(pid)
    libc = pidfd_libc()
    if function := getattr(libc, "pidfd_open", None):
        function.argtypes = [ctypes.c_int, ctypes.c_uint]
        function.restype = ctypes.c_int
        return checked_pidfd_result(function(pid, 0))
    if platform.machine() != "x86_64":
        raise RuntimeError("No verified pidfd_open syscall number for this architecture")
    libc.syscall.restype = ctypes.c_long
    return checked_pidfd_result(libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0)))


def signal_pidfd(descriptor, signum):
    """Signal only the kernel process handle; never fall back to a numeric-PID kill."""
    if native := getattr(signal, "pidfd_send_signal", None):
        return native(descriptor, signum)
    libc = pidfd_libc()
    if function := getattr(libc, "pidfd_send_signal", None):
        function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        return checked_pidfd_result(function(descriptor, signum, None, 0))
    if platform.machine() != "x86_64":
        raise RuntimeError("No verified pidfd_send_signal syscall number for this architecture")
    libc.syscall.restype = ctypes.c_long
    return checked_pidfd_result(
        libc.syscall(
            ctypes.c_long(424), ctypes.c_int(descriptor), ctypes.c_int(signum), ctypes.c_void_p(), ctypes.c_uint(0)
        )
    )


def read_line(stream, timeout=4):
    deadline = time.monotonic() + timeout
    line = bytearray()
    while time.monotonic() < deadline:
        if select.select([stream], [], [], max(0, deadline - time.monotonic()))[0]:
            byte = os.read(stream.fileno(), 1)
            if not byte or byte == b"\n":
                return line.decode()
            line.extend(byte)
    raise TimeoutError("CPU fixture did not produce its expected line")


def wait_not_live(pid, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sample = direct_child.read_process(pid)
        if sample is None or sample.state in {"Z", "X"}:
            return
        time.sleep(0.05)
    pytest.fail(f"Owned CPU fixture {pid} remained live")


@contextmanager
def watchdog(payload, *, duration=5, extra_args=(), extra_fds=(), parent_start_ticks=None):
    parent = direct_child.read_process(os.getpid())
    read_fd, write_fd = os.pipe()
    argv = [
        sys.executable,
        "-m",
        MODULE,
        "--start-fd",
        str(read_fd),
        "--parent-pid",
        str(os.getpid()),
        "--parent-start-ticks",
        str(parent.start_ticks if parent_start_ticks is None else parent_start_ticks),
        "--boot-id",
        direct_child.BOOT_PATH.read_text().strip(),
        "--deadline-monotonic",
        str(time.monotonic() + duration),
        "--grace-seconds",
        "0.3",
        *extra_args,
        "--",
        sys.executable,
        "-c",
        payload,
    ]
    child = subprocess.Popen(
        argv,
        pass_fds=(read_fd, *extra_fds),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.close(read_fd)
    try:
        yield child, write_fd
    finally:
        with suppress(OSError):
            os.close(write_fd)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # This is our still-unreaped child and dedicated group leader.
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
        child.stdout.close()
        child.stderr.close()


@pytest.mark.parametrize("cpu_ids", ["", "0,0", "-1", "0,2", "x"])
def test_invalid_affinity_never_changes_current_mask(monkeypatch, cpu_ids):
    monkeypatch.setattr(direct_child.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(
        direct_child.os, "sched_setaffinity", lambda *_: pytest.fail("invalid CPU mask applied"), raising=False
    )
    with pytest.raises(ValueError, match="cpu_ids"):
        direct_child.configure_affinity(cpu_ids)


def test_affinity_is_a_subset_and_applied_to_self(monkeypatch):
    applied = []
    monkeypatch.setattr(direct_child.os, "sched_getaffinity", lambda pid: {2, 3, 4}, raising=False)
    monkeypatch.setattr(
        direct_child.os, "sched_setaffinity", lambda pid, cpus: applied.append((pid, cpus)), raising=False
    )
    direct_child.configure_affinity("2,4")
    assert applied == [(0, {2, 4})]


def test_missing_python_pidfd_bindings_use_guarded_exact_handle_syscalls(monkeypatch):
    calls = []

    class FakeSyscall:
        def __call__(self, *args):
            values = tuple(arg.value for arg in args)
            calls.append(values)
            return 17 if values[0] == 434 else 0

    class FakeLibc:
        syscall = FakeSyscall()

    monkeypatch.delattr(os, "pidfd_open", raising=False)
    monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: FakeLibc())
    assert open_pidfd(321) == 17
    assert signal_pidfd(17, signal.SIGKILL) == 0
    assert calls == [(434, 321, 0), (424, 17, signal.SIGKILL, None, 0)]
    monkeypatch.setattr(platform, "machine", lambda: "unverified-architecture")
    with pytest.raises(RuntimeError, match="verified pidfd_open"):
        open_pidfd(321)
    with pytest.raises(RuntimeError, match="verified pidfd_send_signal"):
        signal_pidfd(17, signal.SIGKILL)
    assert len(calls) == 2


def test_pidfd_syscall_errors_preserve_process_lookup_semantics(monkeypatch):
    monkeypatch.setattr(ctypes, "get_errno", lambda: 3)  # Linux ESRCH; cleanup may ignore only a gone process.
    with pytest.raises(ProcessLookupError):
        checked_pidfd_result(-1)


@LINUX
def test_barrier_holds_payload_until_exact_release_and_payload_shares_group():
    payload = "import os; print(str(os.getpid()) + ',' + str(os.getpgrp()), flush=True)"
    with watchdog(payload) as (child, release):
        assert not select.select([child.stdout], [], [], 0.3)[0]
        assert child.poll() is None
        os.write(release, b"1")
        payload_pid, group = map(int, read_line(child.stdout).split(","))
        assert payload_pid != child.pid
        assert group == child.pid
        assert child.wait(timeout=3) == 0


@LINUX
@pytest.mark.parametrize("authorization", [b"", b"0"])
def test_closed_or_invalid_barrier_never_starts_payload(authorization):
    with watchdog("print('PAYLOAD-MUST-NOT-START', flush=True)") as (child, release):
        if authorization:
            os.write(release, authorization)
        else:
            os.close(release)
        assert child.wait(timeout=3) == 125
        assert child.stdout.read() == b""


@LINUX
def test_reused_parent_identity_is_rejected_before_payload():
    wrong_ticks = direct_child.read_process(os.getpid()).start_ticks + 1
    with watchdog("print('PAYLOAD-MUST-NOT-START', flush=True)", parent_start_ticks=wrong_ticks) as (child, _):
        assert child.wait(timeout=3) == 125
        assert child.stdout.read() == b""


@LINUX
def test_affinity_and_explicit_descriptor_reach_payload():
    cpu = min(os.sched_getaffinity(0))
    read_fd, payload_fd = os.pipe()
    try:
        payload = f"import os; os.write({payload_fd}, str(sorted(os.sched_getaffinity(0))).encode())"
        with watchdog(
            payload,
            extra_args=("--cpu-ids", str(cpu), "--pass-fd", str(payload_fd)),
            extra_fds=(payload_fd,),
        ) as (child, release):
            os.write(release, b"1")
            assert child.wait(timeout=3) == 0
            assert os.read(read_fd, 100) == str([cpu]).encode()
    finally:
        os.close(read_fd)
        os.close(payload_fd)


@LINUX
def test_deadline_kills_term_ignoring_group_and_not_an_unrelated_session():
    descendant = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    payload = (
        "import signal,subprocess,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"child=subprocess.Popen([sys.executable,'-c',{descendant!r}]); "
        "print(child.pid, flush=True); time.sleep(30)"
    )
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        with watchdog(payload, duration=1.0) as (child, release):
            os.write(release, b"1")
            descendant_pid = int(read_line(child.stdout))
            assert child.wait(timeout=4) == -signal.SIGKILL
            wait_not_live(descendant_pid)
            assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=3)


@LINUX
def test_signal_stops_cooperative_payload_without_waiting_for_deadline():
    with watchdog("import time; print('ready', flush=True); time.sleep(30)") as (child, release):
        os.write(release, b"1")
        assert read_line(child.stdout) == "ready"
        child.terminate()
        assert child.wait(timeout=3) == 128 + signal.SIGTERM


@LINUX
def test_memory_rss_cap_is_independent_of_parent_monitor():
    with watchdog("import time; time.sleep(30)", extra_args=("--memory-limit-bytes", "1")) as (child, release):
        os.write(release, b"1")
        assert child.wait(timeout=3) == 137


@LINUX
def test_parent_death_stops_payload_even_after_start_authorization():
    # This supervisor intentionally has no cleanup handler, mimicking controller
    # SIGKILL. pidfds make test fallback cleanup target exact processes, not reused PIDs.
    # Check exact-handle capability before starting any descendant fixtures.
    capability = open_pidfd(os.getpid())
    try:
        signal_pidfd(capability, 0)
    finally:
        os.close(capability)
    supervisor_source = f"""
import json, os, subprocess, sys, time
from experiments.keyframe_neighborhood_sampling import direct_child
parent = direct_child.read_process(os.getpid())
read_fd, write_fd = os.pipe()
payload = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(json.dumps({{'payload':os.getpid()}}),flush=True); time.sleep(30)"
payload = "import json; " + payload
guard = subprocess.Popen([
    sys.executable, '-m', {MODULE!r}, '--start-fd', str(read_fd),
    '--parent-pid', str(os.getpid()), '--parent-start-ticks', str(parent.start_ticks),
    '--boot-id', direct_child.BOOT_PATH.read_text().strip(),
    '--deadline-monotonic', str(time.monotonic()+5), '--grace-seconds', '0.3',
    '--', sys.executable, '-c', payload,
], pass_fds=(read_fd,), start_new_session=True)
os.close(read_fd)
print(json.dumps({{'guard':guard.pid}}), flush=True)
os.write(write_fd, b'1')
os.close(write_fd)
time.sleep(30)
"""
    owner = subprocess.Popen([sys.executable, "-c", supervisor_source], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pinned = []
    try:
        guard_pid = json.loads(read_line(owner.stdout))["guard"]
        pinned.append(open_pidfd(guard_pid))
        payload_pid = json.loads(read_line(owner.stdout))["payload"]
        pinned.append(open_pidfd(payload_pid))
        owner.kill()
        owner.wait(timeout=3)
        wait_not_live(guard_pid)
        wait_not_live(payload_pid)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=3)
        for descriptor in pinned:
            with suppress(ProcessLookupError):
                signal_pidfd(descriptor, signal.SIGKILL)
            os.close(descriptor)
        owner.stdout.close()
        owner.stderr.close()
