"""Actual Linux CPU-process checks; no CUDA, model, or experimental dispatch."""

import os
import signal
import subprocess
import sys

import pytest

from experiments.keyframe_neighborhood_sampling.direct_runtime import LinuxProcessOperations
from experiments.keyframe_neighborhood_sampling.direct_runtime import OwnedProcessGroup

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Requires real Linux /proc process identity")


def test_actual_linux_owned_child_terminates_without_touching_supervisor():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    owner = None
    try:
        assert child.stdout.readline().strip() == "ready"
        owner = OwnedProcessGroup(child, LinuxProcessOperations())
        assert owner.identity.parent_pid == os.getpid()
        assert owner.identity.uid == os.getuid()
        assert owner.reap_if_finished() is None
        assert owner.stop(grace_seconds=1.0, kill_wait_seconds=1.0) == -signal.SIGTERM
        assert owner.reap_if_finished() == -signal.SIGTERM
        assert LinuxProcessOperations().identity(child.pid) is None
    finally:
        if owner is None or owner.returncode is None:
            # Only this unreaped test child can own this PID here.
            child.kill()
            child.wait(timeout=5)
        child.stdout.close()
