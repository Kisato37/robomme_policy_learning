"""CPU-only architecture lifecycle fixtures; no model, subprocess or GPU launch."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from experiments.uniform_keyframe_expansion import architecture_controller as a
from experiments.uniform_keyframe_expansion.artifacts import ExpansionRunStore, _write
from tests.uniform_keyframe_expansion.test_episode_integration import run_provenance
from tests.uniform_keyframe_expansion.test_resident_controller import fake_plan


def plan_fixture(root):
    plan = fake_plan(root)
    plan.stage, plan.rows = "architecture_smoke", []
    return plan


def test_unvalidated_mapping_is_not_authority():
    with pytest.raises(TypeError):
        a._runtime_plan({"stage": "architecture_smoke", "approved": True})


def test_actual_plan_stage_isolation(tmp_path, monkeypatch):
    from tests.uniform_keyframe_expansion.test_launch_contract import Fixture
    from experiments.uniform_keyframe_expansion.launch_contract import validate_execution_plan
    f = Fixture(tmp_path, monkeypatch)
    cpu = validate_execution_plan(f.plan())
    with pytest.raises(ValueError):
        a._runtime_plan(cpu)
    f.cpu()
    architecture = validate_execution_plan(f.plan("architecture_smoke"))
    assert a._runtime_plan(architecture) is architecture
    f.architecture()
    trajectory = validate_execution_plan(f.plan("end_to_end_smoke"))
    with pytest.raises(ValueError):
        a._runtime_plan(trajectory)


def test_dry_cli_has_no_runtime_side_effects(tmp_path, monkeypatch, capsys):
    plan = plan_fixture(tmp_path)
    monkeypatch.setattr(a, "_runtime_plan", lambda value: value)
    monkeypatch.setattr(a.lifecycle, "load_plan", lambda _: plan)
    def forbidden(*args, **kwargs):
        pytest.fail("Dry-run cannot enter execution or GPU calls")
    monkeypatch.setattr(a, "execute_architecture", forbidden)
    monkeypatch.setattr(a, "run_architecture_worker", forbidden)
    monkeypatch.setattr(a.lifecycle.subprocess, "Popen", forbidden)
    monkeypatch.setattr(a.lifecycle.subprocess, "check_output", forbidden)
    assert a.main(["--plan", "/fixture/plan.json"]) == 0
    assert json.loads(capsys.readouterr().out)["launches_processes"] is False
    assert list(tmp_path.iterdir()) == []


def test_command_has_no_trajectory_or_socket_arguments(tmp_path):
    plan = plan_fixture(tmp_path)
    plan.environment["roles"]["policy"]["command_prefix"] = ["/bin/bash", "/fixture/approved.sh"]
    command = a._command(plan, tmp_path / "execution_plan.json")
    assert command[:2] == ["/bin/bash", "/fixture/approved.sh"]
    assert command[-3:] == ["--execute", "--role", "architecture-worker"]
    assert not any(flag in command for flag in ("--row-id", "--port", "--listen-fd"))


def test_nonlinux_and_source_failures_do_not_query_gpu_or_write_run(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "_runtime_plan", lambda value: value)
    plan = plan_fixture(tmp_path / "absent")
    monkeypatch.setattr(a.sys, "platform", "darwin")
    monkeypatch.setattr(a.lifecycle, "check_gpu_admission", lambda _: pytest.fail("No GPU access"))
    with pytest.raises(RuntimeError, match="Linux"):
        a.execute_architecture(plan)
    monkeypatch.setattr(a.sys, "platform", "linux")
    def refuse(_):
        raise ValueError("source changed")
    monkeypatch.setattr(a.lifecycle, "verify_sources", refuse)
    with pytest.raises(ValueError, match="source changed"):
        a.execute_architecture(plan)
    assert not plan.store_root.exists()


@pytest.mark.parametrize("fault", [None, "start", "wait", "cleanup", "wait_cleanup", "admission", "publish"])
def test_controller_single_owned_worker_cleanup_and_write_once(tmp_path, monkeypatch, fault):
    from experiments.keyframe_neighborhood_sampling import direct_runtime
    root = tmp_path / "uniform_keyframe_expansion" / "architecture-fixture"
    ExpansionRunStore.create(root, stage="smoke", run_manifest=run_provenance())
    plan = plan_fixture(root)
    monkeypatch.setattr(a, "_runtime_plan", lambda value: value)
    monkeypatch.setattr(a.sys, "platform", "linux")
    monkeypatch.setattr(a.lifecycle, "verify_sources", lambda _: {"cpu_fixture": True})
    monkeypatch.setattr(a.lifecycle, "host_lock_directory", lambda _: tmp_path / "host-locks")
    events = []
    class Lease:
        pass_fds = (8,)
        def __init__(self, *args): pass
        def __enter__(self):
            events.append("lease_start")
            return self
        def __exit__(self, *args): events.append("lease_end")
    class Supervisor:
        def __init__(self, *args): pass
        def start(self, name, role, command, deadline):
            assert name == "architecture" and role == "policy"
            events.append("start")
            if fault == "start": raise RuntimeError("fixture start failure")
        def close(self):
            events.append("close")
            if fault in {"cleanup", "wait_cleanup"}: raise RuntimeError("fixture cleanup failure")
    def admission(_):
        events.append("admission")
        if fault == "admission": raise RuntimeError("fixture resource failure")
        return {"cpu_fixture": True}
    def wait(*args, **kwargs):
        events.append("wait")
        if fault in {"wait", "wait_cleanup"}: raise RuntimeError("fixture worker failure")
    def publish(p, directory):
        events.append("publish")
        assert (directory / "controller_complete.json").is_file()
        assert events[-2] == "lease_end"
        if fault == "publish": raise RuntimeError("fixture numerical gate rejected")
        return directory / "architecture_gate.json"  # CPU fixture creates no PASS report.
    monkeypatch.setattr(direct_runtime, "GpuLease", Lease)
    monkeypatch.setattr(a.lifecycle, "Supervisor", Supervisor)
    monkeypatch.setattr(a.lifecycle, "check_gpu_admission", admission)
    monkeypatch.setattr(a, "_wait_worker", wait)
    monkeypatch.setitem(sys.modules, "experiments.uniform_keyframe_expansion.architecture_artifacts",
                        SimpleNamespace(publish_architecture_gate=publish))
    if fault:
        with pytest.raises(RuntimeError): a.execute_architecture(plan)
    else:
        a.execute_architecture(plan)
    directory = a._directory(plan)
    assert events.count("start") == (0 if fault == "admission" else 1)
    assert events.count("close") == (0 if fault == "admission" else 1)
    assert (directory / "controller_complete.json").exists() == (fault in {None, "publish"})
    assert (directory / "controller_failure.json").exists() == bool(fault)
    assert not (directory / "architecture_gate.json").exists()
    assert events.count("publish") == (1 if fault in {None, "publish"} else 0)
    if fault == "wait_cleanup":
        failure = json.loads((directory / "controller_failure.json").read_text())
        assert failure["message"] == "fixture worker failure"
        assert "cleanup failure" in failure["notes"][0]
    with pytest.raises(FileExistsError): a.execute_architecture(plan)


@pytest.mark.parametrize("fault", [None, "ownership", "bootstrap", "probe", "publish_measurement"])
def test_worker_import_order_single_load_artifacts_and_failure(tmp_path, monkeypatch, fault):
    plan = plan_fixture(tmp_path)
    directory = a._directory(plan)
    directory.mkdir(parents=True)
    monkeypatch.setattr(a, "_runtime_plan", lambda value: value)
    events = []
    policy = object()
    def ownership(p, role):
        events.append("ownership")
        assert role == "architecture"
        if fault == "ownership": raise ValueError("not an owned worker")
        return {"execution_identity": plan.execution_identity, "role": "architecture", "worker_ownership_verified": True}
    def load(p):
        events.append("load")
        assert events[0] == "ownership"
        if fault == "bootstrap": raise RuntimeError("strict load failed")
        return SimpleNamespace(policy=policy, provenance={"policy_execution_identity": plan.execution_identity})
    def probe(actual, p, *, output_dir):
        events.append("probe")
        assert actual is policy and output_dir == directory / "probe"
        assert (directory / "live_policy_provenance.json").is_file()
        if fault == "probe": raise RuntimeError("non-finite model output")
        return {"cpu_fixture_only": True}
    monkeypatch.setitem(sys.modules, "experiments.uniform_keyframe_expansion.worker_ownership",
                        SimpleNamespace(verify_worker_ownership=ownership))
    monkeypatch.setitem(sys.modules, "experiments.uniform_keyframe_expansion.server_bootstrap",
                        SimpleNamespace(load_authorized_policy=load))
    monkeypatch.setitem(sys.modules, "experiments.uniform_keyframe_expansion.architecture_probe",
                        SimpleNamespace(run_loaded_probe=probe))
    original_write = a._write
    def write(path, value):
        if fault == "publish_measurement" and path.name == "architecture_measurements.json":
            raise OSError("fixture output error")
        return original_write(path, value)
    monkeypatch.setattr(a, "_write", write)
    if fault:
        with pytest.raises((ValueError, RuntimeError, OSError)): a.run_architecture_worker(plan)
    else:
        assert a.run_architecture_worker(plan) == {"cpu_fixture_only": True}
        a._worker_evidence(plan, directory)
    assert events.count("load") <= 1 and events.count("probe") <= 1
    if fault == "ownership":
        assert list(directory.iterdir()) == []
    else:
        assert (directory / "architecture_worker_failure.json").exists() == bool(fault)
    assert (directory / "architecture_worker_complete.json").exists() == (fault is None)


@pytest.mark.parametrize("status", [0, 1])
def test_worker_wait_does_not_need_a_resident_server(status, monkeypatch, tmp_path):
    calls = []
    supervisor = SimpleNamespace(stop_event=SimpleNamespace(is_set=lambda: False),
        children={"architecture": SimpleNamespace(reap_if_finished=lambda: status)},
        record_exit=lambda *args: calls.append(args))
    monkeypatch.setattr(a.time, "monotonic", lambda: 10)
    monkeypatch.setattr(a, "_worker_evidence", lambda *args: {"cpu_fixture": True})
    if status:
        with pytest.raises(RuntimeError, match="no automatic retry"):
            a._wait_worker(plan_fixture(tmp_path), supervisor, tmp_path, startup_deadline=20, deadline=30)
    else:
        assert a._wait_worker(plan_fixture(tmp_path), supervisor, tmp_path, startup_deadline=20, deadline=30)["cpu_fixture"]
    assert calls == [("architecture", status)]


@pytest.mark.parametrize("mode", ["interrupted", "deadline", "startup"])
def test_wait_interruptions_and_startup_timeout(mode, monkeypatch, tmp_path):
    supervisor = SimpleNamespace(stop_event=SimpleNamespace(is_set=lambda: mode == "interrupted"),
        children={"architecture": SimpleNamespace(reap_if_finished=lambda: None)})
    monkeypatch.setattr(a.time, "monotonic", lambda: 10)
    with pytest.raises((InterruptedError, TimeoutError)):
        a._wait_worker(plan_fixture(tmp_path), supervisor, tmp_path, startup_deadline=9,
                       deadline=9 if mode == "deadline" else 20)


def test_execution_directory_rejects_symlinks(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        a._directory(plan_fixture(alias))
