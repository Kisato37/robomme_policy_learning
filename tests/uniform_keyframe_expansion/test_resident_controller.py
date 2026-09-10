"""CPU-only controller wiring; production process ownership is Linux-gated."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.uniform_keyframe_expansion import resident_controller as r
from experiments.uniform_keyframe_expansion import contract
from experiments.uniform_keyframe_expansion.artifacts import ExpansionRunStore
from tests.uniform_keyframe_expansion.test_episode_integration import run_provenance

GPU = "GPU-11111111-1111-4111-8111-111111111111"


def fake_plan(root):
    role = {"python_executable": "/fixture/python", "command_prefix": [],
            "process_environment": {"CUDA_VISIBLE_DEVICES": GPU, "PYTHONPATH": "/fixture/source"}}
    return SimpleNamespace(
        stage="end_to_end_smoke", gpu_uuid=GPU,
        policy_root=Path("/fixture/source"), store_root=root,
        rows=contract.build_smoke_matrix()["rows"][:2],
        execution_identity={"execution_id": "11111111-1111-4111-8111-111111111111", "dispatch_sha256": "e" * 64},
        environment={"roles": {"policy": deepcopy(role), "simulator": deepcopy(role)},
                     "hardware": {"minimum_free_memory_mib": 16000}},
        payload={"fixture": True}, revalidate=lambda: None,
    )


def test_default_cli_only_inspects_no_process_or_gpu(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(r, "load_plan", lambda _: fake_plan(tmp_path))
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not start processes or inspect GPU")
    monkeypatch.setattr(r, "execute_resident", forbidden)
    monkeypatch.setattr(r, "run_child", forbidden)
    monkeypatch.setattr(r.subprocess, "Popen", forbidden)
    monkeypatch.setattr(r.subprocess, "check_output", forbidden)
    assert r.main(["--plan", "/fixture/plan.json"]) == 0
    assert json.loads(capsys.readouterr().out)["launches_processes"] is False


def test_plain_mapping_cannot_bypass_validated_plan_requirement():
    with pytest.raises(TypeError):
        r._runtime_plan({"stage": "end_to_end_smoke", "approved": True})


def test_child_environment_is_separate_and_exact_gpu_bound(tmp_path, monkeypatch):
    monkeypatch.setenv("UNRELATED_FIXTURE_VARIABLE", "retained")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    plan = fake_plan(tmp_path)
    environment = r._environment(plan, "policy")
    assert environment["CUDA_VISIBLE_DEVICES"] == GPU
    assert environment["UNRELATED_FIXTURE_VARIABLE"] == "retained"
    assert r.os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    plan.environment["roles"]["policy"]["process_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    with pytest.raises(ValueError, match="physical GPU"):
        r._environment(plan, "policy")


def test_stale_old_dispatch_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYFRAME_RUNNER_BACKEND", "direct")
    with pytest.raises(ValueError, match="old experiment"):
        r._environment(fake_plan(tmp_path), "simulator")


def test_commands_keep_role_and_row_listener_bindings(tmp_path):
    plan = fake_plan(tmp_path)
    server = r._command(plan, "policy", tmp_path / "plan.json", 19000, listen_fd=9)
    assert server[-2:] == ["--listen-fd", "9"] and "--execute" in server
    evaluator = r._command(plan, "simulator", tmp_path / "plan.json", 19000, row_id=1)
    assert evaluator[-2:] == ["--row-id", "1"]
    with pytest.raises(ValueError):
        r._command(plan, "policy", tmp_path / "plan.json", 19000)
    with pytest.raises(ValueError):
        r._command(plan, "simulator", tmp_path / "plan.json", 19000, row_id=8)


@pytest.mark.parametrize("free,minimum,okay", [(17000, 16000, True), (16000, 16000, True),
                                            (15999, 16000, False), (32000, 0, False)])
def test_shared_gpu_admission_uses_memory_not_idle_rule(tmp_path, monkeypatch, free, minimum, okay):
    plan = fake_plan(tmp_path)
    plan.environment["hardware"]["minimum_free_memory_mib"] = minimum
    commands = []
    def output(command, **kwargs):
        commands.append(command)
        return f"{GPU}, {free}\n"
    monkeypatch.setattr(r.subprocess, "check_output", output)
    if okay:
        assert r.check_gpu_admission(plan)["shared_gpu_allowed"] is True
        assert "--query-gpu=uuid,memory.free" in commands[0]
        assert not any("utilization" in arg for arg in commands[0])
    else:
        with pytest.raises((ValueError, RuntimeError)):
            r.check_gpu_admission(plan)


def test_existing_attempt_never_inferred_as_retryable(tmp_path):
    store = ExpansionRunStore.create(tmp_path / "uniform_keyframe_expansion" / "no-retry",
                                    stage="smoke", run_manifest=run_provenance())
    rows = contract.build_smoke_matrix()["rows"][:2]
    r._check_rows_unused(store, rows)
    store.new_attempt(rows[0], 0)
    with pytest.raises(RuntimeError, match="review"):
        r._check_rows_unused(store, rows)


@pytest.mark.parametrize("fault", [None, "readiness", "row", "scientific_error", "cleanup", "row_and_cleanup"])
def test_controller_residency_cleanup_and_no_retry(tmp_path, monkeypatch, fault):
    from experiments.keyframe_neighborhood_sampling import direct_runtime
    root = tmp_path / "uniform_keyframe_expansion" / "controller-fixture"
    ExpansionRunStore.create(root, stage="smoke", run_manifest=run_provenance())
    plan = fake_plan(root)
    monkeypatch.setattr(r, "_runtime_plan", lambda value: value)
    monkeypatch.setattr(r.sys, "platform", "linux")
    monkeypatch.setattr(r, "check_gpu_admission", lambda _: {"cpu_fixture": True})
    monkeypatch.setattr(r, "verify_sources", lambda _: {"cpu_fixture": True})
    monkeypatch.setattr(r, "verify_deep_evidence", lambda _: {"cpu_fixture": True})
    monkeypatch.setattr(r, "host_lock_directory", lambda _: tmp_path / "host-locks")
    events = []
    class Lease:
        def __init__(self, *args):
            self.pass_fds = (8,)
        def __enter__(self):
            events.append("lease_start")
            return self
        def __exit__(self, *args):
            events.append("lease_end")
    class Port(Lease):
        def __init__(self):
            self.port, self.pass_fds = 19000, (9,)
    class Supervisor:
        def __init__(self, *args):
            pass
        def start(self, name, role, *args, **kwargs):
            events.append("start:" + name)
        def wait_row(self, name, deadline):
            events.append("wait:" + name)
            if fault in {"row", "row_and_cleanup"}:
                raise RuntimeError("fixture row failed")
        def close(self):
            events.append("close")
            if fault in {"cleanup", "row_and_cleanup"}:
                raise RuntimeError("fixture cleanup uncertain")
    def ready(*args):
        if fault == "readiness":
            raise TimeoutError("fixture readiness")
        return {"fixture": True}
    monkeypatch.setattr(direct_runtime, "GpuLease", Lease)
    monkeypatch.setattr(direct_runtime, "PortReservation", Port)
    monkeypatch.setattr(r, "Supervisor", Supervisor)
    monkeypatch.setattr(r, "_wait_ready", ready)
    monkeypatch.setattr(r.ExpansionRunStore, "audit_attempt", lambda *args: {
        "status": "complete", "smoke_readiness_pass": fault != "scientific_error",
    })
    if fault:
        with pytest.raises((RuntimeError, TimeoutError)):
            r.execute_resident(plan)
    else:
        result = r.execute_resident(plan)
        assert (result / "controller_complete.json").is_file()
    directory = root / "executions" / plan.execution_identity["execution_id"]
    assert events.count("start:policy") == 1
    assert events.count("close") == 1 and events[-1] == "lease_end"
    for row in plan.rows:
        assert events.count(f"start:row_{row['row_id']:04d}") <= 1
    if fault:
        assert (directory / "controller_failure.json").is_file()
        assert not (directory / "controller_complete.json").exists()
    if fault == "row_and_cleanup":
        failure = json.loads((directory / "controller_failure.json").read_text())
        assert failure["message"] == "fixture row failed"
        assert "fixture cleanup uncertain" in failure["notes"][0]
    if fault in ("readiness", "row", "scientific_error"):
        assert "start:row_0001" not in events
    with pytest.raises(FileExistsError):
        r.execute_resident(plan)


def test_actual_controller_refuses_nonlinux_before_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "_runtime_plan", lambda plan: plan)
    monkeypatch.setattr(r.sys, "platform", "darwin")
    def forbidden(*args):
        raise AssertionError("GPU must not be queried")
    monkeypatch.setattr(r, "check_gpu_admission", forbidden)
    with pytest.raises(RuntimeError, match="Linux"):
        r.execute_resident(fake_plan(tmp_path))


@pytest.mark.parametrize("role", ["policy", "simulator"])
def test_child_rejects_unbound_listener_or_row_before_real_bootstrap(tmp_path, monkeypatch, role):
    monkeypatch.setattr(r, "_runtime_plan", lambda plan: plan)
    kwargs = {"listen_fd": None} if role == "policy" else {"row_id": 999}
    with pytest.raises(ValueError):
        r.run_child(fake_plan(tmp_path), role, port=19000, **kwargs)


def test_source_gate_rejects_before_gpu_and_run_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "_runtime_plan", lambda value: value)
    monkeypatch.setattr(r.sys, "platform", "linux")
    def refuse(_):
        raise RuntimeError("fixture source changed")
    def forbidden(_):
        raise AssertionError("No GPU query allowed after failed source gate")
    monkeypatch.setattr(r, "verify_sources", refuse)
    monkeypatch.setattr(r, "check_gpu_admission", forbidden)
    with pytest.raises(RuntimeError, match="source changed"):
        r.execute_resident(fake_plan(tmp_path / "absent"))
    assert not (tmp_path / "absent").exists()


def test_host_lock_directory_is_cross_run_owned_and_private(tmp_path):
    plan = fake_plan(tmp_path / "run-one")
    locks = tmp_path / "shared-locks"
    locks.mkdir(mode=0o700)
    plan.environment["lock_directory"] = str(locks)
    assert r.host_lock_directory(plan) == locks
    plan.store_root = tmp_path / "run-two"
    assert r.host_lock_directory(plan) == locks
    locks.chmod(0o777)
    with pytest.raises(ValueError, match="user-owned"):
        r.host_lock_directory(plan)
    locks.chmod(0o700)
    plan.store_root = tmp_path
    with pytest.raises(ValueError, match="across runs"):
        r.host_lock_directory(plan)
