"""CPU mocks for the production bootstrap, not GPU/checkpoint readiness proof."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from experiments.uniform_keyframe_expansion import server_bootstrap as b
from experiments.uniform_keyframe_expansion.contract import FORMAL_TASKS, build_smoke_matrix


ROOT = Path(__file__).resolve().parents[2]
GPU = "GPU-11111111-1111-4111-8111-111111111111"
IDENTITY = {"execution_id": "11111111-1111-4111-8111-111111111111", "dispatch_sha256": "a" * 64}


class FixturePlan:
    """Mocks only the already validated file-chain layer; no real authority."""
    def __init__(self, stage="end_to_end_smoke"):
        self.stage = stage
        self.policy_root = ROOT
        self.benchmark_root = ROOT / "third_party/robomme_benchmark"
        self.gpu_uuid = GPU
        self.checkpoint_dir = ROOT / "not-a-real-checkpoint/79999"
        self.execution_identity = deepcopy(IDENTITY)
        self.rows = tuple(build_smoke_matrix()["rows"][:3])
        self.policy_source = {"root": str(ROOT), "revision": "a" * 40, "branch": "exp/fixture"}
        self.benchmark_source = {"root": str(self.benchmark_root), "revision": "b" * 40, "branch": "HEAD"}
        self.revalidations = []
        roles = {}
        for role, packages in b._REQUIRED_PACKAGES.items():
            env = {"CUDA_VISIBLE_DEVICES": GPU, "PYTHONPATH": "/recorded/src"}
            if role == "policy":
                env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            roles[role] = {"python_executable": sys.executable, "python_version": "3.11.99",
                           "packages": {name: "1.2.3" for name in packages}, "process_environment": env}
        self.environment = {"host": "fixture-host", "roles": roles,
                            "hardware": {"gpu_uuid": GPU, "gpu_name": "FixtureGPU", "driver_version": "999"}}

    def require_runtime_stage(self):
        if self.stage == "cpu_prepare":
            raise ValueError("No GPU authority")

    def revalidate(self):
        self.revalidations.append("small_file_chain")
        return self


@pytest.fixture
def plan(monkeypatch):
    monkeypatch.setitem(sys.modules, "experiments.uniform_keyframe_expansion.launch_contract",
                        SimpleNamespace(ValidatedExecutionPlan=FixturePlan))
    return FixturePlan()


def fail_if_called(*args, **kwargs):
    pytest.fail("Production import/load/connection must not happen")


def test_module_import_has_no_production_backend_import_or_gpu_commands(monkeypatch):
    def forbidden(name, *args, **kwargs):
        if name in {"jax", "torch", "sapien", "robomme"}:
            pytest.fail("Eager production import")
        return original(name, *args, **kwargs)
    original = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", forbidden)
    monkeypatch.setattr(b.subprocess, "run", fail_if_called)
    importlib.reload(b)


@pytest.mark.parametrize("stage", ["cpu_prepare", "architecture_smoke"])
def test_non_episode_authority_never_imports_simulator(plan, monkeypatch, stage):
    plan.stage = stage
    monkeypatch.setattr(b, "_benchmark_modules", fail_if_called)
    monkeypatch.setattr(b, "_fresh_backends", lambda: None)
    with pytest.raises((ValueError, b.BootstrapError)):
        b.prepare_benchmark_runtime(plan, policy_port=12345)


def test_bare_approval_dict_cannot_load_model(plan, monkeypatch):
    monkeypatch.setattr(b, "_fresh_backends", fail_if_called)
    with pytest.raises(b.BootstrapError, match="validated execution plan"):
        b.load_authorized_policy({"approved": True})


def test_existing_backend_is_rejected_before_environment_changes(monkeypatch):
    monkeypatch.setitem(sys.modules, "sapien", SimpleNamespace())
    with pytest.raises(b.BootstrapError, match="fresh process"):
        b._fresh_backends()


@pytest.mark.parametrize("role,verify", [("policy", True), ("simulator", False)])
def test_preflight_revalidates_then_checks_actual_sources_environment_gpu(plan, monkeypatch, role, verify):
    events = []
    monkeypatch.setattr(b, "_fresh_backends", lambda: events.append("fresh"))
    monkeypatch.setattr(b, "_live_repository", lambda source, **kw: events.append(kw["role"]) or dict(source))
    monkeypatch.setattr(b, "_live_environment", lambda p, r: events.append(r + "env") or {})
    monkeypatch.setattr(b, "_gpu_inventory", lambda p: events.append("gpu") or {})
    monkeypatch.setattr(b, "_live_checkpoint", lambda p: events.append("weight_bytes") or {})
    checked, evidence = b._preflight(plan, role)
    assert checked is plan and plan.revalidations == ["small_file_chain"]
    assert events == ["fresh", "policy", "benchmark", role + "env", "gpu"] + (["weight_bytes"] if verify else [])
    assert evidence["checkpoint_content_verified_in_this_process"] is verify


def test_controller_source_gate_has_no_worker_import_gpu_or_weights(plan, monkeypatch):
    monkeypatch.setattr(b, "_live_repository", lambda source, **kw: {"role": kw["role"], "clean": True})
    for name in ("_fresh_backends", "_live_environment", "_gpu_inventory", "_live_checkpoint", "_benchmark_modules"):
        monkeypatch.setattr(b, name, fail_if_called)
    assert set(b.verify_controller_sources(plan)) == {"policy", "benchmark"}
    assert plan.revalidations == ["small_file_chain"]
    plan.stage = "cpu_prepare"
    with pytest.raises(ValueError):
        b.verify_controller_sources(plan)


def test_evidence_revalidation_failure_precedes_any_actual_import_or_child(plan, monkeypatch):
    def reject():
        raise ValueError("changed source evidence")
    monkeypatch.setattr(plan, "revalidate", reject)
    monkeypatch.setattr(b, "_live_repository", fail_if_called)
    monkeypatch.setattr(b, "_benchmark_modules", fail_if_called)
    with pytest.raises(ValueError, match="changed source evidence"):
        b.verify_controller_sources(plan)


def test_real_validated_plan_interface_and_live_streaming_gate(tmp_path, monkeypatch):
    # These are synthetic approval/source/weight files, never real authority.
    from tests.uniform_keyframe_expansion.test_launch_contract import Fixture
    from experiments.uniform_keyframe_expansion import launch_contract as lc

    fixture = Fixture(tmp_path, monkeypatch)
    cpu_plan = lc.validate_execution_plan(fixture.plan())
    with pytest.raises(lc.ExpansionLaunchError):
        b.verify_controller_sources(cpu_plan)
    fixture.cpu()
    checked = lc.validate_execution_plan(fixture.plan("architecture_smoke"))
    with pytest.raises(b.BootstrapError, match="Architecture"):
        b.prepare_benchmark_runtime(checked, policy_port=12345)
    monkeypatch.setattr(b, "__file__", str(checked.policy_root / "experiments/uniform_keyframe_expansion/server_bootstrap.py"))
    monkeypatch.setattr(b, "_fresh_backends", lambda: None)
    monkeypatch.setattr(b, "_live_repository", lambda source, **kw: dict(source))
    monkeypatch.setattr(b, "_live_environment", lambda p, role: {})
    monkeypatch.setattr(b, "_gpu_inventory", lambda p: {})
    _, proof = b._preflight(checked, "policy")
    assert proof["checkpoint_content_verified_in_this_process"] is True
    assert proof["checkpoint_content_evidence"]["content_tree"] == checked.checkpoint["content_tree"]
    (checked.policy_root / "module.py").write_text("# mutated source after validation\n")
    monkeypatch.setattr(b, "_live_checkpoint", fail_if_called)
    with pytest.raises(lc.ExpansionLaunchError):
        b._preflight(checked, "policy")


@pytest.mark.parametrize("fault", ["dirty", "revision", "branch", "wrong_root"])
def test_live_git_checks_reject_report_only_claims(plan, monkeypatch, fault):
    source = plan.policy_source
    def command(args):
        if args[-1] == "--show-toplevel":
            return "/private/tmp" if fault == "wrong_root" else str(ROOT)
        if args[-1] == "HEAD":
            return "c" * 40 if fault == "revision" else source["revision"]
        if args[-1] == "--show-current":
            return "main" if fault == "branch" else source["branch"]
        return " M src/file.py" if fault == "dirty" else ""
    monkeypatch.setattr(b, "_command", command)
    with pytest.raises(b.BootstrapError):
        b._live_repository(source, role="policy")


def test_detached_benchmark_pin_is_preserved(plan, monkeypatch):
    source = plan.benchmark_source
    def command(args):
        if args[-1] == "--show-toplevel":
            return source["root"]
        if args[-1] == "HEAD":
            return source["revision"]
        return ""
    monkeypatch.setattr(b, "_command", command)
    assert b._live_repository(source, role="benchmark")["branch"] == "HEAD"


@pytest.mark.parametrize("mutate", [None, "archive", "weights"])
def test_live_checkpoint_streams_bytes_and_rejects_equal_size_changes(tmp_path, monkeypatch, mutate):
    from experiments.uniform_keyframe_expansion import contract
    from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_content_tree_identity

    directory = tmp_path / "79999"
    directory.mkdir()
    weights = directory / "weights.bin"
    weights.write_bytes(b"original-weights")
    archive = tmp_path / "79999.zip"
    archive.write_bytes(b"fixture-archive")
    checksum = hashlib.sha256(b"fixture-archive").hexdigest()
    plan = SimpleNamespace(checkpoint_dir=directory, checkpoint={
        "archive": {"path": str(archive), "sha256": checksum},
        "content_tree": checkpoint_content_tree_identity(directory),
    })
    monkeypatch.setattr(contract, "CHECKPOINT_ARCHIVE_SHA256", checksum)
    if mutate == "archive":
        archive.write_bytes(b"changed-archive")
    elif mutate == "weights":
        weights.write_bytes(b"modified-weights")
    monkeypatch.setattr(Path, "read_bytes", fail_if_called)
    if mutate:
        with pytest.raises(b.BootstrapError):
            b._live_checkpoint(plan)
    else:
        assert b._live_checkpoint(plan)["archive_sha256"] == checksum


def mock_live_environment(plan, monkeypatch, role):
    monkeypatch.setattr(b.sys, "platform", "linux")
    monkeypatch.setattr(b.socket, "gethostname", lambda: "fixture-host")
    monkeypatch.setattr(b.platform, "python_version", lambda: "3.11.99")
    monkeypatch.setattr(b.importlib.metadata, "version", lambda _: "1.2.3")
    monkeypatch.setattr(b.os, "environ", dict(plan.environment["roles"][role]["process_environment"]))


@pytest.mark.parametrize("fault", [None, "unrecorded_flag", "different_cuda", "version", "hostname", "preallocate", "missing_package"])
def test_live_environment_exact_and_no_mutation(plan, monkeypatch, fault):
    mock_live_environment(plan, monkeypatch, "policy")
    if fault == "unrecorded_flag":
        b.os.environ["XLA_FLAGS"] = "--unsafe-change"
    elif fault == "different_cuda":
        b.os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif fault == "version":
        monkeypatch.setattr(b.importlib.metadata, "version", lambda _: "9.9.9")
    elif fault == "hostname":
        monkeypatch.setattr(b.socket, "gethostname", lambda: "other-host")
    elif fault == "preallocate":
        b.os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE")
        plan.environment["roles"]["policy"]["process_environment"].pop("XLA_PYTHON_CLIENT_PREALLOCATE")
    elif fault == "missing_package":
        plan.environment["roles"]["policy"]["packages"].pop("jaxlib")
    before = dict(b.os.environ)
    if fault:
        with pytest.raises(b.BootstrapError):
            b._live_environment(plan, "policy")
    else:
        assert b._live_environment(plan, "policy")["role"] == "policy"
    assert b.os.environ == before


def declare_simulator_import_environment(plan):
    profile = plan.environment["roles"]["simulator"]
    profile["process_environment"]["LD_LIBRARY_PATH"] = "/recorded/graphics:/recorded/other"
    profile["post_import_process_environment"] = {
        "LD_LIBRARY_PATH": "/recorded/opencv/lib64:/recorded/graphics:/recorded/other",
        "SAPIEN_PACKAGE_PATH": "/recorded/site-packages/sapien",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }
    return profile


def test_recorded_import_transition_preserves_both_phases_without_mutation(plan, monkeypatch):
    profile = declare_simulator_import_environment(plan)
    original = deepcopy(profile)
    mock_live_environment(plan, monkeypatch, "simulator")
    before = dict(b.os.environ)
    assert b._live_environment(plan, "simulator")["process_environment"] == before
    with pytest.raises(b.BootstrapError, match="process environment"):
        b._live_environment(plan, "simulator", phase="after_benchmark_import")
    assert dict(b.os.environ) == before
    b.os.environ.update(profile["post_import_process_environment"])
    after = dict(b.os.environ)
    evidence = b._live_environment(plan, "simulator", phase="after_benchmark_import")
    assert evidence["process_environment"] == after
    assert evidence["phase"] == "after_benchmark_import"
    with pytest.raises(b.BootstrapError, match="process environment"):
        b._live_environment(plan, "simulator")
    assert profile == original and dict(b.os.environ) == after


@pytest.mark.parametrize("fault", ["undeclared", "library", "package", "logging", "cuda", "xla", "vulkan"])
def test_post_import_does_not_silently_accept_live_changes(plan, monkeypatch, fault):
    profile = declare_simulator_import_environment(plan)
    mock_live_environment(plan, monkeypatch, "simulator")
    b.os.environ.update(profile["post_import_process_environment"])
    if fault == "undeclared":
        del profile["post_import_process_environment"]
    else:
        key = {"library": "LD_LIBRARY_PATH", "package": "SAPIEN_PACKAGE_PATH",
               "logging": "TF_CPP_MIN_LOG_LEVEL", "cuda": "CUDA_VISIBLE_DEVICES",
               "xla": "XLA_FLAGS", "vulkan": "VK_ICD_FILENAMES"}[fault]
        b.os.environ[key] = "unreviewed-change"
    before = dict(b.os.environ)
    with pytest.raises(b.BootstrapError, match="process environment"):
        b._live_environment(plan, "simulator", phase="after_benchmark_import")
    assert dict(b.os.environ) == before


@pytest.mark.parametrize("fault", ["policy", "extra_key", "missing_key", "not_mapping", "not_string",
                                  "relative_package", "wrong_log_level", "discard_launch_path",
                                  "relative_library", "multiple_libraries", "empty_library"])
def test_invalid_import_transition_is_rejected_even_before_import(plan, fault):
    profile = declare_simulator_import_environment(plan)
    post = profile["post_import_process_environment"]
    role = "simulator"
    if fault == "policy":
        role = "policy"
    elif fault == "extra_key":
        post["CUDA_VISIBLE_DEVICES"] = GPU
    elif fault == "missing_key":
        del post["SAPIEN_PACKAGE_PATH"]
    elif fault == "not_mapping":
        profile["post_import_process_environment"] = []
    elif fault == "not_string":
        post["TF_CPP_MIN_LOG_LEVEL"] = 3
    elif fault == "relative_package":
        post["SAPIEN_PACKAGE_PATH"] = "relative/sapien"
    elif fault == "wrong_log_level":
        post["TF_CPP_MIN_LOG_LEVEL"] = "1"
    else:
        post["LD_LIBRARY_PATH"] = {
            "discard_launch_path": "/elsewhere/lib",
            "relative_library": "relative/lib:/recorded/graphics:/recorded/other",
            "multiple_libraries": "/first:/second:/recorded/graphics:/recorded/other",
            "empty_library": ":/recorded/graphics:/recorded/other",
        }[fault]
    with pytest.raises(b.BootstrapError):
        b._expected_process_environment(profile, role)


def test_unknown_environment_phase_rejected(plan):
    profile = declare_simulator_import_environment(plan)
    with pytest.raises(b.BootstrapError, match="Unknown"):
        b._expected_process_environment(profile, "simulator", phase="whatever_is_live")
    with pytest.raises(b.BootstrapError, match="Only the simulator"):
        b._expected_process_environment(plan.environment["roles"]["policy"], "policy", phase="after_benchmark_import")


def test_import_transition_record_validated_by_launch_contract(tmp_path, monkeypatch, plan):
    from tests.uniform_keyframe_expansion.test_launch_contract import Fixture
    from experiments.uniform_keyframe_expansion import launch_contract as lc

    fixture = Fixture(tmp_path, monkeypatch)
    profile = declare_simulator_import_environment(plan)
    fixture.environment["roles"]["simulator"].update({
        "process_environment": profile["process_environment"],
        "post_import_process_environment": profile["post_import_process_environment"],
    })
    ref = fixture.write("recorded-import-environment.json", fixture.environment)
    # The pre-GPU CPU schema gate has no selected GPU yet in this fixture.
    assert lc._environment(ref, "cpu-fixture", None)["roles"]["simulator"] == fixture.environment["roles"]["simulator"]
    fixture.environment["roles"]["simulator"]["post_import_process_environment"]["JAX_PLATFORMS"] = "cpu"
    ref = fixture.write("invalid-import-environment.json", fixture.environment)
    with pytest.raises(lc.ExpansionLaunchError, match="import-environment"):
        lc._environment(ref, "cpu-fixture", None)


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "name", "exclusive"])
def test_selected_physical_gpu_must_match_live_inventory(plan, monkeypatch, fault):
    line = f"{GPU}, 00000000:01:00.0, FixtureGPU, 999, Default"
    if fault == "missing":
        line = line.replace(GPU, "GPU-other")
    elif fault == "duplicate":
        line += "\n" + line
    elif fault == "name":
        line = line.replace("FixtureGPU", "WrongGPU")
    elif fault == "exclusive":
        line = line.replace("Default", "Exclusive_Process")
    monkeypatch.setattr(b, "_command", lambda _: line)
    if fault:
        with pytest.raises(b.BootstrapError):
            b._gpu_inventory(plan)
    else:
        assert b._gpu_inventory(plan)["gpu_uuid"] == GPU


def test_module_shadowing_rejected_without_replacing_existing(tmp_path, monkeypatch):
    module = SimpleNamespace(__file__=str(tmp_path / "wrong.py"))
    monkeypatch.setitem(sys.modules, "fixture_utils", module)
    path = ROOT / "examples/robomme/utils.py"
    with pytest.raises(b.BootstrapError, match="another checkout"):
        b._module_at("fixture_utils", path)
    assert sys.modules["fixture_utils"] is module


def test_production_factory_preserves_original_resolver_and_physical_renderer(plan, monkeypatch):
    from experiments.uniform_keyframe_expansion import serving

    captured = []
    class OriginalRunner:
        def __init__(self, task, directory, **kwargs):
            self.env_id, self.dataset = task, kwargs["dataset"]
            self.renderer_device = None
            captured.append(kwargs)
        def make_env(self, episode_id):
            self.resolved_environment_seed = 40000 + episode_id
            self.difficulty = "original-mapping"
            self.renderer_device = {"matches_dispatch": True, "gpu_uuid": GPU, "expected_gpu_uuid": GPU}
    original_state, original_pack, original_recorder = object(), object(), object()
    utils = SimpleNamespace(TASK_NAME_LIST=list(FORMAL_TASKS), TASK_WITH_VIDEO_DEMO=["InsertPeg"],
                            EpisodeState=original_state, pack_buffer=original_pack, RolloutRecorder=original_recorder)
    monkeypatch.setattr(b, "_preflight", lambda p, r: (p, {"live": True}))
    environment_phases = []
    def environment(p, r, *, phase="startup"):
        environment_phases.append(phase)
        return {"phase": phase}
    monkeypatch.setattr(b, "_live_environment", environment)
    monkeypatch.setattr(b, "_benchmark_modules", lambda p: (utils, OriginalRunner))
    monkeypatch.setattr(serving, "ExpansionClient", lambda *args, **kwargs: (args, kwargs))
    runtime = b.prepare_benchmark_runtime(plan, policy_port=12000)
    assert runtime.provenance["benchmark_import_environment"] == {"phase": "after_benchmark_import"}
    assert runtime.components.episode_state is original_state
    assert runtime.components.pack_buffer is original_pack
    assert runtime.components.recorder is original_recorder
    env = runtime.env_factory("BinFill", Path("/unused"), max_steps=64, dataset="val", require_current_task_index=True)
    env.make_env(0)
    assert env.resolved_environment_seed == 40000 and env.difficulty == "original-mapping"
    assert captured == [{"max_steps": 64, "dataset": "val", "require_current_task_index": True, "expected_render_gpu_uuid": GPU}]
    with pytest.raises(b.BootstrapError):
        env.make_env(0)
    env2 = runtime.env_factory("BinFill", Path("/unused"), max_steps=64, dataset="val", require_current_task_index=True)
    with pytest.raises(b.BootstrapError):
        env2.make_env(1)
    with pytest.raises(b.BootstrapError):
        runtime.env_factory("BinFill", Path("/unused"), max_steps=64, dataset="test", require_current_task_index=True)
    args, kwargs = runtime.client_factory()
    assert args == ("127.0.0.1", 12000) and kwargs == {"expected_execution_identity": IDENTITY}
    assert environment_phases == ["after_benchmark_import"] * 5


@pytest.mark.parametrize("fault", [None, "cpu", "other_physical", "no_process", "load_error", "wrong_checkpoint"])
def test_model_only_strict_new_loader_then_real_pid_device_check(plan, monkeypatch, fault):
    from experiments.uniform_keyframe_expansion import serving

    monkeypatch.setattr(b, "_preflight", lambda p, r: (p, {}))
    monkeypatch.setattr(b, "_package_origin", lambda *args: None)
    device = SimpleNamespace(platform="cpu" if fault == "cpu" else "gpu", local_hardware_id=0)
    fake_jax = SimpleNamespace(devices=lambda: [device])
    monkeypatch.setattr(b.importlib, "import_module", lambda _: fake_jax)
    model = object()
    calls = []
    def load(path):
        calls.append(path)
        if fault == "load_error":
            raise RuntimeError("strict checkpoint shape mismatch")
        return model
    monkeypatch.setattr(serving, "load_expansion_policy", load)
    monkeypatch.setattr(b.os, "getpid", lambda: 123)
    query = "123, " + ("GPU-other" if fault == "other_physical" else GPU)
    monkeypatch.setattr(b, "_command", lambda _: "" if fault == "no_process" else query)
    if fault == "wrong_checkpoint":
        plan.checkpoint_dir = Path("/unused/12345")
    if fault:
        with pytest.raises((RuntimeError, b.BootstrapError)):
            b.load_authorized_policy(plan)
    else:
        loaded = b.load_authorized_policy(plan)
        assert loaded.policy is model
        assert loaded.provenance["model_gpu_uuid"] == GPU
    assert len(calls) == (0 if fault in {"cpu", "wrong_checkpoint"} else 1)
