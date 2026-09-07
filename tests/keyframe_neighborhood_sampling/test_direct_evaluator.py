"""CPU-only direct evaluator integration; all simulator imports are stubbed."""
# ruff: noqa: SLF001
# Tests intentionally exercise private evaluator and renderer contract seams.
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest

from experiments.keyframe_neighborhood_sampling import direct_provenance
from experiments.keyframe_neighborhood_sampling.runner_contract import DirectDispatch
from experiments.keyframe_neighborhood_sampling.runner_contract import canonical_bytes
from experiments.keyframe_neighborhood_sampling.runner_contract import record_digest

REPO = Path(__file__).resolve().parents[2]
POLICY_GPU = "GPU-11111111-1111-4111-8111-111111111111"
SIMULATOR_GPU = "GPU-22222222-2222-4222-8222-222222222222"
BOOT_ID = "33333333-3333-4333-8333-333333333333"


def _load_module(monkeypatch, name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "examples/robomme" / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator_module(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("KEYFRAME_", "SLURM_", "SLURMD_")) or name == "CUDA_VISIBLE_DEVICES":
            monkeypatch.delenv(name)
    modules = {name: types.ModuleType(name) for name in (
        "openpi_client", "utils", "env_runner", "subgoal_predictor", "evaluation_records",
    )}
    modules["openpi_client"].websocket_client_policy = types.SimpleNamespace(MMEVLAWebsocketClientPolicy=object)
    utils = modules["utils"]
    utils.pack_buffer = lambda *args, **kwargs: None
    utils.check_args = lambda args: None
    utils.TASK_NAME_LIST = ["PickXtimes"]
    utils.TASK_WITH_VIDEO_DEMO = set()
    utils.SUBGOAL_TYPES = set()
    utils.EpisodeState = type("EpisodeState", (), {})
    utils.RolloutRecorder = type("RolloutRecorder", (), {})
    modules["env_runner"].EnvRunner = type("EnvRunner", (), {})
    modules["subgoal_predictor"].build_subgoal_predictor = lambda *args, **kwargs: None
    modules["subgoal_predictor"].SubgoalPredictorBase = type("SubgoalPredictorBase", (), {})
    modules["evaluation_records"].EpisodeResultWriter = type("EpisodeResultWriter", (), {})
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return _load_module(monkeypatch, "_direct_evaluator_test", "eval.py")


@pytest.fixture
def direct_case(monkeypatch, tmp_path, evaluator_module):
    root = tmp_path / "runs/keyframe_neighborhood_sampling/direct-unit"
    path = root / "direct/attempt_00/row_0013/dispatch.json"
    dispatch = DirectDispatch(
        execution_id="44444444-4444-4444-8444-444444444444",
        run_root=str(root), repository_commit_sha="a" * 40,
        stage="development_smoke", launch_manifest_sha256="b" * 64,
        submission_plan_sha256="c" * 64, matrix_sha256="d" * 64,
        attempt_id=0, row_id=13, shard_id=None,
        gpu_uuids=(POLICY_GPU, SIMULATOR_GPU),
        host_name="cpu-test-host", host_boot_id=BOOT_ID, policy_port=8011,
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_bytes(dispatch.as_record()))
    digest = record_digest(dispatch.as_record())
    monkeypatch.setenv("KEYFRAME_RUNNER_BACKEND", "direct")
    monkeypatch.setenv("KEYFRAME_DIRECT_DISPATCH_PATH", str(path))
    monkeypatch.setenv("KEYFRAME_DIRECT_DISPATCH_SHA256", digest)
    monkeypatch.setenv("KEYFRAME_SMOKE_ROW_ID", "13")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", SIMULATOR_GPU)
    monkeypatch.setattr(direct_provenance, "live_host_identity", lambda: ("cpu-test-host", BOOT_ID))
    args = evaluator_module.Args(
        model_seed=7, model_ckpt_id=79999, dataset="val", max_steps=64,
        keyframe_selector_arm="OC3", keyframe_run_root=str(root),
        keyframe_trajectory_kind="short", only_tasks="PickXtimes", episode_ids="0",
        direct_dispatch_record=str(path), direct_dispatch_sha256=digest,
        save_dir=str(tmp_path / "evaluation"),
    )
    runner = {
        "backend": "direct", "dispatch_path": str(path),
        "dispatch_sha256": digest, "dispatch": dispatch.as_record(),
    }
    return evaluator_module, args, runner


def _attempt_context(module, runner):
    evaluator = types.SimpleNamespace(
        _validated_run_manifest={
            "protocol_version": module.EXTENSION_PROTOCOL_VERSION,
            "protocol_family": module.EXTENSION_PROTOCOL_FAMILY,
        },
        _validated_direct_runner=runner,
        _seed_table_payload={"entries_sha256": "a" * 64},
    )
    env = types.SimpleNamespace(
        dataset="val", resolved_environment_seed=17,
        resolved_difficulty_hint="hard", difficulty="hard",
        renderer_device={
            "render_backend": "sapien_cuda", "simulation_backend": "physx_cpu",
            "cuda_device_id": 0, "pci_bus_id": "0000:41:00.0",
            "gpu_uuid": SIMULATOR_GPU, "expected_gpu_uuid": SIMULATOR_GPU,
            "can_render": True, "is_cuda": True, "matches_dispatch": True,
        },
    )
    return evaluator, env


def test_direct_evaluator_loads_real_canonical_dispatch_and_explicit_row(direct_case):
    module, args, runner = direct_case
    module.validate_keyframe_args(args)
    assert module._load_direct_evaluator_runner(args) == runner
    evaluator, _ = _attempt_context(module, runner)
    assert module._keyframe_runtime_row_id(args, evaluator) == 13


@pytest.mark.parametrize(("field", "value", "match"), [
    ("direct_dispatch_record", "", "both dispatch"),
    ("direct_dispatch_sha256", "", "both dispatch"),
    ("direct_dispatch_record", "relative.json", "canonical absolute"),
    ("direct_dispatch_sha256", "A" * 64, "lowercase"),
    ("direct_dispatch_sha256", "f" * 64, "launcher environment"),
    ("only_tasks", "", "one explicit task"),
    ("only_tasks", "PickXtimes,StopCube", "one explicit task"),
    ("episode_ids", "", "one explicit nonnegative episode"),
    ("episode_ids", "0,1", "one explicit nonnegative episode"),
    ("episode_ids", "-1", "one explicit nonnegative episode"),
    ("keyframe_selector_arm", "OC", "restricted to governed OC3/OC5"),
])
def test_direct_argument_contract_is_fail_closed(direct_case, field, value, match):
    module, args, _ = direct_case
    with pytest.raises(ValueError, match=match):
        module.validate_keyframe_args(dataclasses.replace(args, **{field: value}))


def test_direct_environment_cannot_enable_ungoverned_evaluation(direct_case):
    module, _, _ = direct_case
    with pytest.raises(ValueError, match="restricted to governed OC3/OC5"):
        module.validate_keyframe_args(module.Args())


@pytest.mark.parametrize("visible", [POLICY_GPU, POLICY_GPU + "," + SIMULATOR_GPU])
def test_evaluator_rejects_policy_or_pair_cuda_visibility(direct_case, monkeypatch, visible):
    module, args, _ = direct_case
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="only its dispatched simulator GPU"):
        module._load_direct_evaluator_runner(args)


@pytest.mark.parametrize(("field", "value"), [("port", 8012), ("keyframe_attempt_id", 1), ("keyframe_trajectory_kind", "formal")])
def test_evaluator_rejects_wrong_live_stage_attempt_or_port(direct_case, field, value):
    module, args, _ = direct_case
    with pytest.raises(ValueError, match="stage, attempt, or policy port"):
        module._load_direct_evaluator_runner(dataclasses.replace(args, **{field: value}))


@pytest.mark.parametrize("fault", ["host", "digest", "slurm"])
def test_invalid_direct_identity_fails_before_outputs_or_simulator(direct_case, monkeypatch, fault):
    module, args, _ = direct_case
    if fault == "host":
        monkeypatch.setattr(direct_provenance, "live_host_identity", lambda: ("different-host", BOOT_ID))
    elif fault == "digest":
        path = Path(args.direct_dispatch_record)
        path.write_bytes(path.read_bytes() + b" ")
    else:
        monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(module, "setup_save_directory", lambda args: pytest.fail("must not create outputs"))
    with pytest.raises(Exception, match=r"host or boot|canonical write-once|impersonate a Slurm"):
        module.evaluate(args)


def test_direct_attempt_and_failure_preserve_host_gpu_and_no_slurm(direct_case, tmp_path):
    module, args, runner = direct_case
    evaluator, env = _attempt_context(module, runner)
    attempt = module._keyframe_attempt_manifest(args, evaluator, env, environment_setup_completed=True)
    assert attempt["runner_backend"] == "direct"
    assert attempt["runner"] == runner
    assert "slurm" not in attempt
    assert attempt["renderer_device"]["gpu_uuid"] == SIMULATOR_GPU
    writer = types.SimpleNamespace(manifest_path=tmp_path / "episode_manifest.json", trace_path=tmp_path / "missing_trace")
    writer.manifest_path.write_text(json.dumps(attempt))
    failure = module._extension_failure_provenance(args, evaluator, writer)
    assert failure["runner"] == runner
    assert failure["runner_backend"] == "direct"
    assert failure["smoke_matrix_row_id"] == 13
    assert failure["renderer_device"] == attempt["renderer_device"]
    assert "slurm" not in failure
    # Copying attempt evidence cannot mutate the validated dispatch in memory.
    attempt["runner"]["dispatch"]["host_name"] = "changed"
    assert runner["dispatch"]["host_name"] == "cpu-test-host"
    writer.manifest_path.write_text(json.dumps(attempt))
    with pytest.raises(RuntimeError, match="changed its validated direct runner"):
        module._extension_failure_provenance(args, evaluator, writer)


def test_successful_direct_setup_requires_renderer_identity(direct_case):
    module, args, runner = direct_case
    evaluator, env = _attempt_context(module, runner)
    env.renderer_device = None
    with pytest.raises(RuntimeError, match="verified physical renderer identity"):
        module._keyframe_attempt_manifest(args, evaluator, env, environment_setup_completed=True)
    failure_attempt = module._keyframe_attempt_manifest(args, evaluator, env, environment_setup_completed=False)
    assert failure_attempt["renderer_device"] is None
    assert "slurm" not in failure_attempt


def test_direct_evaluate_binds_row_before_environment_and_records_setup_failure(direct_case, monkeypatch, tmp_path):
    module, args, runner = direct_case
    evaluator, _ = _attempt_context(module, runner)
    calls = []
    failures = []
    manifests = []

    def prepared(*positional, **kwargs):
        calls.append("prepared")
        return evaluator._validated_run_manifest

    def row_validator(root, **row):
        calls.append("row")
        assert row == {
            "attempt_id": 0, "row_id": 13, "task": "PickXtimes", "episode_id": 0,
            "arm": "OC3", "trajectory_kind": "short", "max_steps": 64, "dataset": "val",
        }

    class BrokenEnvironment:
        num_episodes = 1
        resolved_environment_seed = 17
        resolved_difficulty_hint = "hard"
        difficulty = None
        renderer_device = None

        def __init__(self, task, save_dir, **kwargs):
            calls.append("environment-wrapper")
            self.dataset = kwargs["dataset"]
            assert kwargs["expected_render_gpu_uuid"] == SIMULATOR_GPU

        def make_env(self, episode_id):
            calls.append("make-environment")
            raise RuntimeError("mock renderer setup failure")

        def close_env(self):
            calls.append("close-environment")

    class Store:
        def __init__(self, root):
            assert root == args.keyframe_run_root

        def new_attempt(self, key, attempt_id, manifest):
            manifests.append(manifest)
            path = tmp_path / "mock-attempt-manifest.json"
            path.write_text(json.dumps(manifest))
            return types.SimpleNamespace(manifest_path=path, trace_path=tmp_path / "no-actions")

        def record_failure(self, failure):
            failures.append(failure)

    monkeypatch.setattr(module, "EpisodeEvaluator", lambda *positional: evaluator)
    monkeypatch.setattr(module, "EnvRunner", BrokenEnvironment)
    monkeypatch.setattr(module, "RunArtifactStore", Store)
    monkeypatch.setattr(module, "_keyframe_smoke_validators", lambda arm: (prepared, row_validator))
    with pytest.raises(RuntimeError, match="Attempt failed for PickXtimes/0"):
        module.evaluate(args)
    assert calls == ["prepared", "environment-wrapper", "row", "make-environment", "close-environment"]
    assert len(manifests) == len(failures) == 1
    assert manifests[0]["environment_setup_completed"] is False
    assert manifests[0]["runner"] == failures[0]["runner"] == runner
    assert failures[0]["smoke_matrix_row_id"] == 13
    assert failures[0]["scientific_actions_started"] is False
    assert failures[0]["renderer_device"] is None
    assert "slurm" not in failures[0]


@pytest.mark.parametrize("arm", ["U", "O", "OC", "R", "OC3", "OC5"])
def test_original_and_extension_slurm_routes_keep_existing_schema(evaluator_module, monkeypatch, arm):
    module = evaluator_module
    args = module.Args(
        model_seed=7, model_ckpt_id=79999, dataset="val", max_steps=64,
        keyframe_selector_arm=arm, keyframe_run_root="/unused", keyframe_trajectory_kind="short",
    )
    module.validate_keyframe_args(args)
    assert module._load_direct_evaluator_runner(args) is None
    evaluator, env = _attempt_context(module, None)
    if arm not in {"OC3", "OC5"}:
        evaluator._validated_run_manifest = {"protocol_version": "v1.0"}
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    monkeypatch.setenv("KEYFRAME_SMOKE_ROW_ID", "7")
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    manifest = module._keyframe_attempt_manifest(args, evaluator, env, environment_setup_completed=True)
    assert manifest["slurm"]["job_id"] == "456"
    assert module._keyframe_runtime_row_id(args, evaluator) == 7
    assert "runner" not in manifest
    assert "runner_backend" not in manifest
    assert "renderer_device" not in manifest


@pytest.fixture
def env_runner_module(monkeypatch):
    modules = {name: types.ModuleType(name) for name in (
        "robomme", "robomme.robomme_env", "robomme.env_record_wrapper",
        "robomme.env_record_wrapper.DemonstrationWrapper", "utils", "causal_stage_instrumentation",
    )}
    modules["robomme.env_record_wrapper"].BenchmarkEnvBuilder = type("Builder", (), {})
    modules["robomme.env_record_wrapper.DemonstrationWrapper"].DemonstrationWrapper = type("Demo", (), {})
    modules["utils"].TASK_NAME_LIST = ["PickXtimes"]
    instrumentation = modules["causal_stage_instrumentation"]
    instrumentation.install_current_task_index_instrumentation = lambda cls: None
    instrumentation.validate_aligned_stages = lambda *args: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return _load_module(monkeypatch, "_direct_env_runner_test", "env_runner.py")


def _renderer_runner(module):
    runner = object.__new__(module.EnvRunner)
    runner.expected_render_gpu_uuid = SIMULATOR_GPU
    device = types.SimpleNamespace(
        cuda_id=0, pci_string="0000:41:00.0",
        can_render=lambda: True, is_cuda=lambda: True,
    )
    runner.env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(backend=types.SimpleNamespace(
        render_device=device, render_backend="sapien_cuda", sim_backend="physx_cpu",
    )))
    return runner, device


def test_renderer_verifies_pci_uuid_not_vulkan_ordinal(env_runner_module, monkeypatch):
    module = env_runner_module
    runner, _ = _renderer_runner(module)
    calls = []
    def inventory(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(stdout=f"{POLICY_GPU}, 00000000:81:00.0\n{SIMULATOR_GPU}, 00000000:41:00.0\n")
    monkeypatch.setattr(module.subprocess, "run", inventory)
    runner._verify_renderer_device()
    assert runner.renderer_device["matches_dispatch"] is True
    assert runner.renderer_device["gpu_uuid"] == SIMULATOR_GPU
    assert runner.renderer_device["pci_bus_id"] == "0000:41:00.0"
    assert calls[0][0] == ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader,nounits"]
    assert calls[0][1]["timeout"] == 15


@pytest.mark.parametrize("fault", ["wrong_uuid", "missing_pci", "duplicate_pci", "wrong_cuda", "cpu_renderer", "gpu_physics"])
def test_renderer_mismatch_fails_closed_with_partial_evidence(env_runner_module, monkeypatch, fault):
    module = env_runner_module
    runner, device = _renderer_runner(module)
    output = f"{SIMULATOR_GPU}, 00000000:41:00.0\n"
    if fault == "wrong_uuid":
        output = f"{POLICY_GPU}, 00000000:41:00.0\n"
    elif fault == "missing_pci":
        output = f"{SIMULATOR_GPU}, 00000000:81:00.0\n"
    elif fault == "duplicate_pci":
        output += output
    elif fault == "wrong_cuda":
        device.cuda_id = 1
    elif fault == "cpu_renderer":
        device.is_cuda = lambda: False
    elif fault == "gpu_physics":
        runner.env.unwrapped.backend.sim_backend = "physx_cuda"
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(stdout=output))
    with pytest.raises(RuntimeError, match=r"physical GPU UUID|exactly one physical|renderer/backend"):
        runner._verify_renderer_device()
    assert runner.renderer_device["matches_dispatch"] is False


def test_default_env_builder_receives_no_changed_numerical_or_rendering_arguments(env_runner_module, monkeypatch):
    module = env_runner_module
    created = []
    calls = []
    class Builder:
        def __init__(self, **kwargs):
            created.append(kwargs)
        def make_env_for_episode(self, episode_id):
            calls.append(episode_id)
            return types.SimpleNamespace(unwrapped=types.SimpleNamespace(difficulty="hard"))
    monkeypatch.setattr(module, "BenchmarkEnvBuilder", Builder)
    monkeypatch.setattr(module.EnvRunner, "_verify_renderer_device", lambda self: calls.append("verify"))
    normal = module.EnvRunner("PickXtimes", "/unused", max_steps=64, dataset="val")
    normal.make_env(0)
    direct = module.EnvRunner("PickXtimes", "/unused", max_steps=64, dataset="val", expected_render_gpu_uuid=SIMULATOR_GPU)
    direct.make_env(0)
    expected = {"env_id": "PickXtimes", "dataset": "val", "action_space": "joint_angle", "gui_render": False, "max_steps": 64}
    assert created == [expected, expected]
    assert calls == [0, 0, "verify"]
