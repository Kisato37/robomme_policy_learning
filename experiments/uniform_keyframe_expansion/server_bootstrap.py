"""Validated, lazy production bindings; deliberately no launcher or CLI.

The controller must create separate fresh policy and simulator processes with
their recorded environments. This module never sets CUDA/numerical variables,
imports production backends on import, selects another GPU, or authorizes work.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import threading
from typing import Any, Callable


class BootstrapError(RuntimeError):
    """Live execution differs from validated evidence; never fall back."""


@dataclass(frozen=True)
class RuntimeBindings:
    components: Any
    env_factory: Callable
    client_factory: Callable
    provenance: dict


@dataclass(frozen=True)
class LoadedPolicy:
    policy: Any
    provenance: dict


_IMPORT_LOCK = threading.Lock()
_NUMERICAL_PREFIXES = (
    "JAX_", "XLA_", "TF_", "CUBLAS_", "CUDNN_", "NVIDIA_", "OMP_",
    "MKL_", "OPENBLAS_", "PYTORCH_", "TORCH_", "CUDA_", "VK_", "SAPIEN_",
)
_SENSITIVE_KEYS = {"PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "NVIDIA_VISIBLE_DEVICES"}
_REQUIRED_PACKAGES = {
    "policy": {"numpy", "jax", "jaxlib", "torch"},
    "simulator": {"numpy", "torch", "mani-skill", "sapien"},
}


def _command(arguments: list[str]) -> str:
    # Read-only Git commands must not opportunistically refresh an index.
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(arguments, capture_output=True, text=True, check=True,
                          timeout=30, env=env).stdout.strip()


def _live_repository(source: Mapping, *, role: str) -> dict:
    root = Path(source["root"]).resolve(strict=True)
    if Path(_command(["git", "-C", str(root), "rev-parse", "--show-toplevel"])).resolve() != root:
        raise BootstrapError(f"{role} source is not the declared repository root")
    revision = _command(["git", "-C", str(root), "rev-parse", "HEAD"])
    branch = _command(["git", "-C", str(root), "branch", "--show-current"])
    if role == "benchmark" and not branch:
        branch = "HEAD"  # Explicit detached pin, not permission to switch branches.
    dirty = _command(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"])
    if revision != source["revision"] or branch != source["branch"] or dirty:
        raise BootstrapError(f"Live {role} revision/branch/clean state differs from evidence")
    if role == "policy" and not branch.startswith("exp/"):
        raise BootstrapError("Policy execution requires its dedicated exp/* branch")
    return {"root": str(root), "revision": revision, "branch": branch, "clean": True}


def _live_environment(plan, role: str) -> dict:
    report = plan.environment
    expected = report["roles"][role]
    if sys.platform != "linux" or socket.gethostname() != report["host"]:
        raise BootstrapError("Live Linux host differs from the execution environment")
    if (Path(sys.executable).absolute() != Path(expected["python_executable"]).absolute()
            or platform.python_version() != expected["python_version"]):
        raise BootstrapError(f"Wrong {role} Python executable/version")
    declared = expected["process_environment"]
    if not {"CUDA_VISIBLE_DEVICES", "PYTHONPATH"}.issubset(declared):
        raise BootstrapError("Environment evidence must bind CUDA visibility and Python paths")
    if declared["CUDA_VISIBLE_DEVICES"] != plan.gpu_uuid:
        raise BootstrapError("Only the single explicitly selected physical GPU may be visible")
    changed = [key for key, value in declared.items() if os.environ.get(key) != value]
    unexpected = [key for key in os.environ
                  if (key.startswith(_NUMERICAL_PREFIXES) or key in _SENSITIVE_KEYS) and key not in declared]
    if changed or unexpected:
        raise BootstrapError("Unrecorded/changed process environment: " + ", ".join(sorted(set(changed + unexpected))))
    if role == "policy" and declared.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
        raise BootstrapError("Recorded colocated policy must disable XLA preallocation before import")
    packages = expected["packages"]
    if not _REQUIRED_PACKAGES[role].issubset(packages):
        raise BootstrapError(f"Missing critical {role} package-version evidence")
    for name, version in packages.items():
        if importlib.metadata.version(name) != version:
            raise BootstrapError(f"Live package version differs: {name}")
    return {"role": role, "host": socket.gethostname(), "python_executable": str(Path(sys.executable).absolute()),
            "python_binary_resolved": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(), "packages": dict(packages),
            "process_environment": dict(declared)}


def _gpu_inventory(plan) -> dict:
    output = _command(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name,driver_version,compute_mode",
                       "--format=csv,noheader,nounits"])
    matches = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 5:
            raise BootstrapError("Malformed physical GPU inventory")
        if values[0] == plan.gpu_uuid:
            matches.append(dict(zip(("gpu_uuid", "pci_bus_id", "gpu_name", "driver_version", "compute_mode"), values)))
    if len(matches) != 1:
        raise BootstrapError("Selected physical GPU is absent or duplicated")
    observed = matches[0]
    hardware = plan.environment["hardware"]
    for key in ("gpu_uuid", "gpu_name", "driver_version"):
        if key in hardware and hardware[key] != observed[key]:
            raise BootstrapError(f"Live GPU {key} differs from environment evidence")
    if observed["compute_mode"] != "Default":
        raise BootstrapError("Colocated policy/renderer requires shareable Default compute mode")
    return observed


def _fresh_backends() -> None:
    present = [name for name in ("jax", "jaxlib", "torch", "tensorflow", "cupy", "sapien", "robomme", "mani_skill")
               if name in sys.modules]
    if present:
        raise BootstrapError("Production bindings require a fresh process before backend import: " + ", ".join(present))


def _live_checkpoint(plan) -> dict:
    """Stream actual archive/parameter bytes once per resident model startup."""
    from experiments.keyframe_oracle_sampling.artifacts import sha256_file
    from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_content_tree_identity
    from experiments.uniform_keyframe_expansion.contract import CHECKPOINT_ARCHIVE_SHA256

    checkpoint = plan.checkpoint
    archive = Path(checkpoint["archive"]["path"])
    # Parent helpers stream in 1 MiB chunks; never materialize the archive in RAM.
    actual_archive = sha256_file(archive)
    if actual_archive != CHECKPOINT_ARCHIVE_SHA256 or actual_archive != checkpoint["archive"]["sha256"]:
        raise BootstrapError("Actual checkpoint archive bytes differ from the frozen checkpoint")
    actual_tree = checkpoint_content_tree_identity(plan.checkpoint_dir)
    if actual_tree != checkpoint["content_tree"]:
        raise BootstrapError("Actual checkpoint parameter bytes differ from the recorded content manifest")
    return {"archive_sha256": actual_archive, "content_tree": actual_tree,
            "verification_scope": "streamed actual bytes before this resident model load"}


def verify_controller_sources(plan) -> dict:
    """Recheck authorization/file chains and live Git before any child/wrapper.

    This controller-safe gate never imports GPU/simulator backends, hashes large
    weights, or requires the controller's interpreter to equal a worker's.
    It is not the worker's complete environment/checkpoint/device validation.
    """
    from experiments.uniform_keyframe_expansion.launch_contract import ValidatedExecutionPlan

    if type(plan) is not ValidatedExecutionPlan:
        raise BootstrapError("Production adapters accept only a validated execution plan")
    plan.require_runtime_stage()
    plan = plan.revalidate()
    plan.require_runtime_stage()
    if Path(plan.policy_root).resolve() != Path(__file__).resolve().parents[2]:
        raise BootstrapError("Bootstrap itself is loaded from a different policy checkout")
    return {"policy": _live_repository(plan.policy_source, role="policy"),
            "benchmark": _live_repository(plan.benchmark_source, role="benchmark")}


def _preflight(plan, role: str):
    from experiments.uniform_keyframe_expansion.launch_contract import ValidatedExecutionPlan

    if type(plan) is not ValidatedExecutionPlan:
        raise BootstrapError("Production adapters accept only a validated execution plan")
    plan.require_runtime_stage()
    if role == "simulator" and plan.stage not in {"end_to_end_smoke", "formal"}:
        raise BootstrapError("Architecture checks cannot launch benchmark episodes")
    _fresh_backends()
    source = verify_controller_sources(plan)
    environment = _live_environment(plan, role)
    gpu = _gpu_inventory(plan)
    checkpoint = _live_checkpoint(plan) if role == "policy" else None
    return plan, {"sources": source, "environment": environment, "gpu": gpu,
                  "policy_execution_identity": deepcopy(plan.execution_identity),
                  "checkpoint_content_verified_in_this_process": role == "policy",
                  "checkpoint_content_evidence": checkpoint}


def _module_at(name: str, path: Path):
    """Load original bare-import modules without modifying global search paths."""
    path = path.resolve(strict=True)
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != path:
            raise BootstrapError(f"Module {name} was already loaded from another checkout")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BootstrapError(f"Cannot load original source module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _package_origin(name: str, expected_root: Path) -> None:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(expected_root.resolve()):
        raise BootstrapError(f"Package {name} resolves outside its validated checkout")


def _benchmark_modules(plan):
    with _IMPORT_LOCK:
        _package_origin("robomme", Path(plan.benchmark_root) / "src")
        example = Path(plan.policy_root) / "examples" / "robomme"
        utils = _module_at("utils", example / "utils.py")
        _module_at("causal_stage_instrumentation", example / "causal_stage_instrumentation.py")
        runner = _module_at("_uniform_keyframe_expansion_env_runner", example / "env_runner.py")
        return utils, runner.EnvRunner


def prepare_benchmark_runtime(plan, *, policy_port: int, policy_host: str = "127.0.0.1") -> RuntimeBindings:
    """Bind the unmodified benchmark pipeline after runtime checks, not before.

    Checkpoint contents are verified by the separately bound resident model;
    simulator workers revalidate evidence but do not hash weights every episode.
    """
    plan, provenance = _preflight(plan, "simulator")
    if policy_host != "127.0.0.1" or type(policy_port) is not int or not 1 <= policy_port <= 65535:
        raise BootstrapError("The resident policy endpoint must be explicit loopback")
    utils, runner = _benchmark_modules(plan)
    from experiments.uniform_keyframe_expansion.contract import FORMAL_TASKS
    from experiments.uniform_keyframe_expansion.evaluator import BenchmarkComponents
    from experiments.uniform_keyframe_expansion.serving import ExpansionClient

    if tuple(utils.TASK_NAME_LIST) != tuple(FORMAL_TASKS):
        raise BootstrapError("Original task registry differs from the frozen population")
    allowed = {(r["task"], r["dataset"], r["max_steps"], r["episode_id"]) for r in plan.rows}

    class BoundEnvRunner(runner):
        # Only admission guards change; all numerical environment methods are
        # inherited unchanged, including the official seed/difficulty resolver.
        def make_env(self, episode_id):
            if (type(episode_id) is not int or self._expansion_opened
                    or (self.env_id, self.dataset, self._expansion_limit, episode_id) not in allowed):
                raise BootstrapError("Episode is not authorized, or environment is being reused")
            self._expansion_opened = True
            super().make_env(episode_id)
            if (not isinstance(self.renderer_device, dict) or self.renderer_device.get("matches_dispatch") is not True
                    or self.renderer_device.get("gpu_uuid") != plan.gpu_uuid
                    or self.renderer_device.get("expected_gpu_uuid") != plan.gpu_uuid):
                raise BootstrapError("Original EnvRunner did not verify the physical renderer device")

    def env_factory(task, directory, *, max_steps, dataset, require_current_task_index):
        _live_environment(plan, "simulator")
        if (require_current_task_index is not True or type(max_steps) is not int
                or not any((task, dataset, max_steps) == entry[:3] for entry in allowed)):
            raise BootstrapError("Environment request differs from the authorized row population")
        result = BoundEnvRunner(task, directory, max_steps=max_steps, dataset=dataset,
                                require_current_task_index=True, expected_render_gpu_uuid=plan.gpu_uuid)
        result._expansion_opened = False
        result._expansion_limit = max_steps
        return result

    def client_factory():
        _live_environment(plan, "simulator")
        return ExpansionClient(policy_host, policy_port, expected_execution_identity=plan.execution_identity)

    return RuntimeBindings(BenchmarkComponents(utils.EpisodeState, utils.pack_buffer, utils.RolloutRecorder,
                                              tuple(utils.TASK_WITH_VIDEO_DEMO)),
                           env_factory, client_factory, provenance)


def _loaded_model_gpu(plan, jax) -> dict:
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or devices[0].local_hardware_id != 0:
        raise BootstrapError("Policy requires exactly one remapped local CUDA GPU, with no CPU fallback")
    output = _command(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"])
    actual = set()
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 2 or not values[0].isdecimal():
            raise BootstrapError("Malformed active CUDA process inventory")
        if int(values[0]) == os.getpid():
            actual.add(values[1])
    if actual != {plan.gpu_uuid}:
        raise BootstrapError("Loaded model's actual physical GPU differs from the selected UUID")
    return {"model_process_pid": os.getpid(), "model_gpu_uuid": plan.gpu_uuid,
            "jax_device": str(devices[0]), "logical_cuda_device": 0}


def load_authorized_policy(plan) -> LoadedPolicy:
    """Strictly load checkpoint79999 once, only inside an authorized policy worker."""
    plan, provenance = _preflight(plan, "policy")
    if Path(plan.checkpoint_dir).name != "79999":
        raise BootstrapError("No checkpoint path fallback is allowed")
    _package_origin("mme_vla_suite", Path(plan.policy_root) / "src")
    _package_origin("openpi", Path(plan.policy_root) / "src")
    jax = importlib.import_module("jax")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or devices[0].local_hardware_id != 0:
        raise BootstrapError("Policy backend is not the single declared CUDA device")
    from experiments.uniform_keyframe_expansion.serving import load_expansion_policy

    policy = load_expansion_policy(plan.checkpoint_dir)
    provenance.update(_loaded_model_gpu(plan, jax))
    return LoadedPolicy(policy, provenance)
