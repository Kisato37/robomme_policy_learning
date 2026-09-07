"""Explicit Linux execution for the frozen OC3/OC5 experiment, without Slurm.

Preparation and scientific gates are reused; this entry point only substitutes
resource ownership and process provenance. Run in a persistent foreground
terminal. No automatic resume or retry is permitted after an uncertain start.
"""

# Optional websocket/scientific-gate dependencies are loaded at the boundary
# that needs them, keeping resource/command construction CPU-only and testable.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from contextlib import ExitStack
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import uuid

from experiments.keyframe_neighborhood_sampling.direct_runtime import GpuLease
from experiments.keyframe_neighborhood_sampling.direct_runtime import LinuxProcessOperations
from experiments.keyframe_neighborhood_sampling.direct_runtime import OwnedProcessGroup
from experiments.keyframe_neighborhood_sampling.direct_runtime import PortReservation
from experiments.keyframe_neighborhood_sampling.direct_runtime import ResourceBusyError
from experiments.keyframe_neighborhood_sampling.gpu_admission import admission_policy
from experiments.keyframe_neighborhood_sampling.gpu_telemetry import GpuTelemetry
from experiments.keyframe_neighborhood_sampling.runner_contract import DirectDispatch
from experiments.keyframe_neighborhood_sampling.runner_contract import process_exit_record
from experiments.keyframe_neighborhood_sampling.runner_contract import process_start_record
from experiments.keyframe_neighborhood_sampling.runner_contract import require_gpu_uuid
from experiments.keyframe_neighborhood_sampling.runner_contract import write_once_record
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_RELATIVE
from experiments.keyframe_oracle_sampling.prepare_smoke import REPO

MODULE = "experiments.keyframe_neighborhood_sampling"
READINESS_SECONDS = 240
ROW_SECONDS = 12 * 60 * 60
ARCHITECTURE_SECONDS = 6 * 60 * 60
CPUS_PER_ROW = 12
MEMORY_BYTES_PER_ROW = 96 * 1024**3


class RowInfrastructureError(RuntimeError):
    """An audited infrastructure failure, not a result and not permission to retry."""


def require_row_outcome(run_root: Path, row: dict, attempt_id: int) -> None:
    from experiments.keyframe_neighborhood_sampling.record_launcher_failure import validate_extension_failure_record
    from experiments.keyframe_oracle_sampling.artifacts import EpisodeAttemptWriter
    from experiments.keyframe_oracle_sampling.artifacts import RunArtifactStore
    from experiments.keyframe_oracle_sampling.artifacts import ScientificKey
    from experiments.keyframe_oracle_sampling.artifacts import read_jsonl

    store = RunArtifactStore(run_root)
    key = ScientificKey(row["task"], row["episode_id"], row["arm"], row["trajectory_kind"])
    writer = EpisodeAttemptWriter(store.attempt_dir(key, attempt_id), key, attempt_id)
    if writer.result_path.is_file() and writer.validate_resume() == "complete":
        return  # A valid result, including a scientific failure, is never retried.
    failures = read_jsonl(store.failures_path) if store.failures_path.is_file() else []
    matches = [
        failure
        for failure in failures
        if all(failure.get(field) == value for field, value in key.as_dict().items())
        and failure.get("attempt_id") == attempt_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Row {row['row_id']} has neither a complete result nor one exact recorded failure")
    validate_extension_failure_record(run_root, matches[0], expected_row_id=row["row_id"])
    if matches[0]["classification"] == "infrastructure":
        raise RowInfrastructureError(f"Row {row['row_id']}: {matches[0]['error_type']}")
    raise RuntimeError(f"Row {row['row_id']} is a recorded hard stop: {matches[0]['error_type']}")


def reject_numerical_overrides(environ: dict[str, str]) -> None:
    prefixes = ("JAX_", "XLA_", "TF_", "CUBLAS_", "CUDNN_", "NVIDIA_TF32_", "OMP_", "MKL_", "OPENBLAS_", "PYTORCH_")
    present = sorted(key for key, value in environ.items() if value and key.startswith(prefixes))
    if present:
        raise RuntimeError("Unreviewed numerical environment overrides: " + ", ".join(present))


def bind_host_resources(profile: dict, allocations: list[list[str]]) -> dict:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Direct host resource preparation requires Linux CPU affinity")
    available = sorted(os.sched_getaffinity(0))
    if len(available) < CPUS_PER_ROW * len(allocations):
        raise RuntimeError("Insufficient allowed CPUs for disjoint 12-CPU execution slots")
    return {
        **profile,
        "cpu_ids_by_policy_gpu": {
            pair[0]: available[slot * CPUS_PER_ROW : (slot + 1) * CPUS_PER_ROW] for slot, pair in enumerate(allocations)
        },
        "memory_bytes_per_row": MEMORY_BYTES_PER_ROW,
        "numerical_environment": {},
        "policy_allocator_environment": (
            {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"} if profile.get("gpu_layout") == "colocated" else {}
        ),
    }


def validate_allocations(stage: str, allocations: list[list[str]], gpu_layout: str = "separate") -> None:
    if gpu_layout not in {"separate", "colocated"}:
        raise ValueError("Unknown GPU layout")
    width = 1 if stage == "architecture_smoke" else 2
    if not allocations or len(allocations) > 4 or any(len(pair) != width for pair in allocations):
        raise ValueError(f"Expected 1..4 allocations of exactly {width} physical GPU UUIDs")
    if stage == "architecture_smoke" and len(allocations) != 1:
        raise ValueError("Architecture gate requires exactly one GPU")
    for pair in allocations:
        if width == 2 and ((pair[0] == pair[1]) != (gpu_layout == "colocated")):
            raise ValueError("GPU allocations overlap or contradict the declared layout")
    flat = [gpu for pair in allocations for gpu in dict.fromkeys(pair)]
    for gpu in flat:
        require_gpu_uuid(gpu)
    if len(flat) != len(set(flat)):
        raise ValueError("GPU allocations overlap")


def runtime_profile(environment_sh: Path, graphics_wrapper: Path, lock_directory: Path) -> dict:
    """Bind process-local setup scripts, never mutate profiles or system drivers."""
    if not lock_directory.is_dir() or lock_directory.is_symlink():
        raise ValueError("Use one existing nonsymlink GPU-lock directory across all own checkouts")
    paths = {"environment_sh": environment_sh, "graphics_wrapper": graphics_wrapper}
    for path in paths.values():
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"Runtime script must exist at an absolute path: {path}")
    return {
        **{key: str(path.resolve()) for key, path in paths.items()},
        **{f"{key}_sha256": sha256_file(path) for key, path in paths.items()},
        "lock_directory": str(lock_directory.resolve()),
    }


def verify_runtime_profile(profile: dict) -> None:
    admission_policy(profile)
    if profile.get("policy_lifetime", "per_row") not in {"per_row", "resident"}:
        raise ValueError("Unknown policy lifetime")
    if profile.get("policy_lifetime") == "resident" and profile.get("gpu_layout") != "colocated":
        raise ValueError("Resident policy requires colocated placement")
    for key in ("environment_sh", "graphics_wrapper"):
        if sha256_file(Path(profile[key])) != profile[f"{key}_sha256"]:
            raise RuntimeError(f"Prepared runtime script changed: {key}")


def check_idle_gpus(gpu_uuids: tuple[str, ...]) -> None:
    """Fresh physical inventory: never acquire memory or displace another user."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    inventory = {}
    for line in output.splitlines():
        gpu, memory, utilization = (value.strip() for value in line.split(","))
        inventory[gpu] = (int(memory), int(utilization))
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    occupied = {line.split(",")[0].strip() for line in apps.splitlines() if line.strip()}
    for gpu in gpu_uuids:
        if gpu not in inventory:
            raise ValueError(f"Declared GPU no longer exists: {gpu}")
        used, utilization = inventory[gpu]
        if gpu in occupied or used > 128 or utilization != 0:
            raise ResourceBusyError(f"GPU is in use; nothing will be canceled: {gpu}")


def check_gpu_admission(gpu_uuids: tuple[str, ...], profile: dict) -> dict:
    """Check immediately before launch; neither reserve VRAM nor touch other jobs."""
    policy = admission_policy(profile)
    if policy["mode"] == "exclusive":
        check_idle_gpus(gpu_uuids)
        return {"policy": policy, "checked_utc": utc_now(), "gpu_uuids": list(gpu_uuids)}
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu,compute_mode", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    inventory = {}
    for line in output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 4:
            raise ValueError("GPU inventory lacks explicit compute mode")
        gpu, free, utilization, compute_mode = fields
        free, utilization = int(free), int(utilization)
        if gpu in inventory or free < 0 or not 0 <= utilization <= 100:
            raise ValueError("Invalid or duplicated GPU inventory")
        inventory[gpu] = {"gpu_uuid": gpu, "free_memory_mib": free, "utilization_gpu_percent": utilization,
                          "compute_mode": compute_mode}
    devices = []
    for gpu in dict.fromkeys(gpu_uuids):
        if gpu not in inventory:
            raise ValueError(f"Declared GPU no longer exists: {gpu}")
        device = inventory[gpu]
        if (
            device["compute_mode"] != "Default"
            or device["free_memory_mib"] < policy["min_free_memory_mib"]
            or device["utilization_gpu_percent"] > policy["max_utilization_gpu_percent"]
        ):
            raise ResourceBusyError(f"Shared GPU headroom/usage check failed; no other job is touched: {device}")
        devices.append(device)
    return {"policy": policy, "checked_utc": utc_now(), "devices": devices, "exclusive_capacity_guaranteed": False}


def build_direct_submission(
    run_root: Path,
    stage: str,
    allocations: list[list[str]],
    profile: dict,
    attempt_id: int = 0,
    row_ids: tuple[int, ...] | None = None,
) -> tuple[list[tuple[Path, dict]], dict | None]:
    """Reuse every existing scientific preparation gate before writing anything."""
    from experiments.keyframe_neighborhood_sampling import submit_architecture_smoke
    from experiments.keyframe_neighborhood_sampling import submit_formal
    from experiments.keyframe_neighborhood_sampling import submit_smoke
    from experiments.keyframe_neighborhood_sampling.formal_artifacts import smoke_submission_path
    from experiments.keyframe_neighborhood_sampling.formal_artifacts import submission_path
    from experiments.keyframe_neighborhood_sampling.formal_artifacts import submission_plan_path

    gpu_layout = profile.get("gpu_layout", "separate")
    validate_allocations(stage, allocations, gpu_layout)
    verify_runtime_profile(profile)
    reject_numerical_overrides(dict(os.environ))
    profile = bind_host_resources(profile, allocations)
    manifest = json.loads((run_root / "protocol/launch_manifest.json").read_text())
    if manifest.get("runner_backend") != "direct":
        raise ValueError("Prepare a NEW root with --runner-backend direct; existing roots cannot be converted")
    if manifest.get("gpu_layout", "separate") != gpu_layout:
        raise ValueError("GPU layout must match the prepared run; existing roots cannot be converted")
    command = [
        str(REPO / ".venv/bin/python"),
        "-m",
        f"{MODULE}.submit_direct",
        "--run-root",
        str(run_root),
        "--stage",
        stage,
        "--gpu-layout",
        gpu_layout,
        "--attempt-id",
        str(attempt_id),
        "--environment-sh",
        profile["environment_sh"],
        "--graphics-wrapper",
        profile["graphics_wrapper"],
        "--lock-directory",
        profile["lock_directory"],
    ]
    admission = admission_policy(profile)
    if profile.get("policy_lifetime") == "resident":
        if stage == "formal":
            raise ValueError("Resident formal runs require a separately reviewed execution gate")
        command.append("--resident-policy")
    if admission["mode"] == "shared":
        command.extend([
            "--allow-shared-gpu", "--shared-min-free-mib", str(admission["min_free_memory_mib"]),
            "--shared-max-utilization", str(admission["max_utilization_gpu_percent"]),
        ])
    for pair in allocations:
        command.extend(["--gpu-allocation", ",".join(pair)])
    if row_ids:
        command.extend(["--rows", ",".join(map(str, row_ids))])
    plan = None
    if stage == "architecture_smoke":
        if attempt_id != 0 or row_ids:
            raise ValueError("Architecture gate has no retry rows")
        _, record = submit_architecture_smoke.build_submission(run_root)
        record["smoke_matrix_sha256"] = sha256_file(run_root / "protocol/smoke_matrix.json")
        records = [(run_root / "protocol/architecture_submission_record.json", record)]
    elif stage == "development_smoke":
        _, record = submit_smoke.build_submission(
            run_root,
            max_concurrent=len(allocations),
            attempt_id=attempt_id,
            row_ids=row_ids,
        )
        records = [(smoke_submission_path(run_root, attempt_id), record)]
    elif stage == "formal":
        # No opaque recovery of uncertain direct-process starts, including a
        # controller lost between writing the plan and starting its first row.
        if submission_plan_path(run_root, attempt_id).exists():
            raise FileExistsError("Existing direct plan is pending/completed; audit before explicit recovery")
        submissions, plan = submit_formal.build_submission(
            run_root,
            max_concurrent=len(allocations),
            attempt_id=attempt_id,
            row_ids=row_ids,
        )
        records = [(submission_path(run_root, attempt_id, record["shard_id"]), record) for _, record in submissions]
    else:
        raise ValueError("Unknown direct execution stage")
    for _, record in records:
        record.update(
            runner_backend="direct", direct_run_id=str(uuid.uuid4()), command=command,
            runtime_profile=profile, gpu_layout=gpu_layout,
        )
        if stage == "architecture_smoke":
            record["gpu_uuids"] = allocations[0]
        else:
            record["gpu_pairs"] = allocations
    if plan is not None:
        plan.update(
            runner_backend="direct", direct_run_id=str(uuid.uuid4()), gpu_pairs=allocations,
            runtime_profile=profile, gpu_layout=gpu_layout,
        )
        for shard, (_, record) in zip(plan["shards"], records, strict=True):
            shard["command"] = record["command"]
    if any(path.exists() for path, _ in records):
        raise FileExistsError("Refusing to replace a direct submission record")
    return records, plan


def publish_submission(run_root: Path, records: list[tuple[Path, dict]], plan: dict | None) -> None:
    """Complete plan and all shard authorizations are immutable before any GPU work."""
    if plan is not None:
        from experiments.keyframe_neighborhood_sampling.formal_artifacts import submission_plan_path

        digest = write_once_record(submission_plan_path(run_root, plan["attempt_id"]), plan)
        for _, record in records:
            record["submission_plan_sha256"] = digest
    for path, record in records:
        write_once_record(path, record)


def child_environment(dispatch: DirectDispatch, dispatch_path: Path, digest: str, role: str) -> dict[str, str]:
    environ = {key: value for key, value in os.environ.items() if not key.startswith("KEYFRAME_")}
    # Direct jobs cannot carry a stale scheduler identity inherited from a shell.
    if any(key.startswith("SLURM_") for key in environ):
        raise RuntimeError("Direct execution refuses a Slurm allocation environment")
    environ.update(
        KEYFRAME_RUNNER_BACKEND="direct",
        KEYFRAME_DIRECT_DISPATCH_PATH=str(dispatch_path),
        KEYFRAME_DIRECT_DISPATCH_SHA256=digest,
        PYTHONPATH=f"{REPO}/examples/robomme:{REPO}/src:{REPO}/packages/openpi-client/src:{REPO}",
        CUDA_VISIBLE_DEVICES=(
            dispatch.gpu_uuids[0]
            if role in {"policy", "architecture"}
            else dispatch.gpu_uuids[1]
            if role == "evaluator"
            else ",".join(dict.fromkeys(dispatch.gpu_uuids))
        ),
    )
    if dispatch.stage == "formal":
        environ["KEYFRAME_FORMAL_ROW_ID"] = str(dispatch.row_id)
    elif dispatch.stage == "development_smoke":
        environ["KEYFRAME_SMOKE_ROW_ID"] = str(dispatch.row_id)
    if role == "architecture":
        environ.update(XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_EXPLAIN_CACHE_MISSES="true")
    elif role == "policy" and dispatch.gpu_layout == "colocated":
        # Allocator placement only: no precision, model, solver or seed change.
        # Do not reserve most of the shared device before the renderer starts.
        environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return environ


def wrapped_command(command: list[str], profile: dict, *, graphics: bool) -> list[str]:
    argv = ["bash", "-ec", 'source "$1"; shift; exec "$@"', "direct-runtime", profile["environment_sh"]]
    if graphics:
        argv.extend(["bash", profile["graphics_wrapper"]])
    return argv + command


class Execution:
    """Own only the real children started for one immutable row dispatch."""

    def __init__(
        self, dispatch: DirectDispatch, directory: Path, profile: dict, lease: GpuLease, stop: threading.Event
    ):
        self.dispatch, self.directory, self.profile, self.lease, self.stop = dispatch, directory, profile, lease, stop
        self.path = directory / "dispatch.json"
        self.digest = write_once_record(self.path, dispatch.as_record())
        self.children: dict[str, tuple[OwnedProcessGroup, str]] = {}
        self.roles: dict[str, dict] = {}
        self.timed_out = False
        self.last_memory_check = 0.0
        self.resident = None
        self.deadline = time.monotonic() + (
            ARCHITECTURE_SECONDS if dispatch.stage == "architecture_smoke" else ROW_SECONDS
        )

    def start(self, role: str, command: list[str], extra_fds: tuple[int, ...] = ()) -> OwnedProcessGroup:
        if role != "reconcile" and (self.stop.is_set() or time.monotonic() >= self.deadline):
            raise InterruptedError("Controller stopped or row deadline reached before process startup")
        verify_runtime_profile(self.profile)
        if "numerical_environment" in self.profile:
            reject_numerical_overrides(dict(os.environ))
        env = child_environment(self.dispatch, self.path, self.digest, role)
        protected_keys = {key for key in env if key.startswith("KEYFRAME_")} | {
            "CUDA_VISIBLE_DEVICES",
            "PYTHONPATH",
            "KEYFRAME_SMOKE_ROW_ID",
            "KEYFRAME_FORMAL_ROW_ID",
        }
        environment_record = self.directory / f"{role}_environment.json"
        numerical_prefixes = (
            "JAX_",
            "XLA_",
            "TF_",
            "CUBLAS_",
            "CUDNN_",
            "NVIDIA_TF32_",
            "OMP_",
            "MKL_",
            "OPENBLAS_",
            "PYTORCH_",
        )
        environment_digest = write_once_record(
            environment_record,
            {
                "protected_environment": {key: env.get(key) for key in sorted(protected_keys)},
                "numerical_environment": {
                    key: value for key, value in env.items() if value and key.startswith(numerical_prefixes)
                },
            },
        )
        guarded_command = [
            str(REPO / ".venv/bin/python"),
            "-I",
            str(Path(__file__).with_name("direct_payload.py")),
            "--environment-record",
            str(environment_record),
            "--expected-sha256",
            environment_digest,
            "--",
            *command,
        ]
        payload = wrapped_command(guarded_command, self.profile, graphics=role == "evaluator")
        parent = LinuxProcessOperations().identity(os.getpid())
        if parent is None:
            raise RuntimeError("Cannot establish controller process identity")
        read_fd, write_fd = os.pipe()
        inherited = self.lease.pass_fds + extra_fds
        # The watchdog waits for a start byte. No model can load before its
        # ownership and immutable process-start record have been established.
        argv = [
            str(REPO / ".venv/bin/python"),
            "-m",
            f"{MODULE}.direct_child",
            "--start-fd",
            str(read_fd),
            "--parent-pid",
            str(parent.pid),
            "--parent-start-ticks",
            str(parent.start_ticks),
            "--boot-id",
            parent.boot_id,
            "--deadline-monotonic",
            str(time.monotonic() + 120 if role == "reconcile" else self.deadline),
        ]
        for descriptor in inherited:
            argv.extend(["--pass-fd", str(descriptor)])
        cpu_ids = self.profile.get("cpu_ids_by_policy_gpu", {}).get(self.dispatch.gpu_uuids[0])
        if cpu_ids is not None:
            argv.extend(["--cpu-ids", ",".join(map(str, cpu_ids))])
        if "memory_bytes_per_row" in self.profile:
            argv.extend(["--memory-limit-bytes", str(self.profile["memory_bytes_per_row"])])
        argv.extend(["--", *payload])
        child = None
        owned = None
        try:
            with (
                (self.directory / f"{role}.out").open("xb") as stdout,
                (self.directory / f"{role}.err").open("xb") as stderr,
            ):
                child = subprocess.Popen(
                    argv,
                    cwd=REPO,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    pass_fds=(*inherited, read_fd),
                )
            owned = OwnedProcessGroup(child, LinuxProcessOperations())
            self.children[role] = (owned, "")
            start = process_start_record(
                self.dispatch, process_identity=asdict(owned.identity), command=argv, working_directory=str(REPO)
            )
            digest = write_once_record(self.directory / f"{role}_start.json", start)
            self.children[role] = (owned, digest)
            if role != "reconcile" and (self.stop.is_set() or time.monotonic() >= self.deadline):
                raise InterruptedError("Controller stopped before releasing payload startup barrier")
            os.write(write_fd, b"1")
        except BaseException:
            # Closing the barrier rejects startup even when /proc ownership
            # construction fails. Reap only this exact unreaped Popen child.
            os.close(write_fd)
            write_fd = -1
            if child is not None and owned is None:
                child.wait(timeout=15)
            raise
        finally:
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
        return owned

    def record_exit(self, role: str, status: int) -> None:
        if role in self.roles:
            return
        if role != "reconcile" and time.monotonic() >= self.deadline:
            self.timed_out = True
        _, start_digest = self.children[role]
        exit_record = process_exit_record(
            self.dispatch,
            started_record_sha256=start_digest,
            returncode=status,
            wall_clock_limit_reached=self.timed_out,
        )
        digest = write_once_record(self.directory / f"{role}_exit.json", exit_record)
        self.roles[role] = {"start_sha256": start_digest, "exit_sha256": digest}

    def wait(self, role: str, *, reconciliation: bool = False) -> int:
        child, _ = self.children[role]
        limit = time.monotonic() + 120 if reconciliation else self.deadline
        while True:
            self.check_memory_budget()
            status = child.reap_if_finished()
            if status is not None:
                self.record_exit(role, status)
                return status
            if time.monotonic() >= limit or (self.stop.is_set() and not reconciliation):
                self.timed_out = time.monotonic() >= self.deadline
                status = child.stop(grace_seconds=10, kill_wait_seconds=10)
                self.record_exit(role, status)
                return status
            self.stop.wait(0.2) if not reconciliation else time.sleep(0.2)

    def check_memory_budget(self) -> None:
        if self.resident is not None:
            child, _ = self.resident.execution.children["policy"]
            if child.reap_if_finished() is not None:
                raise RuntimeError("Resident model exited; stop instead of silently restarting it")
        limit = self.profile.get("memory_bytes_per_row")
        if limit is None or time.monotonic() - self.last_memory_check < 2:
            return
        self.last_memory_check = time.monotonic()
        pids = set()
        operations = LinuxProcessOperations()
        children = list(self.children.values())
        if self.resident is not None:
            children += list(self.resident.execution.children.values())
        for child, _ in children:
            if child.returncode is None:
                pids.update(member.pid for member in operations.live_group_members(child.identity.process_group))
        total = 0
        for pid in pids:
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1]) * 1024
            except (FileNotFoundError, ProcessLookupError):
                continue
        if total > limit:
            evidence = self.directory / "memory_limit.json"
            if not evidence.exists():
                write_once_record(
                    evidence,
                    {
                        "observed_rss_bytes": total,
                        "limit_bytes": limit,
                        "recorded_utc": utc_now(),
                        "execution_id": self.dispatch.execution_id,
                    },
                )
            self.stop.set()

    def cleanup(self) -> None:
        errors = []
        for role, (child, _) in reversed(list(self.children.items())):
            try:
                status = child.stop(grace_seconds=10, kill_wait_seconds=10)
                self.record_exit(role, status)
            except BaseException as exc:
                errors.append(f"{role}: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("Unconfirmed cleanup; no retry allowed: " + "; ".join(errors))

    def complete(self) -> None:
        self.cleanup()
        resident_links = {}
        if self.resident is not None:
            resident_links = {"resident_policy": {
                "binding_sha256": sha256_file(self.directory / "policy_session.json"),
                "reset_sha256": sha256_file(self.directory / "resident_reset.json"),
            }}
        write_once_record(
            self.directory / "completion.json",
            {
                "backend": "direct",
                "execution_id": self.dispatch.execution_id,
                "dispatch_sha256": self.digest,
                "cleanup_confirmed": True,
                "roles": self.roles,
                "finished_utc": utc_now(),
                **resident_links,
            },
        )


def readiness(execution: Execution) -> str | None:
    """Read the server's metadata identity, not just whether its TCP port is open."""
    from openpi_client import msgpack_numpy
    from websockets.exceptions import WebSocketException
    from websockets.sync.client import connect

    expected = {"execution_id": execution.dispatch.execution_id, "dispatch_sha256": execution.digest}
    deadline = min(execution.deadline, time.monotonic() + READINESS_SECONDS)
    child, _ = execution.children["policy"]
    while time.monotonic() < deadline and not execution.stop.is_set():
        execution.check_memory_budget()
        status = child.reap_if_finished()
        if status is not None:
            execution.record_exit("policy", status)
            return "PolicyServerExitedBeforeReadiness"
        try:
            with connect(f"ws://127.0.0.1:{execution.dispatch.policy_port}", open_timeout=2, close_timeout=1) as ws:
                metadata = msgpack_numpy.unpackb(ws.recv(timeout=2))
                if metadata.get("direct_execution") != expected:
                    raise RuntimeError("Listening policy server has wrong execution identity")
                return None
        except (OSError, TimeoutError, WebSocketException):
            execution.stop.wait(0.5)
    if execution.stop.is_set():
        raise InterruptedError("Controller interrupted during policy startup; this is not a readiness timeout")
    return "PolicyServerReadinessTimeout"


def row_commands(
    dispatch: DirectDispatch, row: dict, dispatch_path: Path, digest: str, listen_fd: int
) -> dict[str, list[str]]:
    root = Path(dispatch.run_root)
    policy = str(REPO / ".venv/bin/python")
    simulator = str(REPO / "third_party/robomme_benchmark/.venv/bin/python")
    seed_table = str(root / "protocol/seed_table.json")
    fields = {
        "run-root": str(root),
        "seed-table": seed_table,
        "repo-root": str(REPO),
        "attempt-id": dispatch.attempt_id,
        **{key.replace("_", "-"): value for key, value in row.items()},
    }
    if dispatch.stage == "formal":
        fields["formal-authorization"] = dispatch.submission_plan_sha256
    common = [item for key, value in fields.items() for item in (f"--{key}", str(value))]
    preflight = "formal" if dispatch.stage == "formal" else "smoke"
    eval_args = {
        "host": "127.0.0.1",
        "port": dispatch.policy_port,
        "obs-horizon": 16,
        "max-steps": row["max_steps"],
        "dataset": row["dataset"],
        "only-tasks": row["task"],
        "episode-ids": row["episode_id"],
        "model-seed": 7,
        "model-ckpt-id": 79999,
        "policy-name": f"perceptual-framesamp-modul-neighborhood-{preflight}-{row['row_id']}",
        "save-dir": str(root / "legacy_eval" / str(row["row_id"]) / f"attempt_{dispatch.attempt_id}"),
        "keyframe-selector-arm": row["arm"],
        "keyframe-seed-table": seed_table,
        "keyframe-run-root": str(root),
        "keyframe-attempt-id": dispatch.attempt_id,
        "keyframe-trajectory-kind": row["trajectory_kind"],
        "direct-dispatch-record": str(dispatch_path),
        "direct-dispatch-sha256": digest,
    }
    if dispatch.stage == "formal":
        eval_args["keyframe-formal-authorization"] = dispatch.submission_plan_sha256
    return {
        "preflight": [policy, "-m", f"{MODULE}.preflight_{preflight}_row", *common],
        "policy": [
            policy,
            "scripts/serve_policy.py",
            "--seed=7",
            f"--port={dispatch.policy_port}",
            f"--listen-fd={listen_fd}",
            f"--execution-id={dispatch.execution_id}",
            f"--dispatch-sha256={digest}",
            "policy:checkpoint",
            f"--policy.dir={REPO / CHECKPOINT_RELATIVE}",
            "--policy.config=mme_vla_suite",
        ],
        "evaluator": [
            simulator,
            "examples/robomme/eval.py",
            *[f"--args.{key}={value}" for key, value in eval_args.items()],
        ],
        "reconcile": [
            policy,
            "-m",
            f"{MODULE}.record_launcher_failure",
            *common,
            "--policy-port",
            str(dispatch.policy_port),
        ],
    }


def execute_one(
    run_root: Path,
    stage: str,
    submission_path: Path,
    row: dict | None,
    gpu_uuids: tuple[str, ...],
    stop: threading.Event,
    *,
    resident=None,
) -> None:
    record = json.loads(submission_path.read_text())
    profile = record["runtime_profile"]
    verify_runtime_profile(profile)
    attempt = record.get("attempt_id", 0)
    directory = (
        run_root / "direct/architecture"
        if row is None
        else run_root / "direct" / f"attempt_{attempt:02d}" / f"row_{row['row_id']:04d}"
    )
    if directory.exists():
        raise FileExistsError("Existing execution is pending/completed, not permission to repeat it")
    # Recheck the explicit exclusive/shared rule before and after our own lease.
    # This advisory lease prevents overlap with our jobs, not other users' jobs.
    physical_gpus = tuple(dict.fromkeys(gpu_uuids))
    if resident is None:
        check_gpu_admission(physical_gpus, profile)
    lease_context = (
        GpuLease(Path(profile["lock_directory"]), physical_gpus)
        if resident is None else nullcontext(resident.lease)
    )
    with lease_context as lease:
        admission_snapshot = (
            check_gpu_admission(physical_gpus, profile) if resident is None else {
                "mode": "resident_session_continuation", "session_execution_id": resident.execution.dispatch.execution_id,
                "checked_utc": utc_now(), "fresh_admission_required": False,
            }
        )
        if stop.is_set():
            return
        with PortReservation() as port, ExitStack() as monitors:
            dispatch = DirectDispatch(
                execution_id=str(uuid.uuid4()),
                run_root=str(run_root),
                repository_commit_sha=record["repository_commit_sha"],
                stage=stage,
                launch_manifest_sha256=record["launch_manifest_sha256"],
                submission_plan_sha256=sha256_file(submission_path),
                matrix_sha256=sha256_file(
                    run_root / "protocol" / ("formal_matrix.json" if stage == "formal" else "smoke_matrix.json")
                ),
                attempt_id=attempt,
                row_id=None if row is None else row["row_id"],
                shard_id=record.get("shard_id"),
                gpu_uuids=gpu_uuids,
                host_name=socket.gethostname(),
                host_boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                policy_port=None if row is None else port.port,
                gpu_layout=record.get("gpu_layout", "separate"),
            )
            directory.mkdir(parents=True, exist_ok=False)
            write_once_record(directory / "gpu_admission.json", admission_snapshot)
            execution = Execution(dispatch, directory, profile, lease, stop)
            execution.resident = resident
            monitors.enter_context(GpuTelemetry(directory / "gpu_telemetry.jsonl", physical_gpus))
            try:
                if row is None:
                    execution.start(
                        "architecture",
                        [
                            str(REPO / ".venv/bin/python"),
                            "-m",
                            f"{MODULE}.architecture_smoke",
                            "--run-root",
                            str(run_root),
                        ],
                    )
                    status = execution.wait("architecture")
                    execution.complete()
                    if status != 0:
                        raise RuntimeError(f"Architecture gate exited {status}")
                    return
                commands = row_commands(dispatch, row, execution.path, execution.digest, port.pass_fds[0])
                if resident is not None:
                    binding_path = directory / "policy_session.json"
                    write_once_record(binding_path, resident.binding(execution))
                    commands["policy"] = [
                        str(REPO / ".venv/bin/python"), "-m", f"{MODULE}.resident_policy",
                        "--binding", str(binding_path), "--listen-fd", str(port.pass_fds[0]),
                    ]
                execution.start("preflight", commands["preflight"])
                if execution.wait("preflight") != 0:
                    raise RuntimeError("Row preflight failed; no policy or simulator launched")
                execution.start("policy", commands["policy"], port.pass_fds)
                error_type = readiness(execution)
                if error_type is None:
                    execution.start("evaluator", commands["evaluator"])
                    evaluator_status = execution.wait("evaluator")
                    # Existing failure rules consume shell status (128+signal),
                    # while process evidence correctly retains raw -signal.
                    failure_args = [
                        "--evaluator-exit-status",
                        str(128 - evaluator_status if evaluator_status < 0 else evaluator_status),
                    ]
                else:
                    evaluator_status = 70 if error_type == "PolicyServerExitedBeforeReadiness" else 71
                    failure_args = [
                        "--error-type",
                        error_type,
                        "--error",
                        "Direct policy readiness failed; see immutable process logs",
                    ]
                # Release GPU work before the small CPU reconciliation step.
                execution.cleanup()
                execution.start("reconcile", commands["reconcile"] + failure_args)
                status = execution.wait("reconcile", reconciliation=True)
                execution.complete()
                if status != 0:
                    raise RuntimeError(f"Result reconciliation failed with exit status {status}")
                require_row_outcome(run_root, row, attempt)
                if evaluator_status != 0:
                    write_once_record(
                        directory / "post_result_exit_warning.json",
                        {
                            "evaluator_status": evaluator_status,
                            "scientific_result_preserved": True,
                            "retry_authorized": False,
                            "recorded_utc": utc_now(),
                        },
                    )
            except RowInfrastructureError:
                # Like separate Slurm array tasks: a proved infrastructure
                # failure does not erase other rows' existing authorization.
                # Its cleanup has finished; only an explicit later retry may
                # rerun this row. The batch will still exit nonzero.
                raise
            except BaseException as exc:
                stop.set()
                try:
                    execution.cleanup()
                finally:
                    write_once_record(
                        directory / "controller_error.json",
                        {
                            "backend": "direct",
                            "execution_id": dispatch.execution_id,
                            "dispatch_sha256": execution.digest,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "retry_authorized": False,
                            "recorded_utc": utc_now(),
                        },
                    )
                raise


def run_controller(run_root: Path, stage: str, records: list[tuple[Path, dict]], allocations: list[list[str]]) -> None:
    """Bounded foreground controller. Queue is CPU only; no implicit retries."""
    stop = threading.Event()
    infrastructure_failures = []
    failures_lock = threading.Lock()
    prior_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        prior_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
    try:
        if stage == "architecture_smoke":
            execute_one(run_root, stage, records[0][0], None, tuple(allocations[0]), stop)
            return
        from experiments.keyframe_neighborhood_sampling.formal_matrix import load_formal_matrix
        from experiments.keyframe_neighborhood_sampling.smoke_matrix import load_smoke_matrix

        rows = (
            load_formal_matrix(run_root / "protocol/formal_matrix.json")
            if stage == "formal"
            else load_smoke_matrix(run_root / "protocol/smoke_matrix.json")
        )["rows"]
        # Shards execute consecutively, each with its original concurrency cap;
        # this never exceeds the recorded global cap or overbooks a GPU pair.
        for path, record in records:
            if stop.is_set():
                raise RuntimeError("Controller interrupted; unfinished rows remain pending")
            capacity = min(record["max_concurrent"], len(allocations))
            buckets = [record["row_ids"][slot::capacity] for slot in range(capacity)]

            def worker(slot: int, buckets=buckets, path=path, record=record) -> None:
                if record.get("runtime_profile", {}).get("policy_lifetime") == "resident":
                    from experiments.keyframe_neighborhood_sampling.resident_policy import run_resident_rows
                    try:
                        run_resident_rows(run_root, stage, path, [rows[i] for i in buckets[slot]],
                                          tuple(allocations[slot]), stop)
                    except BaseException:
                        stop.set()
                        raise
                    return
                for row_id in buckets[slot]:
                    if stop.is_set():
                        return
                    try:
                        execute_one(run_root, stage, path, rows[row_id], tuple(allocations[slot]), stop)
                    except RowInfrastructureError as exc:
                        with failures_lock:
                            infrastructure_failures.append(str(exc))
                    except BaseException:
                        stop.set()
                        raise

            with ThreadPoolExecutor(max_workers=capacity) as pool:
                for future in as_completed([pool.submit(worker, slot) for slot in range(capacity)]):
                    future.result()
        if stop.is_set():
            raise RuntimeError("Controller interrupted; audit pending rows before any restart")
        if infrastructure_failures:
            raise RuntimeError(
                f"{len(infrastructure_failures)} audited infrastructure failures; audit and explicitly select retries"
            )
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("architecture_smoke", "development_smoke", "formal"), required=True)
    parser.add_argument("--gpu-layout", choices=("separate", "colocated"), default="separate")
    parser.add_argument("--resident-policy", action="store_true", help="Load once per smoke slot; reset each episode")
    parser.add_argument("--allow-shared-gpu", action="store_true", help="Explicitly authorized sharing only; requires colocated layout")
    parser.add_argument("--shared-min-free-mib", type=int, default=49152)
    parser.add_argument("--shared-max-utilization", type=int, default=80)
    parser.add_argument(
        "--gpu-allocation",
        action="append",
        required=True,
        help="Full UUID for architecture; policyUUID,simUUID for each trajectory slot (repeat UUID for colocated)",
    )
    parser.add_argument("--environment-sh", type=Path, required=True)
    parser.add_argument("--graphics-wrapper", type=Path, required=True)
    parser.add_argument("--lock-directory", type=Path, required=True)
    parser.add_argument("--attempt-id", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--rows", help="Explicit infrastructure-failure retries only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-authorized-extension-smoke-v1", action="store_true")
    parser.add_argument("--confirm-authorized-extension-v1", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        authorized = (
            args.confirm_authorized_extension_v1
            if args.stage == "formal"
            else args.confirm_authorized_extension_smoke_v1
        )
        if not authorized:
            parser.error("Live execution requires the explicit stage-specific authorization flag")
        if os.name != "posix" or not Path("/proc/sys/kernel/random/boot_id").is_file():
            parser.error("Live direct execution requires Linux /proc identity")
    allocations = [value.split(",") for value in args.gpu_allocation]
    root = args.run_root.resolve()
    profile = runtime_profile(args.environment_sh, args.graphics_wrapper, args.lock_directory)
    profile["gpu_layout"] = args.gpu_layout
    if args.resident_policy:
        profile["policy_lifetime"] = "resident"
    if args.allow_shared_gpu:
        profile["gpu_admission"] = {
            "mode": "shared",
            "min_free_memory_mib": args.shared_min_free_mib,
            "max_utilization_gpu_percent": args.shared_max_utilization,
        }
    rows = tuple(int(value) for value in args.rows.split(",")) if args.rows else None
    records, plan = build_direct_submission(root, args.stage, allocations, profile, args.attempt_id, rows)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "backend": "direct",
                    "launches_processes": False,
                    "creates_run_root": False,
                    "records": [record for _, record in records],
                    "plan": plan,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    # Check capacity before publication; record admission without claiming that
    # other users' future memory/compute demand is reserved or predictable.
    snapshots = [check_gpu_admission(tuple(dict.fromkeys(pair)), profile) for pair in allocations]
    for _, record in records:
        record["gpu_admission_snapshots"] = snapshots
    if plan is not None:
        plan["gpu_admission_snapshots"] = snapshots
    publish_submission(root, records, plan)
    run_controller(root, args.stage, records, allocations)


if __name__ == "__main__":
    main()
