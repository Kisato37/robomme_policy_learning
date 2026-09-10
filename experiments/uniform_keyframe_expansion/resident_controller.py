"""Explicitly gated Linux resident execution; importing/dry-running launches nothing.

One owned policy process serves all submitted rows sequentially. Each evaluator
is a fresh process/episode. Only lifecycle primitives are reused from the older
family: none of its arm, matrix, budget or artifact contracts are relaxed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time

from experiments.uniform_keyframe_expansion.artifacts import ExpansionRunStore, _write
from experiments.uniform_keyframe_expansion.contract import canonical_sha256

MODULE = "experiments.uniform_keyframe_expansion.resident_controller"
WATCHDOG = "experiments.keyframe_neighborhood_sampling.direct_child"
STARTUP_SECONDS = 1800
ROW_SECONDS = 12 * 60 * 60


def verify_sources(plan):
    from experiments.uniform_keyframe_expansion.server_bootstrap import verify_controller_sources
    return verify_controller_sources(plan)


def verify_deep_evidence(plan):
    from experiments.uniform_keyframe_expansion.launch_contract import deep_validate_runtime_evidence
    return deep_validate_runtime_evidence(plan)


def verify_worker(plan, role, row_id):
    from experiments.uniform_keyframe_expansion.worker_ownership import verify_worker_ownership
    proof = verify_worker_ownership(plan, role, row_id=row_id)
    name = "policy" if role == "policy" else f"row_{row_id:04d}"
    directory = Path(plan.store_root) / "executions" / plan.execution_identity["execution_id"]
    _write(directory / f"{name}_ownership.json", proof)
    return proof


def host_lock_directory(plan):
    directory = Path(plan.environment["lock_directory"])
    if not directory.is_absolute() or any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("GPU lease directory must be an absolute nonsymlink host-wide path")
    info = directory.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ValueError("GPU lease directory must be user-owned and not writable by other users")
    if directory == Path(plan.store_root) or Path(plan.store_root) in directory.parents:
        raise ValueError("GPU lease directory must be shared across runs, not inside this run")
    return directory


def load_plan(path):
    from experiments.uniform_keyframe_expansion.launch_contract import validate_execution_plan
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("Use a canonical absolute plan path, without symlinks")
    return validate_execution_plan(json.loads(path.read_bytes()))


def _runtime_plan(plan):
    from experiments.uniform_keyframe_expansion.launch_contract import ValidatedExecutionPlan
    if not isinstance(plan, ValidatedExecutionPlan):
        raise TypeError("A verified expansion execution plan is required")
    plan.revalidate()
    plan.require_runtime_stage()
    if plan.stage not in {"end_to_end_smoke", "formal"}:
        raise ValueError("Resident trajectory controller accepts only E2E smoke or formal plans")
    return plan


def _environment(plan, role):
    record = plan.environment["roles"][role]
    env = dict(os.environ)
    # The environment evidence is explicit and will be rechecked inside the
    # bootstrap after any hash-bound setup wrapper. Never carry old dispatch IDs.
    if any(key.startswith("KEYFRAME_") for key in env):
        raise ValueError("An old experiment dispatch is present in the inherited environment")
    env.update(record["process_environment"])
    if env.get("CUDA_VISIBLE_DEVICES") != plan.gpu_uuid:
        raise ValueError("Child environment does not bind the requested physical GPU")
    return env


def _command(plan, role, plan_path, port, *, listen_fd=None, row_id=None):
    record = plan.environment["roles"][role]
    command = [*record["command_prefix"], record["python_executable"], "-m", MODULE,
               "--plan", str(plan_path), "--execute", "--role", role, "--port", str(port)]
    if role == "policy":
        if type(listen_fd) is not int or listen_fd < 3:
            raise ValueError("Policy must inherit its already-bound listening socket")
        command += ["--listen-fd", str(listen_fd)]
    elif role == "simulator":
        if type(row_id) is not int or row_id not in {r["row_id"] for r in plan.rows}:
            raise ValueError("Evaluator row is outside the authorized shard")
        command += ["--row-id", str(row_id)]
    else:
        raise ValueError("Unknown child role")
    return command


def check_gpu_admission(plan):
    minimum = plan.environment["hardware"].get("minimum_free_memory_mib")
    if type(minimum) is not int or minimum <= 0:
        raise ValueError("Reviewed positive minimum_free_memory_mib is required")
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    matches = []
    for line in output.splitlines():
        gpu, free = [v.strip() for v in line.split(",")]
        if gpu == plan.gpu_uuid:
            matches.append(int(free))
    if len(matches) != 1 or matches[0] < minimum:
        raise RuntimeError("Selected GPU missing or below its reviewed free-memory threshold")
    return {"gpu_uuid": plan.gpu_uuid, "free_memory_mib": matches[0],
            "minimum_free_memory_mib": minimum, "shared_gpu_allowed": True}


def _check_rows_unused(store, rows):
    # We do not infer resume/retry permission from an old partial attempt.
    # A reviewed new plan must name unused rows; this initial runner uses attempt0.
    for row in rows:
        directory = store.attempt_dir(row, 0).parent
        if directory.exists() and any(directory.glob("attempt_*")):
            raise RuntimeError(f"Row {row['row_id']} already has attempt evidence; review, do not overwrite/retry")


class Supervisor:
    """Pin unreaped child identities and release payloads only after recording ownership."""

    def __init__(self, plan, directory, pass_fds, stop):
        self.plan, self.directory, self.pass_fds, self.stop_event = plan, directory, pass_fds, stop
        self.children = {}
        self.exits = {}

    def start(self, name, role, command, deadline, extra_fds=()):
        from experiments.keyframe_neighborhood_sampling.direct_runtime import LinuxProcessOperations, OwnedProcessGroup
        if self.stop_event.is_set():
            raise InterruptedError("Stopped before child launch")
        verify_sources(self.plan)
        operations = LinuxProcessOperations()
        parent = operations.identity(os.getpid())
        if parent is None:
            raise RuntimeError("Cannot establish supervisor ownership")
        start_read, start_write = os.pipe()
        inherited = (*self.pass_fds, *extra_fds)
        # Watchdog stays in its own process group and kills only that owned
        # group's descendants if the controller disappears or its deadline expires.
        argv = [self.plan.environment["roles"]["policy"]["python_executable"], "-m", WATCHDOG,
                "--start-fd", str(start_read), "--parent-pid", str(parent.pid),
                "--parent-start-ticks", str(parent.start_ticks), "--boot-id", parent.boot_id,
                "--deadline-monotonic", str(deadline)]
        for fd in inherited:
            argv += ["--pass-fd", str(fd)]
        argv += ["--", *command]
        child = owned = None
        try:
            with (self.directory / f"{name}.out").open("xb") as stdout, (self.directory / f"{name}.err").open("xb") as stderr:
                child = subprocess.Popen(argv, cwd=self.plan.policy_root, env=_environment(self.plan, role),
                                         stdout=stdout, stderr=stderr, start_new_session=True,
                                         pass_fds=(*inherited, start_read))
            owned = OwnedProcessGroup(child, operations)
            self.children[name] = owned
            _write(self.directory / f"{name}_start.json", {
                "execution_identity": self.plan.execution_identity, "role": role,
                "process_identity": asdict(owned.identity), "command": argv,
                "controller_identity": asdict(parent),
                "deadline_monotonic": deadline, "cwd": str(self.plan.policy_root),
            })
            if self.stop_event.is_set():
                raise InterruptedError("Stopped before releasing child startup barrier")
            os.write(start_write, b"1")
        except BaseException:
            os.close(start_write)
            start_write = -1
            if child is not None and owned is None:
                child.wait(timeout=15)  # Closed pipe forbids any payload startup.
            raise
        finally:
            os.close(start_read)
            if start_write >= 0:
                os.close(start_write)
        return owned

    def record_exit(self, name, status):
        if name not in self.exits:
            _write(self.directory / f"{name}_exit.json", {
                "execution_identity": self.plan.execution_identity, "returncode": status,
                "start_sha256": _sha(self.directory / f"{name}_start.json"),
            })
            self.exits[name] = status

    def wait_row(self, name, deadline):
        while True:
            if self.stop_event.is_set() or time.monotonic() >= deadline:
                raise InterruptedError("Controller interrupted or evaluator deadline reached")
            status = self.children[name].reap_if_finished()
            if status is not None:
                self.record_exit(name, status)
                if status != 0:
                    raise RuntimeError(f"Evaluator {name} exited {status}; no automatic retry")
                return
            if self.children["policy"].reap_if_finished() is not None:
                raise RuntimeError("Resident policy exited while an episode was running")
            time.sleep(0.1)

    def close(self):
        failures = []
        for name, owned in reversed(list(self.children.items())):
            try:
                status = owned.stop(grace_seconds=10, kill_wait_seconds=10)
                self.record_exit(name, status)
            except Exception as error:
                failures.append(f"{name}: {type(error).__name__}: {error}")
        if failures:
            raise RuntimeError("Owned process cleanup unconfirmed: " + "; ".join(failures))


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _wait_ready(plan, supervisor, port, deadline):
    from experiments.uniform_keyframe_expansion.serving import ExpansionClient
    while True:
        if supervisor.stop_event.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("Resident policy readiness timed out or was interrupted")
        if supervisor.children["policy"].reap_if_finished() is not None:
            raise RuntimeError("Policy exited before readiness")
        try:
            with ExpansionClient("127.0.0.1", port, expected_execution_identity=plan.execution_identity,
                                 connect_timeout=1, response_timeout=1) as client:
                metadata = client.get_server_metadata()
            owned = supervisor.children["policy"]
            members = owned.operations.live_group_members(owned.identity.process_group)
            if metadata["model_process_pid"] not in {m.pid for m in members}:
                raise RuntimeError("Ready policy PID is not inside the owned server process group")
            return metadata
        except (ConnectionRefusedError, ConnectionResetError, TimeoutError):
            time.sleep(0.2)


def execute_resident(plan):
    plan = _runtime_plan(plan)
    if sys.platform != "linux":
        raise RuntimeError("Actual resident execution requires Linux ownership checks")
    source_verification = verify_sources(plan)
    deep_verification = verify_deep_evidence(plan)
    from experiments.keyframe_neighborhood_sampling.direct_runtime import GpuLease, PortReservation
    store = ExpansionRunStore.open(plan.store_root)
    _check_rows_unused(store, plan.rows)
    directory = store.run_root / "executions" / plan.execution_identity["execution_id"]
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("Execution paths must not traverse symlinks")
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "execution_plan.json"
    _write(path, plan.payload)
    _write(directory / "source_verification.json", source_verification)
    _write(directory / "deep_evidence_verification.json", deep_verification)
    # Admission is checked only at startup: useful utilization by this batch is
    # never a reason to evict it. Advisory lease covers cooperating shards only.
    stop = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    supervisor = None
    try:
        with GpuLease(host_lock_directory(plan), [plan.gpu_uuid]) as lease, PortReservation() as port:
            _write(directory / "gpu_admission.json", check_gpu_admission(plan))
            supervisor = Supervisor(plan, directory, lease.pass_fds, stop)
            try:
                total_deadline = time.monotonic() + STARTUP_SECONDS + len(plan.rows) * (ROW_SECONDS + 120)
                supervisor.start("policy", "policy", _command(plan, "policy", path, port.port, listen_fd=port.pass_fds[0]),
                                 total_deadline, extra_fds=port.pass_fds)
                ready = _wait_ready(plan, supervisor, port.port, time.monotonic() + STARTUP_SECONDS)
                _write(directory / "policy_ready.json", ready)
                for row in plan.rows:
                    _check_rows_unused(store, [row])
                    name = f"row_{row['row_id']:04d}"
                    deadline = time.monotonic() + ROW_SECONDS
                    supervisor.start(name, "simulator", _command(plan, "simulator", path, port.port, row_id=row["row_id"]), deadline)
                    supervisor.wait_row(name, deadline)
                    audit = store.audit_attempt(row, 0)
                    if audit.get("status") != "complete":
                        raise RuntimeError("Evaluator exit is not a validated scientific completion")
                    if plan.stage == "end_to_end_smoke" and audit.get("smoke_readiness_pass") is not True:
                        raise RuntimeError("Preserved benchmark outcome fails smoke readiness; review required")
                _write(directory / "shard_outcomes_complete.json", {
                    "execution_identity": plan.execution_identity, "row_ids": [r["row_id"] for r in plan.rows],
                    "scope": "submitted shard only, not whole-run completeness or GPU readiness",
                })
            except BaseException as primary_error:
                try:
                    supervisor.close()
                except BaseException as cleanup_error:
                    primary_error.add_note(f"Owned process cleanup unconfirmed: {type(cleanup_error).__name__}: {cleanup_error}")
                raise
            else:
                supervisor.close()
        _write(directory / "controller_complete.json", {"execution_identity": plan.execution_identity,
                                                        "owned_process_cleanup_confirmed": True})
    except BaseException as error:
        try:
            _write(directory / "controller_failure.json", {
                "exception_type": type(error).__name__, "message": str(error),
                "notes": list(getattr(error, "__notes__", [])),
                "automatic_retry": False, "requires_review": True,
                "scope": "controller failure, never a synthesized episode outcome",
            })
        except Exception as logging_error:
            error.add_note(f"Controller failure log unavailable: {logging_error}")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return directory


def run_child(plan, role, *, port, listen_fd=None, row_id=None):
    plan = _runtime_plan(plan)
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Child port must be an explicit nonprivileged loopback port")
    if role == "policy":
        if type(listen_fd) is not int or listen_fd < 3:
            raise ValueError("A held listener must be inherited, not rebound")
        verify_worker(plan, role, row_id)
        from experiments.uniform_keyframe_expansion.server_bootstrap import load_authorized_policy
        from experiments.uniform_keyframe_expansion.serving import ExpansionPolicyServer
        loaded = load_authorized_policy(plan)
        directory = Path(plan.store_root) / "executions" / plan.execution_identity["execution_id"]
        _write(directory / "live_policy_provenance.json", loaded.provenance)
        ExpansionPolicyServer(loaded.policy, execution_identity=plan.execution_identity,
                              host="127.0.0.1", port=port, listen_fd=listen_fd).serve_forever()
        raise RuntimeError("Resident policy server returned unexpectedly")
    if role != "simulator" or type(row_id) is not int:
        raise ValueError("Unknown/missing evaluator role or row")
    rows = [row for row in plan.rows if row["row_id"] == row_id]
    if len(rows) != 1:
        raise ValueError("Requested row is not in the authorized shard")
    verify_worker(plan, role, row_id)
    from experiments.uniform_keyframe_expansion.server_bootstrap import prepare_benchmark_runtime
    from experiments.uniform_keyframe_expansion.evaluator import evaluate_attempt
    bindings = prepare_benchmark_runtime(plan, policy_port=port)
    store = ExpansionRunStore.open(plan.store_root)
    _check_rows_unused(store, rows)
    return evaluate_attempt(store, rows[0], 0, env_factory=bindings.env_factory,
                            client_factory=bindings.client_factory, components=bindings.components,
                            episode_provenance=bindings.provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--execute", action="store_true", help="Requires a separately authorized, validated plan")
    parser.add_argument("--role", choices=("controller", "policy", "simulator"), default="controller")
    parser.add_argument("--port", type=int)
    parser.add_argument("--listen-fd", type=int)
    parser.add_argument("--row-id", type=int)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    if not args.execute:
        print(json.dumps({"stage": plan.stage, "row_count": len(plan.rows), "launches_processes": False,
                          "execution_identity": plan.execution_identity,
                          "plan_sha256": canonical_sha256(plan.payload)}, indent=2))
        return 0
    if args.role == "controller":
        if args.port is not None or args.listen_fd is not None or args.row_id is not None:
            raise ValueError("Controller reserves its port and follows the exact plan rows")
        execute_resident(plan)
    else:
        run_child(plan, args.role, port=args.port, listen_fd=args.listen_fd, row_id=args.row_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
