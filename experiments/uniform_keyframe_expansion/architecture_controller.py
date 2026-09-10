"""Separately authorized, write-once architecture diagnostics; default CLI is dry.

This is not a trajectory launcher and cannot grant E2E/formal authority. A single
owned policy worker loads the real checkpoint once, runs the numerical probe,
and exits. The readiness-gate publisher must validate the resulting evidence
after confirmed cleanup; this controller never manufactures a PASS report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import time

from experiments.uniform_keyframe_expansion.artifacts import ExpansionRunStore, _write
from experiments.uniform_keyframe_expansion.contract import canonical_sha256
from experiments.uniform_keyframe_expansion import resident_controller as lifecycle

MODULE = "experiments.uniform_keyframe_expansion.architecture_controller"
STARTUP_SECONDS = lifecycle.STARTUP_SECONDS
PROBE_SECONDS = lifecycle.ROW_SECONDS


def _runtime_plan(plan):
    from experiments.uniform_keyframe_expansion.launch_contract import ValidatedExecutionPlan
    if type(plan) is not ValidatedExecutionPlan:
        raise TypeError("An actual validated architecture execution plan is required")
    plan.revalidate()
    plan.require_runtime_stage("architecture_smoke")
    if plan.rows != []:
        raise ValueError("Architecture diagnostics cannot contain trajectory rows")
    return plan


def _directory(plan):
    path = Path(plan.store_root) / "executions" / plan.execution_identity["execution_id"]
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Architecture execution paths cannot traverse symlinks")
    return path


def _command(plan, plan_path):
    role = plan.environment["roles"]["policy"]
    return [*role["command_prefix"], role["python_executable"], "-m", MODULE,
            "--plan", str(plan_path), "--execute", "--role", "architecture-worker"]


def _read(path):
    from experiments.uniform_keyframe_expansion.launch_contract import file_reference
    reference = file_reference(path)
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("Architecture evidence must contain a JSON mapping")
    return value, reference


def _worker_evidence(plan, directory):
    """Check file/identity binding; numerical gate validation remains separate."""
    complete, _ = _read(directory / "architecture_worker_complete.json")
    if complete.get("execution_identity") != plan.execution_identity:
        raise ValueError("Architecture completion belongs to another execution")
    expected = {"architecture_measurements.json", "live_policy_provenance.json", "architecture_ownership.json"}
    if set(complete.get("artifact_sha256", {})) != expected:
        raise ValueError("Architecture completion lacks its exact measurement/provenance sources")
    for name in sorted(expected):
        value, reference = _read(directory / name)
        if reference["sha256"] != complete["artifact_sha256"][name]:
            raise ValueError("Architecture worker source checksum mismatch")
        if name == "live_policy_provenance.json":
            if value.get("policy_execution_identity") != plan.execution_identity:
                raise ValueError("Architecture model provenance belongs to another execution")
        elif name == "architecture_ownership.json":
            if (value.get("execution_identity") != plan.execution_identity or value.get("role") != "architecture"
                    or value.get("worker_ownership_verified") is not True):
                raise ValueError("Architecture ownership evidence is missing")
    return complete


def _wait_worker(plan, supervisor, directory, *, startup_deadline, deadline):
    while True:
        if supervisor.stop_event.is_set() or time.monotonic() >= deadline:
            raise InterruptedError("Architecture controller interrupted or diagnostic deadline reached")
        status = supervisor.children["architecture"].reap_if_finished()
        if status is not None:
            supervisor.record_exit("architecture", status)
            if status != 0:
                raise RuntimeError(f"Architecture worker exited {status}; no automatic retry")
            return _worker_evidence(plan, directory)
        if time.monotonic() >= startup_deadline and not (directory / "live_policy_provenance.json").is_file():
            raise TimeoutError("Architecture checkpoint bootstrap exceeded its startup deadline")
        time.sleep(0.1)


def execute_architecture(plan):
    plan = _runtime_plan(plan)
    if sys.platform != "linux":
        raise RuntimeError("Actual architecture execution requires Linux ownership checks")
    source_verification = lifecycle.verify_sources(plan)
    from experiments.keyframe_neighborhood_sampling.direct_runtime import GpuLease
    ExpansionRunStore.open(plan.store_root)
    directory = _directory(plan)
    directory.mkdir(parents=True, exist_ok=False)
    plan_path = directory / "execution_plan.json"
    _write(plan_path, plan.payload)
    _write(directory / "source_verification.json", source_verification)
    stop = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with GpuLease(lifecycle.host_lock_directory(plan), [plan.gpu_uuid]) as lease:
            _write(directory / "gpu_admission.json", lifecycle.check_gpu_admission(plan))
            supervisor = lifecycle.Supervisor(plan, directory, lease.pass_fds, stop)
            try:
                started = time.monotonic()
                deadline = started + STARTUP_SECONDS + PROBE_SECONDS
                supervisor.start("architecture", "policy", _command(plan, plan_path), deadline)
                _wait_worker(plan, supervisor, directory, startup_deadline=started + STARTUP_SECONDS, deadline=deadline)
            except BaseException as primary:
                try:
                    supervisor.close()
                except BaseException as cleanup:
                    primary.add_note(f"Owned architecture cleanup unconfirmed: {type(cleanup).__name__}: {cleanup}")
                raise
            else:
                supervisor.close()
        _write(directory / "controller_complete.json", {
            "execution_identity": plan.execution_identity, "owned_process_cleanup_confirmed": True,
            "scope": "architecture worker execution only; numerical readiness requires the gate publisher",
        })
        from experiments.uniform_keyframe_expansion.architecture_artifacts import publish_architecture_gate
        publish_architecture_gate(plan, directory)
    except BaseException as error:
        _failure(directory / "controller_failure.json", plan, error, "architecture controller")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return directory


def _failure(path, plan, error, scope):
    try:
        _write(path, {"execution_identity": plan.execution_identity,
                      "exception_type": type(error).__name__, "message": str(error),
                      "notes": list(getattr(error, "__notes__", [])), "automatic_retry": False,
                      "requires_review": True, "scope": scope + "; not a scientific outcome or PASS gate"})
    except Exception as logging_error:
        error.add_note(f"Architecture failure log unavailable: {logging_error}")


def run_architecture_worker(plan):
    plan = _runtime_plan(plan)
    directory = _directory(plan)
    # Ownership must succeed before writing worker evidence or importing GPU code.
    from experiments.uniform_keyframe_expansion.worker_ownership import verify_worker_ownership
    proof = verify_worker_ownership(plan, "architecture")
    try:
        _write(directory / "architecture_ownership.json", proof)
        from experiments.uniform_keyframe_expansion.server_bootstrap import load_authorized_policy
        loaded = load_authorized_policy(plan)
        _write(directory / "live_policy_provenance.json", loaded.provenance)
        from experiments.uniform_keyframe_expansion.architecture_probe import run_loaded_probe
        measurements = run_loaded_probe(loaded.policy, plan, output_dir=directory / "probe")
        _write(directory / "architecture_measurements.json", measurements)
        from experiments.uniform_keyframe_expansion.launch_contract import file_reference
        _write(directory / "architecture_worker_complete.json", {
            "execution_identity": plan.execution_identity,
            "artifact_sha256": {name: file_reference(directory / name)["sha256"] for name in (
                "architecture_measurements.json", "live_policy_provenance.json", "architecture_ownership.json")},
            "scope": "measurements collected; not a readiness gate",
        })
        return measurements
    except BaseException as error:
        _failure(directory / "architecture_worker_failure.json", plan, error, "architecture worker")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--execute", action="store_true", help="Requires separate architecture-smoke authorization")
    parser.add_argument("--role", choices=("controller", "architecture-worker"), default="controller")
    args = parser.parse_args(argv)
    plan = _runtime_plan(lifecycle.load_plan(args.plan))
    if not args.execute:
        print(json.dumps({"stage": plan.stage, "row_count": 0, "launches_processes": False,
                          "execution_identity": plan.execution_identity,
                          "plan_sha256": canonical_sha256(plan.payload)}, indent=2))
        return 0
    if args.role == "controller":
        execute_architecture(plan)
    else:
        run_architecture_worker(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
