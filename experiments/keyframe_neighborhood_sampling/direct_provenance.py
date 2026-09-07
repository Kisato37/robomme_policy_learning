"""Explicit direct-backend bindings shared by runtime gates and offline audits."""

from __future__ import annotations

from collections.abc import Mapping
import datetime as dt
import json
import os
from pathlib import Path
import socket
from typing import Any

from experiments.keyframe_neighborhood_sampling.gpu_admission import admission_policy
from experiments.keyframe_neighborhood_sampling.runner_contract import DirectDispatch
from experiments.keyframe_neighborhood_sampling.runner_contract import canonical_bytes
from experiments.keyframe_neighborhood_sampling.runner_contract import process_exit_record
from experiments.keyframe_neighborhood_sampling.runner_contract import process_start_record
from experiments.keyframe_neighborhood_sampling.runner_contract import require_gpu_uuid
from experiments.keyframe_neighborhood_sampling.runner_contract import require_uuid
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import sha256_file


def runner_backend(payload: Mapping[str, Any]) -> str:
    backend = payload.get("runner_backend", "slurm")
    if not isinstance(backend, str) or backend not in {"slurm", "direct"}:
        raise ArtifactContractError("Unknown or missing explicit runner backend")
    if backend == "direct":
        if any(key == "slurm" or key.startswith("slurm_") for key in payload):
            raise ArtifactContractError("Direct evidence cannot contain fabricated Slurm provenance")
    elif "runner" in payload or "direct_run_id" in payload:
        raise ArtifactContractError("Direct evidence requires explicit runner_backend=direct")
    return backend


def direct_gpu_layout(payload: Mapping[str, Any]) -> str:
    layout = payload.get("gpu_layout", "separate")
    if not isinstance(layout, str) or layout not in {"separate", "colocated"}:
        raise ArtifactContractError("Direct GPU layout must be separate or colocated")
    return layout


def validate_direct_submission(submission: Mapping[str, Any], *, stage: str) -> None:
    if runner_backend(submission) != "direct":
        raise ArtifactContractError("Submission is not explicitly direct")
    require_uuid(submission.get("direct_run_id"), field="direct_run_id")
    layout = direct_gpu_layout(submission)
    profile = submission.get("runtime_profile", {})
    if not isinstance(profile, Mapping) or direct_gpu_layout(profile) != layout:
        raise ArtifactContractError("Direct runtime profile and submission GPU layouts differ")
    try:
        admission_policy(profile)
    except ValueError as exc:
        raise ArtifactContractError("Invalid direct GPU admission policy") from exc
    if profile.get("policy_lifetime", "per_row") not in {"per_row", "resident"}:
        raise ArtifactContractError("Unknown policy lifetime")
    if profile.get("policy_lifetime") == "resident" and (layout != "colocated" or stage == "formal"):
        raise ArtifactContractError("Resident lifecycle is currently a colocated smoke-only backend")
    if stage == "architecture_smoke":
        allocations = [submission.get("gpu_uuids")]
        width = 1
    else:
        allocations = submission.get("gpu_pairs")
        width = 2
    if not isinstance(allocations, list) or not allocations:
        raise ArtifactContractError("Direct submission lacks explicit GPU allocation")
    occupied = set()
    for allocation in allocations:
        if not isinstance(allocation, list) or len(allocation) != width:
            raise ArtifactContractError("Direct GPU allocation shape mismatch")
        for gpu in allocation:
            require_gpu_uuid(gpu)
        physical = set(allocation)
        expected_count = 1 if width == 1 or layout == "colocated" else 2
        if len(physical) != expected_count:
            raise ArtifactContractError("Direct submission GPU allocation overlap or shape differs from its layout")
        if occupied.intersection(physical):
            raise ArtifactContractError("Direct submission GPU allocations overlap across execution slots")
        occupied.update(physical)


def dispatch_path(run_root: Path, *, stage: str, attempt_id: int, row_id: int | None,
                  execution_id: str | None = None) -> Path:
    root = run_root.resolve()
    if stage == "policy_session":
        require_uuid(execution_id, field="execution_id")
        return root / "direct/sessions" / execution_id / "dispatch.json"
    if stage == "architecture_smoke":
        return root / "direct/architecture/dispatch.json"
    if stage not in {"development_smoke", "formal"} or type(row_id) is not int or row_id < 0:
        raise ArtifactContractError("Direct trajectory dispatch requires an explicit stage and row")
    if type(attempt_id) is not int or attempt_id not in {0, 1, 2}:
        raise ArtifactContractError("Invalid direct attempt ID")
    return root / "direct" / f"attempt_{attempt_id:02d}" / f"row_{row_id:04d}" / "dispatch.json"


def _load_canonical(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
        payload = json.loads(data)
    except (OSError, ValueError) as exc:
        raise ArtifactContractError(f"Missing or invalid direct evidence: {path}") from exc
    if not isinstance(payload, dict) or canonical_bytes(payload) != data:
        raise ArtifactContractError(f"Direct evidence must use canonical write-once JSON: {path}")
    return payload


def _load_runner(envelope: Mapping[str, Any], run_root: Path) -> tuple[DirectDispatch, Path]:
    if set(envelope) != {"backend", "dispatch_path", "dispatch_sha256", "dispatch"} or envelope["backend"] != "direct":
        raise ArtifactContractError("Invalid direct runner envelope")
    if not isinstance(envelope["dispatch"], Mapping):
        raise ArtifactContractError("Direct runner lacks dispatch identity")
    dispatch = DirectDispatch.from_record(envelope["dispatch"])
    root = run_root.resolve()
    path = dispatch_path(root, stage=dispatch.stage, attempt_id=dispatch.attempt_id, row_id=dispatch.row_id,
                         execution_id=dispatch.execution_id)
    if envelope["dispatch_path"] != str(path) or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ArtifactContractError("Direct dispatch path escapes or differs from its canonical run location")
    if dispatch.run_root != str(root):
        raise ArtifactContractError("Direct dispatch belongs to a different run root")
    if _load_canonical(path) != envelope["dispatch"] or sha256_file(path) != envelope["dispatch_sha256"]:
        raise ArtifactContractError("Direct dispatch file, embedded identity, and digest disagree")
    return dispatch, path


def live_host_identity() -> tuple[str, str]:
    """No hostname/boot identity is inferred from caller-provided environment variables."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError as exc:
        raise ArtifactContractError("Direct runtime requires live Linux boot identity") from exc
    return socket.gethostname(), boot


def direct_runner_for_runtime(run_root: Path, *, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    environ = dict(os.environ if environ is None else environ)
    if environ.get("KEYFRAME_RUNNER_BACKEND") != "direct":
        raise ArtifactContractError("Direct runtime requires explicit KEYFRAME_RUNNER_BACKEND=direct")
    if any(key.startswith(("SLURM_", "SLURMD_")) for key in environ):
        raise ArtifactContractError("Direct runtime cannot impersonate a Slurm allocation")
    raw_path = environ.get("KEYFRAME_DIRECT_DISPATCH_PATH")
    if not isinstance(raw_path, str) or not raw_path:
        raise ArtifactContractError("Direct runtime lacks its dispatch path")
    path = Path(raw_path)
    envelope = {
        "backend": "direct",
        "dispatch_path": raw_path,
        "dispatch_sha256": environ.get("KEYFRAME_DIRECT_DISPATCH_SHA256"),
        "dispatch": _load_canonical(path),
    }
    dispatch, _ = _load_runner(envelope, Path(run_root))
    if (dispatch.host_name, dispatch.host_boot_id) != live_host_identity():
        raise ArtifactContractError("Direct runtime host or boot differs from its dispatch")
    allocated = tuple(dict.fromkeys(dispatch.gpu_uuids))
    if environ.get("CUDA_VISIBLE_DEVICES") not in {*allocated, ",".join(allocated)}:
        raise ArtifactContractError("Direct CUDA visibility differs from its explicit GPU allocation")
    return envelope


def validate_direct_runner(
    envelope: Mapping[str, Any],
    run_root: Path,
    *,
    stage: str,
    attempt_id: int,
    row_id: int | None,
    submission: Mapping[str, Any],
    submission_path: Path,
) -> DirectDispatch:
    """Offline identity/authorization binding; no live process or GPU is needed."""
    validate_direct_submission(submission, stage=stage)
    if _load_canonical(submission_path) != dict(submission):
        raise ArtifactContractError("Direct submission object differs from its recorded bytes")
    dispatch, _ = _load_runner(envelope, run_root)
    launch_path = run_root / "protocol/launch_manifest.json"
    try:
        launch = json.loads(launch_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ArtifactContractError("Direct execution lacks its launch manifest") from exc
    if not isinstance(launch, Mapping) or direct_gpu_layout(launch) != direct_gpu_layout(submission):
        raise ArtifactContractError("Direct launch manifest and submission GPU layouts differ")
    if sha256_file(launch_path) != dispatch.launch_manifest_sha256:
        raise ArtifactContractError("Direct execution launch manifest bytes differ from its dispatch")
    expected = {
        "stage": stage,
        "attempt_id": attempt_id,
        "row_id": row_id,
        "gpu_layout": direct_gpu_layout(submission),
        "repository_commit_sha": submission.get("repository_commit_sha"),
        "launch_manifest_sha256": submission.get("launch_manifest_sha256"),
        # This field binds the exact stage submission record, not the outer formal plan.
        "submission_plan_sha256": sha256_file(submission_path),
        "matrix_sha256": submission.get("formal_matrix_sha256" if stage == "formal" else "smoke_matrix_sha256"),
        "shard_id": submission.get("shard_id") if stage == "formal" else None,
    }
    if any(getattr(dispatch, field) != value for field, value in expected.items()):
        raise ArtifactContractError("Direct dispatch differs from its submitted stage/row/attempt or artifact identity")
    if stage == "architecture_smoke":
        allocations = [submission["gpu_uuids"]]
    else:
        allocations = submission["gpu_pairs"]
        if row_id not in submission.get("row_ids", []):
            raise ArtifactContractError("Direct row is not authorized by its submission")
    if list(dispatch.gpu_uuids) not in allocations:
        raise ArtifactContractError("Direct dispatch uses a GPU allocation absent from its submission")
    return dispatch


def runtime_direct_provenance(
    run_root: Path,
    *,
    stage: str,
    attempt_id: int,
    row_id: int | None,
    submission: Mapping[str, Any],
    submission_path: Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    envelope = direct_runner_for_runtime(run_root, environ=environ)
    validate_direct_runner(
        envelope,
        run_root,
        stage=stage,
        attempt_id=attempt_id,
        row_id=row_id,
        submission=submission,
        submission_path=submission_path,
    )
    return envelope


def require_runtime_backend(submission: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> str:
    environ = os.environ if environ is None else environ
    backend = runner_backend(submission)
    if environ.get("KEYFRAME_RUNNER_BACKEND", "slurm") != backend:
        raise ArtifactContractError("Live and submitted runner backends differ")
    return backend


def validate_direct_attempt(
    manifest: Mapping[str, Any],
    run_root: Path,
    *,
    attempt_id: int,
    row_id: int,
    trajectory_kind: str,
    required_roles: set[str] | None = None,
) -> DirectDispatch:
    if runner_backend(manifest) != "direct":
        raise ArtifactContractError("Attempt lacks explicit direct provenance")
    envelope = manifest.get("runner", {})
    dispatch, _ = _load_runner(envelope, run_root)
    stage = "formal" if trajectory_kind == "formal" else "development_smoke"
    if dispatch.stage != stage or dispatch.attempt_id != attempt_id or dispatch.row_id != row_id:
        raise ArtifactContractError("Direct attempt stage, row, or attempt identity differs")
    if stage == "formal":
        filename = (
            f"submission_record_shard_{dispatch.shard_id:02d}.json"
            if attempt_id == 0
            else f"submission_record_attempt_{attempt_id:02d}_shard_{dispatch.shard_id:02d}.json"
        )
    else:
        filename = "submission_record.json" if attempt_id == 0 else f"submission_record_attempt_{attempt_id:02d}.json"
    submission_file = run_root / "protocol" / filename
    submission = _load_canonical(submission_file)
    validate_direct_runner(
        envelope,
        run_root,
        stage=stage,
        attempt_id=attempt_id,
        row_id=row_id,
        submission=submission,
        submission_path=submission_file,
    )
    if manifest.get("environment_setup_completed") is True:
        renderer = manifest.get("renderer_device")
        expected_renderer = {
            "cuda_device_id": 0,
            "gpu_uuid": dispatch.gpu_uuids[1],
            "expected_gpu_uuid": dispatch.gpu_uuids[1],
            "can_render": True,
            "is_cuda": True,
            "matches_dispatch": True,
        }
        if (
            not isinstance(renderer, Mapping)
            or any(renderer.get(field) != value for field, value in expected_renderer.items())
            or any(renderer.get(field) is not True for field in ("can_render", "is_cuda", "matches_dispatch"))
            or type(renderer.get("cuda_device_id")) is not int
            or any(
                not isinstance(renderer.get(field), str) or not renderer[field]
                for field in (
                    "render_backend",
                    "simulation_backend",
                    "pci_bus_id",
                )
            )
        ):
            raise ArtifactContractError("Direct simulator renderer identity does not match its allocated GPU")
    if required_roles is not None:
        audit_direct_completion(envelope, run_root, required_roles=required_roles)
    return dispatch


def _validated_role_evidence(dispatch: DirectDispatch, path: Path, role: str) -> tuple[dict, dict]:
    started = _load_canonical(path.parent / f"{role}_start.json")
    exited = _load_canonical(path.parent / f"{role}_exit.json")
    expected_start = process_start_record(
        dispatch,
        process_identity=started.get("process_identity", {}),
        command=started.get("command"),
        working_directory=started.get("working_directory"),
    )
    expected_start["started_utc"] = started.get("started_utc")
    expected_exit = process_exit_record(
        dispatch,
        started_record_sha256=sha256_file(path.parent / f"{role}_start.json"),
        returncode=exited.get("returncode"),
        wall_clock_limit_reached=exited.get("wall_clock_limit_reached"),
    )
    expected_exit["finished_utc"] = exited.get("finished_utc")
    if started != expected_start or exited != expected_exit:
        raise ArtifactContractError("Direct process start/exit evidence has mismatched identity")
    try:
        begin = dt.datetime.fromisoformat(started["started_utc"])
        finish = dt.datetime.fromisoformat(exited["finished_utc"])
        if begin.utcoffset() != dt.timedelta(0) or finish.utcoffset() != dt.timedelta(0) or finish < begin:
            raise ValueError("Invalid UTC interval")
    except (TypeError, ValueError, KeyError) as exc:
        raise ArtifactContractError("Direct process timestamps are invalid") from exc
    return started, exited


def validate_direct_process_exit(envelope: Mapping[str, Any], run_root: Path, *, role: str) -> dict[str, Any]:
    """Validate an already-exited role without requiring the still-running reconciler's completion."""
    dispatch, path = _load_runner(envelope, run_root)
    allowed = (
        {"architecture"}
        if dispatch.stage == "architecture_smoke"
        else {"preflight", "policy", "evaluator", "reconcile"}
    )
    if role not in allowed:
        raise ArtifactContractError("Direct process role differs from its execution stage")
    return _validated_role_evidence(dispatch, path, role)[1]


def audit_direct_completion(envelope: Mapping[str, Any], run_root: Path, *, required_roles: set[str]) -> dict[str, Any]:
    """Require linked process lifecycle and confirmed cleanup, independently of scientific outcome."""
    dispatch, path = _load_runner(envelope, run_root)
    completion = _load_canonical(path.parent / "completion.json")
    expected = {
        "backend": "direct",
        "execution_id": dispatch.execution_id,
        "dispatch_sha256": envelope["dispatch_sha256"],
        "cleanup_confirmed": True,
    }
    if (
        any(completion.get(field) != value for field, value in expected.items())
        or completion.get("cleanup_confirmed") is not True
    ):
        raise ArtifactContractError("Direct execution lacks confirmed matching cleanup evidence")
    roles = completion.get("roles")
    if not isinstance(roles, dict) or not required_roles.issubset(roles):
        raise ArtifactContractError("Direct completion lacks required process roles")
    if set(roles) - {"preflight", "policy", "evaluator", "reconcile", "architecture"}:
        raise ArtifactContractError("Direct completion contains an unknown process role")
    if (dispatch.stage == "architecture_smoke" and set(roles) != {"architecture"}) or (
        dispatch.stage != "architecture_smoke" and "architecture" in roles
    ):
        raise ArtifactContractError("Direct completion roles differ from its execution stage")
    try:
        completed = dt.datetime.fromisoformat(completion["finished_utc"])
        if completed.utcoffset() != dt.timedelta(0):
            raise ValueError("Completion timestamp is not UTC")
    except (TypeError, ValueError, KeyError) as exc:
        raise ArtifactContractError("Direct completion timestamp is invalid") from exc
    process_identities = set()
    for role, links in roles.items():
        start_path = path.parent / f"{role}_start.json"
        exit_path = path.parent / f"{role}_exit.json"
        if links != {"start_sha256": sha256_file(start_path), "exit_sha256": sha256_file(exit_path)}:
            raise ArtifactContractError("Direct completion process digests disagree")
        started, exited = _validated_role_evidence(dispatch, path, role)
        identity = started["process_identity"]
        identity_key = (identity["boot_id"], identity["pid"], identity["start_ticks"])
        if identity_key in process_identities:
            raise ArtifactContractError("Direct process identity is reused across roles")
        process_identities.add(identity_key)
        if completed < dt.datetime.fromisoformat(exited["finished_utc"]):
            raise ArtifactContractError("Direct completion timestamp precedes process exit")
    if dispatch.stage == "policy_session":
        if set(roles) != {"policy"} or "resident_policy" in completion:
            raise ArtifactContractError("Resident session must own exactly one real policy process")
    elif dispatch.stage == "development_smoke":
        name = "submission_record.json" if dispatch.attempt_id == 0 else f"submission_record_attempt_{dispatch.attempt_id:02d}.json"
        submission = _load_canonical(run_root / "protocol" / name)
        resident = submission.get("runtime_profile", {}).get("policy_lifetime") == "resident"
        if resident:
            audit_resident_binding(envelope, run_root, completion)
        elif "resident_policy" in completion or (path.parent / "policy_session.json").exists():
            raise ArtifactContractError("Unsubmitted resident policy evidence")
    return completion


def audit_resident_binding(envelope: Mapping[str, Any], run_root: Path, completion: dict) -> None:
    """Bind each proxy/reset to the real resident process and its final cleanup."""
    from experiments.keyframe_neighborhood_sampling.resident_policy import RESET_STATE  # noqa: PLC0415

    row, path = _load_runner(envelope, run_root)
    binding_path = path.parent / "policy_session.json"
    reset_path = path.parent / "resident_reset.json"
    if completion.get("resident_policy") != {
        "binding_sha256": sha256_file(binding_path), "reset_sha256": sha256_file(reset_path),
    }:
        raise ArtifactContractError("Resident completion lacks exact binding/reset digests")
    binding = _load_canonical(binding_path)
    reset = _load_canonical(reset_path)
    if (binding.get("schema") != "resident-policy-binding-v1"
            or binding.get("row_execution_id") != row.execution_id
            or binding.get("row_dispatch_sha256") != envelope["dispatch_sha256"]):
        raise ArtifactContractError("Resident binding names a different row")
    session, session_path = _load_runner(binding.get("session", {}), run_root)
    if session.stage != "policy_session" or any(getattr(session, field) != getattr(row, field) for field in (
        "run_root", "repository_commit_sha", "launch_manifest_sha256", "submission_plan_sha256",
        "matrix_sha256", "attempt_id", "gpu_uuids", "gpu_layout", "host_name", "host_boot_id",
    )):
        raise ArtifactContractError("Resident policy differs from row source/allocation/submission")
    if binding.get("policy_start_sha256") != sha256_file(session_path.parent / "policy_start.json"):
        raise ArtifactContractError("Resident binding does not name the real process-start evidence")
    plan = _load_canonical(session_path.parent / "row_plan.json")
    if (plan.get("stage") != row.stage or row.row_id not in plan.get("row_ids", [])
            or len(plan["row_ids"]) != len(set(plan["row_ids"]))):
        raise ArtifactContractError("Resident session plan does not contain this row uniquely")
    if (reset.get("schema") != "resident-reset-v1" or reset.get("row_execution_id") != row.execution_id
            or reset.get("session_execution_id") != session.execution_id
            or reset.get("binding_sha256") != sha256_file(binding_path) or reset.get("state") != RESET_STATE
            or type(reset.get("model_process_pid")) is not int or reset["model_process_pid"] <= 0):
        raise ArtifactContractError("Resident reset receipt is incomplete or mismatched")
    matrix = json.loads((run_root / "protocol/smoke_matrix.json").read_bytes())
    config = reset.get("selector_config", {})
    if any(config.get(field) != matrix["rows"][row.row_id][field] for field in ("task", "episode_id", "arm")):
        raise ArtifactContractError("Resident reset configured the wrong trajectory")
    audit_direct_completion(binding["session"], run_root, required_roles={"policy"})
    started, exited = _validated_role_evidence(session, session_path, "policy")
    if not (dt.datetime.fromisoformat(started["started_utc"])
            <= dt.datetime.fromisoformat(reset["recorded_utc"])
            <= dt.datetime.fromisoformat(completion["finished_utc"])
            <= dt.datetime.fromisoformat(exited["finished_utc"])):
        raise ArtifactContractError("Row/reset interval is outside the resident model lifecycle")
