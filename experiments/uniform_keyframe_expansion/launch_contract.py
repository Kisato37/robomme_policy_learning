"""Read-only execution-plan and evidence-chain gates for the expansion family.

No subprocess, socket, model, GPU query, or output writer lives here. File hashes
verify what was reviewed, not who wrote it or whether a host/GPU declaration is
true. The production bootstrap must independently check live Git, interpreter,
packages, environment and physical GPU identity before using a validated plan.
The launcher must reserve/publish its execution UUID once, without replacement.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any
import uuid
import xml.etree.ElementTree as ET

from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_identity, CHECKPOINT_CONTENT_TREE_ALGORITHM
from experiments.uniform_keyframe_expansion import contract as c
from experiments.uniform_keyframe_expansion.artifacts import ExpansionRunStore

STAGES = ("cpu_prepare", "architecture_smoke", "end_to_end_smoke", "formal")
_SEAL = object()
BASE_EVIDENCE = {"protocol", "policy_source", "benchmark_source", "environment", "checkpoint"}
ARCH_CHECKS = ("strict_real_checkpoint_load", "short_long_both_arms", "shapes_masks_finite",
               "same_process_repeatability", "reset_isolation", "padded_u_positional_diagnostic",
               "latency_and_peak_memory", "no_training", "causal_development_inputs")
class ExpansionLaunchError(ValueError):
    pass


def _same(actual, expected, label):
    if c.canonical_json(actual) != c.canonical_json(expected):
        raise ExpansionLaunchError(f"{label} does not match the bound execution")


def _digest(value, label, digits=64):
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{digits}}}", value) is None:
        raise ExpansionLaunchError(f"Invalid {label} digest")
    return value


def _path(value, label, *, directory=False):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ExpansionLaunchError(f"{label} must be an absolute path")
    path = Path(value)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ExpansionLaunchError(f"{label} may not traverse a symbolic link")
    if not (path.is_dir() if directory else path.is_file()):
        raise ExpansionLaunchError(f"Missing {label}: {path}")
    return path


def file_reference(path: str | Path) -> dict[str, str]:
    """Hash an existing regular file; no file is created or modified."""
    path = _path(str(Path(path).absolute()), "evidence file")
    return {"path": str(path), "sha256": sha256_file(path)}


def _read_ref(ref, label):
    if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
        raise ExpansionLaunchError(f"{label} needs an exact path/SHA-256 file reference")
    _digest(ref["sha256"], label)
    path = _path(ref["path"], label)
    _same(sha256_file(path), ref["sha256"], f"{label} file checksum")
    return path


def _json_ref(ref, label, kind=None):
    try:
        value = json.loads(_read_ref(ref, label).read_text())
    except (ValueError, OSError) as error:
        raise ExpansionLaunchError(f"Unreadable {label}") from error
    if not isinstance(value, dict):
        raise ExpansionLaunchError(f"{label} must contain a JSON mapping")
    c.canonical_json(value)
    if kind is not None:
        _same(value.get("schema_version"), 1, f"{label} schema")
        _same(value.get("kind"), kind, f"{label} kind")
    return value


def _sources(value, label):
    if not isinstance(value, list) or not value:
        raise ExpansionLaunchError(f"{label} requires underlying source files, not a PASS declaration")
    paths = [_read_ref(ref, f"{label} source") for ref in value]
    if len(set(paths)) != len(paths):
        raise ExpansionLaunchError(f"Duplicate {label} sources")
    return paths


def _source(ref, role, revision):
    source = _json_ref(ref, f"{role} source", "source_checkout")
    root = _path(source.get("root"), f"{role} repository", directory=True)
    _same(source.get("revision"), revision, f"{role} revision")
    _digest(revision, f"{role} revision", 40)
    _same(source.get("clean"), True, f"{role} clean declaration")
    if not isinstance(source.get("branch"), str) or not source["branch"]:
        raise ExpansionLaunchError("Source branch evidence is required")
    if role == "policy" and not source["branch"].startswith("exp/"):
        raise ExpansionLaunchError("Policy checkout must be on an exp/* branch")
    files = source.get("files")
    if not isinstance(files, list) or not files:
        raise ExpansionLaunchError("Source content manifest cannot be empty")
    seen = set()
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ExpansionLaunchError("Invalid source content entry")
        rel = entry["path"]
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts or rel in seen:
            raise ExpansionLaunchError("Invalid/duplicate relative source path")
        seen.add(rel)
        _read_ref({"path": str(root / rel), "sha256": entry["sha256"]}, "source content")
    _sources(source.get("sources"), f"{role} provenance")
    return source


def _environment(ref, host, gpu_uuid):
    env = _json_ref(ref, "environment", "runtime_environment")
    _same(env.get("host"), host, "execution host")
    _path(env.get("lock_directory"), "host-wide lock directory", directory=True)
    roles = env.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {"policy", "simulator"}:
        raise ExpansionLaunchError("Separate policy/simulator runtime profiles are required")
    source_paths = _sources(env.get("sources"), "environment provenance")
    for role, profile in roles.items():
        if not isinstance(profile, Mapping):
            raise ExpansionLaunchError("Invalid runtime profile")
        executable = profile.get("python_executable")
        # Virtualenv interpreter symlinks are normal. Live bootstrap checks the
        # actual interpreter/venv; evidence files themselves cannot be symlinks.
        if not isinstance(executable, str) or not Path(executable).is_absolute() or not Path(executable).is_file():
            raise ExpansionLaunchError("Runtime interpreter is missing")
        if not isinstance(profile.get("python_version"), str) or not profile["python_version"]:
            raise ExpansionLaunchError("Python version is required")
        packages = profile.get("packages")
        if not isinstance(packages, Mapping) or not packages or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in packages.items()):
            raise ExpansionLaunchError("Pinned package versions are required")
        variables = profile.get("process_environment")
        if not isinstance(variables, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in variables.items()):
            raise ExpansionLaunchError("Process environment must be an explicit string mapping")
        if gpu_uuid is not None:
            _same(variables.get("CUDA_VISIBLE_DEVICES"), gpu_uuid, f"{role} physical GPU mapping")
        # Imports may perform a small, explicitly recorded simulator-only
        # transition. Validate it before any launch; the worker checks its exact
        # post-import values. It is never applied to a child's startup env.
        from experiments.uniform_keyframe_expansion.server_bootstrap import BootstrapError, _expected_process_environment
        try:
            _expected_process_environment(profile, role)
        except BootstrapError as error:
            raise ExpansionLaunchError(str(error)) from error
        prefix = profile.get("command_prefix")
        if not isinstance(prefix, list) or any(not isinstance(part, str) or not part for part in prefix):
            raise ExpansionLaunchError("command_prefix must be an argv list, possibly empty")
        if prefix:
            _path(prefix[0], "runtime wrapper executable")
            if any(part in {"-c", "-lc", "-ic", "--command"} for part in prefix[1:]):
                raise ExpansionLaunchError("Shell command strings are forbidden; use a bound script")
            if Path(prefix[0]).name in {"sh", "bash", "zsh", "dash", "ksh"}:
                if len(prefix) != 2 or _path(prefix[1], "runtime wrapper script") not in source_paths:
                    raise ExpansionLaunchError("Shell wrapper must name exactly one hashed script")
            elif Path(prefix[0]) not in source_paths:
                raise ExpansionLaunchError("Runtime wrapper must have a hashed environment source")
        _sources(profile.get("lockfiles"), f"{role} locks")
    if not isinstance(env.get("hardware"), Mapping) or not env["hardware"]:
        raise ExpansionLaunchError("Hardware provenance is required")
    if gpu_uuid is not None:
        _same(env["hardware"].get("gpu_uuid"), gpu_uuid, "physical GPU evidence")
        minimum = env["hardware"].get("minimum_free_memory_mib")
        if type(minimum) is not int or minimum <= 0:
            raise ExpansionLaunchError("A measured/reviewed positive startup free-memory threshold is required")
    return env


def _checkpoint(ref):
    value = _json_ref(ref, "checkpoint", "checkpoint_source")
    directory = _path(value.get("checkpoint_dir"), "checkpoint", directory=True)
    if directory.name != "79999":
        raise ExpansionLaunchError("Only the frozen 79999 checkpoint may be used")
    archive = value.get("archive")
    if not isinstance(archive, Mapping) or set(archive) != {"path", "sha256"}:
        raise ExpansionLaunchError("Checkpoint archive source reference is required")
    _path(archive["path"], "checkpoint archive")
    _same(value["archive"]["sha256"], c.CHECKPOINT_ARCHIVE_SHA256, "released archive")
    history = _read_ref(value.get("history_config"), "checkpoint history configuration")
    _same(str(history), str(directory.parent / "history_config.txt"), "original checkpoint-side metadata")
    # Cheap census/size check, not live parameter-byte verification. Each model
    # worker must stream the real archive and content tree ONCE before loading.
    # Simulator per-row startup must not reread 12 GB of weights and archive.
    _same(checkpoint_identity(directory), value.get("metadata"), "unpacked metadata")
    manifest = _json_ref(value.get("content_manifest"), "checkpoint content manifest")
    _same(manifest.get("algorithm"), CHECKPOINT_CONTENT_TREE_ALGORITHM, "content tree algorithm")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ExpansionLaunchError("Checkpoint per-file content manifest cannot be empty")
    actual_files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ExpansionLaunchError("Checkpoint tree cannot contain links or special files")
        if path.is_file():
            actual_files.append({"path": path.relative_to(directory).as_posix(), "size": path.stat().st_size})
    recorded_files = []
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "size", "sha256"}:
            raise ExpansionLaunchError("Malformed checkpoint content entry")
        _digest(entry["sha256"], "checkpoint file")
        recorded_files.append({"path": entry["path"], "size": entry["size"]})
    _same(recorded_files, actual_files, "checkpoint file census/sizes")
    expected_tree = {"algorithm": CHECKPOINT_CONTENT_TREE_ALGORITHM,
                     "content_tree_sha256": c.canonical_sha256(manifest),
                     "file_count": len(files), "total_bytes": sum(row["size"] for row in files)}
    _same(value.get("content_tree"), expected_tree, "recorded checkpoint content-tree digest")
    _sources(value.get("sources"), "checkpoint extraction/source provenance")
    return value


def _binding(request, policy_source, benchmark_source, checkpoint):
    return {
        "policy_commit": policy_source["revision"], "benchmark_commit": benchmark_source["revision"],
        "environment_sha256": request["evidence"]["environment"]["sha256"],
        "checkpoint_archive_sha256": c.CHECKPOINT_ARCHIVE_SHA256,
        "checkpoint_content_tree_sha256": checkpoint["content_tree"]["content_tree_sha256"],
        "inference_contract_sha256": c.canonical_sha256(c.frozen_inference_contract()),
    }


def _gate(ref, kind, binding):
    gate = _json_ref(ref, kind, kind)
    _same(gate.get("protocol_family"), c.PROTOCOL_FAMILY, f"{kind} family")
    _same(gate.get("binding"), binding, f"{kind} runtime binding")
    _same(gate.get("status"), "PASS", f"{kind} status")
    _sources(gate.get("sources"), kind)
    return gate


def _cpu_gate(ref, binding):
    gate = _gate(ref, "cpu_gate", binding)
    junit = _read_ref(gate.get("junit"), "CPU test results")
    if gate["junit"] not in gate["sources"]:
        raise ExpansionLaunchError("CPU raw test results must be bound in sources")
    try:
        root = ET.fromstring(junit.read_bytes())
        cases = list(root.iter("testcase"))
        if not cases or any(list(case.iter("failure")) or list(case.iter("error")) for case in cases):
            raise ValueError("no passing test census")
        count = sum(not list(case.iter("skipped")) for case in cases)
        if count < 1 or type(gate.get("passed_test_count")) is not int or gate["passed_test_count"] != count:
            raise ValueError("wrong passing count")
    except (ValueError, ET.ParseError) as error:
        raise ExpansionLaunchError("CPU JUnit evidence is not passing") from error


def _architecture_gate(ref, binding):
    gate = _gate(ref, "architecture_gate", binding)
    raw = _json_ref(gate.get("measurements"), "architecture measurements", "architecture_measurements")
    validate_architecture_measurements(gate, raw, binding)
    from experiments.uniform_keyframe_expansion.architecture_artifacts import validate_execution_record
    validate_execution_record(gate, raw, binding)


def validate_architecture_measurements(gate, raw, binding):
    """Validate small measurement records; actual raw arrays have a deep gate."""
    _same(gate.get("checks"), {key: True for key in ARCH_CHECKS}, "architecture checks")
    if gate["measurements"] not in gate["sources"]:
        raise ExpansionLaunchError("Architecture measurements must be a bound source")
    _same(raw.get("binding"), binding, "architecture measurements binding")
    _same(raw.get("scope"), "development_only", "architecture development scope")
    _same(raw.get("strict_load"), {"missing": [], "extra": [], "random_initialized": []}, "strict load evidence")
    cases = raw.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ExpansionLaunchError("Architecture needs both arms with short and long histories")
    covered = set()
    for case in cases:
        if not isinstance(case, Mapping) or case.get("arm") not in c.EXPANSION_ARMS or type(case.get("history_length")) is not int or case["history_length"] < 1:
            raise ExpansionLaunchError("Invalid architecture case")
        if case["history_length"] not in (16, 64):
            raise ExpansionLaunchError("Architecture uses the fixed 16/64 synthetic history cases")
        for field, expected in (("split", "val"), ("task", "InsertPeg"), ("episode_id", 0)):
            _same(case.get(field), expected, "architecture synthetic context")
        covered.add((case["arm"], "short" if case["history_length"] < 32 else "long" if case["history_length"] > 32 else "boundary"))
        _same(case.get("action_shape"), [20, 8], "architecture action shape")
        _same(case.get("memory_shape"), [1, 768, 1024], "architecture memory shape")
        for field in ("finite_outputs", "memory_digest_repeat_equal", "action_repeat_equal", "reset_verified"):
            _same(case.get(field), True, f"architecture {field}")
        for field in ("cold_latency_ms", "warm_latency_ms", "selector_latency_ms", "full_request_latency_ms", "peak_gpu_bytes"):
            value = case.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (field == "peak_gpu_bytes" and value == 0):
                raise ExpansionLaunchError(f"Missing actual architecture {field}")
    if not {(arm, length) for arm in c.EXPANSION_ARMS for length in ("short", "long")} <= covered:
        raise ExpansionLaunchError("Architecture short/long arm coverage incomplete")
    diagnostic = raw.get("padded_u_diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise ExpansionLaunchError("Padded-U positional diagnostic is missing")
    for key, expected in (("same_input", True), ("same_noise", True), ("u_memory_slots", 512), ("padded_memory_slots", 768)):
        _same(diagnostic.get(key), expected, f"positional diagnostic {key}")
    for key in ("action_difference_linf", "velocity_difference_linf"):
        value = diagnostic.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ExpansionLaunchError("Positional differences must be recorded, not assumed zero")


def _smoke_gate(ref, binding):
    gate = _gate(ref, "end_to_end_gate", binding)
    manifest_path = _read_ref(gate.get("run_manifest"), "smoke run manifest")
    if manifest_path.name != "run_manifest.json" or gate["run_manifest"] not in gate["sources"]:
        raise ExpansionLaunchError("Smoke manifest must be explicitly bound")
    smoke = ExpansionRunStore.open(manifest_path.parent)
    _same(smoke.stage, "smoke", "source smoke stage")
    provenance = smoke.manifest["provenance"]
    for key, expected in (("code_commit", binding["policy_commit"]), ("benchmark_commit", binding["benchmark_commit"]),
                          ("environment_manifest_sha256", binding["environment_sha256"]),
                          ("checkpoint_archive_sha256", binding["checkpoint_archive_sha256"])):
        _same(provenance.get(key), expected, f"smoke source {key}")
    # This is the small-file binding gate, not a substitute for full raw-file
    # audit. deep_validate_runtime_evidence performs that once per controller.
    executions = gate.get("execution_records")
    if not isinstance(executions, list) or not executions:
        raise ExpansionLaunchError("Smoke requires real execution/controller/bootstrap records")
    owner_by_row = {}
    execution_ids = set()
    for execution in executions:
        if not isinstance(execution, Mapping) or set(execution) != {"plan", "controller_result", "policy_bootstrap", "shard_result", "policy_ready"}:
            raise ExpansionLaunchError("Incomplete smoke execution source record")
        source_plan = validate_execution_plan(_json_ref(execution["plan"], "source execution plan"))
        source_plan.require_runtime_stage("end_to_end_smoke")
        _same(str(source_plan.store_root), str(smoke.run_root), "smoke execution source root")
        _same(source_plan.binding, binding, "smoke execution source configuration")
        identity = source_plan.execution_identity
        if identity["execution_id"] in execution_ids:
            raise ExpansionLaunchError("Duplicate smoke execution identity")
        execution_ids.add(identity["execution_id"])
        complete = _json_ref(execution["controller_result"], "controller completion")
        _same(complete.get("execution_identity"), identity, "smoke controller identity")
        _same(complete.get("owned_process_cleanup_confirmed"), True, "smoke process cleanup")
        shard = _json_ref(execution["shard_result"], "shard completion")
        _same(shard.get("execution_identity"), identity, "smoke shard identity")
        _same(shard.get("row_ids"), [row["row_id"] for row in source_plan.rows], "smoke shard coverage")
        bootstrap = _json_ref(execution["policy_bootstrap"], "live model bootstrap")
        _same(bootstrap.get("policy_execution_identity"), identity, "smoke model execution identity")
        _same(bootstrap.get("checkpoint_content_verified_in_this_process"), True, "actual checkpoint load process")
        verified = bootstrap.get("checkpoint_content_evidence") or {}
        _same(verified.get("archive_sha256"), c.CHECKPOINT_ARCHIVE_SHA256, "smoke loaded archive")
        _same(verified.get("content_tree"), source_plan.checkpoint["content_tree"], "smoke loaded parameter tree")
        _same(bootstrap.get("model_gpu_uuid"), source_plan.gpu_uuid, "smoke model physical GPU")
        pid = bootstrap.get("model_process_pid")
        if type(pid) is not int or pid <= 0:
            raise ExpansionLaunchError("Smoke model process identity is missing")
        ready = _json_ref(execution["policy_ready"], "policy readiness")
        from experiments.uniform_keyframe_expansion.serving import validate_server_metadata
        validate_server_metadata(ready, identity, c.policy_variant_for_rows(source_plan.rows))
        _same(ready.get("model_process_pid"), pid, "smoke readiness/model PID")
        for row in source_plan.rows:
            if row["row_id"] in owner_by_row:
                raise ExpansionLaunchError("Smoke execution plans overlap rows")
            owner_by_row[row["row_id"]] = identity
    _same(sorted(owner_by_row), list(range(c.SMOKE_TRAJECTORY_COUNT)), "executed smoke census")
    records = gate.get("results")
    if not isinstance(records, list) or len(records) != c.SMOKE_TRAJECTORY_COUNT:
        raise ExpansionLaunchError(f"Smoke requires all {c.SMOKE_TRAJECTORY_COUNT} bound result records")
    by_row = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"row_id", "result", "manifest", "initial"} or type(record["row_id"]) is not int or record["row_id"] in by_row:
            raise ExpansionLaunchError("Malformed/duplicate smoke result binding")
        by_row[record["row_id"]] = record
    _same(sorted(by_row), list(range(c.SMOKE_TRAJECTORY_COUNT)), "bound smoke result census")
    initial_by_task = {}
    for row in c.build_smoke_matrix()["rows"]:
        record = by_row[row["row_id"]]
        result = _json_ref(record["result"], "smoke result")
        manifest = _json_ref(record["manifest"], "smoke episode manifest")
        initial = _json_ref(record["initial"], "smoke initial conditions")
        _same(result.get("row"), row, "smoke result row")
        _same(manifest.get("row"), row, "smoke manifest row")
        _same(manifest.get("episode_provenance", {}).get("policy_execution_identity"), owner_by_row[row["row_id"]], "smoke episode execution identity")
        attempt = result.get("attempt_id")
        if type(attempt) is not int or not 0 <= attempt <= 2:
            raise ExpansionLaunchError("Invalid smoke result attempt")
        directory = smoke.attempt_dir(row, attempt)
        for name, filename in (("result", "episode_result.json"), ("manifest", "episode_manifest.json"), ("initial", "initial_condition_hashes.json")):
            _same(record[name]["path"], str(directory / filename), f"smoke {name} path")
        declared = result.get("artifact_sha256") or {}
        _same(declared.get("episode_manifest.json"), record["manifest"]["sha256"], "smoke sealed manifest")
        _same(declared.get("initial_condition_hashes.json"), record["initial"]["sha256"], "smoke sealed initial conditions")
        if result.get("terminal_reason") not in ("success", "fail", "timeout", "short_limit", "error"):
            raise ExpansionLaunchError("Invalid smoke outcome")
        _same(result.get("success"), result["terminal_reason"] == "success", "smoke binary outcome")
        if result["terminal_reason"] == "error":
            raise ExpansionLaunchError("Official benchmark error prevents smoke readiness")
        env, reset = initial["environment_provenance"], initial["reset_evidence"]
        signature = {"hashes": initial["hashes"],
                     "environment": {key: env.get(key) for key in ("resolved_environment_seed", "difficulty", "dataset", "resolved_difficulty_hint")},
                     "reset_prefix": {key: reset[key] for key in ("reset_prefix_frame_count", "reset_prefix_stage_count", "reset_prefix_frames_sha256", "reset_prefix_stages_sha256")}}
        if row["task"] in initial_by_task:
            _same(signature, initial_by_task[row["task"]], "smoke cross-arm/scope initial conditions")
        initial_by_task[row["task"]] = signature
    _same(gate.get("completed_count"), c.SMOKE_TRAJECTORY_COUNT, "smoke report census")


def _formal_freeze(evidence, binding):
    freeze = _json_ref(evidence["freeze"], "formal freeze", "protocol_freeze")
    _same(freeze.get("protocol_version"), "v1.1", "formal freeze version")
    _same(freeze.get("binding"), binding, "formal freeze runtime")
    for key in ("protocol", "end_to_end_gate", "architecture_gate", "exposure"):
        _same(freeze.get(f"{key}_sha256"), evidence[key]["sha256"], f"freeze {key}")
    _sources(freeze.get("sources"), "freeze review")
    exposure = _json_ref(evidence["exposure"], "prior exposure", "prior_exposure")
    _same(exposure.get("protocol_family"), c.PROTOCOL_FAMILY, "exposure family")
    _same(exposure.get("present_formal_outcomes_inspected"), False, "new outcome exposure")
    expected = [{"task": task, "episode_id": ep, "dataset": "test"} for task in c.FORMAL_TASKS for ep in range(50)]
    _same(exposure.get("records"), expected, "complete prior-viewed census")
    _sources(exposure.get("sources"), "prior-exposure evidence")


def build_execution_request(*, run_root: str | Path, stage: str, row_ids: list[int],
                            execution_id: str, gpu_uuid: str | None, evidence: Mapping[str, Any]) -> dict:
    """Prepare the exact scope to show the user; it is not execution authority."""
    store = ExpansionRunStore.open(run_root)
    matrix = c.build_formal_matrix() if store.stage == "formal" else c.build_smoke_matrix()
    if not isinstance(row_ids, list) or any(type(i) is not int or not 0 <= i < len(matrix["rows"]) for i in row_ids):
        raise ExpansionLaunchError("Invalid matrix row IDs")
    request = {
        "schema_version": 1, "protocol_family": c.PROTOCOL_FAMILY, "stage": stage,
        "store_root": str(store.run_root), "run_manifest": file_reference(store.run_root / "run_manifest.json"),
        "execution_id": execution_id, "gpu_uuid": gpu_uuid,
        "rows": [matrix["rows"][i] for i in row_ids], "evidence": deepcopy(dict(evidence)),
    }
    _validate_request(request)
    return {**request, "request_sha256": c.canonical_sha256(request)}


def build_execution_plan(request: Mapping[str, Any], authorization=None) -> dict:
    """Attach a separately captured, scope-bound user authorization file."""
    plan = {"request": deepcopy(dict(request)), "authorization": deepcopy(authorization)}
    validate_execution_plan(plan)
    return plan


def _validate_request(request):
    fields = {"schema_version", "protocol_family", "stage", "store_root", "run_manifest", "execution_id", "gpu_uuid", "rows", "evidence"}
    if not isinstance(request, Mapping) or set(request) != fields:
        raise ExpansionLaunchError("Execution request fields differ from the schema")
    _same(request["schema_version"], 1, "request schema")
    _same(request["protocol_family"], c.PROTOCOL_FAMILY, "request family")
    stage = request["stage"]
    if stage not in STAGES:
        raise ExpansionLaunchError("Unknown execution stage")
    identity = request["execution_id"]
    try:
        if not isinstance(identity, str) or str(uuid.UUID(identity)) != identity:
            raise ValueError("noncanonical UUID")
    except (ValueError, AttributeError) as error:
        raise ExpansionLaunchError("Execution identity must be a unique canonical UUID") from error
    gpu_uuid = request["gpu_uuid"]
    if gpu_uuid is not None and (not isinstance(gpu_uuid, str) or re.fullmatch(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", gpu_uuid) is None):
        raise ExpansionLaunchError("GPU must be a physical NVIDIA GPU UUID, not an ordinal or MIG slice")
    if stage != "cpu_prepare" and gpu_uuid is None:
        raise ExpansionLaunchError("Runtime stages require physical GPU identity")
    store = ExpansionRunStore.open(request["store_root"])
    _same(str(_read_ref(request["run_manifest"], "run manifest")), str(store.run_root / "run_manifest.json"), "store manifest path")
    if (stage == "formal" and store.stage != "formal") or (stage in ("architecture_smoke", "end_to_end_smoke") and store.stage != "smoke"):
        raise ExpansionLaunchError("Requested stage differs from store stage")
    rows = request["rows"]
    if not isinstance(rows, list) or (stage == "architecture_smoke" and rows) or (stage in ("end_to_end_smoke", "formal") and not rows):
        raise ExpansionLaunchError("Architecture has no trajectories; trajectory execution needs fixed rows")
    normalized = [c.validate_row(row, store.stage) for row in rows]
    if len({row["row_id"] for row in normalized}) != len(rows):
        raise ExpansionLaunchError("Duplicate execution row")
    if stage in ("end_to_end_smoke", "formal"):
        try:
            c.policy_variant_for_rows(normalized)
        except c.ExpansionContractError as error:
            raise ExpansionLaunchError(str(error)) from error
    evidence = request["evidence"]
    required = BASE_EVIDENCE | ({"cpu_gate"} if stage != "cpu_prepare" else set())
    if stage in ("end_to_end_smoke", "formal"):
        required |= {"architecture_gate"}
    if stage == "formal":
        required |= {"end_to_end_gate", "freeze", "exposure"}
    if not isinstance(evidence, Mapping) or set(evidence) != required:
        raise ExpansionLaunchError(f"Stage {stage} requires exactly evidence {sorted(required)}")
    protocol = _read_ref(evidence["protocol"], "protocol snapshot")
    version = re.search(r"^\*\*Protocol version:\*\* (v[0-9.]+)\s*$", protocol.read_text(), re.MULTILINE)
    if version is None or version.group(1) not in (("v1.1",) if stage == "formal" else ("v0.9", "v1.0", "v1.1")):
        raise ExpansionLaunchError("Formal requires the reviewed v1.1 protocol")
    provenance = store.manifest["provenance"]
    _same(evidence["protocol"]["sha256"], provenance["protocol_sha256"], "stored protocol")
    _same(evidence["environment"]["sha256"], provenance["environment_manifest_sha256"], "stored environment")
    policy = _source(evidence["policy_source"], "policy", provenance["code_commit"])
    benchmark = _source(evidence["benchmark_source"], "benchmark", provenance["benchmark_commit"])
    environment = _environment(evidence["environment"], provenance["host"], gpu_uuid)
    _same(environment["hardware"], provenance["hardware"], "stored hardware evidence")
    checkpoint = _checkpoint(evidence["checkpoint"])
    binding = _binding(request, policy, benchmark, checkpoint)
    if stage != "cpu_prepare":
        _cpu_gate(evidence["cpu_gate"], binding)
    if stage in ("end_to_end_smoke", "formal"):
        _architecture_gate(evidence["architecture_gate"], binding)
    if stage == "formal":
        _smoke_gate(evidence["end_to_end_gate"], binding)
        _formal_freeze(evidence, binding)
    return {"policy_source": policy, "benchmark_source": benchmark,
            "environment": environment, "checkpoint": checkpoint, "binding": binding}


@dataclass(frozen=True, init=False)
class ValidatedExecutionPlan:
    """Defensive immutable snapshot, not a security token or self-authorizing plan."""

    _serialized: str
    _validated: str

    def __init__(self, payload, validated, *, _seal=None):
        if _seal is not _SEAL:
            raise ExpansionLaunchError("Use validate_execution_plan; a bare mapping is not a validated plan")
        object.__setattr__(self, "_serialized", c.canonical_json(payload))
        object.__setattr__(self, "_validated", c.canonical_json(validated))

    @property
    def payload(self): return json.loads(self._serialized)
    @property
    def stage(self): return self.payload["request"]["stage"]
    @property
    def store_root(self): return Path(self.payload["request"]["store_root"])
    @property
    def run_root(self): return self.store_root
    @property
    def rows(self): return self.payload["request"]["rows"]
    @property
    def gpu_uuid(self): return self.payload["request"]["gpu_uuid"]
    @property
    def execution_identity(self):
        request = self.payload["request"]
        return {"execution_id": request["execution_id"], "dispatch_sha256": request["request_sha256"]}
    @property
    def policy_source(self): return json.loads(self._validated)["policy_source"]
    @property
    def benchmark_source(self): return json.loads(self._validated)["benchmark_source"]
    @property
    def environment(self): return json.loads(self._validated)["environment"]
    @property
    def checkpoint(self): return json.loads(self._validated)["checkpoint"]
    @property
    def binding(self): return json.loads(self._validated)["binding"]
    @property
    def policy_root(self): return Path(self.policy_source["root"])
    @property
    def benchmark_root(self): return Path(self.benchmark_source["root"])
    @property
    def checkpoint_dir(self): return Path(self.checkpoint["checkpoint_dir"])
    @property
    def policy_variant(self):
        return c.policy_variant_for_rows(
            self.rows, allow_empty_architecture=self.stage == "architecture_smoke",
        )
    @property
    def live_checkpoint_verification_required(self): return True

    def require_runtime_stage(self, stage=None):
        if self.stage == "cpu_prepare" or (stage is not None and stage != self.stage):
            raise ExpansionLaunchError("Plan does not authorize this runtime stage")
        return self

    def assert_runtime_allowed(self):
        return self.require_runtime_stage()

    def revalidate(self):
        """Recheck small evidence files/census; live weight bytes are a bootstrap gate."""
        return validate_execution_plan(self.payload)


def validate_execution_plan(plan: Mapping[str, Any]) -> ValidatedExecutionPlan:
    if not isinstance(plan, Mapping) or set(plan) != {"request", "authorization"}:
        raise ExpansionLaunchError("Plan requires request and authorization")
    request = deepcopy(plan["request"])
    if not isinstance(request, dict) or "request_sha256" not in request:
        raise ExpansionLaunchError("Missing immutable request digest")
    digest = request.pop("request_sha256")
    _same(c.canonical_sha256(request), digest, "request digest")
    validated = _validate_request(request)
    if request["stage"] == "cpu_prepare":
        if plan["authorization"] is not None:
            raise ExpansionLaunchError("CPU preparation must not masquerade as runtime approval")
    else:
        approval = _json_ref(plan["authorization"], "explicit user authorization", "user_authorization")
        _same(approval.get("protocol_family"), c.PROTOCOL_FAMILY, "authorization family")
        _same(approval.get("stage"), request["stage"], "separate stage authorization")
        _same(approval.get("request_sha256"), digest, "authorized request")
        _same(approval.get("approved"), True, "explicit stage approval")
        instruction = _read_ref(approval.get("instruction"), "user instruction source")
        if not instruction.read_text().strip():
            raise ExpansionLaunchError("Empty user authorization source")
        for key in ("recorded_utc", "recorded_by"):
            if not isinstance(approval.get(key), str) or not approval[key].strip():
                raise ExpansionLaunchError("Authorization capture provenance is required")
    return ValidatedExecutionPlan(plan, validated, _seal=_SEAL)


def deep_validate_runtime_evidence(plan: ValidatedExecutionPlan) -> dict:
    """Controller startup gate, before any child: audit heavyweight raw sources.

    Ordinary validate/revalidate checks only the bound small evidence chain and
    cannot claim this full raw-artifact audit. The controller calls this once,
    not per episode. Actual parameter-byte loading is a separate policy-worker
    bootstrap check; this function never loads a model or contacts a GPU.
    """
    if type(plan) is not ValidatedExecutionPlan:
        raise ExpansionLaunchError("Deep runtime audit requires a validated plan")
    plan = plan.revalidate()
    plan.require_runtime_stage()
    checked = 0
    architecture_raw = 0
    if plan.stage in ("end_to_end_smoke", "formal"):
        from experiments.uniform_keyframe_expansion.architecture_artifacts import validate_execution_record, validate_raw_arrays
        gate = _json_ref(plan.payload["request"]["evidence"]["architecture_gate"], "architecture raw audit")
        raw = _json_ref(gate["measurements"], "architecture measurements")
        audit = validate_execution_record(gate, raw, plan.binding, deep=True)
        validate_raw_arrays(raw)
        architecture_raw = audit["raw_artifact_count"]
    if plan.stage == "formal":
        gate = _json_ref(plan.payload["request"]["evidence"]["end_to_end_gate"], "source smoke audit")
        smoke = ExpansionRunStore.open(Path(gate["run_manifest"]["path"]).parent)
        completed = smoke.completed_rows()  # Includes actual videos, NPZ and every trace.
        _same(sorted(completed), list(range(c.SMOKE_TRAJECTORY_COUNT)), "deep complete smoke census")
        for record in gate["results"]:
            _same(str(completed[record["row_id"]]), record["result"]["path"], "deep smoke accepted attempt")
            _same(sha256_file(completed[record["row_id"]]), record["result"]["sha256"], "deep smoke accepted result")
        checked = c.SMOKE_TRAJECTORY_COUNT
    return {"execution_identity": plan.execution_identity, "stage": plan.stage,
            "scope": "controller-startup source audit; not GPU execution or live checkpoint verification",
            "source_smoke_raw_episodes_audited": checked,
            "architecture_raw_artifacts_audited": architecture_raw, "passed": True}
