"""Publish architecture readiness only from a completed, bound real-worker run.

Hashes establish provenance/integrity, not independent proof of scientific
truth. Numerical checks use the probe's saved arrays; no model or GPU is loaded
here. Large arrays are audited once at publication/controller startup, not at
each subsequent episode's ordinary plan revalidation.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from experiments.uniform_keyframe_expansion import contract as c
from experiments.uniform_keyframe_expansion.artifacts import _write


EXECUTION_FILES = {
    "plan": "execution_plan.json",
    "controller_result": "controller_complete.json",
    "policy_bootstrap": "live_policy_provenance.json",
    "worker_ownership": "architecture_ownership.json",
    "worker_complete": "architecture_worker_complete.json",
}


def validate_execution_record(gate, raw, binding, *, deep=False):
    from experiments.uniform_keyframe_expansion import launch_contract as lc
    execution = gate.get("execution")
    if not isinstance(execution, Mapping) or set(execution) != set(EXECUTION_FILES):
        raise lc.ExpansionLaunchError("Architecture needs actual execution/controller/worker evidence")
    plan = lc.validate_execution_plan(lc._json_ref(execution["plan"], "architecture execution plan"))
    plan.require_runtime_stage("architecture_smoke")
    lc._same(plan.binding, binding, "architecture source configuration")
    identity = plan.execution_identity
    directory = plan.store_root / "executions" / identity["execution_id"]
    for key, name in EXECUTION_FILES.items():
        lc._same(execution[key]["path"], str(directory / name), "architecture execution source path")
        if execution[key] not in gate["sources"]:
            raise lc.ExpansionLaunchError("Architecture execution evidence must be in bound sources")
    lc._same(gate["measurements"]["path"], str(directory / "architecture_measurements.json"), "architecture measurement path")
    if (directory / "controller_failure.json").exists() or (directory / "architecture_worker_failure.json").exists():
        raise lc.ExpansionLaunchError("Failed architecture execution cannot publish readiness")
    lc._same(raw.get("execution_identity"), identity, "architecture measurement execution")
    lc._same(raw.get("input_kind"), "synthetic_causal_history", "architecture input scope")
    lc._same(raw.get("no_task_execution"), True, "architecture excludes task execution")
    lc._same(raw.get("training_performed"), False, "architecture excludes training")
    lc._same(raw.get("formal_test_outcomes_opened"), False, "architecture excludes test outcomes")
    complete = lc._json_ref(execution["controller_result"], "architecture controller completion")
    lc._same(complete.get("execution_identity"), identity, "architecture controller identity")
    lc._same(complete.get("owned_process_cleanup_confirmed"), True, "architecture cleanup")
    ownership = lc._json_ref(execution["worker_ownership"], "architecture worker ownership")
    for key, expected in (("role", "architecture"), ("execution_identity", identity), ("worker_ownership_verified", True)):
        lc._same(ownership.get(key), expected, f"architecture ownership {key}")
    bootstrap = lc._json_ref(execution["policy_bootstrap"], "architecture live policy bootstrap")
    lc._same(bootstrap.get("policy_execution_identity"), identity, "architecture model execution")
    lc._same(bootstrap.get("checkpoint_content_verified_in_this_process"), True, "architecture actual checkpoint load")
    verified = bootstrap.get("checkpoint_content_evidence") or {}
    lc._same(verified.get("archive_sha256"), c.CHECKPOINT_ARCHIVE_SHA256, "architecture actual archive")
    lc._same(verified.get("content_tree"), plan.checkpoint["content_tree"], "architecture actual weight tree")
    lc._same(bootstrap.get("model_gpu_uuid"), plan.gpu_uuid, "architecture physical GPU")
    pid = bootstrap.get("model_process_pid")
    if type(pid) is not int or pid <= 0:
        raise lc.ExpansionLaunchError("Architecture requires the live model process PID")
    lc._same((ownership.get("worker_identity") or {}).get("pid"), pid, "architecture owned model PID")
    lc._same(raw.get("model_process_pid"), pid, "architecture measured model PID")
    worker = lc._json_ref(execution["worker_complete"], "architecture worker completion")
    lc._same(worker.get("execution_identity"), identity, "architecture worker completion identity")
    lc._same(worker.get("artifact_sha256"), {
        "architecture_measurements.json": gate["measurements"]["sha256"],
        "live_policy_provenance.json": execution["policy_bootstrap"]["sha256"],
        "architecture_ownership.json": execution["worker_ownership"]["sha256"],
    }, "architecture worker artifact binding")
    manifest_ref = raw.get("raw_artifact_manifest")
    manifest = lc._json_ref(manifest_ref, "architecture raw artifact manifest", "architecture_raw_artifacts")
    if manifest_ref not in gate["sources"]:
        raise lc.ExpansionLaunchError("Architecture raw manifest must be a bound source")
    lc._same(manifest.get("execution_identity"), identity, "architecture raw execution identity")
    entries = manifest.get("artifacts")
    lc._same(raw.get("raw_artifacts"), entries, "architecture raw file census")
    if not isinstance(entries, list) or not entries:
        raise lc.ExpansionLaunchError("Architecture raw artifacts cannot be empty")
    seen = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256", "size_bytes"}:
            raise lc.ExpansionLaunchError("Invalid architecture raw artifact entry")
        path = Path(entry["path"])
        if not path.is_absolute() or not path.is_relative_to(directory / "probe") or ".." in path.parts or str(path) in seen:
            raise lc.ExpansionLaunchError("Raw architecture artifacts must be unique and inside their own probe directory")
        seen.add(str(path))
        lc._digest(entry["sha256"], "architecture raw artifact")
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] <= 0:
            raise lc.ExpansionLaunchError("Architecture raw artifact size is missing")
        if deep:
            checked = lc._read_ref({"path": str(path), "sha256": entry["sha256"]}, "architecture raw artifact")
            lc._same(checked.stat().st_size, entry["size_bytes"], "architecture raw size")
    return {"execution_identity": identity, "raw_artifact_count": len(entries), "deep_checked": deep}


def validate_raw_arrays(raw):
    """Independently check saved observations, repeated outputs and differences."""
    import numpy as np
    from experiments.uniform_keyframe_expansion import launch_contract as lc
    from mme_vla_suite.shared.keyframe_oracle_sampling import official_uniform_indices
    from mme_vla_suite.shared.uniform_keyframe_expansion import select_expansion_indices
    listed = {item["path"]: item for item in raw["raw_artifacts"]}
    used = set()

    def arrays(ref):
        path = lc._read_ref(ref, "architecture numerical artifact")
        if str(path) not in listed or listed[str(path)]["sha256"] != ref["sha256"]:
            raise lc.ExpansionLaunchError("Numerical artifact is not in the bound raw manifest")
        if str(path) in used:
            raise lc.ExpansionLaunchError("Distinct diagnostic executions need distinct raw artifacts")
        used.add(str(path))
        with np.load(path, allow_pickle=False) as archive:
            result = {name: archive[name] for name in archive.files}
        for value in result.values():
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise lc.ExpansionLaunchError("Non-numerical or nonfinite diagnostic array")
        return result

    def equal(first, second, label):
        if first.shape != second.shape or first.dtype != second.dtype or not np.array_equal(first, second):
            raise lc.ExpansionLaunchError(f"Architecture raw {label} mismatch")

    for case in raw["cases"]:
        first, repeat = arrays(case.get("first_artifact")), arrays(case.get("repeat_artifact"))
        shapes = {"raw_image": (768, 2048), "raw_position": (768, 768), "raw_state": (768, 8),
                  "raw_mask": (768,), "image": (1, 768, 2048), "position": (1, 768, 768),
                  "state": (1, 768, 8), "mask": (1, 768), "final_memory": (1, 768, 1024), "actions": (20, 8)}
        for name, shape in shapes.items():
            for value in (first, repeat):
                if name not in value or value[name].shape != shape:
                    raise lc.ExpansionLaunchError(f"Wrong actual architecture {name} shape")
            equal(first[name], repeat[name], name + " repeat")
        for name in ("initial_noise", "sample_rng_data", "raw_model_actions", "history_images", "history_states",
                     "history_stages", "current_front", "current_wrist", "current_state"):
            if name not in first or name not in repeat:
                raise lc.ExpansionLaunchError(f"Missing actual architecture {name}")
            equal(first[name], repeat[name], name + " repeat")
        length = case["history_length"]
        if first["history_images"].shape[0] != length or first["history_states"].shape[0] != length or first["history_stages"].shape != (length,):
            raise lc.ExpansionLaunchError("Actual history length differs from architecture case")
        mask = first["raw_mask"]
        if mask.dtype != np.bool_ or first["mask"].dtype != np.bool_:
            raise lc.ExpansionLaunchError("Architecture masks must actually be Boolean")
        equal(mask[None, :], first["mask"], "pre/post-normalization mask")
        count = int(mask.sum())
        if count == 0 or count % 16 or not np.array_equal(mask, np.arange(768) < count):
            raise lc.ExpansionLaunchError("Architecture valid-memory packing is invalid")
        if length == 16 and count != 256:
            raise lc.ExpansionLaunchError("Short-history padding must retain all 16 visible frames")
        stages = first["history_stages"]
        if stages.dtype.kind not in "iu" or np.any(stages < 0):
            raise lc.ExpansionLaunchError("Architecture history stages must be nonnegative integers")
        flags = np.r_[True, stages[1:] != stages[:-1]]
        expected, _ = select_expansion_indices(case["arm"], step_idx=length - 1,
            base_uniform_indices=official_uniform_indices(length - 1), boundary_flags=flags,
            split="val", task="InsertPeg", episode_id=0, policy_call_index=0)
        lc._same(case.get("selected_indices"), expected, "architecture causal selected indices")
        lc._same(count, len(expected) * 16, "architecture actual selected memory count")
        for name in ("raw_image", "raw_position", "raw_state"):
            if np.any(first[name][~mask] != 0):
                raise lc.ExpansionLaunchError("Pre-normalization architecture padding is not zero")
        # Do not demand normalized states or encoded padding be numerically zero.
    diagnostic = raw["padded_u_diagnostic"]
    value = arrays(diagnostic.get("artifact"))
    for stem in ("image", "position", "state", "mask"):
        low, high = value.get(stem + "_512"), value.get(stem + "_768")
        if low is None or high is None or low.ndim < 2 or high.ndim != low.ndim or low.shape[1] != 512 or high.shape[1] != 768:
            raise lc.ExpansionLaunchError("Padded-U actual static-memory slots differ")
        equal(low, high[:, :512], "padded-U identical " + stem)
    if value["mask_768"].dtype != np.bool_ or np.any(value["mask_768"][:, 512:]):
        raise lc.ExpansionLaunchError("Padded-U extra slots must be masked")
    for stem, field in (("actions", "action_difference_linf"), ("velocity", "velocity_difference_linf")):
        left, right = value.get(stem + "_512"), value.get(stem + "_768")
        if left is None or right is None or left.shape != right.shape or not left.size:
            raise lc.ExpansionLaunchError("Padded-U actual outputs missing or mismatched")
        if stem == "actions" and left.shape != (20, 8):
            raise lc.ExpansionLaunchError("Padded-U final actions must have shape 20 by 8")
        actual = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        if not np.isclose(actual, diagnostic[field], rtol=1e-6, atol=1e-8):
            raise lc.ExpansionLaunchError("Padded-U reported difference disagrees with actual outputs")
    if "initial_noise" not in value or "time" not in value:
        raise lc.ExpansionLaunchError("Padded-U shared noise/time evidence missing")
    if used != set(listed):
        raise lc.ExpansionLaunchError("Unreferenced architecture raw artifact in manifest")
    return len(used)


def publish_architecture_gate(plan, execution_directory):
    """Write once, after numerical/source verification and confirmed cleanup."""
    from experiments.uniform_keyframe_expansion import launch_contract as lc
    if type(plan) is not lc.ValidatedExecutionPlan:
        raise lc.ExpansionLaunchError("Architecture publication requires a validated execution plan")
    plan = plan.revalidate()
    plan.require_runtime_stage("architecture_smoke")
    directory = lc._path(str(Path(execution_directory).absolute()), "architecture execution directory", directory=True)
    lc._same(str(directory), str(plan.store_root / "executions" / plan.execution_identity["execution_id"]), "architecture publish directory")
    execution = {key: lc.file_reference(directory / name) for key, name in EXECUTION_FILES.items()}
    lc._same(lc._json_ref(execution["plan"], "executed architecture plan"), plan.payload, "actual executed plan")
    measurements = lc.file_reference(directory / "architecture_measurements.json")
    raw = lc._json_ref(measurements, "architecture measurements", "architecture_measurements")
    gate = {"schema_version": 1, "protocol_family": c.PROTOCOL_FAMILY, "kind": "architecture_gate",
            "status": "PASS", "binding": plan.binding, "checks": {key: True for key in lc.ARCH_CHECKS},
            "measurements": measurements, "execution": execution,
            "sources": [measurements, *execution.values(), raw.get("raw_artifact_manifest")]}
    # Validate the object before publishing it; an error must not leave a PASS file.
    lc.validate_architecture_measurements(gate, raw, plan.binding)
    validate_execution_record(gate, raw, plan.binding, deep=True)
    validate_raw_arrays(raw)
    target = directory / "architecture_gate.json"
    _write(target, gate)
    return target
