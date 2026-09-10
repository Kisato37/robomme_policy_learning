"""Synthetic CPU-only evidence graph, never a production/model execution receipt.

Large-shaped arrays are compressed zeros so the actual raw-array contract can
be exercised without importing a GPU runtime or loading any model weights.
"""
from functools import lru_cache
import io
from pathlib import Path
import uuid

import numpy as np

from experiments.uniform_keyframe_expansion import contract as c
from experiments.uniform_keyframe_expansion import launch_contract as lc
from experiments.uniform_keyframe_expansion.artifacts import ExpansionRunStore
from mme_vla_suite.shared.keyframe_oracle_sampling import official_uniform_indices
from mme_vla_suite.shared.uniform_keyframe_expansion import select_expansion_indices


@lru_cache(maxsize=2)
def case_npz(length):
    count = 256 if length == 16 else 528
    image = np.zeros((768, 2048), np.float32)
    position = np.zeros((768, 768), np.float32)
    state = np.zeros((768, 8), np.float32)
    mask = np.arange(768) < count
    payload = {
        "raw_image": image, "raw_position": position, "raw_state": state, "raw_mask": mask,
        "image": image[None], "position": position[None], "state": state[None], "mask": mask[None],
        "final_memory": np.zeros((1, 768, 1024), np.float32), "actions": np.zeros((20, 8), np.float32),
        "initial_noise": np.zeros((1, 20, 32), np.float32), "sample_rng_data": np.array([0, 7], np.uint32),
        "raw_model_actions": np.zeros((1, 20, 32), np.float32),
        "history_images": np.zeros((length, 1, 1, 3), np.uint8),
        "history_states": np.zeros((length, 8), np.float32),
        "history_stages": (np.arange(length) >= 17).astype(np.int64),
        "current_front": np.zeros((1, 1, 3), np.uint8), "current_wrist": np.zeros((1, 1, 3), np.uint8),
        "current_state": np.zeros((8,), np.float32),
    }
    output = io.BytesIO()
    np.savez_compressed(output, **payload)
    return output.getvalue()


@lru_cache(maxsize=1)
def diagnostic_npz():
    payload = {"initial_noise": np.zeros((1, 20, 32), np.float32), "time": np.array(0.5, np.float32)}
    for slots in (512, 768):
        for name, width in (("image", 2048), ("position", 768), ("state", 8)):
            payload[f"{name}_{slots}"] = np.zeros((1, slots, width), np.float32)
        payload[f"mask_{slots}"] = (np.arange(slots) < 256)[None]
        payload[f"actions_{slots}"] = np.full((20, 8), 0 if slots == 512 else 0.25, np.float32)
        payload[f"velocity_{slots}"] = np.full((1, 20, 32), 0 if slots == 512 else 0.5, np.float32)
    output = io.BytesIO()
    np.savez_compressed(output, **payload)
    return output.getvalue()


def create_architecture_fixture(f, *, write_gate=True):
    """Return {plan, directory, gate, raw}; f is the tiny launch Fixture builder."""
    f.cpu()
    f.serial += 1
    store = ExpansionRunStore.create(f.path / "uniform_keyframe_expansion" / f"source-architecture-{f.serial}",
                                     stage="smoke", run_manifest=f.provenance)
    request = lc.build_execution_request(run_root=store.run_root, stage="architecture_smoke", row_ids=[],
        execution_id=str(uuid.uuid4()), gpu_uuid=f.environment["hardware"]["gpu_uuid"],
        evidence={key: f.evidence[key] for key in lc.BASE_EVIDENCE | {"cpu_gate"}})
    approval = f.write(f"architecture-approval-{f.serial}.json", {
        "schema_version": 1, "kind": "user_authorization", "protocol_family": c.PROTOCOL_FAMILY,
        "stage": "architecture_smoke", "approved": True, "request_sha256": request["request_sha256"],
        "instruction": f.raw, "recorded_utc": "2026-09-10T00:00:00Z", "recorded_by": "CPU fixture, not real approval"})
    plan = lc.validate_execution_plan(lc.build_execution_plan(request, approval))
    identity = plan.execution_identity
    directory = store.run_root / "executions" / identity["execution_id"]
    directory.mkdir(parents=True)
    def write(name, value):
        return f.write(str((directory / name).relative_to(f.path)), value)
    execution = {
        "plan": write("execution_plan.json", plan.payload),
        "controller_result": write("controller_complete.json", {"execution_identity": identity,
             "owned_process_cleanup_confirmed": True, "scope": "CPU fixture only"}),
        "policy_bootstrap": write("live_policy_provenance.json", {
             "policy_execution_identity": identity, "checkpoint_content_verified_in_this_process": True,
             "checkpoint_content_evidence": {"archive_sha256": c.CHECKPOINT_ARCHIVE_SHA256,
                                               "content_tree": plan.checkpoint["content_tree"]},
             "model_process_pid": 99, "model_gpu_uuid": plan.gpu_uuid, "cpu_fixture_only": True}),
        "worker_ownership": write("architecture_ownership.json", {"execution_identity": identity,
             "role": "architecture", "worker_ownership_verified": True,
             "worker_identity": {"pid": 99}, "cpu_fixture_only": True}),
    }
    artifacts, cases = [], []
    for arm in c.EXPANSION_ARMS:
        for length in (16, 64):
            refs = []
            for call in ("first", "repeat"):
                ref = write(f"probe/{arm}_{length}_{call}.npz", case_npz(length))
                artifacts.append({**ref, "size_bytes": Path(ref["path"]).stat().st_size})
                refs.append(ref)
            stages = (np.arange(length) >= 17).astype(np.int64)
            selected, _ = select_expansion_indices(arm, step_idx=length - 1,
                base_uniform_indices=official_uniform_indices(length - 1),
                boundary_flags=np.r_[True, stages[1:] != stages[:-1]], split="val", task="InsertPeg",
                episode_id=0, policy_call_index=0)
            cases.append({"arm": arm, "history_length": length, "task": "InsertPeg", "split": "val", "episode_id": 0,
                "selected_indices": selected, "action_shape": [20, 8], "memory_shape": [1, 768, 1024],
                "finite_outputs": True, "memory_digest_repeat_equal": True, "action_repeat_equal": True,
                "reset_verified": True, "cold_latency_ms": 1, "warm_latency_ms": 1, "selector_latency_ms": 0,
                "full_request_latency_ms": 1, "peak_gpu_bytes": 1, "first_artifact": refs[0], "repeat_artifact": refs[1]})
    diagnostic = write("probe/padded_u.npz", diagnostic_npz())
    artifacts.append({**diagnostic, "size_bytes": Path(diagnostic["path"]).stat().st_size})
    raw_manifest = write("probe/raw_manifest.json", {"schema_version": 1, "kind": "architecture_raw_artifacts",
        "execution_identity": identity, "artifacts": artifacts})
    raw = {"schema_version": 1, "kind": "architecture_measurements", "binding": f.binding,
           "scope": "development_only", "execution_identity": identity, "input_kind": "synthetic_causal_history",
           "no_task_execution": True, "cpu_fixture_only": True, "training_performed": False,
           "formal_test_outcomes_opened": False, "model_process_pid": 99,
           "strict_load": {"missing": [], "extra": [], "random_initialized": []}, "cases": cases,
           "padded_u_diagnostic": {"same_input": True, "same_noise": True, "u_memory_slots": 512,
               "padded_memory_slots": 768, "action_difference_linf": 0.25, "velocity_difference_linf": 0.5,
               "artifact": diagnostic}, "raw_artifacts": artifacts, "raw_artifact_manifest": raw_manifest}
    measurements = write("architecture_measurements.json", raw)
    execution["worker_complete"] = write("architecture_worker_complete.json", {
        "execution_identity": identity, "artifact_sha256": {
            "architecture_measurements.json": measurements["sha256"],
            "live_policy_provenance.json": execution["policy_bootstrap"]["sha256"],
            "architecture_ownership.json": execution["worker_ownership"]["sha256"]},
        "scope": "CPU fixture only"})
    gate_payload = f.gate("architecture_gate", checks={name: True for name in lc.ARCH_CHECKS},
        measurements=measurements, execution=execution, sources=[measurements, *execution.values(), raw_manifest])
    gate = write("architecture_gate.json", gate_payload) if write_gate else None
    return {"plan": plan, "directory": directory, "gate": gate, "gate_payload": gate_payload, "raw": raw}
