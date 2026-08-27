from __future__ import annotations

import hashlib
import json

import pytest

from experiments.keyframe_oracle_sampling.artifacts import (
    ArtifactContractError,
    EpisodeAttemptWriter,
    PROTOCOL_VERSION,
    SMOKE_CHECKPOINT_ID,
    SMOKE_CHECKPOINT_PATH,
    SMOKE_EVALUATION_POLICY_SEED,
    ScientificKey,
    audit_smoke_attempt,
    build_smoke_seed_table,
    validate_seed_table,
)
from mme_vla_suite.shared.keyframe_oracle_sampling import (
    SMOKE_SEED_DATASET,
    SMOKE_SEED_SCOPE,
    select_indices,
)


def _sha(value: str = "artifact") -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _strict_attempt(
    tmp_path,
    *,
    arm: str,
    steps: int = 64,
    selection_arm: str | None = None,
    wrong_random_seed: bool = False,
    trace_count: int | None = None,
    final_memory_shape: list[int] | None = None,
    complete_initial_conditions: bool = True,
    valid_mask_contract: bool = True,
):
    task = "InsertPeg"
    key = ScientificKey(task, 0, arm, "short")
    seed_payload = build_smoke_seed_table([task], [0])
    seed_lookup = validate_seed_table(
        seed_payload,
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    row = {
        "task": task,
        "episode_id": 0,
        "arm": arm,
        "trajectory_kind": "short",
        "max_steps": 64,
        "dataset": "val",
    }
    launch = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "val",
        "evaluation_policy_seed": SMOKE_EVALUATION_POLICY_SEED,
        "checkpoint_path": SMOKE_CHECKPOINT_PATH,
        "checkpoint_archive_sha256_actual": _sha("archive"),
        "checkpoint_unpacked_metadata_sha256": _sha("metadata"),
        "checkpoint_unpacked_content_tree_sha256": _sha("tree"),
        "seed_table_scope": SMOKE_SEED_SCOPE,
        "seed_table_dataset": SMOKE_SEED_DATASET,
        "seed_table_derivation": seed_payload["derivation"],
        "seed_table_entries_sha256": seed_payload["entries_sha256"],
    }
    fixed = {
        "dataset": "val",
        "max_steps": 64,
        "executed_action_horizon": 16,
        "evaluation_policy_seed": SMOKE_EVALUATION_POLICY_SEED,
        "checkpoint_id": SMOKE_CHECKPOINT_ID,
    }
    writer = EpisodeAttemptWriter(tmp_path / f"{arm}_attempt", key, 0)
    writer.create(
        {
            **fixed,
            "protocol_version": PROTOCOL_VERSION,
            "seed_table_sha256": seed_payload["entries_sha256"],
        }
    )
    initial_conditions = {
            "front_observations_sha256": _sha("front"),
            "wrist_observations_sha256": _sha("wrist"),
            "robot_states_sha256": _sha("robot"),
            "task_state_sha256": _sha("task-state"),
            "task_instruction_sha256": _sha("instruction"),
    }
    if not complete_initial_conditions:
        initial_conditions.pop("task_state_sha256")
    writer.record_initial_conditions(initial_conditions)

    expected_calls = (steps + 15) // 16
    emitted_calls = expected_calls if trace_count is None else trace_count
    policy_latency_ms = []
    policy_model_latency_ms = []
    history_lengths = []
    for call_index in range(emitted_calls):
        current_history_index = call_index * 16
        visible_boundaries = [0]
        preregistered_seed = seed_lookup[(task, 0, call_index)] if arm == "R" else None
        recorded_seed = (
            preregistered_seed + 1
            if wrong_random_seed and preregistered_seed is not None
            else preregistered_seed
        )
        selected = select_indices(
            selection_arm or arm,
            current_history_index,
            boundary_indices=visible_boundaries,
            random_seed=(
                recorded_seed if (selection_arm or arm) == "R" else None
            ),
        )
        selected_hash = hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        end_to_end_latency = 2.0 + call_index
        model_latency = 1.0 + call_index
        writer.append_trace(
            {
                "schema_version": 1,
                "task": task,
                "episode_id": 0,
                "selector_name": arm,
                "selector_seed": recorded_seed,
                "seed_table_sha256": seed_payload["entries_sha256"],
                "seed_table_scope": SMOKE_SEED_SCOPE,
                "seed_table_dataset": SMOKE_SEED_DATASET,
                "policy_call_index": call_index,
                "environment_step": call_index * 16,
                "history_length": current_history_index + 1,
                "current_history_index": current_history_index,
                "selected_frame_indices": selected,
                "selected_indices_sha256": selected_hash,
                "visible_boundary_indices": visible_boundaries,
                "valid_frame_count": len(selected),
                "padding_frame_count": 32 - len(selected),
                "valid_memory_token_count": 16 * len(selected),
                "mask_shape": [512],
                "mask_dtype": "bool",
                "mask_valid_prefix_all_true": True,
                "mask_padding_all_false": valid_mask_contract,
                "mask_sha256": _sha(f"mask-{call_index}"),
                "image_tensor_shape": [512, 2048],
                "image_tensor_dtype": "bfloat16",
                "image_tensor_sha256": _sha(f"image-{call_index}"),
                "position_tensor_shape": [512, 768],
                "position_tensor_dtype": "float32",
                "position_tensor_sha256": _sha(f"position-{call_index}"),
                "state_tensor_shape": [512, 8],
                "state_tensor_dtype": "float32",
                "state_tensor_sha256": _sha(f"state-{call_index}"),
                "prepared_memory_component_shapes": [
                    [512, 2048],
                    [512, 768],
                    [512, 8],
                    [512],
                ],
                "prepared_memory_components_sha256": _sha(
                    f"components-{call_index}"
                ),
                "prepared_memory_input_shape": [512, 2824],
                "prepared_memory_input_dtype": "float32",
                "prepared_memory_input_sha256": _sha(f"input-{call_index}"),
                "final_memory_tensor_shape": final_memory_shape or [1, 512, 1024],
                "final_memory_tensor_dtype": "bfloat16",
                "final_memory_tensor_is_floating": True,
                "final_memory_tensor_finite": True,
                "final_memory_tensor_sha256": _sha(f"memory-{call_index}"),
                "boundary_lookup_latency_ms": 0.1,
                "selector_decision_latency_ms": 0.2,
                "selector_bookkeeping_latency_ms": 0.3,
                "selector_latency_ms": 0.6,
                "model_latency_ms": model_latency,
                "end_to_end_request_latency_ms": end_to_end_latency,
            }
        )
        policy_latency_ms.append(end_to_end_latency)
        policy_model_latency_ms.append(model_latency)
        history_lengths.append(current_history_index + 1)

    # Keep the result lists aligned to the scientific number of calls even in
    # the deliberately truncated-trace negative fixture.
    while len(policy_latency_ms) < expected_calls:
        policy_latency_ms.append(100.0 + len(policy_latency_ms))
        policy_model_latency_ms.append(50.0 + len(policy_model_latency_ms))
        history_lengths.append(1 + 16 * len(history_lengths))
    writer.finalize(
        {
            **fixed,
            "task": task,
            "episode_id": 0,
            "selector_arm": arm,
            "steps": steps,
            "success": False,
            "terminal_reason": "timeout",
            "timeout": True,
            "policy_latency_ms": policy_latency_ms,
            "policy_model_latency_ms": policy_model_latency_ms,
            "history_lengths_at_policy_calls": history_lengths,
        }
    )
    return writer, key, row, launch, seed_payload, seed_lookup


def _audit(fixture):
    writer, key, row, launch, seed_payload, seed_lookup = fixture
    return audit_smoke_attempt(
        writer,
        expected_key=key,
        expected_row=row,
        launch_manifest=launch,
        smoke_seed_payload=seed_payload,
        smoke_seed_lookup=seed_lookup,
    )


@pytest.mark.parametrize("arm", ("U", "O", "OC", "R"))
def test_strict_smoke_attempt_recomputes_each_frozen_arm(tmp_path, arm):
    report = _audit(_strict_attempt(tmp_path, arm=arm))
    assert report["strict_smoke_contract"] is True
    assert report["policy_call_count"] == 4


@pytest.mark.parametrize("arm", ("O", "OC", "R"))
def test_strict_smoke_attempt_rejects_all_arms_running_uniform(tmp_path, arm):
    fixture = _strict_attempt(tmp_path, arm=arm, selection_arm="U")
    with pytest.raises(ArtifactContractError, match="does not match arm"):
        _audit(fixture)


def test_strict_smoke_attempt_rejects_wrong_random_seed(tmp_path):
    fixture = _strict_attempt(tmp_path, arm="R", wrong_random_seed=True)
    with pytest.raises(ArtifactContractError, match="wrong preregistered seed"):
        _audit(fixture)


def test_strict_smoke_attempt_rejects_too_few_trace_records(tmp_path):
    fixture = _strict_attempt(tmp_path, arm="U", steps=32, trace_count=1)
    with pytest.raises(ArtifactContractError, match="ceil"):
        _audit(fixture)


def test_strict_smoke_attempt_rejects_bad_final_memory_tensor(tmp_path):
    fixture = _strict_attempt(
        tmp_path,
        arm="U",
        final_memory_shape=[1, 512, 2048],
    )
    with pytest.raises(ArtifactContractError, match="shape"):
        _audit(fixture)


def test_strict_smoke_attempt_requires_all_five_initial_condition_hashes(tmp_path):
    fixture = _strict_attempt(
        tmp_path,
        arm="U",
        complete_initial_conditions=False,
    )
    with pytest.raises(ArtifactContractError, match="exact five"):
        _audit(fixture)


def test_strict_smoke_attempt_rejects_invalid_padding_mask_contract(tmp_path):
    fixture = _strict_attempt(tmp_path, arm="U", valid_mask_contract=False)
    with pytest.raises(ArtifactContractError, match="valid prefix"):
        _audit(fixture)
