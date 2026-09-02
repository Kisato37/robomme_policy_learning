from __future__ import annotations

import hashlib
import json

import pytest

from experiments.keyframe_oracle_sampling.aggregate_formal import AGGREGATE_FILENAMES
from experiments.keyframe_oracle_sampling.aggregate_formal import audit_formal_attempt
from experiments.keyframe_oracle_sampling.aggregate_formal import build_per_episode_rows
from experiments.keyframe_oracle_sampling.aggregate_formal import build_per_task_rows
from experiments.keyframe_oracle_sampling.aggregate_formal import publish_aggregate_directory
from experiments.keyframe_oracle_sampling.artifacts import ALL_ARMS
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_CHECKPOINT_ID
from experiments.keyframe_oracle_sampling.artifacts import SMOKE_CHECKPOINT_PATH
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import EpisodeAttemptWriter
from experiments.keyframe_oracle_sampling.artifacts import ScientificKey
from experiments.keyframe_oracle_sampling.artifacts import build_seed_table
from experiments.keyframe_oracle_sampling.artifacts import released_prepared_component_dtypes
from experiments.keyframe_oracle_sampling.artifacts import validate_seed_table
from experiments.keyframe_oracle_sampling.formal_matrix import FORMAL_MAX_STEPS
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import select_indices


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _formal_attempt(
    tmp_path,
    *,
    arm: str,
    selected_as: str | None = None,
    wrong_random_seed: bool = False,
):
    task = "InsertPeg"
    episode_id = 0
    key = ScientificKey(task, episode_id, arm, "formal")
    seed_payload = build_seed_table([task], [episode_id])
    seed_lookup = validate_seed_table(
        seed_payload,
        expected_scope=FORMAL_SEED_SCOPE,
        expected_dataset=FORMAL_SEED_DATASET,
    )
    launch = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "test",
        "evaluation_policy_seed": 7,
        "executed_action_horizon": 16,
        "checkpoint_path": SMOKE_CHECKPOINT_PATH,
        "seed_table_scope": FORMAL_SEED_SCOPE,
        "seed_table_dataset": FORMAL_SEED_DATASET,
        "seed_table_entries_sha256": seed_payload["entries_sha256"],
    }
    row = {
        "row_id": 0,
        "task": task,
        "episode_id": episode_id,
        "arm": arm,
        "trajectory_kind": "formal",
        "max_steps": FORMAL_MAX_STEPS,
        "dataset": "test",
    }
    fixed = {
        "dataset": "test",
        "max_steps": FORMAL_MAX_STEPS,
        "executed_action_horizon": 16,
        "evaluation_policy_seed": 7,
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
    writer.record_initial_conditions(
        {
            "front_observations_sha256": _sha("front"),
            "wrist_observations_sha256": _sha("wrist"),
            "robot_states_sha256": _sha("robot"),
            "task_state_sha256": _sha("task-state"),
            "task_instruction_sha256": _sha("instruction"),
        }
    )

    policy_latency = []
    model_latency = []
    history_lengths = []
    for call_index in range(4):
        current_history_index = call_index * 16
        boundaries = ([0], [0, 8], [0, 8, 24], [0, 8, 24, 40])[call_index]
        expected_seed = seed_lookup[(task, episode_id, call_index)] if arm == "R" else None
        recorded_seed = expected_seed + 1 if wrong_random_seed and expected_seed is not None else expected_seed
        selection_arm = selected_as or arm
        selected = select_indices(
            selection_arm,
            current_history_index,
            boundary_indices=boundaries,
            random_seed=recorded_seed if selection_arm == "R" else None,
        )
        component_dtypes = released_prepared_component_dtypes(len(selected))
        end_to_end = 2.0 + call_index
        model = 1.0 + call_index
        writer.append_trace(
            {
                "schema_version": 1,
                "task": task,
                "episode_id": episode_id,
                "selector_name": arm,
                "selector_seed": recorded_seed,
                "seed_table_sha256": seed_payload["entries_sha256"],
                "seed_table_scope": FORMAL_SEED_SCOPE,
                "seed_table_dataset": FORMAL_SEED_DATASET,
                "policy_call_index": call_index,
                "environment_step": call_index * 16,
                "history_length": current_history_index + 1,
                "current_history_index": current_history_index,
                "selected_frame_indices": selected,
                "selected_indices_sha256": hashlib.sha256(
                    json.dumps(selected, separators=(",", ":")).encode("ascii")
                ).hexdigest(),
                "visible_boundary_indices": boundaries,
                "valid_frame_count": len(selected),
                "padding_frame_count": 32 - len(selected),
                "valid_memory_token_count": 16 * len(selected),
                "mask_shape": [512],
                "mask_dtype": "bool",
                "mask_valid_prefix_all_true": True,
                "mask_padding_all_false": True,
                "mask_sha256": _sha(f"mask-{call_index}"),
                "image_tensor_dtype": component_dtypes[0],
                "image_tensor_sha256": _sha(f"image-{call_index}"),
                "position_tensor_dtype": component_dtypes[1],
                "position_tensor_sha256": _sha(f"position-{call_index}"),
                "state_tensor_dtype": component_dtypes[2],
                "state_tensor_sha256": _sha(f"state-{call_index}"),
                "prepared_memory_component_shapes": [
                    [512, 2048],
                    [512, 768],
                    [512, 8],
                    [512],
                ],
                "prepared_memory_components_sha256": _sha(f"components-{call_index}"),
                "prepared_memory_input_shape": [512, 2824],
                "prepared_memory_input_sha256": _sha(f"input-{call_index}"),
                "final_memory_tensor_shape": [1, 512, 1024],
                "final_memory_tensor_dtype": "bfloat16",
                "final_memory_tensor_is_floating": True,
                "final_memory_tensor_finite": True,
                "final_memory_tensor_sha256": _sha(f"memory-{call_index}"),
                "boundary_lookup_latency_ms": 0.1,
                "selector_decision_latency_ms": 0.2,
                "selector_bookkeeping_latency_ms": 0.3,
                "selector_latency_ms": 0.6,
                "model_latency_ms": model,
                "end_to_end_request_latency_ms": end_to_end,
            }
        )
        policy_latency.append(end_to_end)
        model_latency.append(model)
        history_lengths.append(current_history_index + 1)
    writer.finalize(
        {
            **fixed,
            "task": task,
            "episode_id": episode_id,
            "selector_arm": arm,
            "steps": 49,
            "success": False,
            "terminal_reason": "fail",
            "timeout": False,
            "collision": False,
            "policy_latency_ms": policy_latency,
            "policy_model_latency_ms": model_latency,
            "history_lengths_at_policy_calls": history_lengths,
        }
    )
    return writer, key, row, launch, seed_payload, seed_lookup


def _audit(fixture):
    writer, key, row, launch, seed_payload, seed_lookup = fixture
    return audit_formal_attempt(
        writer,
        expected_key=key,
        expected_row=row,
        launch=launch,
        seed_payload=seed_payload,
        seed_lookup=seed_lookup,
    )


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_formal_attempt_recomputes_each_selector_arm(tmp_path, arm):
    report, latency, _, _ = _audit(_formal_attempt(tmp_path, arm=arm))
    assert report["strict_formal_contract"] is True
    assert report["policy_call_count"] == 4
    assert latency["selector"] == [0.6, 0.6, 0.6, 0.6]


@pytest.mark.parametrize("arm", ["O", "OC"])
def test_formal_attempt_rejects_uniform_trace_for_semantic_arm(tmp_path, arm):
    with pytest.raises(ArtifactContractError, match="does not match arm"):
        _audit(_formal_attempt(tmp_path, arm=arm, selected_as="U"))


def test_formal_attempt_rejects_wrong_preregistered_random_seed(tmp_path):
    with pytest.raises(ArtifactContractError, match="wrong preregistered seed"):
        _audit(_formal_attempt(tmp_path, arm="R", wrong_random_seed=True))


def _formal_records():
    records = {}
    for task in FORMAL_TASKS:
        for episode_id in range(50):
            for arm_index, arm in enumerate(ALL_ARMS):
                key = ScientificKey(task, episode_id, arm, "formal")
                records[key] = {
                    "success": episode_id < 10 + arm_index * 5,
                    "terminal_reason": "success" if episode_id < 10 + arm_index * 5 else "fail",
                    "steps": 20 + episode_id,
                    "collision": False,
                    "attempt_id": 0,
                }
    return records


def test_formal_csv_tables_have_exact_paired_shapes_and_differences():
    records = _formal_records()
    per_episode = build_per_episode_rows(records)
    per_task = build_per_task_rows(records)
    assert len(per_episode) == 800
    assert len(per_task) == 16
    assert per_task[0]["U_success_count"] == 10
    assert per_task[0]["OC_minus_U_pp"] == pytest.approx(20.0)
    assert per_episode[0]["OC_minus_U"] == 0


def test_aggregate_directory_is_atomic_and_write_once(tmp_path):
    payloads = {filename: f"{filename}\n".encode() for filename in AGGREGATE_FILENAMES}
    target = publish_aggregate_directory(tmp_path, payloads)
    assert sorted(path.name for path in target.iterdir()) == sorted(AGGREGATE_FILENAMES)
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_aggregate_directory(tmp_path, payloads)
