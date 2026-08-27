from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np
import pytest

from mme_vla_suite.policies.policy import MME_VLA_Policy
from mme_vla_suite.models.representation.percep_mem import PerceptualMemory
from mme_vla_suite.shared.keyframe_oracle_sampling import (
    FORMAL_SEED_DATASET,
    FORMAL_SEED_SCOPE,
    SMOKE_SEED_DATASET,
    SMOKE_SEED_SCOPE,
    SelectorArm,
    derive_random_seed,
    derive_smoke_random_seed,
    official_uniform_indices,
)
from mme_vla_suite.shared.mem_buffer import MemoryBuffer


def _fixture_buffer(history_length: int, stages=None) -> MemoryBuffer:
    buffer = MemoryBuffer(
        num_views=1,
        img_emb_dim=2,
        pos_emb_dim=1,
        state_emb_dim=1,
        prepare_buffer=False,
    )
    for index in range(history_length):
        buffer._history_feats[index] = {
            "image_emb_4x4": np.full((1, 16, 2), index, dtype=np.float32),
            "pos_emb_4x4": np.full((1, 16, 1), 1000 + index, dtype=np.float32),
            "state_emb": np.asarray([2000 + index], dtype=np.float32),
        }
    stages = list(stages if stages is not None else [0] * history_length)
    pending = buffer._validated_stage_metadata(stages, list(range(history_length)))
    buffer._history_metadata.update(pending)
    return buffer


@pytest.mark.parametrize("history_length", (1, 2, 16, 32, 33, 64, 1301))
def test_literal_released_uniform_method_matches_pure_expected_fixture(history_length):
    buffer = _fixture_buffer(1)
    step_idx = history_length - 1
    assert buffer.get_frame_sampling_indices(step_idx, 512, 16) == official_uniform_indices(step_idx)


def test_oracle_only_padding_false_mask_and_original_alignment():
    buffer = _fixture_buffer(64, stages=[0] * 20 + [1] * 20 + [2] * 24)

    def oracle_only(step_idx, token_budget, token_per_image):
        assert (step_idx, token_budget, token_per_image) == (63, 512, 16)
        return [0, 20, 40]

    with buffer.temporary_frame_sampling_selector(oracle_only):
        image, position, state, mask = buffer.prepare_frame_sampling(
            63, 512, 16, buffer.default_history_feats_gather_fn
        )
    assert image.shape == (512, 2)
    assert position.shape == (512, 1)
    assert state.shape == (512, 1)
    assert mask.shape == (512,)
    assert mask.dtype == np.bool_
    assert mask.sum() == 3 * 16
    assert not mask[3 * 16 :].any()
    assert np.all(image[:16] == 0)
    assert np.all(image[16:32] == 20)
    assert np.all(image[32:48] == 40)
    assert np.all(position[16:32] == 1020)
    assert np.all(state[32:48] == 2040)


def test_selector_latency_excludes_feature_gather_and_padding(monkeypatch):
    buffer = _fixture_buffer(4)
    monotonic_values = iter((10.0, 10.25))
    monkeypatch.setattr(
        "mme_vla_suite.shared.mem_buffer.time.monotonic",
        lambda: next(monotonic_values),
    )
    buffer.prepare_frame_sampling(3, 512, 16, buffer.default_history_feats_gather_fn)
    assert buffer._last_frame_sampling_selector_latency_ms == pytest.approx(250.0)


def test_policy_selector_latency_includes_boundary_lookup_and_bookkeeping(monkeypatch):
    buffer = _fixture_buffer(4, stages=[0, 0, 1, 1])
    policy = _fixture_policy(SelectorArm.ORACLE_COVERAGE, buffer)
    shared_clock = iter((1.0, 1.01, 2.0, 2.03, 3.0, 3.02))
    monkeypatch.setattr(
        "mme_vla_suite.policies.policy.time.monotonic",
        lambda: next(shared_clock),
    )

    policy._prepare_experiment_frame_sampling(
        buffer.default_history_feats_gather_fn, 512, 16
    )
    trace = policy._pending_selector_trace
    assert trace["boundary_lookup_latency_ms"] == pytest.approx(10.0)
    assert trace["selector_decision_latency_ms"] == pytest.approx(30.0)
    assert trace["selector_bookkeeping_latency_ms"] == pytest.approx(20.0)
    assert trace["selector_latency_ms"] == pytest.approx(60.0)


def test_temporary_override_restores_after_success_and_exception():
    buffer = _fixture_buffer(4)
    with buffer.temporary_frame_sampling_selector(lambda *_: [0, 3]):
        buffer.prepare_frame_sampling(3, 512, 16, buffer.default_history_feats_gather_fn)
    assert buffer._frame_sampling_selector is None

    def explode(*_):
        raise RuntimeError("selector exploded")

    with pytest.raises(RuntimeError, match="selector exploded"):
        with buffer.temporary_frame_sampling_selector(explode):
            buffer.prepare_frame_sampling(3, 512, 16, buffer.default_history_feats_gather_fn)
    assert buffer._frame_sampling_selector is None


def test_memory_metadata_boundaries_cross_segments_and_clear_without_model_exposure():
    buffer = _fixture_buffer(0)
    first = buffer._validated_stage_metadata([2, 2, 3], [0, 1, 2])
    buffer._history_metadata.update(first)
    second = buffer._validated_stage_metadata([3, 4], [3, 4])
    buffer._history_metadata.update(second)
    assert buffer.get_boundary_indices(4) == [0, 2, 4]
    assert all("stage" not in values and "boundary" not in values for values in buffer._history_feats.values())
    buffer._frame_sampling_selector = object()
    buffer.clear()
    assert buffer._history_feats == {}
    assert buffer._history_metadata == {}
    assert buffer._frame_sampling_selector is None


def _fixture_policy(arm: SelectorArm, buffer: MemoryBuffer) -> MME_VLA_Policy:
    policy = MME_VLA_Policy.__new__(MME_VLA_Policy)
    policy.mem_buffer = buffer
    policy.step_idx = len(buffer._history_feats) - 1
    policy._selector_call_index = 0
    policy._selector_rng = None
    task = "InsertPeg"
    policy._keyframe_selector_config = {
        "arm": arm,
        "task": task,
        "episode_id": 0,
        "random_seeds": tuple(derive_random_seed(task, 0, i) for i in range(82)),
        "seed_table_sha256": "a" * 64,
        "seed_table_scope": FORMAL_SEED_SCOPE,
        "seed_table_dataset": FORMAL_SEED_DATASET,
    }
    policy._pending_selector_trace = None
    return policy


def _configurable_policy() -> MME_VLA_Policy:
    policy = MME_VLA_Policy.__new__(MME_VLA_Policy)
    policy.step_idx = -1
    policy.mem_buffer = types.SimpleNamespace(_history_feats={})
    policy.config = types.SimpleNamespace(
        budget=512,
        num_views=1,
        token_per_image=16,
        representation_type="perceptual",
        perceptual_memory=types.SimpleNamespace(type="frame_sampling"),
        integration_type="modulation",
        use_state_emb=False,
        streaming_obs_horizon=16,
    )
    policy._model = types.SimpleNamespace(action_horizon=20)
    return policy


@pytest.mark.parametrize(
    ("scope", "dataset", "derive_seed"),
    [
        (FORMAL_SEED_SCOPE, FORMAL_SEED_DATASET, derive_random_seed),
        (SMOKE_SEED_SCOPE, SMOKE_SEED_DATASET, derive_smoke_random_seed),
    ],
)
def test_configure_selector_validates_seed_derivation_by_explicit_scope(
    scope, dataset, derive_seed
):
    policy = _configurable_policy()
    task = "InsertPeg"
    policy.configure_keyframe_selector(
        {
            "arm": "R",
            "task": task,
            "episode_id": 0,
            "random_seeds": tuple(derive_seed(task, 0, i) for i in range(82)),
            "seed_table_sha256": "a" * 64,
            "seed_table_scope": scope,
            "seed_table_dataset": dataset,
        }
    )
    assert policy._keyframe_selector_config["seed_table_scope"] == scope
    assert policy._keyframe_selector_config["seed_table_dataset"] == dataset


def test_configure_selector_rejects_cross_scope_seed_table():
    policy = _configurable_policy()
    task = "InsertPeg"
    with pytest.raises(ValueError, match="RandomSamp seed mismatch"):
        policy.configure_keyframe_selector(
            {
                "arm": "R",
                "task": task,
                "episode_id": 0,
                "random_seeds": tuple(
                    derive_random_seed(task, 0, i) for i in range(82)
                ),
                "seed_table_sha256": "a" * 64,
                "seed_table_scope": SMOKE_SEED_SCOPE,
                "seed_table_dataset": SMOKE_SEED_DATASET,
            }
        )


def test_runtime_u_calls_literal_method_and_emits_512_token_trace():
    buffer = _fixture_buffer(64, stages=[0] * 32 + [1] * 32)
    calls = []
    literal = buffer.get_frame_sampling_indices

    def spy(*args):
        calls.append(args)
        return literal(*args)

    buffer.get_frame_sampling_indices = spy
    policy = _fixture_policy(SelectorArm.OFFICIAL_UNIFORM, buffer)
    prepared = policy._prepare_experiment_frame_sampling(
        buffer.default_history_feats_gather_fn, 512, 16
    )
    assert calls == [(63, 512, 16)]
    assert prepared[3].sum() == 512
    assert policy._pending_selector_trace["valid_frame_count"] == 32
    assert policy._pending_selector_trace["valid_memory_token_count"] == 512
    assert policy._pending_selector_trace["visible_boundary_indices"] == [0, 32]


def test_runtime_o_uses_only_boundaries_and_keeps_padding_masked():
    buffer = _fixture_buffer(64, stages=[0] * 32 + [1] * 32)
    policy = _fixture_policy(SelectorArm.ORACLE_ONLY, buffer)
    _, _, _, mask = policy._prepare_experiment_frame_sampling(
        buffer.default_history_feats_gather_fn, 512, 16
    )
    trace = policy._pending_selector_trace
    assert trace["selected_frame_indices"] == [0, 32]
    assert trace["valid_memory_token_count"] == 32
    assert trace["padding_frame_count"] == 30
    assert mask[:32].all() and not mask[32:].any()


@pytest.mark.parametrize("history_length", (16, 64))
@pytest.mark.parametrize("arm", tuple(SelectorArm))
def test_all_runtime_arms_preserve_shapes_masks_and_cardinality(arm, history_length):
    stages = [index // 7 for index in range(history_length)]
    buffer = _fixture_buffer(history_length, stages=stages)
    policy = _fixture_policy(arm, buffer)
    image, position, state, mask = policy._prepare_experiment_frame_sampling(
        buffer.default_history_feats_gather_fn, 512, 16
    )
    trace = policy._pending_selector_trace
    assert image.shape == (512, 2)
    assert position.shape == (512, 1)
    assert state.shape == (512, 1)
    assert mask.shape == (512,)
    assert mask.dtype == np.bool_
    boundary_count = len(buffer.get_boundary_indices(history_length - 1))
    expected_frames = (
        min(32, boundary_count)
        if arm is SelectorArm.ORACLE_ONLY
        else min(32, history_length)
    )
    assert trace["valid_frame_count"] == expected_frames
    assert trace["valid_memory_token_count"] == 16 * expected_frames
    assert trace["padding_frame_count"] == 32 - expected_frames


def test_same_random_selection_reproduces_indices_and_memory_digest_in_process():
    first_buffer = _fixture_buffer(64, stages=[0] * 64)
    second_buffer = _fixture_buffer(64, stages=[0] * 64)
    first_policy = _fixture_policy(SelectorArm.RANDOM, first_buffer)
    second_policy = _fixture_policy(SelectorArm.RANDOM, second_buffer)
    first_policy._prepare_experiment_frame_sampling(
        first_buffer.default_history_feats_gather_fn, 512, 16
    )
    second_policy._prepare_experiment_frame_sampling(
        second_buffer.default_history_feats_gather_fn, 512, 16
    )
    first_trace = first_policy._pending_selector_trace
    second_trace = second_policy._pending_selector_trace
    assert first_trace["selected_frame_indices"] == second_trace["selected_frame_indices"]
    assert (
        first_trace["prepared_memory_components_sha256"]
        == second_trace["prepared_memory_components_sha256"]
    )


def test_trace_distinguishes_prepared_inputs_from_final_1024d_memory_tensor():
    buffer = _fixture_buffer(4, stages=[0, 0, 1, 1])
    policy = _fixture_policy(SelectorArm.OFFICIAL_UNIFORM, buffer)
    policy.config = types.SimpleNamespace(budget=512, memory_token_dim=1024)
    policy._prepare_experiment_frame_sampling(
        buffer.default_history_feats_gather_fn, 512, 16
    )
    assert "prepared_memory_components_sha256" in policy._pending_selector_trace
    assert "final_memory_tensor_sha256" not in policy._pending_selector_trace

    final_memory = np.zeros((1, 512, 1024), dtype=np.float32)
    final_memory[:, 0, 0] = 1.0
    policy._record_final_memory_tensor(final_memory)
    trace = policy._pending_selector_trace
    assert trace["final_memory_tensor_shape"] == [1, 512, 1024]
    assert trace["final_memory_tensor_dtype"] == "float32"
    assert trace["final_memory_tensor_is_floating"] is True
    assert trace["final_memory_tensor_finite"] is True
    assert len(trace["final_memory_tensor_sha256"]) == 64

    policy._record_final_memory_tensor(
        np.asarray(jnp.zeros((1, 512, 1024), dtype=jnp.bfloat16))
    )
    assert policy._pending_selector_trace["final_memory_tensor_dtype"] == "bfloat16"
    assert policy._pending_selector_trace["final_memory_tensor_is_floating"] is True

    invalid_memory = np.zeros((1, 512, 1024), dtype=np.float32)
    invalid_memory[0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        policy._record_final_memory_tensor(invalid_memory)


def test_perceptual_memory_audit_method_is_exact_feature_encoder_output():
    class Encoder:
        def encode_perceptual_memory(self, image, position, state):
            assert position is positions
            assert state is states
            return image + 7

    memory = types.SimpleNamespace(
        config=types.SimpleNamespace(budget=3),
        feature_encoder=Encoder(),
    )
    images = np.zeros((1, 3, 2), dtype=np.float32)
    positions = np.ones((1, 3, 1), dtype=np.float32)
    states = np.ones((1, 3, 1), dtype=np.float32)
    observed = PerceptualMemory.encode_tokens(memory, images, positions, states)
    np.testing.assert_array_equal(observed, images + 7)


def test_policy_reset_clears_history_configuration_call_index_and_rng():
    policy = MME_VLA_Policy.__new__(MME_VLA_Policy)
    old_buffer = _fixture_buffer(2, stages=[0, 1])
    policy.mem_buffer = old_buffer
    policy._seed = 7
    policy._keyframe_selector_config = {"arm": SelectorArm.RANDOM}
    policy._selector_call_index = 12
    policy._selector_rng = np.random.default_rng(9)
    policy._pending_selector_trace = {"old": True}
    new_buffer = _fixture_buffer(0)
    policy._prepare_mem_buffer = types.MethodType(
        lambda self: setattr(self, "mem_buffer", new_buffer), policy
    )
    policy.reset()
    assert policy.mem_buffer is new_buffer
    assert policy.step_idx == -1
    assert policy._keyframe_selector_config is None
    assert policy._selector_call_index == 0
    assert policy._selector_rng is None
    assert policy._pending_selector_trace is None


def test_recurrent_baseline_add_buffer_does_not_receive_stage_keyword():
    calls = []

    class RecurrentBuffer:
        def add_buffer(self, images, states, step_indices):
            calls.append((images, states, step_indices))

    policy = MME_VLA_Policy.__new__(MME_VLA_Policy)
    policy.mem_buffer = RecurrentBuffer()
    policy.config = types.SimpleNamespace(representation_type="recurrent")
    policy._keyframe_selector_config = None
    policy.step_idx = -1
    policy.exec_start_idx = 0
    images = np.zeros((2, 1, 4, 4, 3), dtype=np.uint8)
    states = np.zeros((2, 8), dtype=np.float32)
    policy.add_buffer({"images": images, "state": states})
    assert calls[0][2] == [0, 1]
    assert policy.step_idx == 1

    with pytest.raises(RuntimeError, match="unsupported for recurrent"):
        policy.add_buffer(
            {"images": images, "state": states, "current_task_index": [0, 0]}
        )


def test_out_of_table_policy_call_is_a_hard_stop():
    buffer = _fixture_buffer(1)
    policy = _fixture_policy(SelectorArm.RANDOM, buffer)
    policy._selector_call_index = 82
    with pytest.raises(RuntimeError, match="exceeds preregistered seed table"):
        policy._prepare_experiment_frame_sampling(
            buffer.default_history_feats_gather_fn, 512, 16
        )
