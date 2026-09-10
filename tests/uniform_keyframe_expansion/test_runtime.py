from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import types

import jax
import numpy as np
from omegaconf import OmegaConf
import pytest

from mme_vla_suite.policies.policy import MME_VLA_Policy
from mme_vla_suite.policies.uniform_keyframe_expansion_policy import UniformKeyframeExpansionPolicy
from mme_vla_suite.shared.mem_buffer import MemoryBuffer
from mme_vla_suite.shared.uniform_keyframe_config import RELEASED_HISTORY_CONFIG
from mme_vla_suite.shared.uniform_keyframe_config import expanded_history_mapping
from mme_vla_suite.shared.uniform_keyframe_config import payload_digest
from mme_vla_suite.shared.uniform_keyframe_expansion import ExpansionInvariantError
from mme_vla_suite.shared.uniform_keyframe_expansion import derive_expansion_seed


def selector_config(arm="UK48", *, split="val", task="InsertPeg", episode_id=0):
    seeds = [derive_expansion_seed(split, task, episode_id, i) for i in range(82)]
    return {"arm": arm, "split": split, "task": task, "episode_id": episode_id,
            "random_seeds": seeds, "seed_table_sha256": payload_digest(seeds)}


def fixture_buffer(length=0, boundaries=(0,), *, real_dimensions=False):
    dims = (2048, 768, 8) if real_dimensions else (2, 1, 1)
    buffer = MemoryBuffer(img_emb_dim=dims[0], pos_emb_dim=dims[1], state_emb_dim=dims[2], prepare_buffer=False)
    stages = []
    stage = -1
    for i in range(length):
        if i in boundaries:
            stage += 1
        stages.append(stage)
        buffer._history_feats[i] = {
            "image_emb_4x4": np.full((1, 16, dims[0]), i + 10, np.float32),
            "pos_emb_4x4": np.full((1, 16, dims[1]), i + 1000, np.float32),
            "state_emb": np.full((dims[2],), i + 2000, np.float32),
        }
    buffer._history_metadata.update(buffer._validated_stage_metadata(stages, list(range(length))))
    return buffer


def fixture_policy(arm="UK48", length=64, boundaries=(0, 17, 35), *, real_dimensions=False):
    policy = UniformKeyframeExpansionPolicy.__new__(UniformKeyframeExpansionPolicy)
    policy._seed = 7
    policy._model = types.SimpleNamespace(action_horizon=20)
    policy.config = OmegaConf.create(expanded_history_mapping(RELEASED_HISTORY_CONFIG)[0])
    policy.mem_buffer = fixture_buffer()
    policy.step_idx = -1
    policy.exec_start_idx = 0
    policy._keyframe_selector_config = None
    policy._pending_selector_trace = None
    policy.last_expansion_failure = None
    policy._rng = jax.random.key(7)
    policy.use_quantiles = False
    policy.state_norm_stats = types.SimpleNamespace(mean=0, std=1)
    policy.configure_uniform_keyframe_expansion(selector_config(arm))
    policy.mem_buffer = fixture_buffer(length, boundaries, real_dimensions=real_dimensions)
    policy.step_idx = length - 1
    return policy


@pytest.mark.parametrize("arm", ("UK48", "UN48"))
@pytest.mark.parametrize("length", (1, 2, 16, 32, 33, 64, 1301))
def test_real_buffer_preserves_literal_u_and_pads_48_slots(arm, length):
    boundaries = tuple(i for i in (0, 17, 35, 127, 256, 511, 1023) if i < length)
    policy = fixture_policy(arm, length, boundaries)
    buffer = policy.mem_buffer
    calls = []
    literal = buffer.get_frame_sampling_indices

    def spy(*args):
        calls.append(args)
        return literal(*args)

    buffer.get_frame_sampling_indices = spy
    before_rng = np.asarray(jax.random.key_data(policy._rng)).copy()
    inputs = policy._prepare_history({"prompt": "unchanged task"})
    trace = policy._pending_selector_trace
    indices = trace["selected_frame_indices"]
    base = literal(length - 1, 512, 16)
    assert calls == [(length - 1, 512, 16)]
    assert set(base).issubset(indices)
    assert len(indices) == len(base) + len(set(boundaries) - set(base))
    assert inputs["static_mask"].shape == (768,)
    assert inputs["static_mask"].dtype == np.bool_
    valid = len(indices) * 16
    assert inputs["static_mask"][:valid].all()
    assert not inputs["static_mask"][valid:].any()
    assert np.all(inputs["static_image_emb"][valid:] == 0)
    assert np.all(inputs["static_pos_emb"][valid:] == 0)
    assert trace["padding_frame_count"] == 48 - len(indices)
    assert trace["valid_memory_token_count"] == valid
    for position, index in enumerate(indices):
        assert np.all(inputs["static_image_emb"][16 * position:16 * (position + 1)] == index + 10)
        assert np.all(inputs["static_pos_emb"][16 * position:16 * (position + 1)] == index + 1000)
    assert policy.mem_buffer._frame_sampling_selector is None
    assert policy._selector_call_index == 1
    np.testing.assert_array_equal(jax.random.key_data(policy._rng), before_rng)
    assert set(inputs) == {"prompt", "static_image_emb", "static_pos_emb", "static_state_emb", "static_mask"}


def test_same_history_equal_additions_and_different_contents():
    key, random = fixture_policy(), fixture_policy("UN48")
    key_inputs, random_inputs = key._prepare_history({}), random._prepare_history({})
    kt, rt = key._pending_selector_trace, random._pending_selector_trace
    assert kt["base_uniform_indices"] == rt["base_uniform_indices"]
    assert kt["extra_count"] == rt["extra_count"] == 2
    assert kt["selected_extra_indices"] == [17, 35]
    assert not set(rt["selected_extra_indices"]) & (set(rt["base_uniform_indices"]) | set(rt["visible_boundary_indices"]))
    np.testing.assert_array_equal(key_inputs["static_mask"], random_inputs["static_mask"])
    again = fixture_policy("UN48")
    again._prepare_history({})
    assert rt["selected_frame_indices"] == again._pending_selector_trace["selected_frame_indices"]
    assert rt["prepared_memory_components_sha256"] == again._pending_selector_trace["prepared_memory_components_sha256"]


def test_full_48_slots_records_normalized_state_dtype_without_changing_state_memory():
    buffer = fixture_buffer()
    base = set(buffer.get_frame_sampling_indices(95, 512, 16))
    extras = sorted(set(range(96)) - base)[:16]
    policy = fixture_policy(length=96, boundaries=tuple([0, *extras]))
    policy.state_norm_stats = types.SimpleNamespace(mean=np.zeros(1, np.float64), std=np.ones(1, np.float64))
    inputs = policy._prepare_history({})
    assert inputs["static_mask"].all()
    assert inputs["static_state_emb"].dtype == np.float64
    assert policy._pending_selector_trace["state_tensor_dtype"] == "float64"
    assert policy.config.use_state_emb is False


def test_production_dimensions_and_final_memory_audit():
    policy = fixture_policy(length=3, boundaries=(0,), real_dimensions=True)
    inputs = policy._prepare_history({})
    assert inputs["static_image_emb"].shape == (768, 2048)
    assert inputs["static_pos_emb"].shape == (768, 768)
    assert inputs["static_state_emb"].shape == (768, 8)
    assert policy._pending_selector_trace["prepared_memory_input_shape"] == [768, 2824]
    policy._record_final_memory_tensor(np.zeros((1, 768, 1024), np.float32))
    assert policy._pending_selector_trace["final_memory_tensor_shape"] == [1, 768, 1024]
    with pytest.raises(RuntimeError, match="shape mismatch"):
        policy._record_final_memory_tensor(np.zeros((1, 512, 1024), np.float32))


@pytest.mark.parametrize("arm", ("UK48", "UN48"))
@pytest.mark.parametrize(("length", "boundaries", "code"), [
    (64, tuple(range(64)), "capacity_overflow"),
    (33, tuple(range(33)), "nonkey_candidate_shortage"),
])
def test_hard_stop_no_gather_no_fallback_and_exception_restores_hook(arm, length, boundaries, code):
    policy = fixture_policy(arm, length, boundaries)

    def forbidden_gather(*_):
        pytest.fail("No features may be gathered after selector invariant failure")

    with pytest.raises(ExpansionInvariantError) as caught:
        policy._prepare_experiment_frame_sampling(forbidden_gather, 768, 16)
    assert caught.value.code == code
    assert policy.last_expansion_failure["error_code"] == code
    assert policy.mem_buffer._frame_sampling_selector is None
    assert policy._selector_call_index == 0
    with pytest.raises(RuntimeError, match="invariant failed"):
        policy._prepare_history({})


def test_gather_exception_restores_hook():
    policy = fixture_policy()

    def fail(*_):
        raise RuntimeError("gather failure")

    with pytest.raises(RuntimeError, match="gather failure"):
        policy._prepare_experiment_frame_sampling(fail, 768, 16)
    assert policy.mem_buffer._frame_sampling_selector is None


def test_reset_clears_memory_config_failures_rng_and_blocks_unconfigured_uniform48():
    policy = fixture_policy("UN48")
    policy._prepare_history({})
    policy.last_expansion_failure = {"error_code": "example"}
    policy._prepare_mem_buffer = types.MethodType(lambda self: setattr(self, "mem_buffer", fixture_buffer()), policy)
    policy.reset()
    assert policy.last_expansion_failure is None
    assert policy.reset_evidence()["selector_unconfigured"] is True
    with pytest.raises(RuntimeError, match="Configure an expansion"):
        policy._prepare_history({})
    policy.configure_uniform_keyframe_expansion(selector_config("UK48"))
    assert policy._keyframe_selector_config["arm"] == "UK48"
    with pytest.raises(RuntimeError, match="fresh"):
        policy.configure_uniform_keyframe_expansion(selector_config("UN48"))
    with pytest.raises(ValueError, match="not an old-family"):
        policy.configure_keyframe_selector({"arm": "U"})


def test_invalid_configuration_and_missing_live_metadata_fail_closed():
    policy = fixture_policy(length=0)
    policy._keyframe_selector_config = None
    config = selector_config()
    config["random_seeds"][0] += 1
    with pytest.raises(ValueError, match="seed list"):
        policy.configure_uniform_keyframe_expansion(config)
    config = selector_config()
    config["seed_table_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        policy.configure_uniform_keyframe_expansion(config)
    policy.config.budget = 512
    with pytest.raises(ValueError, match="48/768"):
        policy.configure_uniform_keyframe_expansion(selector_config())
    policy = fixture_policy()
    del policy.mem_buffer._history_metadata[17]
    with pytest.raises(ValueError, match="Missing live"):
        policy._prepare_history({})
    policy = fixture_policy()
    policy._selector_call_index = 82
    with pytest.raises(RuntimeError, match="0..81"):
        policy._prepare_history({})


def test_original_policy_still_requires_512_and_original_yaml_unchanged():
    config_path = Path("src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml")
    assert OmegaConf.to_container(OmegaConf.load(config_path)) == RELEASED_HISTORY_CONFIG
    policy = MME_VLA_Policy.__new__(MME_VLA_Policy)
    policy.config = OmegaConf.create(expanded_history_mapping(RELEASED_HISTORY_CONFIG)[0])
    policy._model = types.SimpleNamespace(action_horizon=20)
    policy.mem_buffer = fixture_buffer()
    policy.step_idx = -1
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        policy.configure_keyframe_selector({"arm": "U"})


def test_configuration_copy_changes_budget_only():
    original = deepcopy(RELEASED_HISTORY_CONFIG)
    expanded, evidence = expanded_history_mapping(original)
    assert original == RELEASED_HISTORY_CONFIG
    restored = deepcopy(expanded)
    restored["budget"] = 512
    assert restored == original
    assert evidence["changed_fields"] == {"budget": {"from": 512, "to": 768}}
    for field, value in (("budget", 512.0), ("num_views", True), ("use_state_emb", True)):
        bad = deepcopy(original)
        bad[field] = value
        with pytest.raises(ValueError, match="exact released"):
            expanded_history_mapping(bad)
