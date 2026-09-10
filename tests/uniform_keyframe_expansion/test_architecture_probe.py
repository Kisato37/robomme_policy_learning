"""CPU helper tests only: no test here produces real-checkpoint/GPU PASS evidence."""
from __future__ import annotations

from copy import deepcopy
import dataclasses
import types

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from flax import nnx, struct
from omegaconf import OmegaConf

from experiments.uniform_keyframe_expansion import architecture_probe as probe
from mme_vla_suite.shared.uniform_keyframe_config import RELEASED_HISTORY_CONFIG, expanded_history_mapping


def test_public_entry_rejects_unvalidated_plan_before_backend_or_policy(monkeypatch):
    monkeypatch.setattr(jax, "devices", lambda: pytest.fail("Backend must not be queried"))
    with pytest.raises(probe.ArchitectureProbeError, match="sealed"):
        probe.run_loaded_probe(object(), {"approved": True})


@pytest.mark.parametrize("length", (16, 64))
def test_synthetic_fixture_is_exact_parent_generator_and_causal(length):
    from experiments.keyframe_oracle_sampling.architecture_smoke import _synthetic_history
    before = np.random.get_state()
    actual = probe._synthetic_history(length)
    expected = _synthetic_history(length)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)
    assert actual[0].shape == (length, 1, 256, 256, 3)
    assert actual[1].dtype == np.float32
    np.testing.assert_array_equal(actual[2], np.arange(length) // 7)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize("length", (0, 15, 32, 65, 1301))
def test_only_frozen_history_lengths(length):
    with pytest.raises(probe.ArchitectureProbeError):
        probe._synthetic_history(length)


def test_seed_context_is_fixed_development_and_identical_across_arms():
    first, second = probe._selector_config("UK48"), probe._selector_config("UN48")
    assert first["split"] == "val" and first["episode_id"] == 0 and first["task"] == "InsertPeg"
    assert first["random_seeds"] == second["random_seeds"]
    assert len(first["random_seeds"]) == 82
    assert first["seed_table_sha256"] == second["seed_table_sha256"]


@pytest.mark.parametrize("stats", (None, {}, {"peak_bytes_in_use": 0}, {"peak_bytes_in_use": -1},
                                  {"peak_bytes_in_use": False}, {"peak_bytes_in_use": float("nan")}))
def test_unavailable_or_fake_zero_peak_cannot_be_reported(stats):
    with pytest.raises(probe.ArchitectureProbeError):
        probe._peak_memory(types.SimpleNamespace(memory_stats=lambda: stats))


def test_actual_peak_scope_is_cumulative_not_per_case():
    result = probe._peak_memory(types.SimpleNamespace(memory_stats=lambda: {"peak_bytes_in_use": 4096,
                                                                          "bytes_in_use": 2048}))
    assert result["peak_gpu_bytes"] == 4096
    assert "cumulative" in result["scope"]


def test_capture_blocks_actual_sample_and_restores_inherited_methods_on_error():
    calls = []

    class Policy:
        def _prepare_experiment_frame_sampling(self):
            return (np.ones((1,), np.float32),) * 4

        def _prepare_history(self, inputs):
            return {field: np.ones((1,), np.float32) for field in probe.MEMORY_FIELDS}

        def _sample_actions(self, rng, observation):
            calls.append("sample")
            return np.ones((1, 20, 8), np.float32)

        def _perceptual_memory_encode(self, *args):
            return np.zeros((1, 768, 1024), np.float32)

    policy = Policy()
    fake_jax = types.SimpleNamespace(block_until_ready=lambda value: calls.append("block"))
    with pytest.raises(RuntimeError, match="fixture failure"):
        with probe._capture(policy, fake_jax) as capture:
            policy._prepare_experiment_frame_sampling()
            policy._prepare_history({})
            policy._sample_actions("rng", "observation")
            policy._perceptual_memory_encode()
            assert calls[:3] == ["block", "sample", "block"]
            assert capture["raw_model_actions"].shape == (1, 20, 8)
            assert capture["final_memory"].shape == (1, 768, 1024)
            assert capture["synchronized_sample_latency_ms"] >= 0
            raise RuntimeError("fixture failure")
    assert policy.__dict__ == {}
    assert policy._sample_actions.__func__ is Policy._sample_actions


def array_fixture():
    arrays = {}
    trace = {"valid_memory_token_count": 256}
    for key, shape in zip(probe.MEMORY_KEYS, probe.RAW_SHAPES, strict=True):
        value = np.zeros(shape, dtype=np.bool_ if key == "mask" else np.float64)
        value[:256] = 1
        arrays[f"raw_{key}"] = value.copy()
        pre = value.copy()
        if key == "state":
            pre -= 3.0  # normalized padding may be nonzero
        arrays[f"pre_model_{key}"] = pre
        arrays[key] = np.asarray(jnp.asarray(pre))[None, ...].copy()
        trace[f"{key}_tensor_sha256" if key != "mask" else "mask_sha256"] = probe._digest(pre)
    arrays["final_memory"] = np.full((1, 768, 1024), 2.0, ml_dtypes.bfloat16)
    trace["final_memory_tensor_sha256"] = probe._digest(arrays["final_memory"])
    arrays["actions"] = np.ones((20, 8), np.float32)
    arrays["raw_model_actions"] = np.ones((1, 20, 32), np.float32)
    return arrays, trace


def test_actual_stage_hashes_allow_original_jax_downcast_and_normalized_nonzero_padding():
    arrays, trace = array_fixture()
    assert arrays["pre_model_image"].dtype == np.float64
    probe._validate_arrays(arrays, trace, jax=jax)


@pytest.mark.parametrize("mutation", ("raw_padding", "mask", "final_memory", "action_shape", "trace_hash", "input_cast"))
def test_captured_actual_arrays_reject_mutation(mutation):
    arrays, trace = array_fixture()
    if mutation == "raw_padding":
        arrays["raw_image"][400, 0] = 1
    elif mutation == "mask":
        arrays["raw_mask"][400] = True
    elif mutation == "final_memory":
        arrays["final_memory"][0, 2, 3] = float("nan")
    elif mutation == "action_shape":
        arrays["actions"] = np.ones((20, 32))
    elif mutation == "trace_hash":
        trace["state_tensor_sha256"] = "0" * 64
    else:
        arrays["image"][0, 2, 0] += 1
    with pytest.raises(probe.ArchitectureProbeError):
        probe._validate_arrays(arrays, trace, jax=jax)


def test_bfloat16_audit_conversion_is_value_preserving_and_write_once(tmp_path):
    value = np.asarray([1.234, -3.5, 0.0], ml_dtypes.bfloat16)
    original = value.copy()
    path = tmp_path / "raw.npz"
    record = probe._write_npz(path, {"final_memory": value})
    with np.load(path, allow_pickle=False) as archive:
        saved = archive["final_memory"]
    assert saved.dtype == np.float32
    np.testing.assert_array_equal(saved.astype(ml_dtypes.bfloat16), original)
    np.testing.assert_array_equal(value, original)
    metadata = probe._encoding_metadata({"final_memory": value})["final_memory"]
    assert metadata["original_dtype"] == "bfloat16" and metadata["stored_dtype"] == "float32"
    assert record == probe._file_record(path)
    with pytest.raises(FileExistsError):
        probe._write_npz(path, {"final_memory": value})
    assert probe._file_record(path) == record
    assert not list(tmp_path.glob(".raw.npz.*"))


@pytest.mark.parametrize("value", (np.asarray([object()]), np.asarray(["fake"]), np.asarray([np.inf])))
def test_raw_artifact_rejects_unsupported_array(value, tmp_path):
    with pytest.raises(probe.ArchitectureProbeError):
        probe._write_npz(tmp_path / "bad.npz", {"value": value})
    assert not list(tmp_path.iterdir())


@dataclasses.dataclass(frozen=True)
class TinyConfig:
    history_config: object


class TinyMemory(nnx.Module):
    def __init__(self, config):
        self.config = config


class TinyModel(nnx.Module):
    """Explicit CPU toy, NOT a HistoryPi0 and not accepted by public probe."""
    def __init__(self):
        config = OmegaConf.create(expanded_history_mapping(RELEASED_HISTORY_CONFIG)[0])
        self.history_config = config
        self.config = TinyConfig(config)
        self.mem_encoder = TinyMemory(config)
        self.weight = nnx.Param(jnp.asarray([2.0]))
        self.action_horizon, self.action_dim = 20, 32

    def sample_actions(self, rng, obs, *, noise=None, num_steps=10):
        assert obs.static_image_emb.shape[1] == self.mem_encoder.config.budget
        if noise is None:
            noise = jax.random.normal(rng, (1, 20, 32))
        return noise + self.weight.value[0] * (self.history_config.budget / 1000)

    def predict_velocity_from_preprocessed(self, obs, x_t, time):
        assert obs.static_image_emb.shape[1] == self.mem_encoder.config.budget
        return x_t + self.weight.value[0] * (self.history_config.budget / 1000), None


@struct.dataclass
class TinyObservation:
    static_image_emb: object
    static_pos_emb: object
    static_state_emb: object
    static_mask: object
    state: object


def toy_observation():
    return TinyObservation(jnp.zeros((1, 768, 2048)), jnp.zeros((1, 768, 768)),
                           jnp.zeros((1, 768, 8)), jnp.arange(768)[None, :] < 256, jnp.zeros((1, 8)))


def test_independent_512_graph_shares_real_array_objects_without_mutating_primary():
    model = TinyModel()
    clone, before = probe._make_512_graph(model, nnx, jax)
    assert clone is not model and clone.mem_encoder is not model.mem_encoder
    assert model.history_config.budget == model.config.history_config.budget == model.mem_encoder.config.budget == 768
    assert clone.history_config.budget == clone.config.history_config.budget == clone.mem_encoder.config.budget == 512
    assert clone.weight.value is model.weight.value
    clone.history_config.budget = 513
    assert model.history_config.budget == 768
    probe._same_parameter_arrays(before, probe._parameter_snapshot(model, nnx, jax))
    clone.weight.value = clone.weight.value + 1
    with pytest.raises(probe.ArchitectureProbeError, match="same JAX arrays"):
        probe._same_parameter_arrays(before, probe._parameter_snapshot(clone, nnx, jax))


def test_all_four_memory_fields_are_sliced_only_false_padding():
    original = toy_observation()
    low = probe._slice_u_observation(original)
    assert low.state is original.state
    for field in probe.MEMORY_FIELDS:
        assert getattr(low, field).shape[1] == 512
        np.testing.assert_array_equal(getattr(low, field), getattr(original, field)[:, :512])
    with pytest.raises(probe.ArchitectureProbeError):
        probe._slice_u_observation(original.replace(static_mask=jnp.ones((1, 768), dtype=bool)))


def test_padding_diagnostic_executes_both_models_and_actual_velocity_same_noise_cpu_only(monkeypatch):
    from mme_vla_suite.models.integration import history_observation
    from openpi.shared import nnx_utils

    monkeypatch.setattr(history_observation, "preprocess_observation", lambda _, obs, train: obs)
    model = TinyModel()
    observation = toy_observation()
    rng = jax.random.split(jax.random.key(7))[1]
    noise = jax.random.normal(rng, (1, 20, 32))
    real_infer_raw = nnx_utils.module_jit(model.sample_actions)(rng, observation)
    policy = types.SimpleNamespace(_model=model, _sample_kwargs={},
                                   _output_transform=lambda x: {"actions": x["actions"][:, :8]})
    case = {"trace": {"history_length": 16, "selected_frame_indices": list(range(16)), "extra_count": 0},
            "observation": observation, "initial_noise": noise, "sample_rng": rng,
            "arrays": {"raw_model_actions": np.asarray(real_infer_raw)}}
    measurement, arrays = probe._padding_diagnostic(policy, case, jax)
    assert measurement["action_difference_linf"] > 0
    assert measurement["velocity_difference_linf"] > 0
    assert measurement["shared_parameter_array_identity"] is True
    assert measurement["primary_budget_after"] == 768
    assert arrays["actions_512"].shape == arrays["actions_768"].shape == (20, 8)
    assert arrays["velocity_512"].shape == arrays["velocity_768"].shape == (1, 20, 32)
    np.testing.assert_array_equal(arrays["initial_noise"], noise)
    # Internal numerical toy results are not a complete real architecture report.
    assert "strict_load" not in measurement and "passed" not in measurement


def test_repeat_detects_changes_to_any_actual_capture_and_unexpected_compile():
    first = {"trace": {"selected_frame_indices": [0, 1], "selector_seed": 123},
             "arrays": {"actions": np.ones((20, 8)), "initial_noise": np.ones((1, 20, 32))},
             "cache_before": {"sample": 1}, "cache_after": {"sample": 1}}
    repeat = deepcopy(first)
    probe._compare_repeat(first, repeat)
    repeat["arrays"]["initial_noise"][0, 0, 0] = 2
    with pytest.raises(probe.ArchitectureProbeError, match="initial_noise"):
        probe._compare_repeat(first, repeat)
    repeat = deepcopy(first)
    repeat["cache_after"]["sample"] += 1
    with pytest.raises(probe.ArchitectureProbeError, match="recompiled"):
        probe._compare_repeat(first, repeat)
