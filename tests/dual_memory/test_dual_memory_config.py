import copy
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
from mme_vla_suite.shared.mem_buffer import MemoryBuffer
from mme_vla_suite.training.config import TokenizePromptWithSymbolicMemory


N_CONFIG = None
S_CONFIG = "symbolic-grounded-subgoal.yaml"
P_CONFIG = "perceptual-framesamp-modul.yaml"
SP_CONFIG = "dual-grounded-framesamp-modul.yaml"


def _model_config(history_config):
    return HistoryPi0Config(
        use_history=history_config is not None,
        history_config=history_config,
        discrete_state_input=False,
        action_horizon=20,
    )


def test_n_s_p_sp_input_specs_are_distinct_and_complete():
    specs = {}
    for name, history_config in {
        "N": N_CONFIG,
        "S": S_CONFIG,
        "P": P_CONFIG,
        "SP": SP_CONFIG,
    }.items():
        specs[name], _ = _model_config(history_config).inputs_spec(batch_size=2)

    assert not isinstance(specs["N"], HistAugObservation)
    assert specs["S"].symbolic_tokenized_prompt.shape == (2, 128)
    assert specs["S"].static_image_emb is None
    assert specs["P"].symbolic_tokenized_prompt is None
    assert specs["P"].static_image_emb.shape == (2, 512, 2048)
    assert specs["SP"].symbolic_tokenized_prompt.shape == (2, 128)
    assert specs["SP"].static_image_emb.shape == (2, 512, 2048)


def test_hybrid_yaml_round_trip_and_runtime_source_validation():
    config = get_history_config(SP_CONFIG)
    serialized = OmegaConf.to_yaml(config, resolve=True)
    restored = OmegaConf.create(serialized)
    assert OmegaConf.to_container(restored, resolve=True) == OmegaConf.to_container(
        config, resolve=True
    )

    model_config = _model_config(SP_CONFIG)
    assert model_config._resolve_symbolic_options(restored) == (
        True,
        "grounded_subgoal",
        "none",
    )

    invalid = copy.deepcopy(restored)
    invalid.symbolic_source = "test_truth"
    with pytest.raises(ValueError, match="symbolic_source"):
        model_config._resolve_symbolic_options(invalid)

    metadata = {
        "history_config": SP_CONFIG,
        "use_symbolic_prompt": True,
        "symbolic_prompt_type": restored.symbolic_prompt_type,
        "symbolic_source": restored.symbolic_source,
    }
    assert json.loads(json.dumps(metadata)) == metadata


class _RecordingTokenizer:
    def tokenize(self, prompt, state=None, subgoal=None):
        text = f"{prompt}|{subgoal}|{state}"
        values = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
        return values, np.ones(values.shape, dtype=np.bool_)


def test_hybrid_symbolic_prompt_is_token_identical_to_symbolic_only():
    sample = {
        "prompt": "insert the peg",
        "state": np.zeros(8, dtype=np.float32),
        "simple_subgoal": "grasp peg",
        "grounded_subgoal": "grasp peg at <120, 80>",
    }
    symbolic_transform = TokenizePromptWithSymbolicMemory(
        tokenizer=_RecordingTokenizer(),
        discrete_state_input=False,
        symbolic_memory_type="grounded_subgoal",
    )
    hybrid_transform = TokenizePromptWithSymbolicMemory(
        tokenizer=_RecordingTokenizer(),
        discrete_state_input=False,
        symbolic_memory_type=get_history_config(SP_CONFIG).symbolic_prompt_type,
    )

    symbolic = symbolic_transform(copy.deepcopy(sample))
    hybrid = hybrid_transform(copy.deepcopy(sample))
    np.testing.assert_array_equal(
        hybrid["symbolic_tokenized_prompt"], symbolic["symbolic_tokenized_prompt"]
    )
    np.testing.assert_array_equal(
        hybrid["symbolic_tokenized_prompt_mask"],
        symbolic["symbolic_tokenized_prompt_mask"],
    )


def _history_features(step_count, num_views=1, token_count=16):
    features = {}
    for step in range(step_count):
        features[step] = {
            "image_emb_4x4": np.full(
                (num_views, token_count, 3), step, dtype=np.float32
            ),
            "pos_emb_4x4": np.full(
                (num_views, token_count, 2), step + 0.5, dtype=np.float32
            ),
            "state_emb": np.full((2,), step + 1, dtype=np.float32),
        }
    return features


def test_hybrid_framesamp_tensors_match_perceptual_baseline():
    p_config = get_history_config(P_CONFIG)
    sp_config = get_history_config(SP_CONFIG)
    assert p_config.budget == sp_config.budget == 512
    assert p_config.token_per_image == sp_config.token_per_image == 16

    history = _history_features(50)
    outputs = []
    for config in [p_config, sp_config]:
        buffer = MemoryBuffer(
            num_views=config.num_views,
            img_emb_dim=3,
            pos_emb_dim=2,
            state_emb_dim=2,
        )
        outputs.append(
            buffer.prepare_frame_sampling(
                49,
                config.budget,
                config.token_per_image,
                lambda indices: {index: history[index] for index in indices},
            )
        )

    for hybrid_value, perceptual_value in zip(outputs[1], outputs[0], strict=True):
        np.testing.assert_array_equal(hybrid_value, perceptual_value)


def test_framesamp_budget_boundaries_and_reset():
    buffer = MemoryBuffer(num_views=1, img_emb_dim=3, pos_emb_dim=2, state_emb_dim=2)
    assert buffer.get_frame_sampling_indices(0, 512, 16) == [0]
    assert buffer.get_frame_sampling_indices(3, 512, 16) == [0, 1, 2, 3]
    indices = buffer.get_frame_sampling_indices(100, 512, 16)
    assert len(indices) == 32
    assert indices[0] == 0 and indices[-1] == 100

    buffer._history_feats[0] = {"sentinel": True}
    buffer.clear()
    assert buffer._history_feats == {}
