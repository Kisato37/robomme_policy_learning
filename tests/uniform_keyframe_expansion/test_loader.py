"""CPU wiring tests with mocked checkpoint I/O; not real-checkpoint evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import types

from omegaconf import OmegaConf
import numpy as np
import pytest

from mme_vla_suite.policies import policy_config
from mme_vla_suite.policies import uniform_keyframe_expansion_policy
from mme_vla_suite.shared.uniform_keyframe_config import EXPANSION_FAMILY
from mme_vla_suite.shared.uniform_keyframe_config import RELEASED_HISTORY_CONFIG, expanded_history_mapping


@dataclass(frozen=True)
class FakeModel:
    calls: list
    history_config: object = "perceptual-framesamp-modul.yaml"
    use_history: bool = True
    pi05: bool = True
    action_horizon: int = 20
    use_symbolic_prompt: bool = False
    symbolic_source: str = "none"

    def load(self, params, *, remove_extra_params=True):
        self.calls.append({"model_config": self, "params": params, "remove_extra_params": remove_extra_params})
        return object()


@dataclass(frozen=True)
class FakeTrainConfig:
    model: FakeModel
    data: object
    assets_dirs: tuple = ()
    policy_metadata: object = None


@pytest.fixture
def setup_loader(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint" / "79999"
    checkpoint.mkdir(parents=True)
    metadata = checkpoint.parent / "history_config.txt"
    metadata.write_text("perceptual-framesamp-modul.yaml")
    calls = []
    params = {"sentinel": "unmodified weights"}
    restored = []

    def restore(path, dtype):
        restored.append((path, dtype))
        return params

    monkeypatch.setattr(policy_config._model, "restore_params", restore)

    def factory(kind):
        return lambda model, **kwargs: {"kind": kind, "model": model, **kwargs}

    monkeypatch.setattr(policy_config._policy, "MME_VLA_Policy", factory("legacy"))
    monkeypatch.setattr(uniform_keyframe_expansion_policy, "UniformKeyframeExpansionPolicy", factory("expanded"))
    data_config = types.SimpleNamespace(
        asset_id="example", use_quantile_norm=False,
        data_transforms=policy_config.transforms.Group(), model_transforms=policy_config.transforms.Group(),
    )
    train_config = FakeTrainConfig(
        model=FakeModel(calls), data=types.SimpleNamespace(create=lambda *_: data_config),
        policy_metadata={"existing": "preserved"},
    )
    return checkpoint, metadata, train_config, calls, restored, params


def test_opt_in_copies_config_strictly_loads_and_preserves_checkpoint_metadata(setup_loader):
    checkpoint, metadata, config, calls, restored, params = setup_loader
    original_bytes = metadata.read_bytes()
    original_yaml = Path("src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml").read_bytes()
    policy = policy_config.create_trained_policy(
        config, str(checkpoint), seed=7, norm_stats={}, experimental_memory_expansion=EXPANSION_FAMILY,
    )
    assert policy["kind"] == "expanded"
    assert calls[0]["remove_extra_params"] is False
    assert calls[0]["params"] is params
    assert restored[0][0] == checkpoint / "params"
    effective = OmegaConf.to_container(calls[0]["model_config"].history_config)
    assert effective["budget"] == 768
    assert effective["token_per_image"] == 16
    assert effective["use_state_emb"] is False
    assert config.model.history_config == "perceptual-framesamp-modul.yaml"
    assert metadata.read_bytes() == original_bytes
    assert Path("src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml").read_bytes() == original_yaml
    evidence = policy["metadata"]["uniform_keyframe_expansion"]
    assert evidence["source_history_config"]["budget"] == 512
    assert evidence["effective_history_config"]["budget"] == 768
    assert evidence["strict_weight_tree_load"] is True
    assert policy["metadata"]["existing"] == "preserved"
    assert config.policy_metadata == {"existing": "preserved"}


def test_default_loader_retains_legacy_behavior(setup_loader):
    checkpoint, metadata, config, calls, restored, params = setup_loader
    policy = policy_config.create_trained_policy(config, checkpoint, seed=7, norm_stats={})
    assert policy["kind"] == "legacy"
    assert calls[0]["remove_extra_params"] is True
    assert calls[0]["model_config"].history_config == "perceptual-framesamp-modul.yaml"
    assert "uniform_keyframe_expansion" not in policy["metadata"]


@pytest.mark.parametrize("invalid_family", ("48", "UN48", "uniform48", ""))
def test_invalid_override_fails_before_checkpoint_read(setup_loader, invalid_family):
    checkpoint, _, config, calls, restored, _ = setup_loader
    with pytest.raises(ValueError, match="Unknown"):
        policy_config.create_trained_policy(
            config, checkpoint, seed=7, norm_stats={}, experimental_memory_expansion=invalid_family,
        )
    assert not restored and not calls


def test_missing_original_metadata_fails_before_checkpoint_read(setup_loader):
    checkpoint, metadata, config, calls, restored, _ = setup_loader
    # Test-owned temporary data only.
    metadata.unlink()
    with pytest.raises(ValueError, match="original checkpoint"):
        policy_config.create_trained_policy(
            config, checkpoint, seed=7, norm_stats={}, experimental_memory_expansion=EXPANSION_FAMILY,
        )
    assert not restored and not calls


def test_wrong_seed_fails_before_checkpoint_read(setup_loader):
    checkpoint, _, config, calls, restored, _ = setup_loader
    with pytest.raises(ValueError, match="seed 7"):
        policy_config.create_trained_policy(
            config, checkpoint, seed=42, norm_stats={}, experimental_memory_expansion=EXPANSION_FAMILY,
        )
    assert not restored and not calls


def test_mismatched_weight_tree_failure_is_not_caught_or_reinitialized(setup_loader, monkeypatch):
    checkpoint, _, config, _, _, _ = setup_loader

    def mismatch(self, params, *, remove_extra_params=True):
        assert remove_extra_params is False
        raise ValueError("checkpoint tree mismatch")

    monkeypatch.setattr(FakeModel, "load", mismatch)
    with pytest.raises(ValueError, match="checkpoint tree mismatch"):
        policy_config.create_trained_policy(
            config, checkpoint, seed=7, norm_stats={}, experimental_memory_expansion=EXPANSION_FAMILY,
        )


def test_real_model_input_spec_accepts_768_without_changing_action_or_current_inputs():
    # This resolves the real configuration only; no model/weights are loaded.
    from mme_vla_suite.training.config import get_config
    from mme_vla_suite.policies.robomme_policy import RoboMMEOutputs
    base = get_config("mme_vla_suite").model
    original = replace(base, history_config=OmegaConf.create(RELEASED_HISTORY_CONFIG))
    expanded = replace(base, history_config=OmegaConf.create(expanded_history_mapping(RELEASED_HISTORY_CONFIG)[0]))
    before, actions_before = original.inputs_spec()
    after, actions_after = expanded.inputs_spec()
    assert before.static_image_emb.shape == (1, 512, 2048)
    assert after.static_image_emb.shape == (1, 768, 2048)
    assert after.static_pos_emb.shape == (1, 768, 768)
    assert after.static_state_emb.shape == (1, 768, 8)
    assert after.static_mask.shape == (1, 768)
    assert before.state.shape == after.state.shape
    assert before.tokenized_prompt.shape == after.tokenized_prompt.shape
    assert {key: value.shape for key, value in before.images.items()} == {
        key: value.shape for key, value in after.images.items()
    }
    assert actions_before.shape == actions_after.shape == (1, 20, 32)
    assert RoboMMEOutputs()({"actions": np.zeros((20, 32))})["actions"].shape == (20, 8)
