import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import dataclasses

import openpi.models.model as _model
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms

import mme_vla_suite.policies.policy as _policy
import mme_vla_suite.training.config as _config



def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    seed: int = 42,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    experimental_memory_expansion: str | None = None,
    strict_weight_tree_load: bool = False,
) -> _policy.MME_VLA_Policy:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    repack_transforms = repack_transforms or transforms.Group()
    
    logging.info("Checking history config")
    history_config = None
    history_config_path = checkpoint_dir.parent / "history_config.txt"
    if history_config_path.exists():
        with open(history_config_path, "r") as f:
            history_config = f.read()
    
    if train_config.model.history_config != history_config:
        print(f" == You are using {train_config.model.history_config}, changing to {history_config} ==")
        train_config = dataclasses.replace(
            train_config, 
            model=dataclasses.replace(train_config.model, history_config=history_config, use_history=history_config is not None)
        )
    

    policy_class = _policy.MME_VLA_Policy
    policy_metadata = dict(train_config.policy_metadata or {})
    if experimental_memory_expansion is not None:
        # Deliberately opt-in and family-specific. Never edit checkpoint metadata
        # or widen the completed experiments' 32/512 validation rules.
        from omegaconf import OmegaConf
        from mme_vla_suite.models.config.utils import get_history_config
        from mme_vla_suite.policies.uniform_keyframe_expansion_policy import UniformKeyframeExpansionPolicy
        from mme_vla_suite.shared.uniform_keyframe_config import EXPANSION_FAMILY, expanded_history_mapping

        if experimental_memory_expansion != EXPANSION_FAMILY:
            raise ValueError("Unknown experimental memory expansion family")
        if seed != 7 or isinstance(seed, bool):
            raise ValueError("Uniform/keyframe expansion requires evaluation policy seed 7")
        if history_config is None or not history_config_path.is_file():
            raise ValueError("Expansion requires the original checkpoint history_config.txt")
        if (not train_config.model.pi05 or train_config.model.action_horizon != 20
                or train_config.model.use_symbolic_prompt
                or train_config.model.symbolic_source != "none"):
            raise ValueError("Expansion requires unchanged pi0.5 action horizon and no symbolic prompting")
        source = OmegaConf.to_container(get_history_config(history_config.strip()), resolve=True)
        effective, evidence = expanded_history_mapping(source)
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, history_config=OmegaConf.create(effective), use_history=True),
        )
        policy_class = UniformKeyframeExpansionPolicy
        policy_metadata["uniform_keyframe_expansion"] = evidence
        logging.info("Explicit test-time expansion: %s", evidence)

    logging.info("Loading model...")
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    if type(strict_weight_tree_load) is not bool:
        raise ValueError("strict_weight_tree_load must be a boolean")
    if experimental_memory_expansion is None and not strict_weight_tree_load:
        model = train_config.model.load(params)
    else:
        # Missing, extra or shape-mismatched parameters must fail, not initialize
        # a new component or silently intersect the checkpoint tree.
        model = train_config.model.load(params, remove_extra_params=False)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    print("Training config: ", train_config)
    print("Data config: ", data_config)

    return policy_class(
        model,
        seed=seed,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=policy_metadata,
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm
    )
