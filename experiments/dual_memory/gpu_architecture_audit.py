#!/usr/bin/env python3
"""Eight-GPU architecture gate for the SP dual-memory policy."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as openpi_model
import openpi.training.sharding as sharding
from mme_vla_suite.models.integration.history_observation import preprocess_observation
import mme_vla_suite.training.config as training_config
import mme_vla_suite.training.dataloader as data_loader
from scripts import train


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_audit(state, base_params_path: Path) -> dict:
    expected = traverse_util.flatten_dict(state.params.to_pure_dict(), sep="/")
    restored = openpi_model.restore_params(base_params_path, restore_type=np.ndarray)
    loaded = traverse_util.flatten_dict(restored, sep="/")
    expected_keys = set(expected)
    loaded_keys = set(loaded)
    missing = sorted(expected_keys - loaded_keys)
    unexpected = sorted(loaded_keys - expected_keys)
    shape_mismatches = sorted(
        {
            key: {"expected": list(expected[key].shape), "loaded": list(loaded[key].shape)}
            for key in expected_keys & loaded_keys
            if tuple(expected[key].shape) != tuple(loaded[key].shape)
        }.items()
    )
    disallowed_missing = [key for key in missing if "mem" not in key.lower()]
    return {
        "base_params_path": str(base_params_path.resolve()),
        "expected_key_count": len(expected_keys),
        "loaded_key_count": len(loaded_keys),
        "matched_key_count": len(expected_keys & loaded_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": dict(shape_mismatches),
        "allowed_missing_rule": "new SP keys must contain 'mem'",
        "disallowed_missing_keys": disallowed_missing,
        "status": "pass" if not unexpected and not shape_mismatches and not disallowed_missing else "fail",
    }


def scalar_dict(tree) -> dict[str, float | int]:
    result = {}
    for key, value in jax.device_get(tree).items():
        scalar = np.asarray(value).item()
        result[key] = int(scalar) if np.issubdtype(type(scalar), np.integer) else float(scalar)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, default=Path("runs/assets"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite architecture audit: {args.output}")

    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    base = training_config.get_config("mme_vla_suite")
    model = dataclasses.replace(
        base.model,
        use_history=True,
        history_config="dual-grounded-framesamp-modul.yaml",
    )
    config = dataclasses.replace(
        base,
        exp_name="sp-architecture-audit",
        model=model,
        dataset_path=str(args.dataset),
        assets_base_dir=str(args.assets_base_dir),
        checkpoint_base_dir=str(args.output.parent / "scratch-checkpoints"),
        weight_loader=training_config.weight_loaders.CheckpointWeightLoader(str(args.base_params)),
        seed=args.seed,
        batch_size=8,
        num_workers=4,
        fsdp_devices=8,
        wandb_enabled=False,
        num_train_steps=1,
    )
    if jax.device_count() != 8:
        raise RuntimeError(f"Architecture audit requires exactly 8 visible GPUs, got {jax.device_count()}")

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    history_config = training_config.get_history_config(config.model.history_config)
    data_config = config.data.create(config.assets_dirs, config.model)
    loader = data_loader.create_data_loader(
        str(args.dataset),
        data_config,
        history_config=config.model.history_config,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=data_sharding,
        shuffle=False,
        num_batches=1,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    observation, actions = next(iter(loader))
    input_shapes = jax.tree.map(
        lambda value: list(value.shape) if value is not None else None,
        observation.to_dict(),
    )
    input_shapes["actions"] = list(actions.shape)

    rng = jax.random.key(config.seed)
    train_rng, init_rng, causal_rng = jax.random.split(rng, 3)
    state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    checkpoint = key_audit(state, args.base_params)

    step = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated, replicated),
        donate_argnums=(1,),
    )
    state, info, _ = step(train_rng, state, (observation, actions))
    jax.block_until_ready(state)
    gradient = scalar_dict(info)
    gradient_status = (
        all(np.isfinite(gradient[name]) and gradient[name] > 0 for name in ["memory_grad_norm", "action_grad_norm", "vlm_grad_norm"])
        and gradient["siglip_grad_leaf_count"] == 0
    )

    policy = nnx.merge(state.model_def, state.params)
    policy.eval()
    processed = preprocess_observation(None, observation, train=False)
    noise = jax.random.normal(causal_rng, actions.shape)
    time = jnp.full(actions.shape[0], 0.5, dtype=jnp.float32)
    x_t = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions

    @nnx.jit
    def velocity(model, obs, noisy_actions, timestep):
        return model.predict_velocity_from_preprocessed(obs, noisy_actions, timestep)[0]

    baseline_velocity = velocity(policy, processed, x_t, time)
    symbolic_tokens = processed.symbolic_tokenized_prompt
    valid_mask_host = np.asarray(jax.device_get(processed.symbolic_tokenized_prompt_mask[0]))
    valid_indices = np.flatnonzero(valid_mask_host)
    if len(valid_indices) < 2:
        raise RuntimeError("Cannot choose a non-BOS valid symbolic token for intervention")
    symbolic_index = int(valid_indices[1])
    symbolic_intervention = processed.replace(
        symbolic_tokenized_prompt=symbolic_tokens.at[:, symbolic_index].set(
            (symbolic_tokens[:, symbolic_index] + 1) % 257152
        )
    )
    history_delta = processed.static_mask[..., None].astype(processed.static_image_emb.dtype) * 0.125
    perceptual_intervention = processed.replace(
        static_image_emb=processed.static_image_emb + history_delta
    )
    symbolic_velocity = velocity(policy, symbolic_intervention, x_t, time)
    perceptual_velocity = velocity(policy, perceptual_intervention, x_t, time)
    symbolic_abs = np.abs(np.asarray(jax.device_get(symbolic_velocity - baseline_velocity), dtype=np.float32))
    perceptual_abs = np.abs(np.asarray(jax.device_get(perceptual_velocity - baseline_velocity), dtype=np.float32))
    causal = {
        "fixed_time": 0.5,
        "fixed_noise_seed": args.seed,
        "symbolic_token_index": symbolic_index,
        "symbolic_velocity_max_abs_delta": float(symbolic_abs.max()),
        "symbolic_velocity_mean_abs_delta": float(symbolic_abs.mean()),
        "perceptual_valid_feature_offset": 0.125,
        "perceptual_velocity_max_abs_delta": float(perceptual_abs.max()),
        "perceptual_velocity_mean_abs_delta": float(perceptual_abs.mean()),
    }
    causal["status"] = "pass" if causal["symbolic_velocity_max_abs_delta"] > 0 and causal["perceptual_velocity_max_abs_delta"] > 0 else "fail"

    report = {
        "status": "pass" if checkpoint["status"] == "pass" and gradient_status and causal["status"] == "pass" else "fail",
        "device_count": jax.device_count(),
        "devices": [str(device) for device in jax.devices()],
        "model": "SP",
        "history_config": "dual-grounded-framesamp-modul.yaml",
        "dataset": str(args.dataset.resolve()),
        "dataset_stats_sha256": sha256_file(args.dataset / "meta" / "stats.json"),
        "input_shapes": input_shapes,
        "checkpoint_compatibility": checkpoint,
        "gradient_flow": {**gradient, "status": "pass" if gradient_status else "fail"},
        "causal_interventions": causal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if report["status"] != "pass":
        raise RuntimeError(f"SP architecture audit failed; see {args.output}")


if __name__ == "__main__":
    main()
