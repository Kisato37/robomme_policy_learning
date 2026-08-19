#!/usr/bin/env python3
"""Construct only preregistered N/S/P/SP training configurations and run them."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding
import openpi.training.weight_loaders as weight_loaders
from mme_vla_suite.models.integration.history_observation import preprocess_observation
import mme_vla_suite.training.config as training_config
from scripts import train


MODEL_CONFIGS = {
    "N": None,
    "S": "symbolic-grounded-subgoal.yaml",
    "P": "perceptual-framesamp-modul.yaml",
    "SP": "dual-grounded-framesamp-modul.yaml",
}


def build_config(args):
    base = training_config.get_config("mme_vla_suite")
    history_config = MODEL_CONFIGS[args.model]
    model = dataclasses.replace(
        base.model,
        use_history=history_config is not None,
        history_config=history_config,
    )
    phase = "formal" if args.mode == "formal" else f"smoke{args.num_steps}"
    return dataclasses.replace(
        base,
        exp_name=f"{phase}_{args.model}_seed{args.seed}",
        model=model,
        dataset_path=str(args.dataset),
        assets_base_dir=str(args.assets_base_dir),
        checkpoint_base_dir=str(args.run_root / "training/checkpoints"),
        seed=args.seed,
        batch_size=64,
        num_workers=4,
        fsdp_devices=8,
        num_train_steps=args.num_steps,
        log_interval=10 if args.mode == "smoke" else base.log_interval,
        wandb_enabled=False,
        overwrite=False,
        resume=False,
    )


def config_record(config, args) -> dict:
    return {
        "model_id": args.model,
        "training_seed": args.seed,
        "mode": args.mode,
        "history_config": MODEL_CONFIGS[args.model],
        "base_checkpoint": str(config.weight_loader.params_path),
        "dataset_path": str(Path(config.dataset_path).resolve()),
        "checkpoint_dir": str(config.checkpoint_dir),
        "num_train_steps": config.num_train_steps,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "fsdp_devices": config.fsdp_devices,
        "optimizer": "AdamW(clip_gradient_norm=1.0)",
        "learning_rate": {
            "type": "cosine_decay",
            "warmup_steps": 10000,
            "peak_lr": 5e-5,
            "decay_steps": 100000,
            "decay_lr": 5e-5,
        },
        "ema_decay": config.ema_decay,
        "action_horizon": config.model.action_horizon,
        "checkpoint_selection": f"final_step_{config.num_train_steps - 1}",
        "wandb_enabled": False,
        "local_metrics": str(config.checkpoint_dir / "training_metrics.jsonl"),
    }


def write_config_snapshot(config, args) -> Path:
    destination = args.run_root / "configs" / f"{args.mode}_{args.model}_{args.seed}.yaml"
    if destination.exists():
        if json.loads(destination.read_text()) != config_record(config, args):
            raise RuntimeError(f"Refusing to change preregistered config snapshot: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config_record(config, args), indent=2, sort_keys=True) + "\n")
    return destination


def fixed_velocity(state, config) -> np.ndarray:
    policy = nnx.merge(state.model_def, state.params)
    policy.eval()
    observation = preprocess_observation(None, config.model.fake_obs(batch_size=1), train=False)
    noisy_actions = jnp.ones((1, config.model.action_horizon, config.model.action_dim), dtype=jnp.float32)
    timestep = jnp.full((1,), 0.5, dtype=jnp.float32)

    @nnx.jit
    def infer(model, obs, x_t, time):
        return model.predict_velocity_from_preprocessed(obs, x_t, time)[0]

    return np.asarray(jax.device_get(infer(policy, observation, noisy_actions, timestep)), dtype=np.float32)


def reload_audit(state, config, args) -> None:
    final_step = args.num_steps - 1
    checkpoint = config.checkpoint_dir / str(final_step) / "params"
    reference = fixed_velocity(state, config)
    del state
    gc.collect()
    jax.clear_caches()

    reload_config = dataclasses.replace(
        config,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(checkpoint)),
    )
    mesh = sharding.make_mesh(config.fsdp_devices)
    reloaded, _ = train.init_train_state(
        reload_config,
        jax.random.key(args.seed + 100000),
        mesh,
        resume=False,
    )
    jax.block_until_ready(reloaded)
    restored = fixed_velocity(reloaded, reload_config)
    delta = np.abs(restored - reference)
    report = {
        "status": "pass" if np.allclose(restored, reference, rtol=1e-5, atol=1e-5) else "fail",
        "model_id": args.model,
        "training_seed": args.seed,
        "checkpoint": str(checkpoint),
        "fixed_input": "HistoryPi0Config.fake_obs(batch=1), x_t=ones, time=0.5",
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
        "rtol": 1e-5,
        "atol": 1e-5,
    }
    output = args.run_root / "audits" / f"checkpoint_reload_{args.mode}_{args.model}_seed{args.seed}.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if report["status"] != "pass":
        raise RuntimeError(f"Checkpoint reload audit failed: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, default=Path("runs/assets"))
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "formal" and args.num_steps != 80000:
        raise ValueError("Formal runs are locked to 80,000 steps")
    if args.mode == "formal" and args.seed not in {42, 43, 44}:
        raise ValueError("Formal runs are locked to seeds 42, 43, and 44")
    config = build_config(args)
    write_config_snapshot(config, args)
    state = train.main(config, tentative_run=False)
    if args.mode == "smoke":
        reload_audit(state, config, args)


if __name__ == "__main__":
    main()
