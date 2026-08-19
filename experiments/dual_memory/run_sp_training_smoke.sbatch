#!/usr/bin/env bash
#SBATCH --job-name=dual-sp-smoke
#SBATCH --partition=athena-small
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=384G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_id=20260818-214308_ecf086c
run_root="$repo/outputs/dual_memory/$run_id"
export OPENPI_DATA_HOME=/home/jp673/robomme_dual_memory/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=/home/jp673/robomme_dual_memory/jax_cache
ulimit -n 65536
cd "$repo"

.venv/bin/python experiments/dual_memory/run_training.py \
  --model SP --seed 42 --mode smoke --num-steps 100 \
  --dataset data/robomme_preprocessed_data \
  --assets-base-dir runs/assets \
  --run-root "$run_root"
