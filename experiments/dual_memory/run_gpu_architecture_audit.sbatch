#!/usr/bin/env bash
#SBATCH --job-name=dual-sp-audit
#SBATCH --partition=athena
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_id=20260818-214308_ecf086c
run_root="$repo/outputs/dual_memory/$run_id"
export OPENPI_DATA_HOME=/home/jp673/robomme_dual_memory/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=/home/jp673/robomme_dual_memory/jax_cache
cd "$repo"

mkdir -p runs/assets/mme_vla_suite
if [[ -e runs/assets/mme_vla_suite/norm_stats.json ]]; then
  cmp assets/norm_stats.json runs/assets/mme_vla_suite/norm_stats.json
else
  cp assets/norm_stats.json runs/assets/mme_vla_suite/norm_stats.json
fi

.venv/bin/python experiments/dual_memory/gpu_architecture_audit.py \
  --dataset data/robomme_preprocessed_data \
  --base-params /home/jp673/robomme_dual_memory/openpi_data/openpi-assets/checkpoints/pi05_base/params \
  --assets-base-dir runs/assets \
  --output "$run_root/audits/sp_gpu_architecture_audit.json"
