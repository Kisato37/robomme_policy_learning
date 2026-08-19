#!/usr/bin/env bash
#SBATCH --job-name=dual-models
#SBATCH --partition=athena
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_id=20260818-214308_ecf086c
run_root="$repo/outputs/dual_memory/$run_id"
export OPENPI_DATA_HOME=/home/jp673/robomme_dual_memory/openpi_data
export HF_HOME=/home/jp673/robomme_dual_memory/.hf-cache
export HF_XET_HIGH_PERFORMANCE=1
cd "$repo"
.venv/bin/python experiments/dual_memory/download_assets.py models \
  --repo "$repo" \
  --output "$run_root/environment/model_assets.json"
