#!/usr/bin/env bash
#SBATCH --job-name=dual-data-audit
#SBATCH --partition=athena
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_id=20260818-214308_ecf086c
cd "$repo"
.venv/bin/python experiments/dual_memory/data_audit.py \
  --dataset data/robomme_preprocessed_data \
  --output-dir "outputs/dual_memory/$run_id/audits" \
  --sample-count 100 \
  --visual-count 20 \
  --seed 20260818
