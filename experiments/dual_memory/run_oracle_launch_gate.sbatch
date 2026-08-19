#!/usr/bin/env bash
#SBATCH --job-name=dual-oracle-launch
#SBATCH --partition=athena
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_root="$repo/outputs/dual_memory/20260818-214308_ecf086c"
cd "$repo"
.venv/bin/python experiments/dual_memory/record_phase_status.py \
  --run-root "$run_root" --phase Phase-5 --status completed \
  --evidence environment/formal_jobs.json \
  --next "Run the complete matched Oracle evaluation."
.venv/bin/python experiments/dual_memory/launch_oracle_evaluation.py --repo "$repo" --run-root "$run_root"
