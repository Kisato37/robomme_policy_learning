#!/usr/bin/env bash
#SBATCH --job-name=dual-formal-launch
#SBATCH --partition=athena
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_root="$repo/outputs/dual_memory/20260818-214308_ecf086c"
cd "$repo"
.venv/bin/python experiments/dual_memory/collect_environment.py --repo "$repo" --run-root "$run_root"
.venv/bin/python experiments/dual_memory/launch_formal_training.py --repo "$repo" --run-root "$run_root"
