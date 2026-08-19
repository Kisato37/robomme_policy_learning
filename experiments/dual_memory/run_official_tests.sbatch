#!/usr/bin/env bash
#SBATCH --job-name=dual-official-tests
#SBATCH --partition=athena-mini
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_root="$repo/outputs/dual_memory/20260818-214308_ecf086c"
report_name="${OFFICIAL_TEST_REPORT:-official_test_report.txt}"
cd "$repo"
.venv/bin/python experiments/dual_memory/run_official_tests.py \
  --repo "$repo" --output "$run_root/tests/$report_name"
