#!/usr/bin/env bash
#SBATCH --job-name=dual-qwen-setup
#SBATCH --partition=athena-small
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_id=20260818-214308_ecf086c
run_root="$repo/outputs/dual_memory/$run_id"
source_venv="$repo/third_party/robomme_benchmark/.venv"
qwen_venv="$repo/third_party/robomme_benchmark/.venv_qwen"
export HF_HOME=/home/jp673/robomme_dual_memory/.hf-cache
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS=4
ulimit -n 65536
cd "$repo"

test ! -e "$qwen_venv"
cp -a --reflink=auto "$source_venv" "$qwen_venv"
"$qwen_venv/bin/python" -m pip install \
  ninja==1.13.0 \
  ms_swift==3.11.1 \
  qwen-vl-utils==0.0.14 \
  transformers==4.57.3 \
  moviepy==2.2.1 \
  setuptools==80.9.0

.venv/bin/python experiments/dual_memory/setup_qwen.py \
  --repo "$repo" \
  --qwen-python "$qwen_venv/bin/python" \
  --output "$run_root/environment/qwen_setup.json"
