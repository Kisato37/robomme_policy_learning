#!/usr/bin/env bash
#SBATCH --job-name=dual-unzip-models
#SBATCH --partition=athena
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
cd "$repo"
.venv/bin/python scripts/unzip_ckpt.py runs/ckpts --processes 8
