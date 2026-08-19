#!/usr/bin/env bash
#SBATCH --job-name=dual-unzip-data
#SBATCH --partition=athena
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
cd "$repo"
.venv/bin/python scripts/unzip_data.py data/robomme_preprocessed_data --processes 16
