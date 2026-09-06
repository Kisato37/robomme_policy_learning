#!/usr/bin/env bash
#SBATCH --job-name=keyframe-neighborhood-arch
#SBATCH --partition=athena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --no-requeue

set -euo pipefail

REPO_CANDIDATE="${KEYFRAME_NEIGHBORHOOD_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if ! REPO_ROOT="$(cd "${REPO_CANDIDATE}" 2>/dev/null && pwd -P)"; then
  echo "Repository root does not exist: ${REPO_CANDIDATE}" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/experiments/keyframe_neighborhood_sampling/EXPERIMENT_EXTENSION_PROTOCOL.md" ]]; then
  echo "Invalid extension repository root: ${REPO_ROOT}" >&2
  exit 2
fi
POLICY_PYTHON="${REPO_ROOT}/.venv/bin/python"
BENCHMARK_PYTHON="${REPO_ROOT}/third_party/robomme_benchmark/.venv/bin/python"
CHECKPOINT="${REPO_ROOT}/runs/test_time_scaling/checkpoints/perceptual-framesamp-modul/79999"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src:${REPO_ROOT}"

if [[ ! -x "${POLICY_PYTHON}" ]]; then
  echo "Missing pinned policy environment: ${POLICY_PYTHON}" >&2
  exit 2
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  "${POLICY_PYTHON}" -m experiments.keyframe_neighborhood_sampling.architecture_smoke --dry-run
  echo "Dry run only: no checkpoint load, GPU inference, simulator, or Slurm submission started."
  exit 0
fi

RUN_ROOT="${1:?usage: run_architecture_smoke.sbatch RUN_ROOT}"
if [[ ! -d "${RUN_ROOT}/protocol" ]]; then
  echo "Prepared extension smoke protocol directory is missing: ${RUN_ROOT}/protocol" >&2
  exit 2
fi
for _ in $(seq 1 30); do
  [[ -f "${RUN_ROOT}/protocol/architecture_submission_record.json" ]] && break
  sleep 1
done
if [[ ! -f "${RUN_ROOT}/protocol/architecture_submission_record.json" ]]; then
  echo "Missing write-once extension architecture submission record" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "Missing frozen protocol checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -x "${BENCHMARK_PYTHON}" ]]; then
  echo "Missing pinned benchmark environment: ${BENCHMARK_PYTHON}" >&2
  exit 2
fi
case "${SLURM_JOB_PARTITION:-}" in
  athena|athena-mini) ;;
  *)
    echo "Hard stop: expected athena or athena-mini, got ${SLURM_JOB_PARTITION:-unset}" >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_EXPLAIN_CACHE_MISSES=true
"${BENCHMARK_PYTHON}" -c 'import sys; print("benchmark_python", sys.executable)'
"${POLICY_PYTHON}" -c 'import jax,sys; print("policy_python",sys.executable); print("devices",jax.devices())'
"${POLICY_PYTHON}" -m experiments.keyframe_neighborhood_sampling.architecture_smoke \
  --run-root "${RUN_ROOT}"
