#!/usr/bin/env bash
#SBATCH --job-name=keyframe-smoke
#SBATCH --partition=athena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --no-requeue

set -euo pipefail

REPO_CANDIDATE="${KEYFRAME_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if ! REPO_ROOT="$(cd "${REPO_CANDIDATE}" 2>/dev/null && pwd -P)"; then
  echo "Repository root does not exist: ${REPO_CANDIDATE}" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/experiments/keyframe_oracle_sampling/EXPERIMENT_PROTOCOL.md" ]]; then
  echo "Invalid repository root from Slurm submission context: ${REPO_ROOT}" >&2
  exit 2
fi
CHECKPOINT="${REPO_ROOT}/runs/test_time_scaling/checkpoints/perceptual-framesamp-modul/79999"
POLICY_PYTHON="${REPO_ROOT}/.venv/bin/python"
BENCHMARK_PYTHON="${REPO_ROOT}/third_party/robomme_benchmark/.venv/bin/python"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src:${REPO_ROOT}"
if [[ ! -x "${POLICY_PYTHON}" ]]; then
  echo "Missing pinned policy Python environment: ${POLICY_PYTHON}" >&2
  exit 2
fi

resolve_allocated_gpus() {
  local visible_devices="${1:-}"
  local policy_device=""
  local simulator_device=""
  if [[ ! "${visible_devices}" =~ ^[^,[:space:]]+,[^,[:space:]]+$ ]]; then
    echo "Expected exactly two comma-separated Slurm CUDA devices, got: ${visible_devices:-<unset>}" >&2
    return 1
  fi
  IFS=',' read -r policy_device simulator_device <<<"${visible_devices}"
  if [[ "${policy_device}" == "${simulator_device}" ]]; then
    echo "Policy and simulator CUDA devices must be distinct" >&2
    return 1
  fi
  printf '%s\t%s\n' "${policy_device}" "${simulator_device}"
}

if [[ "${1:-}" == "--gpu-binding-dry-run" ]]; then
  resolve_allocated_gpus "${CUDA_VISIBLE_DEVICES:-}"
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  "${POLICY_PYTHON}" -m experiments.keyframe_oracle_sampling.prepare_smoke --dry-run
  "${POLICY_PYTHON}" -m experiments.keyframe_oracle_sampling.smoke_matrix --row 0 --row-tsv
  echo "Dry run only: no server, simulator, GPU inference, or Slurm submission started."
  exit 0
fi

RUN_ROOT="${1:?usage: run_smoke.sbatch RUN_ROOT SEED_TABLE [ATTEMPT_ID]}"
SEED_TABLE="${2:?usage: run_smoke.sbatch RUN_ROOT SEED_TABLE [ATTEMPT_ID]}"
ATTEMPT_ID="${3:-0}"
if [[ ! "${ATTEMPT_ID}" =~ ^[012]$ ]]; then
  echo "ATTEMPT_ID must be 0, 1, or 2" >&2
  exit 1
fi
ROW_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
ROW_TSV="$("${POLICY_PYTHON}" -m experiments.keyframe_oracle_sampling.smoke_matrix \
  --row "${ROW_ID}" --row-tsv)"
IFS=$'\t' read -r TASK ARM TRAJECTORY_KIND MAX_STEPS DATASET <<<"${ROW_TSV}"
GPU_BINDING_TSV="$(resolve_allocated_gpus "${CUDA_VISIBLE_DEVICES:-}")"
IFS=$'\t' read -r POLICY_CUDA_DEVICE SIMULATOR_CUDA_DEVICE <<<"${GPU_BINDING_TSV}"
echo "GPU binding: allocated=${CUDA_VISIBLE_DEVICES}; policy=${POLICY_CUDA_DEVICE}; simulator=${SIMULATOR_CUDA_DEVICE}"

ATTEMPT_ROOT="${RUN_ROOT}/trajectories/${TASK}/episode_00/${ARM}/${TRAJECTORY_KIND}/attempt_$(printf '%02d' "${ATTEMPT_ID}")"
if [[ -e "${ATTEMPT_ROOT}" ]]; then
  echo "Refusing to reuse existing attempt directory: ${ATTEMPT_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "Missing frozen checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -x "${BENCHMARK_PYTHON}" ]]; then
  echo "Missing pinned benchmark Python environment: ${BENCHMARK_PYTHON}" >&2
  exit 1
fi
"${POLICY_PYTHON}" - "${CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_identity

checkpoint_identity(Path(sys.argv[1]))
PY
if [[ "${ATTEMPT_ID}" == 0 ]]; then
  SUBMISSION_RECORD="${RUN_ROOT}/protocol/submission_record.json"
else
  SUBMISSION_RECORD="${RUN_ROOT}/protocol/submission_record_attempt_$(printf '%02d' "${ATTEMPT_ID}").json"
fi
for _ in $(seq 1 30); do
  [[ -f "${SUBMISSION_RECORD}" ]] && break
  sleep 1
done
if [[ ! -f "${SUBMISSION_RECORD}" ]]; then
  echo "Missing write-once smoke submission record" >&2
  exit 1
fi
"${POLICY_PYTHON}" -m experiments.keyframe_oracle_sampling.preflight_smoke_row \
  --run-root "${RUN_ROOT}" \
  --seed-table "${SEED_TABLE}" \
  --repo-root "${REPO_ROOT}" \
  --attempt-id "${ATTEMPT_ID}" \
  --row-id "${ROW_ID}" \
  --task "${TASK}" \
  --episode-id 0 \
  --arm "${ARM}" \
  --trajectory-kind "${TRAJECTORY_KIND}" \
  --max-steps "${MAX_STEPS}" \
  --dataset "${DATASET}"

PORT="$((20000 + SLURM_JOB_ID % 20000))"
SERVER_PID=""
EVALUATOR_PID=""
TERMINATION_STATUS=""
cleanup() {
  if [[ -n "${EVALUATOR_PID}" ]]; then
    kill "${EVALUATOR_PID}" 2>/dev/null || true
    wait "${EVALUATOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
handle_signal() {
  local signal_name="$1"
  local signal_status="$2"
  TERMINATION_STATUS="${signal_status}"
  if [[ -n "${EVALUATOR_PID}" ]]; then
    kill -s "${signal_name}" "${EVALUATOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID}" ]]; then
    kill -s "${signal_name}" "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${POLICY_CUDA_DEVICE}" "${POLICY_PYTHON}" scripts/serve_policy.py \
  --seed=7 --port="${PORT}" policy:checkpoint \
  --policy.dir="${CHECKPOINT}" --policy.config=mme_vla_suite \
  >"${RUN_ROOT}/server-${SLURM_ARRAY_JOB_ID}-${ROW_ID}.out" \
  2>"${RUN_ROOT}/server-${SLURM_ARRAY_JOB_ID}-${ROW_ID}.err" &
SERVER_PID="$!"

set +e
"${POLICY_PYTHON}" - "${PORT}" "${SERVER_PID}" <<'PY'
import os
import socket
import sys
import time

port = int(sys.argv[1])
server_pid = int(sys.argv[2])
deadline = time.monotonic() + 240
while time.monotonic() < deadline:
    try:
        os.kill(server_pid, 0)
    except ProcessLookupError:
        print("Policy server exited before readiness", file=sys.stderr)
        raise SystemExit(70)
    try:
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
        break
    except OSError:
        time.sleep(2)
else:
    print("Policy server readiness timeout", file=sys.stderr)
    raise SystemExit(71)
PY
READINESS_STATUS="$?"
set -e
if [[ "${READINESS_STATUS}" -ne 0 ]]; then
  if [[ "${READINESS_STATUS}" -eq 70 ]]; then
    READINESS_ERROR_TYPE="PolicyServerExitedBeforeReadiness"
  elif [[ "${READINESS_STATUS}" -eq 71 ]]; then
    READINESS_ERROR_TYPE="PolicyServerReadinessTimeout"
  else
    READINESS_ERROR_TYPE="PolicyServerReadinessProbeError"
  fi
  "${POLICY_PYTHON}" -m experiments.keyframe_oracle_sampling.record_launcher_failure \
    --run-root "${RUN_ROOT}" \
    --seed-table "${SEED_TABLE}" \
    --repo-root "${REPO_ROOT}" \
    --attempt-id "${ATTEMPT_ID}" \
    --row-id "${ROW_ID}" \
    --task "${TASK}" \
    --episode-id 0 \
    --arm "${ARM}" \
    --trajectory-kind "${TRAJECTORY_KIND}" \
    --max-steps "${MAX_STEPS}" \
    --dataset "${DATASET}" \
    --error-type "${READINESS_ERROR_TYPE}" \
    --error "Policy server readiness failed with exit status ${READINESS_STATUS}; inspect the immutable Slurm/server logs" \
    --policy-port "${PORT}"
  exit "${READINESS_STATUS}"
fi

if [[ -n "${TERMINATION_STATUS}" ]]; then
  EVALUATOR_STATUS="${TERMINATION_STATUS}"
else
  CUDA_VISIBLE_DEVICES="${SIMULATOR_CUDA_DEVICE}" \
    PYTHONPATH="${REPO_ROOT}/examples/robomme:${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src:${REPO_ROOT}" \
    "${BENCHMARK_PYTHON}" examples/robomme/eval.py \
    --args.host=127.0.0.1 \
    --args.port="${PORT}" \
    --args.obs-horizon=16 \
    --args.max-steps="${MAX_STEPS}" \
    --args.dataset="${DATASET}" \
    --args.only-tasks="${TASK}" \
    --args.episode-ids=0 \
    --args.model-seed=7 \
    --args.model-ckpt-id=79999 \
    --args.policy-name=perceptual-framesamp-modul-keyframe-smoke-${ROW_ID} \
    --args.save-dir="${RUN_ROOT}/legacy_eval/${ROW_ID}/attempt_${ATTEMPT_ID}" \
    --args.keyframe-selector-arm="${ARM}" \
    --args.keyframe-seed-table="${SEED_TABLE}" \
    --args.keyframe-run-root="${RUN_ROOT}" \
    --args.keyframe-attempt-id="${ATTEMPT_ID}" \
    --args.keyframe-trajectory-kind="${TRAJECTORY_KIND}" &
  EVALUATOR_PID="$!"
  set +e
  wait "${EVALUATOR_PID}"
  EVALUATOR_STATUS="$?"
  set -e
  EVALUATOR_PID=""
  if [[ -n "${TERMINATION_STATUS}" ]]; then
    EVALUATOR_STATUS="${TERMINATION_STATUS}"
  fi
fi

set +e
"${POLICY_PYTHON}" -m experiments.keyframe_oracle_sampling.record_launcher_failure \
  --run-root "${RUN_ROOT}" \
  --seed-table "${SEED_TABLE}" \
  --repo-root "${REPO_ROOT}" \
  --attempt-id "${ATTEMPT_ID}" \
  --row-id "${ROW_ID}" \
  --task "${TASK}" \
  --episode-id 0 \
  --arm "${ARM}" \
  --trajectory-kind "${TRAJECTORY_KIND}" \
  --max-steps "${MAX_STEPS}" \
  --dataset "${DATASET}" \
  --policy-port "${PORT}" \
  --evaluator-exit-status "${EVALUATOR_STATUS}"
RECONCILE_STATUS="$?"
set -e
if [[ "${RECONCILE_STATUS}" -ne 0 ]]; then
  exit "${RECONCILE_STATUS}"
fi
if [[ "${EVALUATOR_STATUS}" -ne 0 ]]; then
  exit "${EVALUATOR_STATUS}"
fi
