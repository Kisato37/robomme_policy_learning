#!/usr/bin/env bash
#SBATCH --job-name=dual-pilot
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --no-requeue

set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 N|S|P|SP" >&2
  exit 2
fi
model="$1"
case "$model" in
  N|S|P|SP) ;;
  *) echo "unsupported pilot model: $model" >&2; exit 2 ;;
esac

repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_id=20260818-214308_ecf086c
run_root="$repo/outputs/dual_memory/$run_id"
checkpoint="$run_root/training/checkpoints/mme_vla_suite/smoke100_${model}_seed42/99"
port=$((18200 + SLURM_JOB_ID % 500))
export OPENPI_DATA_HOME=/home/jp673/robomme_dual_memory/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
cd "$repo"
test -d "$checkpoint"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py \
  --seed=7 --port="$port" policy:checkpoint \
  --policy.dir="$checkpoint" --policy.config=mme_vla_suite \
  >"$run_root/training/logs/pilot-${model}-server-${SLURM_JOB_ID}.log" 2>&1 &
server_pid=$!
trap 'kill '"$server_pid"' 2>/dev/null || true' EXIT

.venv/bin/python -c "import socket,time; deadline=time.time()+900
while time.time()<deadline:
    try:
        socket.create_connection(('127.0.0.1', $port), 2).close(); break
    except OSError: time.sleep(2)
else: raise SystemExit('policy server timeout')"

extra_args=()
source=none
if [[ "$model" == N ]]; then
  extra_args=(--args.no-use-history)
elif [[ "$model" == S || "$model" == SP ]]; then
  source=oracle
  extra_args=(--args.use-oracle --args.subgoal-type=grounded_subgoal)
fi

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=examples/robomme \
  third_party/robomme_benchmark/.venv/bin/python examples/robomme/eval.py \
  --args.host=127.0.0.1 --args.port="$port" \
  --args.save-dir="$run_root/pilot/evaluation" \
  --args.policy-name="pilot-${model}" --args.model-seed=7 --args.model-ckpt-id=99 \
  --args.only-tasks=VideoUnmask,VideoRepick,InsertPeg,VideoPlaceOrder,RouteStick \
  --args.episode-ids=0,1,2,3,4 \
  --args.dual-memory-run-root="$run_root" --args.model-id="pilot-${model}" \
  --args.training-seed=42 --args.symbolic-source="$source" "${extra_args[@]}"

kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
trap - EXIT
