#!/usr/bin/env bash
#SBATCH --job-name=dual-eval
#SBATCH --gres=gpu:2
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --no-requeue

set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "usage: $0 N|S|P|SP seed none|oracle|qwenvl full|diagnostic" >&2
  exit 2
fi
model="$1"
seed="$2"
source="$3"
scope="$4"
case "$model" in N|S|P|SP) ;; *) exit 2 ;; esac
case "$seed" in 42|43|44) ;; *) exit 2 ;; esac
case "$source" in none|oracle|qwenvl) ;; *) exit 2 ;; esac
case "$scope" in full|diagnostic) ;; *) exit 2 ;; esac

repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_id=20260818-214308_ecf086c
run_root="$repo/outputs/dual_memory/$run_id"
checkpoint="$run_root/training/checkpoints/mme_vla_suite/formal_${model}_seed${seed}/79999"
port=$((19000 + SLURM_JOB_ID % 1000))
export OPENPI_DATA_HOME=/home/jp673/robomme_dual_memory/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
cd "$repo"
test -d "$checkpoint"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py \
  --seed=7 --port="$port" policy:checkpoint \
  --policy.dir="$checkpoint" --policy.config=mme_vla_suite \
  >"$run_root/training/logs/formal-eval-${model}-${seed}-${source}-server-${SLURM_JOB_ID}.log" 2>&1 &
server_pid=$!
trap 'kill '"$server_pid"' 2>/dev/null || true' EXIT

.venv/bin/python -c "import socket,time; deadline=time.time()+900
while time.time()<deadline:
    try:
        socket.create_connection(('127.0.0.1', $port), 2).close(); break
    except OSError: time.sleep(2)
else: raise SystemExit('policy server timeout')"

python_env=third_party/robomme_benchmark/.venv/bin/python
extra_args=(--args.symbolic-source="$source")
if [[ "$model" == N ]]; then
  extra_args+=(--args.no-use-history)
fi
if [[ "$source" == oracle ]]; then
  extra_args+=(--args.use-oracle --args.subgoal-type=grounded_subgoal)
elif [[ "$source" == qwenvl ]]; then
  python_env=third_party/robomme_benchmark/.venv_qwen/bin/python
  extra_args+=(
    --args.use-qwenvl --args.subgoal-type=grounded_subgoal
    --args.qwen-cache-dir="$run_root/evaluation/qwen_cache"
  )
fi
if [[ "$scope" == diagnostic ]]; then
  extra_args+=(
    --args.only-tasks=VideoUnmask,VideoRepick,InsertPeg,VideoPlaceOrder,RouteStick
    --args.episode-ids=0
  )
fi

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=examples/robomme "$python_env" examples/robomme/eval.py \
  --args.host=127.0.0.1 --args.port="$port" \
  --args.save-dir="$run_root/formal/evaluation" \
  --args.policy-name="formal-${model}-seed${seed}" \
  --args.model-seed=7 --args.model-ckpt-id=79999 \
  --args.dual-memory-run-root="$run_root" --args.model-id="$model" \
  --args.training-seed="$seed" --args.evaluation-scope="$scope" \
  "${extra_args[@]}"

kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
trap - EXIT
