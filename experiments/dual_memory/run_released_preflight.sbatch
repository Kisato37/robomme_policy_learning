#!/usr/bin/env bash
#SBATCH --job-name=dual-preflight
#SBATCH --partition=athena-small
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_id=20260818-214308_ecf086c
run_root="$repo/outputs/dual_memory/$run_id"
export OPENPI_DATA_HOME=/home/jp673/robomme_dual_memory/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
cd "$repo"

run_reference() {
  local checkpoint_name="$1"
  local model_id="$2"
  local source="$3"
  local port="$4"
  local policy_name="released-${checkpoint_name}"
  local checkpoint="runs/ckpts/mme_vla_suite/${checkpoint_name}/79999"
  test -d "$checkpoint"

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py \
    --seed=7 --port="$port" policy:checkpoint \
    --policy.dir="$checkpoint" --policy.config=mme_vla_suite \
    >"$run_root/training/logs/preflight-${model_id}-server.log" 2>&1 &
  local server_pid=$!
  trap "kill $server_pid 2>/dev/null || true" EXIT

  .venv/bin/python -c "import socket,time; deadline=time.time()+900
while time.time()<deadline:
    try:
        socket.create_connection(('127.0.0.1', int('$port')), 2).close(); break
    except OSError: time.sleep(2)
else: raise SystemExit('policy server timeout')"

  local source_args=()
  if [[ "$source" == oracle ]]; then
    source_args=(--args.use-oracle --args.subgoal-type=grounded_subgoal --args.symbolic-source=oracle)
  fi
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=examples/robomme \
    third_party/robomme_benchmark/.venv/bin/python examples/robomme/eval.py \
    --args.host=127.0.0.1 --args.port="$port" \
    --args.save-dir="$run_root/preflight/released_reference" \
    --args.policy-name="$policy_name" --args.model-seed=7 --args.model-ckpt-id=79999 \
    --args.only-tasks=InsertPeg --args.episode-ids=0 \
    --args.dual-memory-run-root="$run_root" --args.model-id="$model_id" \
    --args.training-seed=-1 "${source_args[@]}"
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - EXIT
}

run_reference symbolic-grounded-subgoal released-S oracle 18111
run_reference perceptual-framesamp-modul released-P none 18112

.venv/bin/python experiments/dual_memory/build_preflight_report.py \
  --run-root "$run_root" --output "$run_root/audits/preflight_report.md"
