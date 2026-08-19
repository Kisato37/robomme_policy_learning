#!/usr/bin/env bash
#SBATCH --job-name=dual-sp-rollout
#SBATCH --partition=athena-mini
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --no-requeue

set -euo pipefail
repo=/home/jp673/robomme_dual_memory/robomme_policy_learning
run_id=20260818-214308_ecf086c
run_root="$repo/outputs/dual_memory/$run_id"
checkpoint="$run_root/training/checkpoints/mme_vla_suite/smoke100_SP_seed42/99"
port=18121
export OPENPI_DATA_HOME=/home/jp673/robomme_dual_memory/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
cd "$repo"
test -d "$checkpoint"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py \
  --seed=7 --port="$port" policy:checkpoint \
  --policy.dir="$checkpoint" --policy.config=mme_vla_suite \
  >"$run_root/training/logs/sp-rollout-smoke-server.log" 2>&1 &
server_pid=$!
trap 'kill '"$server_pid"' 2>/dev/null || true' EXIT

.venv/bin/python -c "import socket,time; deadline=time.time()+900
while time.time()<deadline:
    try:
        socket.create_connection(('127.0.0.1', $port), 2).close(); break
    except OSError: time.sleep(2)
else: raise SystemExit('policy server timeout')"

common_args=(
  --args.host=127.0.0.1 --args.port="$port"
  --args.save-dir="$run_root/smoke/evaluation"
  --args.policy-name=SP-smoke --args.model-seed=7 --args.model-ckpt-id=99
  --args.only-tasks=InsertPeg --args.episode-ids=0
  --args.dual-memory-run-root="$run_root" --args.model-id=SP-smoke --args.training-seed=42
  --args.subgoal-type=grounded_subgoal
)

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=examples/robomme \
  third_party/robomme_benchmark/.venv/bin/python examples/robomme/eval.py \
  "${common_args[@]}" --args.use-oracle --args.symbolic-source=oracle

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=examples/robomme \
  third_party/robomme_benchmark/.venv_qwen/bin/python examples/robomme/eval.py \
  "${common_args[@]}" --args.use-qwenvl --args.symbolic-source=qwenvl \
  --args.qwen-cache-dir="$run_root/evaluation/qwen_cache"

.venv/bin/python experiments/dual_memory/build_qwen_smoke_audit.py \
  --eval-root "$run_root/smoke/evaluation" \
  --output "$run_root/audits/qwen_smoke_audit.json"
.venv/bin/python experiments/dual_memory/record_phase_status.py \
  --run-root "$run_root" --phase Phase-3 --status completed \
  --evidence audits/qwen_smoke_audit.json --evidence audits/checkpoint_reload_smoke_SP_seed42.json \
  --next "Run the paired five-task pilot."

kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
trap - EXIT
