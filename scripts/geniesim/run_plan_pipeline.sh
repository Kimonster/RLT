#!/usr/bin/env bash
set -Eeuo pipefail

# Unattended continuation for the plan run.  It never starts a robot or an
# online collector; the post-Stage-1 steps consume only the recorded rollout.
repo_root=/root/workspace/rlt/RLT
python_bin=/root/workspace/envs/openpi/bin/python
plan_root=/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan
stage1_status="$plan_root/analysis/stage1_training_status.json"
pipeline_log="$plan_root/analysis/pipeline_continuation.log"
stage1_eval_log="$plan_root/analysis/stage1_evaluation.log"
replay_log="$plan_root/analysis/stage2_replay_export.log"
stage2_log="$plan_root/analysis/stage2_training.log"
finalize_log="$plan_root/analysis/finalize_plan.log"

export PYTHONPATH="$repo_root/src:$repo_root/rlt_online_rl/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$plan_root/analysis"
exec > >(tee -a "$pipeline_log") 2>&1
printf '%s continuation_started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

find_stage1_pid() {
  # The Stage-1 command is launched in a tmux window.  Matching the complete
  # command line with pgrep can return the shared tmux server instead of the
  # Python worker, which would never exit while this continuation window is
  # alive.  Select only the actual Python process.
  ps -eo pid=,args= | awk '$0 ~ /[s]cripts\/geniesim\/train_rlt_plan\.py/ && $0 !~ /tmux/ && $0 !~ /run_plan_pipeline\.sh/ && $0 !~ /awk/ {print $1; exit}'
}

stage1_pid="$(find_stage1_pid || true)"
if [[ -n "$stage1_pid" ]]; then
  printf 'waiting for Stage-1 pid=%s\n' "$stage1_pid"
  while kill -0 "$stage1_pid" 2>/dev/null; do
    sleep 60
  done
fi

if [[ ! -f "$stage1_status" ]]; then
  printf 'Stage-1 status file is missing; refusing to start Stage-2\n' >&2
  exit 2
fi
stage1_step="$($python_bin - "$stage1_status" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("step", -1)))
PY
)"
if [[ "$stage1_step" -lt 20000 ]]; then
  printf 'Stage-1 stopped at step=%s; refusing to start Stage-2\n' "$stage1_step" >&2
  exit 3
fi

printf '%s Stage-1 complete; evaluating fixed validation set\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$python_bin" -u "$repo_root/scripts/geniesim/evaluate_stage1_plan.py" \
  --config rlt_pi05_geniesim_stack_three_blocks_plan \
  --output-root "$plan_root" \
  --max-val-samples 512 \
  --batch-size 64 \
  --num-workers 4 2>&1 | tee "$stage1_eval_log"

selected_stage1="$($python_bin - "$plan_root/analysis/selected_stage1_checkpoint.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])
PY
)"
test -d "$selected_stage1"

printf '%s exporting recorded 50-success/100-failure replay\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$python_bin" -u "$repo_root/scripts/geniesim/export_rollout_replay.py" \
  --config rlt_pi05_geniesim_stack_three_blocks \
  --checkpoint-dir "$selected_stage1" \
  --rollout-root /mnt/pfs/kk/kk/data/data/geniesim/rollout/stack_three_blocks \
  --cache-root /mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/cache_224 \
  --output-dir "$plan_root/stage2/replay_source" \
  --batch-size 8 2>&1 | tee "$replay_log"

printf '%s starting Stage-2 offline Actor/Critic\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
"$python_bin" -u "$repo_root/scripts/geniesim/train_stage2_plan.py" \
  --replay-path "$plan_root/stage2/replay_source/replay/replay_journal.pkl" \
  --output-root "$plan_root/stage2" \
  --steps 30000 \
  --batch-size 128 \
  --seed 42 \
  --num-devices 8 \
  --reference-dropout-prob 0 \
  --actor-residual-scale 0.05 \
  --target-actor-deterministic \
  --bc-weight 10 \
  --q-weight 0.001 \
  --delta-weight 100 \
  --actor-q-start-step 5000 \
  --eval-every 1000 2>&1 | tee "$stage2_log"

printf '%s generating final summary and source diff\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$python_bin" -u "$repo_root/scripts/geniesim/finalize_plan.py" \
  --plan-root "$plan_root" 2>&1 | tee "$finalize_log"

printf '%s plan_complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
