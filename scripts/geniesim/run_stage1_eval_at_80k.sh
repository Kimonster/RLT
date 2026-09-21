#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=/root/workspace/rlt/RLT
python_bin=/root/workspace/envs/openpi/bin/python
plan_root=/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan
checkpoint_root="$plan_root/checkpoints/rlt_pi05_geniesim_stack_three_blocks_plan/stage1_rl_token_20k"
analysis_dir="$plan_root/analysis/stage1_interim_eval_20k_80k"
checkpoint_metadata="$checkpoint_root/80000/_CHECKPOINT_METADATA"
log_path="$analysis_dir/run_20k_80k.log"

export PYTHONPATH="$repo_root/src:$repo_root/rlt_online_rl/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=7
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-chedroybendit5-chinese-university-of-hong-kong-shenzhen}"

mkdir -p "$analysis_dir"
exec > >(tee -a "$log_path") 2>&1

printf '%s waiting_for_stage1_checkpoint_80000\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
until [[ -f "$checkpoint_metadata" ]]; do
  if ! pgrep -f '[t]rain_rlt_plan.py.*--num-train-steps 200000' >/dev/null; then
    printf '%s training_stopped_before_checkpoint_80000\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
    exit 1
  fi
  sleep 60
done

printf '%s evaluating_stage1_checkpoints_20k_40k_60k_80k\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$python_bin" -u "$repo_root/scripts/geniesim/evaluate_stage1_plan.py" \
  --config rlt_pi05_geniesim_stack_three_blocks_plan \
  --output-root "$plan_root" \
  --analysis-dir "$analysis_dir" \
  --max-val-samples 150 \
  --batch-size 5 \
  --num-workers 1 \
  --steps 20000 40000 60000 80000 \
  --wandb-run-name stage1_interim_eval_20k_80k

printf '%s stage1_checkpoint_80000_evaluation_complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
