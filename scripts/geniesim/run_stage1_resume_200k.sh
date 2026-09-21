#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=/root/workspace/rlt/RLT
python_bin=/root/workspace/envs/openpi/bin/python
plan_root=/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan
analysis_subdir=stage1_resume_20k_to_200k
analysis_dir="$plan_root/analysis/$analysis_subdir"
log_path="$analysis_dir/run.log"

export PYTHONPATH="$repo_root/src:$repo_root/rlt_online_rl/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-chedroybendit5-chinese-university-of-hong-kong-shenzhen}"

mkdir -p "$analysis_dir"
exec > >(tee -a "$log_path") 2>&1
printf '%s stage1_resume_started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$python_bin" -u "$repo_root/scripts/geniesim/train_rlt_plan.py" \
  --config rlt_pi05_geniesim_stack_three_blocks_plan \
  --output-root "$plan_root" \
  --analysis-subdir "$analysis_subdir" \
  --resume \
  --num-train-steps 200000 \
  --num-workers 8 \
  --save-interval 5000 \
  --keep-period 2500 \
  --wandb-run-name stage1_rl_token_resume_20k_to_200k

printf '%s stage1_resume_training_complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$python_bin" -u "$repo_root/scripts/geniesim/evaluate_stage1_plan.py" \
  --config rlt_pi05_geniesim_stack_three_blocks_plan \
  --output-root "$plan_root" \
  --analysis-dir "$analysis_dir" \
  --max-val-samples 500 \
  --batch-size 50 \
  --num-workers 4 \
  --steps 5000 10000 15000 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000 \
  --wandb-run-name stage1_rl_token_20k_to_200k_evaluation

printf '%s stage1_resume_evaluation_complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
