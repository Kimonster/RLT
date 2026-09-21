#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=/root/workspace/rlt/RLT
python_bin=/root/workspace/envs/openpi/bin/python
plan_root=/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan
replay_path="$plan_root/stage2/replay_source/replay/replay_journal.pkl"
output_root="$plan_root/stage2_retrain_conservative_8gpu"
log_path="$output_root/stage2_retrain.log"

export PYTHONPATH="$repo_root/src:$repo_root/rlt_online_rl/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-chedroybendit5-chinese-university-of-hong-kong-shenzhen}"

if [[ -e "$output_root" ]]; then
  printf 'Output already exists; refusing to overwrite: %s\n' "$output_root" >&2
  exit 2
fi
if [[ ! -f "$replay_path" ]]; then
  printf 'Replay journal does not exist: %s\n' "$replay_path" >&2
  exit 3
fi

mkdir -p "$output_root"
exec > >(tee -a "$log_path") 2>&1
printf '%s stage2_retrain_started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$python_bin" -u "$repo_root/scripts/geniesim/train_stage2_plan.py" \
  --replay-path "$replay_path" \
  --output-root "$output_root" \
  --steps 30000 \
  --batch-size 128 \
  --seed 42 \
  --eval-every 1000 \
  --num-devices 8 \
  --reference-dropout-prob 0 \
  --actor-residual-scale 0.05 \
  --target-actor-deterministic \
  --bc-weight 10 \
  --q-weight 0.001 \
  --delta-weight 100 \
  --actor-q-start-step 5000 \
  --wandb-run-name stage2_actor_critic_conservative_8gpu_30k

printf '%s stage2_retrain_complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
