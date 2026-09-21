#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=/root/workspace/rlt/RLT
python_bin=/root/workspace/envs/openpi/bin/python3.11
run_root=/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks
config_name=rlt_pi05_geniesim_stack_three_blocks
rlt_exp=rlt_frozen_5k_seed42
rlt_checkpoint=$run_root/checkpoints/$config_name/$rlt_exp/4999
task_run=$run_root/runs/geniesim_stack_three_blocks
ac_output=$run_root/ac_offline_10k
log_dir=$run_root/logs
status_file=$run_root/pipeline.status

mkdir -p "$log_dir"
trap 'rc=$?; printf "%s exit_code=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" > "$status_file"' EXIT
printf '%s running\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"
export PYTHONPATH=$repo_root/src:$repo_root/rlt_online_rl/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE=disabled

"$python_bin" -u "$repo_root/scripts/geniesim/prepare_rollout_cache.py" \
  --workers 4 2>&1 | tee "$log_dir/01_prepare_cache.log"

"$python_bin" -u "$repo_root/scripts/train_rlt.py" "$config_name" \
  --resume 2>&1 | tee "$log_dir/02_train_rl_token.log"

test -d "$rlt_checkpoint"
"$python_bin" -u "$repo_root/scripts/geniesim/export_rollout_replay.py" \
  --checkpoint-dir "$rlt_checkpoint" \
  --output-dir "$task_run" \
  --batch-size 8 2>&1 | tee "$log_dir/03_export_replay.log"

"$python_bin" -u "$repo_root/rlt_online_rl/scripts/offline/offline_train_from_replay.py" \
  --replay-path "$task_run/replay/replay_journal.pkl" \
  --steps 10000 \
  --batch-size 128 \
  --seed 42 \
  --bc-weight 10.0 \
  --q-weight 0.1 \
  --delta-weight 10.0 \
  --eval-every 500 \
  --output-dir "$ac_output" 2>&1 | tee "$log_dir/04_train_actor_critic.log"

printf 'RLT checkpoint: %s\nAC output: %s\n' "$rlt_checkpoint" "$ac_output"
