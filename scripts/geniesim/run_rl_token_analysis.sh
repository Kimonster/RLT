#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=/root/workspace/rlt/RLT
python_bin=/root/workspace/envs/openpi/bin/python
output_root=${RLT_TOKEN_ANALYSIS_OUTPUT_ROOT:-/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/rlt_token_analysis_endpoint_v3}
checkpoint_root=/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/checkpoints/rlt_pi05_geniesim_stack_three_blocks_plan/stage1_rl_token_20k
script="$repo_root/scripts/geniesim/analyze_rl_tokens.py"
visual_script="$repo_root/scripts/geniesim/make_rl_token_visuals_v2.py"
steps=(10000 30000 50000 70000 90000 110000 130000 150000 170000)

export PYTHONPATH="$repo_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.72
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$output_root/.matplotlib"

mkdir -p "$output_root/logs" "$MPLCONFIGDIR"
exec > >(tee -a "$output_root/logs/pipeline.log") 2>&1

if ps -eo comm=,args= | awk '$1 ~ /^python/ && $0 ~ /train_rlt_plan.py/ { found = 1 } END { exit !found }'; then
  printf '%s refusing_to_run_while_stage1_training_is_active\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
  exit 2
fi

printf '%s prepare_fixed_manifests\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$python_bin" -u "$script" prepare \
  --output-root "$output_root" \
  --checkpoint-root "$checkpoint_root" \
  --steps "${steps[@]}" \
  --episodes-per-outcome 20 \
  --frames-per-episode 16 \
  --seed 42

printf '%s cache_frozen_vla_prefix\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda "$python_bin" -u "$script" cache-prefix \
  --output-root "$output_root" \
  --checkpoint-root "$checkpoint_root" \
  --steps "${steps[@]}" \
  --reference-step 10000 \
  --batch-size 8

printf '%s stage1_smoke_test_2_success_2_failure\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda "$python_bin" -u "$script" checkpoint \
  --output-root "$output_root" \
  --checkpoint-root "$checkpoint_root" \
  --step 10000 \
  --batch-size 8 \
  --bootstrap-replicates 100 \
  --permutation-replicates 200 \
  --smoke-episodes-per-outcome 2 \
  --target-dir "$output_root/smoke_test/ckpt_10k"

printf '%s stage2_parallel_checkpoint_analysis\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  step=${steps[$gpu]}
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    export JAX_PLATFORMS=cuda
    "$python_bin" -u "$script" checkpoint \
      --output-root "$output_root" \
      --checkpoint-root "$checkpoint_root" \
      --step "$step" \
      --batch-size 8 \
      --bootstrap-replicates 500 \
      --permutation-replicates 2000 2>&1 | tee "$output_root/logs/ckpt_$((step / 1000))k.log"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  printf '%s first_checkpoint_wave_failed\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
  exit 3
fi

step=${steps[8]}
env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda "$python_bin" -u "$script" checkpoint \
  --output-root "$output_root" \
  --checkpoint-root "$checkpoint_root" \
  --step "$step" \
  --batch-size 8 \
  --bootstrap-replicates 500 \
  --permutation-replicates 2000 2>&1 | tee "$output_root/logs/ckpt_$((step / 1000))k.log"

printf '%s stage3_cross_checkpoint_finalize\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$python_bin" -u "$script" finalize \
  --output-root "$output_root" \
  --steps "${steps[@]}" \
  --cka-max-samples 512

printf '%s create_endpoint_aligned_visual_gallery\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$python_bin" -u "$visual_script" \
  --output-root "$output_root" \
  --steps "${steps[@]}"

printf '%s rl_token_analysis_complete output=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$output_root"
