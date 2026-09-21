# GenieSim Stack-Three-Blocks Reproduction

This reproduction uses the frozen pi0.5 checkpoint and the first rollout set
listed in `1.txt`.

Inputs:

- pi0.5 checkpoint: `/mnt/pfs/kk/kk/ckpt/geniesim3/spatialpi05`
- first-round rollout: `/mnt/pfs/kk/kk/data/data/geniesim/rollout/stack_three_blocks`
- rollout records: 4,122 policy calls from 50 successful and 100 failed episodes
- action format: 50 x 16, with 14 arm joints represented as deltas and two
  grippers represented as absolute values during training

The pipeline performs four stages:

1. Cache the 4,122 recorded three-camera policy calls at 224x224.
2. Train one 2,048-dimensional RL token for 5,000 steps with the VLA frozen.
3. Encode the rollouts and export a replay journal with terminal reward 1 for
   successful episodes and 0 otherwise.
4. Offline-train the 16-dimensional, 50-step actor/critic for 10,000 updates.

Run the complete pipeline with:

```bash
bash scripts/geniesim/run_reproduction.sh
```

Outputs and logs are written under
`/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks`.

The pipeline is resumable. Stage-1 training restores the latest saved
checkpoint, and the image cache skips records that are already present. Check
the current state and logs with:

```bash
cat /mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/pipeline.status
tail -f /mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/logs/02_train_rl_token.log
tail -f /mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/logs/03_export_replay.log
tail -f /mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/logs/04_train_actor_critic.log
```

After completion, preserve these artifacts for the next rollout round:

- RLT/VLA checkpoint:
  `/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/checkpoints/rlt_pi05_geniesim_stack_three_blocks/rlt_frozen_5k_seed42/4999`
- AC online-compatible bundle:
  `/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/ac_offline_10k`
- actor snapshot used by rollout:
  `/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/ac_offline_10k/actor_snapshot/actor_snapshot.pkl`

The complete RLT checkpoint directory is needed because serving reads both
`params/` and `assets/`. The AC bundle contains the actor and critic snapshots,
action normalization statistics, optimizer checkpoint, resolved online-RL
config, and the replay journal used to initialize continued online training.
Its internal paths are relative, so the entire AC directory can be moved to the
rollout machine.

Start the RLT feature/reference server from the repository root with:

```bash
/root/workspace/envs/openpi/bin/python3.11 scripts/serve_rlt_policy.py \
  --config rlt_pi05_geniesim_stack_three_blocks \
  --checkpoint-dir /path/to/rlt_checkpoint/4999 \
  --port 8000
```

Start the AC actor service from a second process with:

```bash
/root/workspace/envs/openpi/bin/python3.11 \
  rlt_online_rl/scripts/run_online_rl.py \
  --config /path/to/ac_offline_10k/checkpoints/online_rl_config.yaml \
  --system.role actor_service
```

This reproduction script stops after AC training. It does not start a second
rollout round automatically.
