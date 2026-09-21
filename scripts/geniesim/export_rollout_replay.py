#!/usr/bin/env python3
"""Encode recorded GenieSim rollouts and export an offline RLT replay journal."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import jax
import numpy as np
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "rlt_online_rl" / "src"))

import openpi.models.model as model_lib
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as train_config
from openpi.training.geniesim_rollout_dataset import GenieSimRolloutDataset
import openpi.transforms as transforms
from rlt_online_rl.replay import RLTTransition
from rlt_online_rl.replay import TransitionSource
from scripts.serve_rlt_policy import load_rlt_model

DEFAULT_CONFIG = "rlt_pi05_geniesim_stack_three_blocks"
DEFAULT_ROLLOUT_ROOT = Path("/mnt/pfs/kk/kk/data/data/geniesim/rollout/stack_three_blocks")
DEFAULT_CACHE_ROOT = Path("/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/cache_224")
DEFAULT_OUTPUT_DIR = Path("/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/runs/geniesim_stack_three_blocks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _stack(items: list[dict[str, Any]]) -> dict[str, Any]:
    return jax.tree.map(lambda *values: np.stack([np.asarray(value) for value in values]), *items)


def _encode_features(
    dataset: GenieSimRolloutDataset,
    config: train_config.TrainConfig,
    checkpoint_dir: Path,
    *,
    batch_size: int,
) -> np.ndarray:
    data_config = config.data.create(config.assets_dirs, config.model)
    input_transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    model = load_rlt_model(config, str(checkpoint_dir))
    encode_fn = nnx_utils.module_jit(model.encode_rl_token)
    rng = jax.random.key(config.seed)
    features: list[np.ndarray] = []

    for start in tqdm.tqdm(range(0, len(dataset), batch_size), desc="Encoding RL tokens"):
        end = min(start + batch_size, len(dataset))
        transformed = [input_transform(dataset[index]) for index in range(start, end)]
        batch = _stack(transformed)
        observation = model_lib.Observation.from_dict(batch)
        rng, batch_rng = jax.random.split(rng)
        tokens = encode_fn(batch_rng, observation)
        token_array = np.asarray(tokens, dtype=np.float32).reshape(end - start, -1)
        features.append(token_array)

    result = np.concatenate(features, axis=0)
    expected_dim = (config.rlt_num_tokens or 1) * (config.rlt_embed_dim or 2048)
    if result.shape != (len(dataset), expected_dim):
        raise ValueError(f"Unexpected RL-token feature shape: {result.shape}")
    return result


def _write_replay(
    dataset: GenieSimRolloutDataset,
    features: np.ndarray,
    replay_path: Path,
) -> dict[str, int]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        grouped[(record.category, record.trajectory_id)].append(index)

    temporary_path = replay_path.with_suffix(f"{replay_path.suffix}.tmp")
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    success_episodes = 0
    failure_episodes = 0
    transition_count = 0
    with temporary_path.open("wb") as stream:
        for episode_id, ((category, _trajectory_id), indices) in enumerate(sorted(grouped.items())):
            successful = category == "successful"
            success_episodes += int(successful)
            failure_episodes += int(not successful)
            samples = [dataset[index] for index in indices]
            for local_step, (index, sample) in enumerate(zip(indices, samples, strict=True)):
                terminal = local_step == len(indices) - 1
                next_index = indices[local_step + 1] if not terminal else index
                next_sample = samples[local_step + 1] if not terminal else sample
                rewards = np.zeros((50,), dtype=np.float32)
                if terminal and successful:
                    rewards[-1] = 1.0
                ref_chunk = np.asarray(sample["actions"], dtype=np.float32)[:50, :16]
                next_ref_chunk = (
                    np.zeros_like(ref_chunk)
                    if terminal
                    else np.asarray(next_sample["actions"], dtype=np.float32)[:50, :16]
                )
                transition = RLTTransition(
                    z_rl=features[index],
                    proprio=np.asarray(sample["state"], dtype=np.float32)[:16],
                    ref_chunk=ref_chunk,
                    action_chunk=ref_chunk.copy(),
                    rewards=rewards,
                    done=terminal,
                    next_z_rl=features[next_index],
                    next_proprio=np.asarray(next_sample["state"], dtype=np.float32)[:16],
                    next_ref_chunk=next_ref_chunk,
                    source=int(TransitionSource.BASE),
                    source_chunk=np.full((50,), int(TransitionSource.BASE), dtype=np.uint8),
                    collection_phase="warmup",
                    success=int(successful),
                    intervention_flag=False,
                    episode_id=episode_id,
                    step_id=local_step,
                )
                pickle.dump(transition.to_journal_record(), stream, protocol=pickle.HIGHEST_PROTOCOL)
                transition_count += 1
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, replay_path)
    return {
        "episodes": len(grouped),
        "success_episodes": success_episodes,
        "failure_episodes": failure_episodes,
        "transitions": transition_count,
    }


def main() -> None:
    args = parse_args()
    config = train_config.get_config(args.config)
    dataset = GenieSimRolloutDataset(args.rollout_root, cache_root=args.cache_root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features = _encode_features(
        dataset,
        config,
        args.checkpoint_dir.expanduser().resolve(),
        batch_size=args.batch_size,
    )
    feature_path = output_dir / "rl_token_features.npz"
    np.savez_compressed(feature_path, z_rl=features.astype(np.float16))
    replay_path = output_dir / "replay" / "replay_journal.pkl"
    counts = _write_replay(dataset, features, replay_path)
    manifest = {
        "schema_version": 1,
        "config": args.config,
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "rollout_root": str(args.rollout_root.expanduser().resolve()),
        "cache_root": str(args.cache_root.expanduser().resolve()),
        "feature_path": str(feature_path),
        "replay_path": str(replay_path),
        "feature_shape": list(features.shape),
        **counts,
        "reward_semantics": "terminal success=1; failure and non-terminal steps=0",
    }
    (output_dir / "replay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
