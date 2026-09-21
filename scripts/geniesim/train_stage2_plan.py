#!/usr/bin/env python3
"""Train and diagnose the offline Actor/Critic stage required by the RLT plan.

The script intentionally keeps the algorithm in ``rlt_online_rl`` unchanged.  It
adds the episode-level split, fixed validation diagnostics, checkpoint comparison,
and W&B logging needed for an auditable reproduction.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import math
from pathlib import Path
import pickle
import random
import shutil
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "rlt_online_rl" / "src"))

from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.config import resolve_rl_config_paths
from rlt_online_rl.networks import build_td_target
from rlt_online_rl.trainer import init_train_state
from scripts.geniesim.plan_utils import DEFAULT_PLAN_ROOT
from scripts.geniesim.plan_utils import atomic_write_json
from scripts.geniesim.plan_utils import init_wandb
from scripts.geniesim.plan_utils import log_artifact
from scripts.geniesim.plan_utils import sha256_file


_OFFLINE_SCRIPT = REPO_ROOT / "rlt_online_rl" / "scripts" / "offline" / "offline_train_from_replay.py"
_OFFLINE_SPEC = importlib.util.spec_from_file_location("rlt_offline_train_from_replay", _OFFLINE_SCRIPT)
if _OFFLINE_SPEC is None or _OFFLINE_SPEC.loader is None:
    raise ImportError(f"Could not load offline trainer from {_OFFLINE_SCRIPT}")
offline = importlib.util.module_from_spec(_OFFLINE_SPEC)
sys.path.insert(0, str(_OFFLINE_SCRIPT.parent))
_OFFLINE_SPEC.loader.exec_module(offline)


DEFAULT_STAGE2_ROOT = DEFAULT_PLAN_ROOT / "stage2"
DEFAULT_REPLAY_PATH = DEFAULT_STAGE2_ROOT / "replay_source" / "replay" / "replay_journal.pkl"
DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "rlt_online_rl" / "configs" / "tasks" / "geniesim_stack_three_blocks" / "online_rl.yaml"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-path", type=Path, default=DEFAULT_REPLAY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_STAGE2_ROOT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--reference-dropout-prob", type=float, default=None)
    parser.add_argument("--actor-residual-scale", type=float, default=None)
    parser.add_argument("--target-actor-deterministic", action="store_true")
    parser.add_argument("--bc-weight", type=float, default=None)
    parser.add_argument("--q-weight", type=float, default=None)
    parser.add_argument("--delta-weight", type=float, default=None)
    parser.add_argument("--actor-q-start-step", type=int, default=0)
    parser.add_argument("--wandb-run-name", default="stage2_actor_critic_30k")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-eval-batches", type=int, default=None)
    return parser.parse_args()


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        while True:
            try:
                value = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(value, dict):
                raise TypeError(f"Replay entry is {type(value).__name__}, expected a mapping")
            records.append(value)
    if not records:
        raise RuntimeError(f"Replay journal is empty: {path}")
    return records


def _validate_records(records: list[dict[str, Any]], replay_path: Path) -> dict[str, Any]:
    required = {
        "z_rl",
        "proprio",
        "ref_chunk",
        "action_chunk",
        "rewards",
        "done",
        "next_z_rl",
        "next_proprio",
        "next_ref_chunk",
        "episode_id",
        "step_id",
        "success",
    }
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise KeyError(f"Replay record {index} is missing fields: {missing}")
        episode_id = int(record["episode_id"])
        by_episode.setdefault(episode_id, []).append(record)
        expected_shapes = {
            "z_rl": (2048,),
            "proprio": (16,),
            "ref_chunk": (50, 16),
            "action_chunk": (50, 16),
            "rewards": (50,),
            "next_z_rl": (2048,),
            "next_proprio": (16,),
            "next_ref_chunk": (50, 16),
        }
        for key, shape in expected_shapes.items():
            value = np.asarray(record[key])
            if value.shape != shape:
                raise ValueError(f"Replay record {index} field {key} has {value.shape}, expected {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"Replay record {index} field {key} contains NaN/Inf")
        if bool(record["done"]) not in (True, False):
            raise ValueError(f"Replay record {index} has a non-boolean done field")

    episode_success: dict[int, int] = {}
    terminal_errors: list[str] = []
    reward_errors: list[str] = []
    for episode_id, episode_records in sorted(by_episode.items()):
        episode_records.sort(key=lambda row: int(row["step_id"]))
        step_ids = [int(row["step_id"]) for row in episode_records]
        if step_ids != list(range(len(step_ids))):
            terminal_errors.append(f"episode {episode_id}: step ids {step_ids[:3]}... are not contiguous")
        done_positions = [index for index, row in enumerate(episode_records) if bool(row["done"])]
        if done_positions != [len(episode_records) - 1]:
            terminal_errors.append(f"episode {episode_id}: done positions={done_positions}")
        labels = {int(row["success"]) for row in episode_records}
        if labels - {0, 1} or len(labels) != 1:
            terminal_errors.append(f"episode {episode_id}: inconsistent success labels={labels}")
        success = int(next(iter(labels)))
        episode_success[episode_id] = success
        nonzero = np.argwhere(np.abs(np.asarray([row["rewards"] for row in episode_records])) > 1e-7)
        if success == 1:
            if nonzero.shape[0] != 1 or int(nonzero[-1, 0]) != len(episode_records) - 1 or int(nonzero[-1, 1]) != 49:
                reward_errors.append(f"episode {episode_id}: expected only terminal reward at chunk index 49")
        elif nonzero.shape[0] != 0:
            reward_errors.append(f"episode {episode_id}: failure has non-zero reward")
    if terminal_errors or reward_errors:
        raise ValueError("Replay validation failed: " + "; ".join((terminal_errors + reward_errors)[:8]))

    success_count = sum(value == 1 for value in episode_success.values())
    failure_count = sum(value == 0 for value in episode_success.values())
    if (success_count, failure_count) != (50, 100):
        raise ValueError(f"Expected 50 success + 100 failure episodes, got {success_count} + {failure_count}")
    return {
        "schema_version": 1,
        "replay_path": str(replay_path.resolve()),
        "replay_sha256": sha256_file(replay_path),
        "transition_count": len(records),
        "episode_count": len(episode_success),
        "success_episode_count": success_count,
        "failure_episode_count": failure_count,
        "episode_ids": sorted(episode_success),
        "episode_success": {str(key): value for key, value in sorted(episode_success.items())},
        "action_source": "recorded outputs/actions; no separate executed-action field exists in source records",
        "reward_semantics": "success: zero until terminal chunk then reward[49]=1; failure: all zero",
    }


def _episode_split(
    records: list[dict[str, Any]], *, seed: int, output_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    labels: dict[int, int] = {}
    for record in records:
        episode_id = int(record["episode_id"])
        label = int(record["success"])
        if episode_id in labels and labels[episode_id] != label:
            raise ValueError(f"Episode {episode_id} has multiple labels")
        labels[episode_id] = label
    train_ids: list[int] = []
    val_ids: list[int] = []
    rng = random.Random(seed)
    for label in (1, 0):
        ids = sorted(episode_id for episode_id, value in labels.items() if value == label)
        rng.shuffle(ids)
        val_count = max(1, int(round(len(ids) * 0.20)))
        val_ids.extend(sorted(ids[:val_count]))
        train_ids.extend(sorted(ids[val_count:]))
    train_ids.sort()
    val_ids.sort()
    train_id_set, val_id_set = set(train_ids), set(val_ids)
    train = [record for record in records if int(record["episode_id"]) in train_id_set]
    val = [record for record in records if int(record["episode_id"]) in val_id_set]
    payload = {
        "schema_version": 1,
        "seed": seed,
        "validation_ratio": 0.20,
        "train_episode_ids": train_ids,
        "validation_episode_ids": val_ids,
        "train_episode_count": len(train_ids),
        "validation_episode_count": len(val_ids),
        "train_success_episode_count": sum(labels[index] == 1 for index in train_ids),
        "train_failure_episode_count": sum(labels[index] == 0 for index in train_ids),
        "validation_success_episode_count": sum(labels[index] == 1 for index in val_ids),
        "validation_failure_episode_count": sum(labels[index] == 0 for index in val_ids),
    }
    atomic_write_json(output_root / "analysis" / "stage2_episode_split.json", payload)
    return train, val, payload


def _records_to_arrays(records: list[dict[str, Any]], gamma: float) -> dict[str, np.ndarray]:
    dataset = offline._stack_records(records)
    # Compute exact chunk-level Monte-Carlo returns without crossing episode boundaries.
    mc_return = np.zeros((len(records),), dtype=np.float32)
    discounted_reward = np.power(gamma, np.arange(50, dtype=np.float32))
    by_episode: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        by_episode.setdefault(int(record["episode_id"]), []).append(index)
    for indices in by_episode.values():
        indices.sort(key=lambda index: int(records[index]["step_id"]))
        future = 0.0
        for index in reversed(indices):
            reward_value = float(np.asarray(records[index]["rewards"], dtype=np.float32).dot(discounted_reward))
            mc_return[index] = reward_value + (gamma**50) * future
            future = float(mc_return[index])
    dataset["mc_return"] = mc_return
    return dataset


def _sample_batch(
    dataset: dict[str, np.ndarray], *, batch_size: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    indices = rng.integers(0, dataset["z_rl"].shape[0], size=batch_size, endpoint=False)
    keys = (
        "z_rl",
        "proprio",
        "ref_chunk",
        "action_chunk",
        "rewards",
        "done",
        "next_z_rl",
        "next_proprio",
        "next_ref_chunk",
        "source",
        "source_chunk",
        "success",
        "intervention_flag",
        "episode_id",
        "step_id",
    )
    return {key: dataset[key][indices] for key in keys}


def _replicate_train_state(state: Any, devices: tuple[jax.Device, ...]) -> Any:
    replicated = jax.device_put_replicated(state, devices)
    device_rngs = [jax.random.fold_in(state.rng, index) for index in range(len(devices))]
    return replicated.replace(rng=jax.device_put_sharded(device_rngs, devices))


def _shard_batch(batch: dict[str, np.ndarray], devices: tuple[jax.Device, ...]) -> dict[str, jax.Array]:
    num_devices = len(devices)
    batch_size = next(iter(batch.values())).shape[0]
    if batch_size % num_devices != 0:
        raise ValueError(f"Batch size {batch_size} is not divisible by {num_devices} devices")
    per_device = batch_size // num_devices
    shards = [
        {key: np.asarray(value[start : start + per_device]) for key, value in batch.items()}
        for start in range(0, batch_size, per_device)
    ]
    return jax.device_put_sharded(shards, devices)


def _unreplicate(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: value[0], tree)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _finite_scalar(value: float) -> bool:
    return bool(np.isfinite(value))


def _eval_dataset(
    actor: Any,
    critic: Any,
    state: Any,
    adapter: ActionRepresentationAdapter,
    dataset: dict[str, np.ndarray],
    rl_config: Any,
    *,
    batch_size: int,
    num_batches: int | None = None,
) -> dict[str, Any]:
    actor_params = state.actor_params
    critic_params = state.critic_params
    target_actor_params = state.target_actor_params
    target_critic_params = state.target_critic_params
    train_losses: list[float] = []
    q_values: list[np.ndarray] = []
    q1_values: list[np.ndarray] = []
    q2_values: list[np.ndarray] = []
    mc_values: list[np.ndarray] = []
    success_values: list[np.ndarray] = []
    actor_mse_values: list[np.ndarray] = []
    normalized_delta_values: list[np.ndarray] = []
    delta_values: list[np.ndarray] = []
    actor_step_delta_values: list[np.ndarray] = []
    ref_step_delta_values: list[np.ndarray] = []
    stochastic_delta_values: list[np.ndarray] = []
    stochastic_step_delta_values: list[np.ndarray] = []
    stochastic_noise_values: list[np.ndarray] = []
    q_actor_values: list[np.ndarray] = []
    q_ref_values: list[np.ndarray] = []
    rng = jax.random.PRNGKey(17)
    total_batches = math.ceil(dataset["z_rl"].shape[0] / batch_size)
    if num_batches is not None:
        total_batches = min(total_batches, num_batches)
    for batch_index, start in enumerate(range(0, dataset["z_rl"].shape[0], batch_size)):
        if batch_index >= total_batches:
            break
        end = min(start + batch_size, dataset["z_rl"].shape[0])
        raw = {key: value[start:end] for key, value in dataset.items()}
        normalized = adapter.prepare_training_batch(raw)
        batch = {key: jnp.asarray(value) for key, value in normalized.items() if key != "mc_return"}
        q1, q2 = critic.q_values(critic_params, batch["z_rl"], batch["proprio"], batch["action_chunk"])
        pred_norm = actor.actor_mean(actor_params, batch["z_rl"], batch["proprio"], batch["ref_chunk"])
        q_actor_1, q_actor_2 = critic.q_values(critic_params, batch["z_rl"], batch["proprio"], pred_norm)
        q_ref_1, q_ref_2 = critic.q_values(critic_params, batch["z_rl"], batch["proprio"], batch["ref_chunk"])
        rng, target_rng, sample_rng = jax.random.split(rng, 3)
        target = build_td_target(
            actor,
            target_actor_params,
            critic,
            target_critic_params,
            batch["next_z_rl"],
            batch["next_proprio"],
            batch["next_ref_chunk"],
            batch["rewards"],
            batch["done"],
            rl_config.gamma,
            target_rng,
            target_actor_deterministic=rl_config.target_actor_deterministic,
        )
        td = 0.5 * (jnp.mean(jnp.square(q1 - target)) + jnp.mean(jnp.square(q2 - target)))
        train_losses.append(float(jax.device_get(td)))
        q1_np = np.asarray(jax.device_get(q1), dtype=np.float32)
        q2_np = np.asarray(jax.device_get(q2), dtype=np.float32)
        q1_values.append(q1_np)
        q2_values.append(q2_np)
        q_values.append(0.5 * (q1_np + q2_np))
        mc_values.append(np.asarray(raw["mc_return"], dtype=np.float32))
        success_values.append(np.asarray(raw["success"], dtype=np.int8))
        pred_norm_np = np.asarray(jax.device_get(pred_norm), dtype=np.float32)
        action_norm_np = np.asarray(normalized["action_chunk"], dtype=np.float32)
        actor_mse_values.append(np.mean(np.square(pred_norm_np - action_norm_np), axis=(1, 2)))
        normalized_delta_values.append(pred_norm_np - np.asarray(normalized["ref_chunk"], dtype=np.float32))
        pred_abs = adapter.denormalize_to_abs_chunk(pred_norm_np, raw["proprio"])
        ref_abs = np.asarray(raw["ref_chunk"], dtype=np.float32)
        stochastic_norm = actor.sample_action(
            actor_params,
            sample_rng,
            batch["z_rl"],
            batch["proprio"],
            batch["ref_chunk"],
            deterministic=False,
        )
        stochastic_abs = adapter.denormalize_to_abs_chunk(
            np.asarray(jax.device_get(stochastic_norm), dtype=np.float32), raw["proprio"]
        )
        delta_values.append(pred_abs - ref_abs)
        actor_step_delta_values.append(np.diff(pred_abs, axis=1))
        ref_step_delta_values.append(np.diff(ref_abs, axis=1))
        stochastic_delta_values.append(stochastic_abs - ref_abs)
        stochastic_step_delta_values.append(np.diff(stochastic_abs, axis=1))
        stochastic_noise_values.append(stochastic_abs - pred_abs)
        q_actor_values.append(0.5 * (np.asarray(jax.device_get(q_actor_1)) + np.asarray(jax.device_get(q_actor_2))))
        q_ref_values.append(0.5 * (np.asarray(jax.device_get(q_ref_1)) + np.asarray(jax.device_get(q_ref_2))))

    q = np.concatenate(q_values)
    q1 = np.concatenate(q1_values)
    q2 = np.concatenate(q2_values)
    mc = np.concatenate(mc_values)
    success = np.concatenate(success_values).astype(bool)
    actor_mse = np.concatenate(actor_mse_values)
    normalized_deltas = np.concatenate(normalized_delta_values, axis=0)
    deltas = np.concatenate(delta_values, axis=0)
    actor_step_deltas = np.concatenate(actor_step_delta_values, axis=0)
    ref_step_deltas = np.concatenate(ref_step_delta_values, axis=0)
    stochastic_deltas = np.concatenate(stochastic_delta_values, axis=0)
    stochastic_step_deltas = np.concatenate(stochastic_step_delta_values, axis=0)
    stochastic_noise = np.concatenate(stochastic_noise_values, axis=0)
    q_actor = np.concatenate(q_actor_values)
    q_ref = np.concatenate(q_ref_values)
    gap = np.abs(q1 - q2)
    success_q = q[success]
    failure_q = q[~success]
    flat_delta = np.abs(deltas).reshape(-1)
    flat_normalized_delta = np.abs(normalized_deltas).reshape(-1)
    flat_actor_step_delta = np.abs(actor_step_deltas).reshape(-1)
    flat_ref_step_delta = np.abs(ref_step_deltas).reshape(-1)
    flat_stochastic_delta = np.abs(stochastic_deltas).reshape(-1)
    flat_stochastic_step_delta = np.abs(stochastic_step_deltas).reshape(-1)
    flat_stochastic_noise = np.abs(stochastic_noise).reshape(-1)
    ref_step_delta_mean = float(np.mean(flat_ref_step_delta))
    deployment_prefix = min(10, deltas.shape[1])
    per_dim = np.abs(deltas).mean(axis=(0, 1))
    per_dim_rmse = np.sqrt(np.square(deltas).mean(axis=(0, 1)))
    q_mc_mse = float(np.mean(np.square(q - mc)))
    q_actor_minus_ref = float(np.mean(q_actor - q_ref))
    q_success_minus_failure = (
        float(np.mean(success_q) - np.mean(failure_q)) if success_q.size and failure_q.size else float("nan")
    )
    metrics: dict[str, Any] = {
        "critic_train_val_td_loss": float(np.mean(train_losses)),
        "critic_val_td_loss": float(np.mean(train_losses)),
        "q_vs_mc_mse": q_mc_mse,
        "q_vs_mc_pearson": _pearson(q, mc),
        "q_vs_mc_spearman": _spearman(q, mc),
        "q_success_mean": float(np.mean(success_q)) if success_q.size else float("nan"),
        "q_success_median": float(np.median(success_q)) if success_q.size else float("nan"),
        "q_failure_mean": float(np.mean(failure_q)) if failure_q.size else float("nan"),
        "q_failure_median": float(np.median(failure_q)) if failure_q.size else float("nan"),
        "q_success_minus_failure": q_success_minus_failure,
        "q1_q2_gap_mean": float(np.mean(gap)),
        "q1_q2_gap_median": float(np.median(gap)),
        "q1_q2_gap_p90": float(np.percentile(gap, 90)),
        "q_abs_max": float(np.max(np.abs(np.concatenate([q1, q2])))),
        "actor_val_action_mse": float(np.mean(actor_mse)),
        "actor_delta_mean": float(np.mean(flat_delta)),
        "actor_delta_median": float(np.median(flat_delta)),
        "actor_delta_p90": float(np.percentile(flat_delta, 90)),
        "actor_delta_max": float(np.max(flat_delta)),
        "actor_normalized_delta_p90": float(np.percentile(flat_normalized_delta, 90)),
        "actor_normalized_delta_max": float(np.max(flat_normalized_delta)),
        "actor_delta_prefix10_mean": float(np.mean(np.abs(deltas[:, :deployment_prefix]))),
        "actor_delta_tail10_mean": float(np.mean(np.abs(deltas[:, -deployment_prefix:]))),
        "reference_step_delta_mean": ref_step_delta_mean,
        "actor_step_delta_mean": float(np.mean(flat_actor_step_delta)),
        "actor_step_delta_p90": float(np.percentile(flat_actor_step_delta, 90)),
        "actor_step_delta_ratio": float(np.mean(flat_actor_step_delta) / max(ref_step_delta_mean, 1e-8)),
        "actor_stochastic_delta_mean": float(np.mean(flat_stochastic_delta)),
        "actor_stochastic_delta_p90": float(np.percentile(flat_stochastic_delta, 90)),
        "actor_stochastic_step_delta_mean": float(np.mean(flat_stochastic_step_delta)),
        "actor_stochastic_step_delta_p90": float(np.percentile(flat_stochastic_step_delta, 90)),
        "actor_stochastic_step_delta_ratio": float(
            np.mean(flat_stochastic_step_delta) / max(ref_step_delta_mean, 1e-8)
        ),
        "actor_stochastic_noise_mae": float(np.mean(flat_stochastic_noise)),
        "actor_stochastic_noise_p90": float(np.percentile(flat_stochastic_noise, 90)),
        "q_actor_mean": float(np.mean(q_actor)),
        "q_ref_mean": float(np.mean(q_ref)),
        "q_actor_minus_ref": q_actor_minus_ref,
        "critic_exploitation_or_ood": bool(
            float(np.percentile(flat_normalized_delta, 90)) > 3.0
            or float(np.max(np.abs(q_actor))) > 10.0
            or (
                np.isfinite(q_success_minus_failure)
                and q_actor_minus_ref > max(0.02, 2.0 * max(q_success_minus_failure, 0.0))
            )
        ),
        "sample_count": int(q.shape[0]),
    }
    for index, value in enumerate(per_dim):
        metrics[f"actor_delta_dim_{index}_mean"] = float(value)
        metrics[f"actor_delta_dim_{index}_rmse"] = float(per_dim_rmse[index])
    metrics["finite"] = bool(
        all(
            _finite_scalar(float(value))
            for key, value in metrics.items()
            if key != "critic_exploitation_or_ood" and isinstance(value, (float, int))
        )
    )
    metrics["critic_divergence"] = bool(
        not metrics["finite"] or metrics["q_abs_max"] > 100.0 or metrics["critic_val_td_loss"] > 1e4
    )
    metrics["actor_temporal_safe"] = bool(metrics["actor_step_delta_ratio"] <= 1.5)
    metrics["actor_residual_safe"] = bool(metrics["actor_normalized_delta_max"] <= 0.5)
    metrics["valid_checkpoint"] = bool(
        metrics["finite"]
        and not metrics["critic_divergence"]
        and not metrics["critic_exploitation_or_ood"]
        and metrics["actor_temporal_safe"]
        and metrics["actor_residual_safe"]
    )
    return metrics


def _train_metrics(
    actor: Any,
    critic: Any,
    state: Any,
    adapter: ActionRepresentationAdapter,
    dataset: dict[str, np.ndarray],
    rl_config: Any,
    *,
    batch_size: int,
    num_batches: int | None = None,
) -> dict[str, Any]:
    return _eval_dataset(
        actor, critic, state, adapter, dataset, rl_config, batch_size=batch_size, num_batches=num_batches
    )


def _save_stage2_checkpoint(
    output_root: Path, state: Any, rl_config: Any, step: int, split: dict[str, Any], *, latest: bool = False
) -> Path:
    actor_version = int(jax.device_get(state.actor_version))
    name = "latest" if latest else f"C{step:05d}_A{actor_version:05d}"
    path = output_root / "checkpoints" / name
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "critic_updates": int(step),
        "actor_updates": actor_version,
        "rl_config": dataclasses.asdict(rl_config),
        "episode_split": split,
        "state": {
            "actor_params": offline._tree_to_numpy(state.actor_params),
            "target_actor_params": offline._tree_to_numpy(state.target_actor_params),
            "critic_params": offline._tree_to_numpy(state.critic_params),
            "target_critic_params": offline._tree_to_numpy(state.target_critic_params),
            "actor_opt_state": offline._tree_to_numpy(state.actor_opt_state),
            "critic_opt_state": offline._tree_to_numpy(state.critic_opt_state),
            "rng": offline._tree_to_numpy(state.rng),
            "global_step": int(jax.device_get(state.global_step)),
            "actor_version": actor_version,
        },
    }
    with (path / "checkpoint.pkl").open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    offline._save_actor_snapshot(path / "actor_snapshot.pkl", actor_version, rl_config, state.actor_params)
    offline._save_critic_snapshot(path / "critic_snapshot.pkl", actor_version, rl_config, state.critic_params)
    atomic_write_json(
        path / "checkpoint.json", {"critic_updates": step, "actor_updates": actor_version, "path": str(path)}
    )
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_plots(rows: list[dict[str, Any]], analysis_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    steps = [int(row["critic_updates"]) for row in rows]
    specs = [
        (
            "critic_train_val_loss.png",
            "Critic train/validation TD loss",
            ("train_critic_val_td_loss", "val_critic_val_td_loss"),
        ),
        ("q_mc_correlation.png", "Q versus Monte-Carlo return", ("val_q_vs_mc_pearson", "val_q_vs_mc_spearman")),
        ("q_success_failure.png", "Success/failure Q separation", ("val_q_success_mean", "val_q_failure_mean")),
        ("twin_q_gap.png", "Twin-Q disagreement", ("val_q1_q2_gap_mean", "val_q1_q2_gap_p90")),
        ("actor_action_mse.png", "Actor validation action MSE", ("val_actor_val_action_mse",)),
        (
            "actor_reference_delta.png",
            "Actor deviation from VLA reference",
            ("val_actor_delta_mean", "val_actor_delta_p90", "val_actor_delta_max"),
        ),
        (
            "actor_temporal_smoothness.png",
            "Actor temporal action delta",
            ("val_reference_step_delta_mean", "val_actor_step_delta_mean", "val_actor_stochastic_step_delta_mean"),
        ),
        ("q_actor_vs_reference.png", "Q(actor) versus Q(reference)", ("val_q_actor_mean", "val_q_ref_mean")),
    ]
    paths: list[Path] = []
    for filename, title, keys in specs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key in keys:
            values = [row.get(key, float("nan")) for row in rows]
            ax.plot(steps, values, marker="o", label=key.removeprefix("val_").removeprefix("train_"))
        ax.set_title(title)
        ax.set_xlabel("critic updates")
        ax.grid(True, alpha=0.25)
        if len(keys) > 1:
            ax.legend()
        fig.tight_layout()
        path = analysis_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _select_checkpoint(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    def _is_true(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes"}

    candidates = [
        row
        for row in rows
        # Evaluation metrics are namespaced under ``val_`` when rows are
        # assembled in ``main``.  The CSV writer also serializes booleans as
        # strings, so parse both representations explicitly.
        if _is_true(row.get("val_valid_checkpoint", row.get("valid_checkpoint", False)))
        and np.isfinite(float(row.get("val_q_vs_mc_spearman", np.nan)))
        and np.isfinite(float(row.get("val_q_vs_mc_pearson", np.nan)))
    ]
    if not candidates:
        return None, "all Stage-2 checkpoints failed finite/divergence/OOD safety checks"
    best_spearman = max(float(row["val_q_vs_mc_spearman"]) for row in candidates)
    candidates = [row for row in candidates if float(row["val_q_vs_mc_spearman"]) >= best_spearman - 0.02]
    best_pearson = max(float(row["val_q_vs_mc_pearson"]) for row in candidates)
    candidates = [row for row in candidates if float(row["val_q_vs_mc_pearson"]) >= best_pearson - 0.03]
    min_td = min(float(row["val_critic_val_td_loss"]) for row in candidates)
    near_td = [row for row in candidates if float(row["val_critic_val_td_loss"]) <= min_td * 1.20 + 1e-8]
    candidates = near_td or candidates
    smoothness_values = [
        float(row.get("val_actor_step_delta_ratio", np.nan))
        for row in candidates
        if np.isfinite(float(row.get("val_actor_step_delta_ratio", np.nan)))
    ]
    if smoothness_values:
        best_smoothness = min(smoothness_values)
        candidates = [
            row
            for row in candidates
            if float(row.get("val_actor_step_delta_ratio", np.inf)) <= best_smoothness * 1.10 + 1e-8
        ]
    selected = sorted(candidates, key=lambda row: int(row["critic_updates"]))[0]
    if smoothness_values:
        reason = (
            "highest Q/MC ranking within Pearson/Spearman tolerance, stable TD loss, and near-best actor "
            "temporal smoothness; earliest near-equivalent checkpoint"
        )
    else:
        reason = (
            "highest Q/MC ranking within Pearson/Spearman tolerance and stable TD loss; "
            "earliest near-equivalent checkpoint (temporal smoothness metric unavailable)"
        )
    return selected, reason


def main() -> None:
    args = _parse_args()
    replay_path = args.replay_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    analysis_dir = output_root / "analysis"
    if args.overwrite and output_root.exists():
        try:
            replay_path.relative_to(output_root)
        except ValueError:
            shutil.rmtree(output_root)
        else:
            raise ValueError("Refusing --overwrite because the replay journal is inside output-root")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if not replay_path.is_file():
        raise FileNotFoundError(f"Replay journal not found: {replay_path}")

    records = _load_records(replay_path)
    replay_manifest = _validate_records(records, replay_path)
    atomic_write_json(analysis_dir / "stage2_replay_manifest.json", replay_manifest)
    train_records, val_records, split = _episode_split(records, seed=args.seed, output_root=output_root)
    system = load_system_config_yaml(str(args.config_path.expanduser().resolve()))
    rl_config = resolve_rl_config_paths(system.rl, str(args.config_path.expanduser().resolve()), require_exists=True)
    if args.reference_dropout_prob is not None:
        rl_config = dataclasses.replace(rl_config, reference_dropout_prob=args.reference_dropout_prob)
    if args.actor_residual_scale is not None:
        rl_config = dataclasses.replace(rl_config, actor_residual_scale=args.actor_residual_scale)
    if args.target_actor_deterministic:
        rl_config = dataclasses.replace(rl_config, target_actor_deterministic=True)
    if rl_config.actor_residual_scale > 0.0 and rl_config.reference_dropout_prob > 0.0:
        raise ValueError("Residual actor requires reference_dropout_prob=0 because the reference is its identity path")
    if args.num_devices < 1:
        raise ValueError("--num-devices must be positive")
    available_devices = tuple(jax.local_devices())
    if len(available_devices) < args.num_devices:
        raise RuntimeError(
            f"Requested {args.num_devices} devices, but JAX sees only {len(available_devices)}: {available_devices}"
        )
    devices = available_devices[: args.num_devices]
    if args.batch_size % args.num_devices != 0:
        raise ValueError(f"Batch size {args.batch_size} must be divisible by {args.num_devices} devices")
    print(
        json.dumps(
            {
                "training_parallelism": "single_device" if args.num_devices == 1 else "synchronous_data_parallel_pmap",
                "num_devices": args.num_devices,
                "per_device_batch_size": args.batch_size // args.num_devices,
                "devices": [str(device) for device in devices],
            }
        ),
        flush=True,
    )
    adapter = ActionRepresentationAdapter.from_config(rl_config)
    if adapter is None:
        raise RuntimeError("Stage-2 requires the configured action normalization statistics")
    train_ds = _records_to_arrays(train_records, rl_config.gamma)
    val_ds = _records_to_arrays(val_records, rl_config.gamma)
    # The recorded replay is the warmup/offline dataset.  Reuse the exact
    # weights from the task config (the previous reproduction used 10.0/0.1),
    # rather than silently falling back to the standalone script defaults.
    bc_weight = float(rl_config.warmup_bc_weight if args.bc_weight is None else args.bc_weight)
    q_weight = float(rl_config.warmup_q_weight if args.q_weight is None else args.q_weight)
    delta_weight = float(rl_config.delta_weight if args.delta_weight is None else args.delta_weight)
    # Checkpoints must carry the effective values used by this run; the online
    # learner restores these fields when training continues after deployment.
    rl_config = dataclasses.replace(
        rl_config,
        warmup_bc_weight=bc_weight,
        warmup_q_weight=q_weight,
        delta_weight=delta_weight,
    )
    atomic_write_json(
        analysis_dir / "stage2_training_config.json",
        {
            "rl_config": dataclasses.asdict(rl_config),
            "bc_weight": bc_weight,
            "q_weight": q_weight,
            "delta_weight": delta_weight,
            "actor_q_start_step": args.actor_q_start_step,
            "batch_size": args.batch_size,
            "num_devices": args.num_devices,
            "per_device_batch_size": args.batch_size // args.num_devices,
            "devices": [str(device) for device in devices],
            "parallelism": "single_device" if args.num_devices == 1 else "synchronous_data_parallel_pmap",
            "steps": args.steps,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "replay_manifest": replay_manifest,
            "episode_split": split,
        },
    )

    run = init_wandb(
        project="rlt-geniesim-stack-three-blocks-reproduction",
        name=args.wandb_run_name,
        config={
            "stage": 2,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "num_devices": args.num_devices,
            "per_device_batch_size": args.batch_size // args.num_devices,
            "devices": [str(device) for device in devices],
            "parallelism": "single_device" if args.num_devices == 1 else "synchronous_data_parallel_pmap",
            "seed": args.seed,
            "bc_weight": bc_weight,
            "q_weight": q_weight,
            "delta_weight": delta_weight,
            "actor_q_start_step": args.actor_q_start_step,
            "rl_config": dataclasses.asdict(rl_config),
            "replay_manifest": replay_manifest,
            "episode_split": split,
        },
        output_dir=output_root,
    )
    atomic_write_json(
        analysis_dir / "wandb_stage2_run.json",
        {"id": run.id, "name": run.name, "project": run.project, "entity": run.entity, "url": run.url},
    )
    log_artifact(
        run,
        name="rlt-stage2-replay",
        artifact_type="dataset",
        files=[replay_path, analysis_dir / "stage2_replay_manifest.json", analysis_dir / "stage2_episode_split.json"],
        metadata=replay_manifest,
    )

    actor_critic = init_train_state(rl_config, rng=jax.random.PRNGKey(args.seed))
    state, actor, critic = actor_critic
    action_q01 = jnp.asarray(adapter.stats.q01, dtype=jnp.float32)
    action_q99 = jnp.asarray(adapter.stats.q99, dtype=jnp.float32)
    train_step = offline._make_train_step(
        actor,
        critic,
        rl_config,
        bc_weight=bc_weight,
        q_weight=q_weight,
        delta_weight=delta_weight,
        disable_ref_input=False,
        use_action_adapter=True,
        action_q01=action_q01,
        action_q99=action_q99,
        actor_q_start_step=args.actor_q_start_step,
        axis_name="devices" if args.num_devices > 1 else None,
        devices=devices if args.num_devices > 1 else None,
    )
    if args.num_devices > 1:
        state = _replicate_train_state(state, devices)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    metrics_path = analysis_dir / "stage2_training_metrics.jsonl"
    for step in range(1, args.steps + 1):
        batch_np = adapter.prepare_training_batch(_sample_batch(train_ds, batch_size=args.batch_size, rng=rng))
        batch = (
            _shard_batch(batch_np, devices)
            if args.num_devices > 1
            else {key: jnp.asarray(value) for key, value in batch_np.items()}
        )
        state, raw = train_step(state, batch)
        if step % args.eval_every != 0 and step != args.steps:
            continue
        eval_state = _unreplicate(state) if args.num_devices > 1 else state
        raw_host = _unreplicate(raw) if args.num_devices > 1 else raw
        raw_metrics = {key: float(value) for key, value in jax.device_get(raw_host).items()}
        train_eval = _train_metrics(
            actor,
            critic,
            eval_state,
            adapter,
            train_ds,
            rl_config,
            batch_size=args.batch_size,
            num_batches=args.num_eval_batches,
        )
        val_eval = _train_metrics(
            actor,
            critic,
            eval_state,
            adapter,
            val_ds,
            rl_config,
            batch_size=args.batch_size,
            num_batches=args.num_eval_batches,
        )
        actor_updates = int(jax.device_get(eval_state.actor_version))
        row: dict[str, Any] = {"critic_updates": step, "actor_updates": actor_updates, **raw_metrics}
        row.update({f"train_{key}": value for key, value in train_eval.items()})
        row.update({f"val_{key}": value for key, value in val_eval.items()})
        row["critic_pearson_generalization_gap"] = row["train_q_vs_mc_pearson"] - row["val_q_vs_mc_pearson"]
        row["critic_spearman_generalization_gap"] = row["train_q_vs_mc_spearman"] - row["val_q_vs_mc_spearman"]
        row["checkpoint_label"] = f"C{step:05d}_A{actor_updates:05d}" if step in {10_000, 20_000, 30_000} else ""
        rows.append(row)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        run.log({key: value for key, value in row.items() if isinstance(value, (int, float, bool))}, step=step)
        atomic_write_json(
            analysis_dir / "stage2_training_status.json", {"latest": row, "steps": step, "target_steps": args.steps}
        )
        _save_stage2_checkpoint(output_root, eval_state, rl_config, step, split, latest=True)
        if step in {10_000, 20_000, 30_000}:
            _save_stage2_checkpoint(output_root, eval_state, rl_config, step, split)
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "critic_updates",
                        "actor_updates",
                        "val_critic_val_td_loss",
                        "val_q_vs_mc_spearman",
                        "val_q_success_minus_failure",
                        "val_actor_delta_p90",
                    )
                },
                ensure_ascii=False,
            )
        )

    _write_csv(analysis_dir / "stage2_training_metrics.csv", rows)
    checkpoint_rows = [row for row in rows if row.get("checkpoint_label")]
    _write_csv(analysis_dir / "stage2_checkpoint_comparison.csv", checkpoint_rows)
    if not checkpoint_rows and args.steps >= 30_000:
        raise RuntimeError("Expected C10k/A5k, C20k/A10k and C30k/A15k rows but none were produced")
    plot_paths = _write_plots(rows, analysis_dir)
    if not checkpoint_rows:
        run.finish()
        print(
            json.dumps(
                {"diagnostic_run": True, "steps": args.steps, "plots": [str(path) for path in plot_paths]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    selected, reason = _select_checkpoint(checkpoint_rows)
    if selected is None:
        (output_root / "FAILED_STAGE2.md").write_text("# Stage-2 failed\n\n" + reason + "\n", encoding="utf-8")
        run.finish()
        raise RuntimeError(reason)
    selection = {
        "checkpoint_label": selected["checkpoint_label"],
        "critic_updates": int(selected["critic_updates"]),
        "actor_updates": int(selected["actor_updates"]),
        "checkpoint": str(output_root / "checkpoints" / selected["checkpoint_label"]),
        "reason": reason,
    }
    atomic_write_json(analysis_dir / "selected_stage2_checkpoint.json", selection)
    (analysis_dir / "selected_stage2_checkpoint.txt").write_text(
        f"checkpoint_label={selection['checkpoint_label']}\ncritic_updates={selection['critic_updates']}\n"
        f"actor_updates={selection['actor_updates']}\ncheckpoint={selection['checkpoint']}\nreason={reason}\n",
        encoding="utf-8",
    )
    for path in plot_paths:
        run.log({path.stem: __import__("wandb").Image(str(path))})
    log_artifact(
        run,
        name="rlt-stage2-diagnostics",
        artifact_type="evaluation",
        files=[
            analysis_dir / "stage2_training_metrics.csv",
            analysis_dir / "stage2_checkpoint_comparison.csv",
            analysis_dir / "selected_stage2_checkpoint.txt",
            *plot_paths,
        ],
        metadata=selection,
    )
    run.summary.update(selection)
    run.finish()
    print(
        json.dumps({"selection": selection, "plots": [str(path) for path in plot_paths]}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
