from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Any

from _common import PHASE_CHOICES
from _common import SOURCE_CHOICES
from _common import ActionRepresentationAdapter
from _common import build_model_ref_chunk
from _common import filter_replay_records
from _common import infer_task_dir_from_replay_path
from _common import load_critic_snapshot
from _common import load_replay_journal
from _common import load_snapshot
from _common import resolve_default_actor_checkpoint_path
from _common import resolve_default_critic_snapshot_path
import jax
import jax.numpy as jnp
import numpy as np

from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.networks import TwinCritic
from rlt_online_rl.networks import conservative_value_quantile_from_logits_pair

SOURCE_HUMAN = 2
SOURCE_MIXED = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate task-Q, distributional value, and autonomy diagnostics on replay data."
    )
    parser.add_argument("--replay-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--actor-path", type=Path, default=None)
    parser.add_argument("--critic-path", type=Path, default=None)
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="online")
    parser.add_argument("--source", choices=tuple(SOURCE_CHOICES), default="all")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _build_models(rl_config) -> tuple[ChunkActor, TwinCritic]:
    actor = ChunkActor(
        z_dim=rl_config.z_dim,
        proprio_dim=rl_config.proprio_dim,
        chunk_len=rl_config.chunk_len,
        action_dim=rl_config.action_dim,
        hidden_dim=rl_config.actor_hidden_dim,
        num_layers=rl_config.actor_num_layers,
        fixed_std=rl_config.fixed_std,
        residual_scale=rl_config.actor_residual_scale,
    )
    critic = TwinCritic(
        z_dim=rl_config.z_dim,
        proprio_dim=rl_config.proprio_dim,
        chunk_len=rl_config.chunk_len,
        action_dim=rl_config.action_dim,
        hidden_dim=rl_config.critic_hidden_dim,
        num_layers=rl_config.critic_num_layers,
        critic_mode=rl_config.critic_mode,
        value_num_atoms=rl_config.value_num_atoms,
    )
    return actor, critic


def _build_step_source_maps(records: list[dict[str, Any]]) -> dict[int, dict[int, int]]:
    step_sources_by_episode: dict[int, dict[int, int]] = {}
    for record in records:
        chunk_len = int(np.asarray(record["ref_chunk"]).shape[0])
        episode_id = int(record["episode_id"])
        step_id = int(record["step_id"])
        source_chunk = np.asarray(
            record.get("source_chunk", np.full((chunk_len,), int(record["source"]), dtype=np.uint8)),
            dtype=np.uint8,
        )
        step_sources = step_sources_by_episode.setdefault(episode_id, {})
        for offset, source_value in enumerate(source_chunk):
            step_sources.setdefault(step_id + offset, int(source_value))
    return step_sources_by_episode


def _next_source_chunk(step_sources_by_episode: dict[int, dict[int, int]], record: dict[str, Any]) -> np.ndarray:
    if "next_source_chunk" in record:
        return np.asarray(record["next_source_chunk"], dtype=np.uint8)
    chunk_len = int(np.asarray(record["ref_chunk"]).shape[0])
    episode_id = int(record["episode_id"])
    step_id = int(record["step_id"])
    step_sources = step_sources_by_episode.get(episode_id, {})
    return np.asarray(
        [step_sources.get(step_id + chunk_len + offset, 0) for offset in range(chunk_len)], dtype=np.uint8
    )


def _human_ratio(source_chunk: np.ndarray) -> np.ndarray:
    source_chunk = np.asarray(source_chunk, dtype=np.uint8)
    return np.mean(
        np.logical_or(source_chunk == SOURCE_HUMAN, source_chunk == SOURCE_MIXED).astype(np.float32), axis=-1
    )


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    comparison = pos[:, None] - neg[None, :]
    return float(np.mean((comparison > 0.0).astype(np.float32) + 0.5 * (comparison == 0.0).astype(np.float32)))


def _normalize_action_chunk(
    adapter: ActionRepresentationAdapter | None,
    action_chunk_abs: np.ndarray,
    proprio: np.ndarray,
) -> np.ndarray:
    if adapter is None:
        return np.asarray(action_chunk_abs, dtype=np.float32)
    return adapter.normalize_chunk(action_chunk_abs, proprio)


def _success_diff_and_z(episode_values: dict[int, float], episode_success: dict[int, int]) -> dict[str, float]:
    success_values = np.asarray(
        [value for eid, value in episode_values.items() if episode_success.get(eid, 0) > 0], dtype=np.float32
    )
    failure_values = np.asarray(
        [value for eid, value in episode_values.items() if episode_success.get(eid, 0) <= 0], dtype=np.float32
    )
    if success_values.size == 0 or failure_values.size == 0:
        return {
            "diff": float("nan"),
            "z": float("nan"),
            "success_count": float(success_values.size),
            "failure_count": float(failure_values.size),
        }
    diff = float(success_values.mean() - failure_values.mean())
    success_var = float(np.var(success_values, ddof=1)) if success_values.size > 1 else 0.0
    failure_var = float(np.var(failure_values, ddof=1)) if failure_values.size > 1 else 0.0
    se = float(np.sqrt(success_var / max(success_values.size, 1) + failure_var / max(failure_values.size, 1)))
    z = float(diff / se) if se > 1e-8 else float("nan")
    return {
        "diff": diff,
        "z": z,
        "success_count": float(success_values.size),
        "failure_count": float(failure_values.size),
    }


def _predict_arrays(
    actor: ChunkActor,
    critic: TwinCritic,
    actor_params: Any,
    critic_params: Any,
    adapter: ActionRepresentationAdapter | None,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    value_min: float,
    value_max: float,
    value_tau_base: float,
    value_tau_alpha: float,
    value_tau_min: float,
    value_tau_max: float,
) -> dict[str, np.ndarray]:
    q_data: list[np.ndarray] = []
    q_ref: list[np.ndarray] = []
    q_actor: list[np.ndarray] = []
    auto_q_data: list[np.ndarray] = []
    auto_q_ref: list[np.ndarray] = []
    auto_q_actor: list[np.ndarray] = []
    q_gap: list[np.ndarray] = []
    actor_ref_delta: list[np.ndarray] = []
    value_data_quantile: list[np.ndarray] = []
    value_ref_quantile: list[np.ndarray] = []
    value_actor_quantile: list[np.ndarray] = []
    value_data_entropy: list[np.ndarray] = []
    value_ref_entropy: list[np.ndarray] = []
    value_actor_entropy: list[np.ndarray] = []
    value_data_tau: list[np.ndarray] = []
    value_ref_tau: list[np.ndarray] = []
    value_actor_tau: list[np.ndarray] = []
    value_gap: list[np.ndarray] = []

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        z_rl = np.stack([np.asarray(record["z_rl"], dtype=np.float32) for record in batch], axis=0)
        proprio = np.stack([np.asarray(record["proprio"], dtype=np.float32) for record in batch], axis=0)
        ref_abs = np.stack([np.asarray(record["ref_chunk"], dtype=np.float32) for record in batch], axis=0)
        action_abs = np.stack([np.asarray(record["action_chunk"], dtype=np.float32) for record in batch], axis=0)
        ref_model = build_model_ref_chunk(adapter, ref_abs, proprio, disable_ref_input=False)
        action_model = _normalize_action_chunk(adapter, action_abs, proprio)

        actor_mean = actor.actor_mean(
            actor_params,
            jnp.asarray(z_rl),
            jnp.asarray(proprio),
            jnp.asarray(ref_model),
        )
        data_q1, data_q2 = critic.q_values(
            critic_params, jnp.asarray(z_rl), jnp.asarray(proprio), jnp.asarray(ref_model), jnp.asarray(action_model)
        )
        ref_q1, ref_q2 = critic.q_values(
            critic_params, jnp.asarray(z_rl), jnp.asarray(proprio), jnp.asarray(ref_model), jnp.asarray(ref_model)
        )
        actor_q1, actor_q2 = critic.q_values(
            critic_params, jnp.asarray(z_rl), jnp.asarray(proprio), jnp.asarray(ref_model), actor_mean
        )
        auto_data_q1, auto_data_q2 = critic.auto_q_values(
            critic_params,
            jnp.asarray(z_rl),
            jnp.asarray(proprio),
            jnp.asarray(ref_model),
            jnp.asarray(action_model),
        )
        auto_ref_q1, auto_ref_q2 = critic.auto_q_values(
            critic_params,
            jnp.asarray(z_rl),
            jnp.asarray(proprio),
            jnp.asarray(ref_model),
            jnp.asarray(ref_model),
        )
        auto_actor_q1, auto_actor_q2 = critic.auto_q_values(
            critic_params,
            jnp.asarray(z_rl),
            jnp.asarray(proprio),
            jnp.asarray(ref_model),
            actor_mean,
        )
        value_data_logits1, value_data_logits2 = critic.value_logits_pair(
            critic_params,
            jnp.asarray(z_rl),
            jnp.asarray(proprio),
            jnp.asarray(ref_model),
            jnp.asarray(action_model),
        )
        value_ref_logits1, value_ref_logits2 = critic.value_logits_pair(
            critic_params,
            jnp.asarray(z_rl),
            jnp.asarray(proprio),
            jnp.asarray(ref_model),
            jnp.asarray(ref_model),
        )
        value_actor_logits1, value_actor_logits2 = critic.value_logits_pair(
            critic_params,
            jnp.asarray(z_rl),
            jnp.asarray(proprio),
            jnp.asarray(ref_model),
            actor_mean,
        )
        value_data_q, value_data_h, value_data_t, _ = conservative_value_quantile_from_logits_pair(
            value_data_logits1,
            value_data_logits2,
            value_min=value_min,
            value_max=value_max,
            tau_base=value_tau_base,
            tau_alpha=value_tau_alpha,
            tau_min=value_tau_min,
            tau_max=value_tau_max,
        )
        value_ref_q, value_ref_h, value_ref_t, _ = conservative_value_quantile_from_logits_pair(
            value_ref_logits1,
            value_ref_logits2,
            value_min=value_min,
            value_max=value_max,
            tau_base=value_tau_base,
            tau_alpha=value_tau_alpha,
            tau_min=value_tau_min,
            tau_max=value_tau_max,
        )
        value_actor_q, value_actor_h, value_actor_t, _ = conservative_value_quantile_from_logits_pair(
            value_actor_logits1,
            value_actor_logits2,
            value_min=value_min,
            value_max=value_max,
            tau_base=value_tau_base,
            tau_alpha=value_tau_alpha,
            tau_min=value_tau_min,
            tau_max=value_tau_max,
        )

        q1_np = np.asarray(jax.device_get(data_q1), dtype=np.float32)
        q2_np = np.asarray(jax.device_get(data_q2), dtype=np.float32)
        q_data.append(np.minimum(q1_np, q2_np))
        q_ref.append(
            np.minimum(
                np.asarray(jax.device_get(ref_q1), dtype=np.float32),
                np.asarray(jax.device_get(ref_q2), dtype=np.float32),
            )
        )
        q_actor.append(
            np.minimum(
                np.asarray(jax.device_get(actor_q1), dtype=np.float32),
                np.asarray(jax.device_get(actor_q2), dtype=np.float32),
            )
        )
        auto_q_data.append(
            np.minimum(
                np.asarray(jax.device_get(auto_data_q1), dtype=np.float32),
                np.asarray(jax.device_get(auto_data_q2), dtype=np.float32),
            )
        )
        auto_q_ref.append(
            np.minimum(
                np.asarray(jax.device_get(auto_ref_q1), dtype=np.float32),
                np.asarray(jax.device_get(auto_ref_q2), dtype=np.float32),
            )
        )
        auto_q_actor.append(
            np.minimum(
                np.asarray(jax.device_get(auto_actor_q1), dtype=np.float32),
                np.asarray(jax.device_get(auto_actor_q2), dtype=np.float32),
            )
        )
        q_gap.append(np.abs(q1_np - q2_np))
        value_data_quantile.append(np.asarray(jax.device_get(value_data_q), dtype=np.float32))
        value_ref_quantile.append(np.asarray(jax.device_get(value_ref_q), dtype=np.float32))
        value_actor_quantile.append(np.asarray(jax.device_get(value_actor_q), dtype=np.float32))
        value_data_entropy.append(np.asarray(jax.device_get(value_data_h), dtype=np.float32))
        value_ref_entropy.append(np.asarray(jax.device_get(value_ref_h), dtype=np.float32))
        value_actor_entropy.append(np.asarray(jax.device_get(value_actor_h), dtype=np.float32))
        value_data_tau.append(np.asarray(jax.device_get(value_data_t), dtype=np.float32))
        value_ref_tau.append(np.asarray(jax.device_get(value_ref_t), dtype=np.float32))
        value_actor_tau.append(np.asarray(jax.device_get(value_actor_t), dtype=np.float32))
        value_gap.append(
            np.abs(
                np.asarray(jax.device_get(value_actor_logits1), dtype=np.float32)
                - np.asarray(jax.device_get(value_actor_logits2), dtype=np.float32)
            ).mean(axis=-1)
        )

        actor_mean_np = np.asarray(jax.device_get(actor_mean), dtype=np.float32)
        if adapter is not None:
            actor_mean_np = adapter.denormalize_to_abs_chunk(actor_mean_np, proprio)
        actor_ref_delta.append(np.abs(actor_mean_np - ref_abs).reshape(len(batch), -1).mean(axis=-1))

    return {
        "q_data": np.concatenate(q_data, axis=0),
        "q_ref": np.concatenate(q_ref, axis=0),
        "q_actor": np.concatenate(q_actor, axis=0),
        "auto_q_data": np.concatenate(auto_q_data, axis=0),
        "auto_q_ref": np.concatenate(auto_q_ref, axis=0),
        "auto_q_actor": np.concatenate(auto_q_actor, axis=0),
        "q_gap": np.concatenate(q_gap, axis=0),
        "actor_ref_delta": np.concatenate(actor_ref_delta, axis=0),
        "value_quantile": np.concatenate(value_actor_quantile, axis=0),
        "value_entropy": np.concatenate(value_actor_entropy, axis=0),
        "value_tau": np.concatenate(value_actor_tau, axis=0),
        "value_data_quantile": np.concatenate(value_data_quantile, axis=0),
        "value_ref_quantile": np.concatenate(value_ref_quantile, axis=0),
        "value_actor_quantile": np.concatenate(value_actor_quantile, axis=0),
        "value_data_entropy": np.concatenate(value_data_entropy, axis=0),
        "value_ref_entropy": np.concatenate(value_ref_entropy, axis=0),
        "value_actor_entropy": np.concatenate(value_actor_entropy, axis=0),
        "value_data_tau": np.concatenate(value_data_tau, axis=0),
        "value_ref_tau": np.concatenate(value_ref_tau, axis=0),
        "value_actor_tau": np.concatenate(value_actor_tau, axis=0),
        "value_logits_gap": np.concatenate(value_gap, axis=0),
    }


def _apply_split(records: list[dict[str, Any]], *, split: str, val_ratio: float, seed: int) -> list[dict[str, Any]]:
    if split == "all":
        return records
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(len(records) * val_ratio))
    selected = indices[:val_size] if split == "val" else indices[val_size:]
    return [records[index] for index in selected]


def main() -> None:
    args = _parse_args()
    replay_path = args.replay_path.resolve()
    task_dir = infer_task_dir_from_replay_path(replay_path)
    model_dir = args.model_dir.resolve() if args.model_dir is not None else task_dir
    actor_path = (
        args.actor_path.resolve() if args.actor_path is not None else resolve_default_actor_checkpoint_path(model_dir)
    )
    critic_path = (
        args.critic_path.resolve() if args.critic_path is not None else resolve_default_critic_snapshot_path(model_dir)
    )

    all_records = load_replay_journal(replay_path)
    step_sources_by_episode = _build_step_source_maps(all_records)
    full_episode_success: dict[int, int] = {}
    for record in all_records:
        episode_id = int(record["episode_id"])
        full_episode_success[episode_id] = max(full_episode_success.get(episode_id, 0), int(record["success"]))
    filtered_records = filter_replay_records(all_records, phase=args.phase, source=args.source)
    records = _apply_split(filtered_records, split=args.split, val_ratio=args.val_ratio, seed=args.seed)
    if not records:
        raise RuntimeError(
            f"No replay records left after filtering phase={args.phase}, source={args.source}, split={args.split}."
        )

    actor_cfg, actor_params = load_snapshot(actor_path, model_dir)
    critic_cfg, critic_params = load_critic_snapshot(critic_path, model_dir)
    if actor_cfg != critic_cfg:
        raise ValueError("Actor and critic snapshots were saved with different rl_config values.")
    rl_config = actor_cfg
    adapter = ActionRepresentationAdapter.from_config(rl_config)
    actor, critic = _build_models(rl_config)
    payload = _predict_arrays(
        actor,
        critic,
        actor_params,
        critic_params,
        adapter,
        records,
        batch_size=args.batch_size,
        value_min=rl_config.value_min,
        value_max=rl_config.value_max,
        value_tau_base=rl_config.value_tau_base,
        value_tau_alpha=rl_config.value_tau_alpha,
        value_tau_min=rl_config.value_tau_min,
        value_tau_max=rl_config.value_tau_max,
    )

    source_chunk = np.stack(
        [
            np.asarray(
                record.get(
                    "source_chunk",
                    np.full((np.asarray(record["ref_chunk"]).shape[0],), int(record["source"]), dtype=np.uint8),
                ),
                dtype=np.uint8,
            )
            for record in records
        ],
        axis=0,
    )
    next_source_chunk = np.stack([_next_source_chunk(step_sources_by_episode, record) for record in records], axis=0)
    human_ratio = _human_ratio(source_chunk)
    next_human_ratio = _human_ratio(next_source_chunk)
    threshold = float(rl_config.human_correction_min_ratio)
    human_mask = human_ratio >= threshold
    autonomous_mask = human_ratio < threshold
    next_intervention = next_human_ratio >= threshold

    episode_success: dict[int, int] = {}
    episode_q_actor_last3: dict[int, float] = {}
    episode_q_data_last3: dict[int, float] = {}
    episode_auto_actor_last3: dict[int, float] = {}
    episode_auto_data_last3: dict[int, float] = {}
    episode_value_actor_last3: dict[int, float] = {}
    episode_value_data_last3: dict[int, float] = {}
    episode_value_ref_last3: dict[int, float] = {}
    for episode_id in sorted({int(record["episode_id"]) for record in records}):
        indices = [idx for idx, record in enumerate(records) if int(record["episode_id"]) == episode_id]
        indices.sort(key=lambda idx: int(records[idx]["step_id"]))
        last = np.asarray(indices[-min(3, len(indices)) :], dtype=np.int32)
        episode_success[episode_id] = int(
            full_episode_success.get(episode_id, max(int(records[idx]["success"]) for idx in indices))
        )
        episode_q_actor_last3[episode_id] = float(payload["q_actor"][last].mean())
        episode_q_data_last3[episode_id] = float(payload["q_data"][last].mean())
        episode_auto_actor_last3[episode_id] = float(payload["auto_q_actor"][last].mean())
        episode_auto_data_last3[episode_id] = float(payload["auto_q_data"][last].mean())
        episode_value_actor_last3[episode_id] = float(payload["value_actor_quantile"][last].mean())
        episode_value_data_last3[episode_id] = float(payload["value_data_quantile"][last].mean())
        episode_value_ref_last3[episode_id] = float(payload["value_ref_quantile"][last].mean())

    q_actor_success = _success_diff_and_z(episode_q_actor_last3, episode_success)
    q_data_success = _success_diff_and_z(episode_q_data_last3, episode_success)
    auto_actor_success = _success_diff_and_z(episode_auto_actor_last3, episode_success)
    auto_data_success = _success_diff_and_z(episode_auto_data_last3, episode_success)
    value_actor_success = _success_diff_and_z(episode_value_actor_last3, episode_success)
    value_data_success = _success_diff_and_z(episode_value_data_last3, episode_success)
    value_ref_success = _success_diff_and_z(episode_value_ref_last3, episode_success)
    auto_labels = next_intervention[autonomous_mask]
    auto_actor_q = payload["auto_q_actor"][autonomous_mask]
    auto_data_q = payload["auto_q_data"][autonomous_mask]
    auto_ref_q = payload["auto_q_ref"][autonomous_mask]
    task_actor_q = payload["q_actor"][autonomous_mask]
    task_data_q = payload["q_data"][autonomous_mask]
    task_ref_q = payload["q_ref"][autonomous_mask]
    task_actor_minus_data = task_actor_q - task_data_q
    task_actor_minus_ref = task_actor_q - task_ref_q
    task_data_minus_ref = task_data_q - task_ref_q
    auto_actor_minus_ref = auto_actor_q - auto_ref_q
    auto_data_minus_ref = auto_data_q - auto_ref_q
    auto_actor_minus_data = auto_actor_q - auto_data_q
    auto_actor_relative_penalty = np.maximum(auto_ref_q - auto_actor_q, 0.0)
    auto_data_relative_penalty = np.maximum(auto_ref_q - auto_data_q, 0.0)
    metrics = {
        "model_dir": os.path.relpath(str(model_dir), start=str(Path.cwd())),
        "actor_path": os.path.relpath(str(actor_path), start=str(Path.cwd())),
        "critic_path": os.path.relpath(str(critic_path), start=str(Path.cwd())),
        "phase": args.phase,
        "source": args.source,
        "split": args.split,
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "num_records": int(len(records)),
        "num_episodes": int(len(episode_success)),
        "episode_success_count": int(sum(episode_success.values())),
        "episode_failure_count": int(len(episode_success) - sum(episode_success.values())),
        "human_chunk_ratio": float(np.mean(human_mask)),
        "autonomous_chunk_ratio": float(np.mean(autonomous_mask)),
        "next_intervention_ratio_in_autonomous": float(np.mean(auto_labels)) if auto_labels.size else float("nan"),
        "fit_actor_ref_mean_abs_delta": float(payload["actor_ref_delta"].mean()),
        "human_rank_acc": float(np.mean(payload["q_data"][human_mask] > payload["q_ref"][human_mask]))
        if np.any(human_mask)
        else float("nan"),
        "human_gap": float(np.mean(payload["q_data"][human_mask] - payload["q_ref"][human_mask]))
        if np.any(human_mask)
        else float("nan"),
        "q_data_last3_diff": q_data_success["diff"],
        "q_actor_last3_diff": q_actor_success["diff"],
        "q_actor_success_z": q_actor_success["z"],
        "q_data_success_z": q_data_success["z"],
        "auto_q_actor_last3_diff": auto_actor_success["diff"],
        "auto_q_actor_success_z": auto_actor_success["z"],
        "auto_q_data_last3_diff": auto_data_success["diff"],
        "auto_q_data_success_z": auto_data_success["z"],
        "value_quantile_last3_diff": value_actor_success["diff"],
        "value_quantile_success_z": value_actor_success["z"],
        "value_actor_last3_diff": value_actor_success["diff"],
        "value_actor_success_z": value_actor_success["z"],
        "value_data_last3_diff": value_data_success["diff"],
        "value_data_success_z": value_data_success["z"],
        "value_ref_last3_diff": value_ref_success["diff"],
        "value_ref_success_z": value_ref_success["z"],
        "value_quantile_mean": float(payload["value_quantile"].mean()),
        "value_quantile_std": float(payload["value_quantile"].std()),
        "value_entropy_mean": float(payload["value_entropy"].mean()),
        "value_entropy_std": float(payload["value_entropy"].std()),
        "value_tau_mean": float(payload["value_tau"].mean()),
        "value_tau_std": float(payload["value_tau"].std()),
        "value_logits_gap": float(payload["value_logits_gap"].mean()),
        "value_human_rank_acc": float(
            np.mean(payload["value_data_quantile"][human_mask] > payload["value_ref_quantile"][human_mask])
        )
        if np.any(human_mask)
        else float("nan"),
        "value_human_gap": float(
            np.mean(payload["value_data_quantile"][human_mask] - payload["value_ref_quantile"][human_mask])
        )
        if np.any(human_mask)
        else float("nan"),
        "value_actor_minus_ref_mean": float(np.mean(payload["value_actor_quantile"] - payload["value_ref_quantile"])),
        "value_data_minus_ref_mean": float(np.mean(payload["value_data_quantile"] - payload["value_ref_quantile"])),
        "value_warning_auc": _binary_auc(auto_labels, -payload["value_quantile"][autonomous_mask]),
        "value_actor_warning_auc": _binary_auc(auto_labels, -payload["value_actor_quantile"][autonomous_mask]),
        "value_data_warning_auc": _binary_auc(auto_labels, -payload["value_data_quantile"][autonomous_mask]),
        "value_entropy_warning_auc": _binary_auc(auto_labels, payload["value_entropy"][autonomous_mask]),
        "q_warning_auc": _binary_auc(auto_labels, -payload["q_actor"][autonomous_mask]),
        "q_data_warning_auc": _binary_auc(auto_labels, -task_data_q),
        "q_ref_warning_auc": _binary_auc(auto_labels, -task_ref_q),
        "q_actor_minus_ref_mean": float(np.mean(task_actor_minus_ref)) if task_actor_minus_ref.size else float("nan"),
        "q_data_minus_ref_mean": float(np.mean(task_data_minus_ref)) if task_data_minus_ref.size else float("nan"),
        "q_actor_minus_data_mean": float(np.mean(task_actor_minus_data))
        if task_actor_minus_data.size
        else float("nan"),
        "q_actor_boundary_mean": float(np.mean(task_actor_q[auto_labels])) if np.any(auto_labels) else float("nan"),
        "q_actor_clean_mean": float(np.mean(task_actor_q[~auto_labels])) if np.any(~auto_labels) else float("nan"),
        "q_actor_clean_minus_boundary": (
            float(np.mean(task_actor_q[~auto_labels]) - np.mean(task_actor_q[auto_labels]))
            if np.any(auto_labels) and np.any(~auto_labels)
            else float("nan")
        ),
        "q_data_boundary_mean": float(np.mean(task_data_q[auto_labels])) if np.any(auto_labels) else float("nan"),
        "q_data_clean_mean": float(np.mean(task_data_q[~auto_labels])) if np.any(~auto_labels) else float("nan"),
        "q_data_clean_minus_boundary": (
            float(np.mean(task_data_q[~auto_labels]) - np.mean(task_data_q[auto_labels]))
            if np.any(auto_labels) and np.any(~auto_labels)
            else float("nan")
        ),
        "q_actor_minus_data_boundary_mean": (
            float(np.mean(task_actor_minus_data[auto_labels])) if np.any(auto_labels) else float("nan")
        ),
        "q_actor_minus_data_clean_mean": (
            float(np.mean(task_actor_minus_data[~auto_labels])) if np.any(~auto_labels) else float("nan")
        ),
        "q_actor_better_than_data_boundary_rate": (
            float(np.mean(task_actor_minus_data[auto_labels] > 0.0)) if np.any(auto_labels) else float("nan")
        ),
        "q_actor_better_than_data_clean_rate": (
            float(np.mean(task_actor_minus_data[~auto_labels] > 0.0)) if np.any(~auto_labels) else float("nan")
        ),
        "auto_q_actor_warning_auc": _binary_auc(auto_labels, -auto_actor_q),
        "auto_q_data_warning_auc": _binary_auc(auto_labels, -auto_data_q),
        "auto_q_ref_warning_auc": _binary_auc(auto_labels, -auto_ref_q),
        "auto_q_actor_minus_ref_mean": float(np.mean(auto_actor_minus_ref))
        if auto_actor_minus_ref.size
        else float("nan"),
        "auto_q_data_minus_ref_mean": float(np.mean(auto_data_minus_ref)) if auto_data_minus_ref.size else float("nan"),
        "auto_q_actor_minus_data_mean": float(np.mean(auto_actor_minus_data))
        if auto_actor_minus_data.size
        else float("nan"),
        "auto_q_actor_relative_warning_auc": _binary_auc(auto_labels, -auto_actor_minus_ref),
        "auto_q_data_relative_warning_auc": _binary_auc(auto_labels, -auto_data_minus_ref),
        "auto_q_actor_relative_penalty_mean": float(np.mean(auto_actor_relative_penalty))
        if auto_actor_relative_penalty.size
        else float("nan"),
        "auto_q_actor_relative_penalty_rate": float(np.mean(auto_actor_relative_penalty > 0.0))
        if auto_actor_relative_penalty.size
        else float("nan"),
        "auto_q_data_relative_penalty_mean": float(np.mean(auto_data_relative_penalty))
        if auto_data_relative_penalty.size
        else float("nan"),
        "auto_q_data_relative_penalty_rate": float(np.mean(auto_data_relative_penalty > 0.0))
        if auto_data_relative_penalty.size
        else float("nan"),
        "auto_q_actor_boundary_mean": float(np.mean(auto_actor_q[auto_labels]))
        if np.any(auto_labels)
        else float("nan"),
        "auto_q_actor_clean_mean": float(np.mean(auto_actor_q[~auto_labels])) if np.any(~auto_labels) else float("nan"),
        "auto_q_actor_clean_minus_boundary": (
            float(np.mean(auto_actor_q[~auto_labels]) - np.mean(auto_actor_q[auto_labels]))
            if np.any(auto_labels) and np.any(~auto_labels)
            else float("nan")
        ),
        "auto_q_actor_relative_boundary_mean": float(np.mean(auto_actor_minus_ref[auto_labels]))
        if np.any(auto_labels)
        else float("nan"),
        "auto_q_actor_relative_clean_mean": float(np.mean(auto_actor_minus_ref[~auto_labels]))
        if np.any(~auto_labels)
        else float("nan"),
        "auto_q_actor_relative_clean_minus_boundary": (
            float(np.mean(auto_actor_minus_ref[~auto_labels]) - np.mean(auto_actor_minus_ref[auto_labels]))
            if np.any(auto_labels) and np.any(~auto_labels)
            else float("nan")
        ),
        "auto_q_actor_minus_data_boundary_mean": (
            float(np.mean(auto_actor_minus_data[auto_labels])) if np.any(auto_labels) else float("nan")
        ),
        "auto_q_actor_minus_data_clean_mean": (
            float(np.mean(auto_actor_minus_data[~auto_labels])) if np.any(~auto_labels) else float("nan")
        ),
        "auto_q_actor_better_than_data_boundary_rate": (
            float(np.mean(auto_actor_minus_data[auto_labels] > 0.0)) if np.any(auto_labels) else float("nan")
        ),
        "auto_q_actor_better_than_data_clean_rate": (
            float(np.mean(auto_actor_minus_data[~auto_labels] > 0.0)) if np.any(~auto_labels) else float("nan")
        ),
        "auto_q_actor_relative_penalty_boundary_mean": (
            float(np.mean(auto_actor_relative_penalty[auto_labels])) if np.any(auto_labels) else float("nan")
        ),
        "auto_q_actor_relative_penalty_clean_mean": (
            float(np.mean(auto_actor_relative_penalty[~auto_labels])) if np.any(~auto_labels) else float("nan")
        ),
        "auto_q_data_boundary_mean": float(np.mean(auto_data_q[auto_labels])) if np.any(auto_labels) else float("nan"),
        "auto_q_data_clean_mean": float(np.mean(auto_data_q[~auto_labels])) if np.any(~auto_labels) else float("nan"),
        "auto_q_data_clean_minus_boundary": (
            float(np.mean(auto_data_q[~auto_labels]) - np.mean(auto_data_q[auto_labels]))
            if np.any(auto_labels) and np.any(~auto_labels)
            else float("nan")
        ),
        "auto_q_data_relative_boundary_mean": float(np.mean(auto_data_minus_ref[auto_labels]))
        if np.any(auto_labels)
        else float("nan"),
        "auto_q_data_relative_clean_mean": float(np.mean(auto_data_minus_ref[~auto_labels]))
        if np.any(~auto_labels)
        else float("nan"),
        "auto_q_data_relative_clean_minus_boundary": (
            float(np.mean(auto_data_minus_ref[~auto_labels]) - np.mean(auto_data_minus_ref[auto_labels]))
            if np.any(auto_labels) and np.any(~auto_labels)
            else float("nan")
        ),
        "task_auto_actor_gap_boundary": (
            float(np.mean((payload["q_actor"][autonomous_mask] - auto_actor_q)[auto_labels]))
            if np.any(auto_labels)
            else float("nan")
        ),
        "task_auto_actor_gap_clean": (
            float(np.mean((payload["q_actor"][autonomous_mask] - auto_actor_q)[~auto_labels]))
            if np.any(~auto_labels)
            else float("nan")
        ),
        "q1_q2_gap": float(payload["q_gap"].mean()),
        "critic_mode": str(rl_config.critic_mode),
        "value_bootstrap": str(rl_config.value_bootstrap),
        "value_bootstrap_mix": float(rl_config.value_bootstrap_mix),
        "value_dist_weight": float(rl_config.value_dist_weight),
        "value_tau_alpha": float(rl_config.value_tau_alpha),
        "autonomy_value_weight": float(rl_config.autonomy_value_weight),
        "autonomy_actor_weight": float(rl_config.autonomy_actor_weight),
        "autonomy_warning_actor_weight": float(rl_config.autonomy_warning_actor_weight),
        "autonomy_actor_scope": str(rl_config.autonomy_actor_scope),
        "autonomy_actor_human_weight": float(rl_config.autonomy_actor_human_weight),
        "autonomy_intervention_cost": float(rl_config.autonomy_intervention_cost),
    }

    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
