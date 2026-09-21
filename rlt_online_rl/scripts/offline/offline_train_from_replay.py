from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

"""
Offline actor/critic training from a replay journal.

Inputs:

- data is selected by `--replay-path`
- task config is inferred from `<task-dir>/actor_snapshot/actor_snapshot.pkl`
- replay subsets are selected only by `--phase` and `--source`

The script:

1. reads the replay journal
2. filters samples by `phase/source`
3. initializes actor and critic from scratch
4. runs offline training
5. periodically evaluates train/validation actor fit
6. exports snapshots, checkpoints, and a manifest
7. exports an online-compatible bundle for continued online RL
"""

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _common import PHASE_CHOICES
from _common import SOURCE_CHOICES
from _common import default_filter_suffix
from _common import filter_replay_records
from _common import infer_task_dir_from_replay_path
from _common import resolve_stats_path
from _common import write_replay_journal

from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.action_representation import jax_denormalize_to_abs_chunk
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.config import relativize_rl_config_paths
from rlt_online_rl.config import resolve_rl_config_paths
from rlt_online_rl.config import save_system_config_yaml
from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.networks import PyTree
from rlt_online_rl.networks import TwinCritic
from rlt_online_rl.networks import compute_critic_loss
from rlt_online_rl.replay import TransitionSource
from rlt_online_rl.trainer import RLTTrainState
from rlt_online_rl.trainer import init_train_state
from rlt_online_rl.trainer import soft_update_targets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline-train actor/critic from a replay journal with custom BC/Q weights."
    )
    parser.add_argument(
        "--replay-path",
        type=Path,
        required=True,
        help="Replay journal to train from, usually runs/<task>/replay/replay_journal.pkl",
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-weight", type=float, default=2.0)
    parser.add_argument("--q-weight", type=float, default=0.1)
    parser.add_argument("--delta-weight", type=float, default=None)
    parser.add_argument("--fixed-std", type=float, default=None)
    parser.add_argument("--actor-hidden-dim", type=int, default=None)
    parser.add_argument("--actor-num-layers", type=int, default=None)
    parser.add_argument("--critic-hidden-dim", type=int, default=None)
    parser.add_argument("--critic-num-layers", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--disable-ref-input",
        action="store_true",
        help="Train/evaluate the actor with ref_chunk input replaced by zeros.",
    )
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    parser.add_argument("--source", choices=tuple(SOURCE_CHOICES), default="all")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional offline output directory.")
    return parser.parse_args()


def _load_snapshot_config(task_dir: Path) -> RLTOnlineRLConfig:
    snapshot_path = task_dir / "actor_snapshot" / "actor_snapshot.pkl"
    if snapshot_path.is_file():
        with snapshot_path.open("rb") as f:
            payload = pickle.load(f)
        cfg = RLTOnlineRLConfig(**payload["rl_config"])
        cfg = resolve_rl_config_paths(cfg, str(snapshot_path), require_exists=True)
        return dataclasses.replace(cfg, action_norm_stats_path=resolve_stats_path(cfg.action_norm_stats_path, task_dir))

    config_path = ROOT / "configs" / "tasks" / task_dir.name / "online_rl.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Neither an actor snapshot nor a task config exists for {task_dir}: {snapshot_path}, {config_path}"
        )
    system_config = load_system_config_yaml(str(config_path))
    return resolve_rl_config_paths(system_config.rl, str(config_path), require_exists=True)


def _load_replay_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as f:
        try:
            while True:
                item = pickle.load(f)
                if isinstance(item, dict):
                    records.append(item)
        except EOFError:
            pass
    if not records:
        raise RuntimeError(f"No replay records found in {path}")
    return records


def _stack_records(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
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
        "success",
        "intervention_flag",
        "episode_id",
        "step_id",
    )
    batch: dict[str, np.ndarray] = {}
    for key in keys:
        batch[key] = np.stack([np.asarray(record[key]) for record in records], axis=0)
    batch["source_chunk"] = np.stack(
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
    batch["done"] = batch["done"].astype(np.float32, copy=False)
    batch["source"] = batch["source"].astype(np.uint8, copy=False)
    batch["source_chunk"] = batch["source_chunk"].astype(np.uint8, copy=False)
    batch["success"] = batch["success"].astype(np.int8, copy=False)
    batch["intervention_flag"] = batch["intervention_flag"].astype(np.bool_, copy=False)
    batch["episode_id"] = batch["episode_id"].astype(np.int32, copy=False)
    batch["step_id"] = batch["step_id"].astype(np.int32, copy=False)
    return batch


def _split_dataset(
    dataset: dict[str, np.ndarray], *, val_ratio: float, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    size = dataset["z_rl"].shape[0]
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(size * val_ratio))
    val_idx = np.asarray(indices[:val_size], dtype=np.int32)
    train_idx = np.asarray(indices[val_size:], dtype=np.int32)
    if train_idx.size == 0:
        raise RuntimeError("Validation split consumed the entire dataset.")
    train = {key: value[train_idx] for key, value in dataset.items()}
    val = {key: value[val_idx] for key, value in dataset.items()}
    return train, val


def _sample_batch(
    dataset: dict[str, np.ndarray], *, batch_size: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    size = dataset["z_rl"].shape[0]
    indices = rng.integers(0, size, size=batch_size, endpoint=False)
    return {key: value[indices] for key, value in dataset.items()}


def _maybe_zero_model_ref_input(ref_chunk: np.ndarray, *, disable_ref_input: bool) -> np.ndarray:
    ref_chunk = np.asarray(ref_chunk, dtype=np.float32)
    if disable_ref_input:
        return np.zeros_like(ref_chunk, dtype=np.float32)
    return ref_chunk


def _predict_abs_chunk_batch(
    actor: ChunkActor,
    actor_params: PyTree,
    adapter: ActionRepresentationAdapter | None,
    batch: dict[str, np.ndarray],
    *,
    disable_ref_input: bool = False,
) -> np.ndarray:
    z_rl = np.asarray(batch["z_rl"], dtype=np.float32)
    proprio = np.asarray(batch["proprio"], dtype=np.float32)
    ref_chunk = np.asarray(batch["ref_chunk"], dtype=np.float32)
    model_ref_chunk = adapter.normalize_ref_chunk(ref_chunk, proprio) if adapter is not None else ref_chunk
    model_ref_chunk = _maybe_zero_model_ref_input(model_ref_chunk, disable_ref_input=disable_ref_input)
    pred = actor.actor_mean(actor_params, jnp.asarray(z_rl), jnp.asarray(proprio), jnp.asarray(model_ref_chunk))
    pred_np = np.asarray(jax.device_get(pred), dtype=np.float32)
    if adapter is not None:
        pred_np = adapter.denormalize_to_abs_chunk(pred_np, proprio)
    return pred_np


def _evaluate_fit(
    actor: ChunkActor,
    actor_params: PyTree,
    adapter: ActionRepresentationAdapter | None,
    dataset: dict[str, np.ndarray],
    *,
    batch_size: int = 256,
    disable_ref_input: bool = False,
) -> dict[str, float]:
    total_abs = []
    size = dataset["z_rl"].shape[0]
    for start in range(0, size, batch_size):
        end = min(start + batch_size, size)
        batch = {key: value[start:end] for key, value in dataset.items()}
        pred_abs = _predict_abs_chunk_batch(actor, actor_params, adapter, batch, disable_ref_input=disable_ref_input)
        ref_abs = np.asarray(batch["ref_chunk"], dtype=np.float32)
        total_abs.append(np.abs(pred_abs - ref_abs))
    abs_delta = np.concatenate(total_abs, axis=0)
    metrics = {
        "mean_abs_delta": float(abs_delta.mean()),
        "median_abs_delta": float(np.median(abs_delta)),
        "p95_abs_delta": float(np.percentile(abs_delta, 95)),
        "max_abs_delta": float(abs_delta.max()),
    }
    for joint_idx in range(abs_delta.shape[-1]):
        joint_vals = abs_delta[:, :, joint_idx]
        metrics[f"joint{joint_idx + 1}_mean_abs_delta"] = float(joint_vals.mean())
        metrics[f"joint{joint_idx + 1}_max_abs_delta"] = float(joint_vals.max())
    return metrics


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def _relativize_path(path_value: str | Path | None, anchor_path: Path) -> str | None:
    if path_value is None:
        return None
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        return str(candidate)
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        return os.path.relpath(str(candidate), start=str(anchor_path.resolve().parent))
    return os.path.relpath(str(candidate), start=str(ROOT))


def _portable_rl_config_dict(rl_config: RLTOnlineRLConfig, anchor_path: Path) -> dict[str, Any]:
    portable = relativize_rl_config_paths(rl_config, str(anchor_path))
    return dataclasses.asdict(portable)


def _bundle_action_norm_stats(output_dir: Path, rl_config: RLTOnlineRLConfig) -> RLTOnlineRLConfig:
    if rl_config.action_norm_stats_path is None:
        return rl_config

    source_path = Path(rl_config.action_norm_stats_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Action normalization stats do not exist: {source_path}")
    target_path = (output_dir / "assets" / "norm_stats.json").resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path != target_path:
        temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
        shutil.copy2(source_path, temporary_path)
        os.replace(temporary_path, target_path)
    return dataclasses.replace(rl_config, action_norm_stats_path=str(target_path))


def _save_actor_snapshot(path: Path, version: int, rl_config: RLTOnlineRLConfig, actor_params: PyTree) -> None:
    payload = {
        "version": int(version),
        "rl_config": _portable_rl_config_dict(rl_config, path),
        "actor_params": jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), actor_params),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_critic_snapshot(path: Path, version: int, rl_config: RLTOnlineRLConfig, critic_params: PyTree) -> None:
    payload = {
        "version": int(version),
        "rl_config": _portable_rl_config_dict(rl_config, path),
        "critic_params": jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), critic_params),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _tree_to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def _warmup_required_updates(rl_config: RLTOnlineRLConfig, warmup_ready_adds_total: int) -> int:
    if rl_config.warmup_post_collect_updates is not None:
        return int(rl_config.warmup_post_collect_updates)
    return int(warmup_ready_adds_total) * int(rl_config.grad_updates_per_cycle)


def _save_online_checkpoint_bundle(
    *,
    output_dir: Path,
    rl_config: RLTOnlineRLConfig,
    state: RLTTrainState,
    replay_size: int,
) -> tuple[Path, Path, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    actor_snapshot_dir = output_dir / "actor_snapshot"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    actor_snapshot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    global_step = int(jax.device_get(state.global_step))
    actor_version = int(jax.device_get(state.actor_version))
    warmup_ready_adds_total = int(replay_size)
    warmup_required_updates = _warmup_required_updates(rl_config, warmup_ready_adds_total)
    learner_ready_for_online = global_step >= warmup_required_updates
    latest_path = checkpoint_dir / "latest.pkl"
    step_path = checkpoint_dir / f"step_{global_step}.pkl"

    checkpoint_payload = {
        "rl_config": _portable_rl_config_dict(rl_config, latest_path),
        "state": {
            "actor_params": _tree_to_numpy(state.actor_params),
            "target_actor_params": _tree_to_numpy(state.target_actor_params),
            "critic_params": _tree_to_numpy(state.critic_params),
            "target_critic_params": _tree_to_numpy(state.target_critic_params),
            "actor_opt_state": _tree_to_numpy(state.actor_opt_state),
            "critic_opt_state": _tree_to_numpy(state.critic_opt_state),
            "rng": _tree_to_numpy(state.rng),
            "global_step": global_step,
            "actor_version": actor_version,
        },
        "progress": {
            "warmup_ready_adds_total": warmup_ready_adds_total,
        },
    }
    with latest_path.open("wb") as f:
        pickle.dump(checkpoint_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with step_path.open("wb") as f:
        pickle.dump(checkpoint_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    actor_snapshot_path = actor_snapshot_dir / "actor_snapshot.pkl"
    _save_actor_snapshot(actor_snapshot_path, actor_version, rl_config, state.actor_params)

    learner_status = {
        "global_step": global_step,
        "actor_version": actor_version,
        "replay_size": replay_size,
        "adds_total": replay_size,
        "pending_update_budget": 0,
        "warmup_ready_adds_total": warmup_ready_adds_total,
        "warmup_required_updates": warmup_required_updates,
        "warmup_post_collect_updates": rl_config.warmup_post_collect_updates,
        "ready_for_online": learner_ready_for_online,
        "training_frozen": False,
    }
    _atomic_write_json(metrics_dir / "learner_status.json", learner_status)
    return latest_path, actor_snapshot_path, metrics_dir / "learner_status.json"


def _save_bundle_config(
    *,
    output_dir: Path,
    run_dir: Path,
    rl_config: RLTOnlineRLConfig,
) -> Path | None:
    source_config_candidates = (
        run_dir / "checkpoints" / "online_rl_config.yaml",
        ROOT / "configs" / "tasks" / run_dir.name / "online_rl.yaml",
    )
    source_config_path = next((path for path in source_config_candidates if path.exists()), None)
    if source_config_path is None:
        return None
    system = load_system_config_yaml(str(source_config_path))
    bundle_actor_snapshot = (output_dir / "actor_snapshot" / "actor_snapshot.pkl").resolve()
    bundle_checkpoint_dir = (output_dir / "checkpoints").resolve()
    bundle_replay_path = (output_dir / "replay" / "replay_journal.pkl").resolve()
    bundle_wandb_dir = (output_dir / "wandb").resolve()
    bundle_wandb_dir.mkdir(parents=True, exist_ok=True)
    system = dataclasses.replace(
        system,
        rl=rl_config,
        actor_service=dataclasses.replace(system.actor_service, snapshot_path=str(bundle_actor_snapshot)),
        learner_service=dataclasses.replace(
            system.learner_service,
            checkpoint_dir=str(bundle_checkpoint_dir),
            actor_snapshot_path=str(bundle_actor_snapshot),
        ),
        replay=dataclasses.replace(system.replay, journal_path=str(bundle_replay_path)),
        monitoring=dataclasses.replace(system.monitoring, wandb_dir=str(bundle_wandb_dir)),
    )
    target_path = bundle_checkpoint_dir / "online_rl_config.yaml"
    save_system_config_yaml(system, str(target_path))
    return target_path


def _write_filtered_replay(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    replay_dir = output_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    target_path = replay_dir / "replay_journal.pkl"
    write_replay_journal(target_path, records)
    return target_path


def _write_bundle_manifest(
    *,
    output_dir: Path,
    run_dir: Path,
    source_replay_path: Path,
    bundle_replay_path: Path,
    rl_config: RLTOnlineRLConfig,
    checkpoint_path: Path,
    actor_snapshot_path: Path,
    learner_status_path: Path,
    bundle_config_path: Path | None,
    state: RLTTrainState,
    published_model_tag: str,
) -> None:
    payload = {
        "format_version": 1,
        "source_run_dir": _relativize_path(run_dir, output_dir / "manifest.json"),
        "source_replay_path": _relativize_path(source_replay_path, output_dir / "manifest.json"),
        "bundle_replay_path": _relativize_path(bundle_replay_path, output_dir / "manifest.json"),
        "published_model_tag": published_model_tag,
        "rl_config": _portable_rl_config_dict(rl_config, output_dir / "manifest.json"),
        "global_step": int(jax.device_get(state.global_step)),
        "actor_version": int(jax.device_get(state.actor_version)),
        "checkpoint_path": _relativize_path(checkpoint_path, output_dir / "manifest.json"),
        "actor_snapshot_path": _relativize_path(actor_snapshot_path, output_dir / "manifest.json"),
        "learner_status_path": _relativize_path(learner_status_path, output_dir / "manifest.json"),
        "bundle_config_path": None
        if bundle_config_path is None
        else _relativize_path(bundle_config_path, output_dir / "manifest.json"),
    }
    _atomic_write_json(output_dir / "manifest.json", payload)


def _custom_actor_loss(
    actor: ChunkActor,
    actor_params: PyTree,
    critic: TwinCritic,
    critic_params: PyTree,
    z_rl: jax.Array,
    proprio: jax.Array,
    ref_chunk: jax.Array,
    behavior_chunk: jax.Array,
    source_chunk: jax.Array,
    *,
    bc_weight: float,
    q_weight: float,
    delta_weight: float,
    reference_dropout_prob: float,
    disable_ref_input: bool,
    rng: jax.Array,
    use_action_adapter: bool,
    action_q01: jax.Array | None,
    action_q99: jax.Array | None,
    action_representation: str,
    delta_action_dims: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    dropout_rng, sample_rng = jax.random.split(rng)
    if reference_dropout_prob > 0.0:
        keep_mask = jax.random.bernoulli(dropout_rng, 1.0 - reference_dropout_prob, (ref_chunk.shape[0], 1, 1))
        dropped_ref = ref_chunk * keep_mask.astype(ref_chunk.dtype)
    else:
        dropped_ref = ref_chunk
    model_ref_input = jnp.zeros_like(dropped_ref) if disable_ref_input else dropped_ref
    # Explore for the Q term, while supervising the deterministic policy used at deployment.
    action_mean, action_std = actor.actor_dist(actor_params, z_rl, proprio, model_ref_input)
    sampled_action = action_mean + action_std * jax.random.normal(
        sample_rng, action_mean.shape, dtype=action_mean.dtype
    )
    q1, q2 = critic.q_values(critic_params, z_rl, proprio, sampled_action)
    human_mask = jnp.logical_or(
        source_chunk == int(TransitionSource.HUMAN),
        source_chunk == int(TransitionSource.MIXED),
    )
    human_mask_f = human_mask.astype(jnp.float32)
    policy_mask_f = 1.0 - human_mask_f
    bc_target = jnp.where(human_mask[..., None], behavior_chunk, ref_chunk)
    bc_error = jnp.mean(jnp.square(action_mean - bc_target), axis=-1)
    ref_error = jnp.mean(jnp.square(action_mean - ref_chunk), axis=-1)
    human_error = jnp.mean(jnp.square(action_mean - behavior_chunk), axis=-1)
    bc_penalty = jnp.mean(bc_error)
    bc_ref_penalty = jnp.sum(ref_error * policy_mask_f) / jnp.maximum(jnp.sum(policy_mask_f), 1.0)
    bc_human_penalty = jnp.sum(human_error * human_mask_f) / jnp.maximum(jnp.sum(human_mask_f), 1.0)
    human_mask_ratio = jnp.mean(human_mask_f)
    if not use_action_adapter:
        pred_abs_chunk = action_mean
        target_abs_chunk = bc_target
    else:
        pred_abs_chunk = jax_denormalize_to_abs_chunk(
            action_mean,
            proprio,
            action_q01,
            action_q99,
            action_representation=action_representation,
            delta_action_dims=delta_action_dims,
        )
        target_abs_chunk = jax_denormalize_to_abs_chunk(
            bc_target,
            proprio,
            action_q01,
            action_q99,
            action_representation=action_representation,
            delta_action_dims=delta_action_dims,
        )
    pred_step_delta = pred_abs_chunk[:, 1:, :delta_action_dims] - pred_abs_chunk[:, :-1, :delta_action_dims]
    target_step_delta = target_abs_chunk[:, 1:, :delta_action_dims] - target_abs_chunk[:, :-1, :delta_action_dims]
    delta_penalty = jnp.mean(jnp.square(pred_step_delta - target_step_delta))
    actor_q = jnp.mean(jnp.minimum(q1, q2))
    weighted_bc = jnp.asarray(bc_weight, dtype=jnp.float32) * bc_penalty
    weighted_q = jnp.asarray(q_weight, dtype=jnp.float32) * actor_q
    weighted_delta = jnp.asarray(delta_weight, dtype=jnp.float32) * delta_penalty
    actor_loss = weighted_bc - weighted_q + weighted_delta
    metrics = {
        "actor_loss": actor_loss,
        "actor_q": actor_q,
        "bc_penalty": bc_penalty,
        "bc_ref_penalty": bc_ref_penalty,
        "bc_human_penalty": bc_human_penalty,
        "human_mask_ratio": human_mask_ratio,
        "policy_mask_ratio": 1.0 - human_mask_ratio,
        "delta_penalty": delta_penalty,
        "weighted_bc": weighted_bc,
        "weighted_delta": weighted_delta,
        "weighted_q": weighted_q,
        "effective_q_weight": jnp.asarray(q_weight, dtype=jnp.float32),
    }
    return actor_loss, metrics


def _update_critic(
    state: RLTTrainState,
    batch: dict[str, jax.Array],
    actor: ChunkActor,
    critic: TwinCritic,
    rl_config: RLTOnlineRLConfig,
    *,
    disable_ref_input: bool,
    axis_name: str | None = None,
) -> tuple[RLTTrainState, dict[str, jax.Array]]:
    critic_rng, next_rng = jax.random.split(state.rng)

    def loss_fn(critic_params: PyTree) -> tuple[jax.Array, dict[str, jax.Array]]:
        return compute_critic_loss(
            critic,
            critic_params,
            actor,
            state.target_actor_params,
            state.target_critic_params,
            batch["z_rl"],
            batch["proprio"],
            batch["action_chunk"],
            batch["rewards"],
            batch["done"],
            batch["next_z_rl"],
            batch["next_proprio"],
            jnp.zeros_like(batch["next_ref_chunk"]) if disable_ref_input else batch["next_ref_chunk"],
            rl_config.gamma,
            critic_rng,
            target_actor_deterministic=rl_config.target_actor_deterministic,
        )

    (critic_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.critic_params)
    if axis_name is not None:
        grads = jax.lax.pmean(grads, axis_name)
    updates, critic_opt_state = state.critic_tx.update(grads, state.critic_opt_state, state.critic_params)
    critic_params = optax.apply_updates(state.critic_params, updates)
    new_state = state.replace(critic_params=critic_params, critic_opt_state=critic_opt_state, rng=next_rng)
    metrics = {**metrics, "critic_loss": critic_loss}
    if axis_name is not None:
        metrics = jax.lax.pmean(metrics, axis_name)
    return new_state, metrics


@jax.jit
def _zero_metrics() -> dict[str, jax.Array]:
    nan = jnp.array(jnp.nan, dtype=jnp.float32)
    return {
        "actor_loss": nan,
        "actor_q": nan,
        "bc_penalty": nan,
        "bc_ref_penalty": nan,
        "bc_human_penalty": nan,
        "human_mask_ratio": nan,
        "policy_mask_ratio": nan,
        "delta_penalty": nan,
        "weighted_bc": nan,
        "weighted_delta": nan,
        "weighted_q": nan,
        "effective_q_weight": nan,
    }


def _make_train_step(
    actor: ChunkActor,
    critic: TwinCritic,
    rl_config: RLTOnlineRLConfig,
    *,
    bc_weight: float,
    q_weight: float,
    delta_weight: float,
    disable_ref_input: bool,
    use_action_adapter: bool,
    action_q01: jax.Array | None,
    action_q99: jax.Array | None,
    actor_q_start_step: int = 0,
    axis_name: str | None = None,
    devices: tuple[jax.Device, ...] | None = None,
):
    def train_step(state: RLTTrainState, batch: dict[str, jax.Array]) -> tuple[RLTTrainState, dict[str, jax.Array]]:
        state_after_critic, critic_metrics = _update_critic(
            state,
            batch,
            actor,
            critic,
            rl_config,
            disable_ref_input=disable_ref_input,
            axis_name=axis_name,
        )
        should_update_actor = ((state_after_critic.global_step + 1) % rl_config.actor_update_period) == 0

        def do_actor_update(train_state: RLTTrainState) -> tuple[RLTTrainState, dict[str, jax.Array]]:
            actor_rng, next_rng = jax.random.split(train_state.rng)
            effective_q_weight = jnp.where(
                train_state.global_step + 1 >= actor_q_start_step,
                jnp.asarray(q_weight, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            )

            def loss_fn(actor_params: PyTree) -> tuple[jax.Array, dict[str, jax.Array]]:
                return _custom_actor_loss(
                    actor,
                    actor_params,
                    critic,
                    train_state.critic_params,
                    batch["z_rl"],
                    batch["proprio"],
                    batch["ref_chunk"],
                    batch["action_chunk"],
                    batch["source_chunk"],
                    bc_weight=bc_weight,
                    q_weight=effective_q_weight,
                    delta_weight=delta_weight,
                    reference_dropout_prob=rl_config.reference_dropout_prob,
                    disable_ref_input=disable_ref_input,
                    rng=actor_rng,
                    use_action_adapter=use_action_adapter,
                    action_q01=action_q01,
                    action_q99=action_q99,
                    action_representation=rl_config.action_representation,
                    delta_action_dims=rl_config.delta_action_dims,
                )

            (actor_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.actor_params)
            if axis_name is not None:
                grads = jax.lax.pmean(grads, axis_name)
            updates, actor_opt_state = train_state.actor_tx.update(
                grads, train_state.actor_opt_state, train_state.actor_params
            )
            actor_params = optax.apply_updates(train_state.actor_params, updates)
            updated_state = train_state.replace(
                actor_params=actor_params,
                actor_opt_state=actor_opt_state,
                rng=next_rng,
            )
            updated_state = updated_state.replace(
                target_actor_params=soft_update_targets(
                    updated_state.target_actor_params, updated_state.actor_params, rl_config.target_tau
                ),
                target_critic_params=soft_update_targets(
                    updated_state.target_critic_params, updated_state.critic_params, rl_config.target_tau
                ),
                actor_version=updated_state.actor_version + 1,
            )
            metrics = {**metrics, "actor_loss": actor_loss}
            if axis_name is not None:
                metrics = jax.lax.pmean(metrics, axis_name)
            return updated_state, metrics

        state_after_actor, actor_metrics = jax.lax.cond(
            should_update_actor,
            do_actor_update,
            lambda train_state: (train_state, _zero_metrics()),
            state_after_critic,
        )
        state_after_actor = state_after_actor.replace(global_step=state_after_actor.global_step + 1)
        metrics = {
            **critic_metrics,
            **actor_metrics,
            "did_actor_update": should_update_actor.astype(jnp.float32),
            "global_step": state_after_actor.global_step.astype(jnp.float32),
            "actor_version": state_after_actor.actor_version.astype(jnp.float32),
        }
        return state_after_actor, metrics

    if axis_name is None:
        return jax.jit(train_step)
    return jax.pmap(train_step, axis_name=axis_name, devices=devices)


def main() -> None:
    args = _parse_args()
    replay_path = args.replay_path.resolve()
    task_dir = infer_task_dir_from_replay_path(replay_path)
    default_output_name = "offline_train_bcq"
    if args.disable_ref_input:
        default_output_name += "_noref"
    default_output_name += default_filter_suffix(phase=args.phase, source=args.source)
    output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else (task_dir / default_output_name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rl_config = _load_snapshot_config(task_dir)
    if args.actor_hidden_dim is not None:
        rl_config = dataclasses.replace(rl_config, actor_hidden_dim=args.actor_hidden_dim)
    if args.actor_num_layers is not None:
        rl_config = dataclasses.replace(rl_config, actor_num_layers=args.actor_num_layers)
    if args.fixed_std is not None:
        rl_config = dataclasses.replace(rl_config, fixed_std=args.fixed_std)
    if args.critic_hidden_dim is not None:
        rl_config = dataclasses.replace(rl_config, critic_hidden_dim=args.critic_hidden_dim)
    if args.critic_num_layers is not None:
        rl_config = dataclasses.replace(rl_config, critic_num_layers=args.critic_num_layers)
    rl_config = _bundle_action_norm_stats(output_dir, rl_config)
    records = filter_replay_records(
        _load_replay_records(replay_path),
        phase=args.phase,
        source=args.source,
    )
    if not records:
        raise RuntimeError(f"No replay samples left after filtering: replay={replay_path}")
    dataset = _stack_records(records)
    train_ds, val_ds = _split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    collection_phase_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for record in records:
        phase = str(record.get("collection_phase", "unknown"))
        collection_phase_counts[phase] = collection_phase_counts.get(phase, 0) + 1
        source_key = str(int(record["source"]))
        source_counts[source_key] = source_counts.get(source_key, 0) + 1

    state, actor, critic = init_train_state(rl_config, rng=jax.random.PRNGKey(args.seed))
    adapter = ActionRepresentationAdapter.from_config(rl_config)
    delta_weight = float(args.delta_weight if args.delta_weight is not None else rl_config.delta_weight)
    if adapter is None:
        action_q01 = None
        action_q99 = None
    else:
        action_q01 = jnp.asarray(adapter.stats.q01, dtype=jnp.float32)
        action_q99 = jnp.asarray(adapter.stats.q99, dtype=jnp.float32)
    train_step = _make_train_step(
        actor,
        critic,
        rl_config,
        bc_weight=args.bc_weight,
        q_weight=args.q_weight,
        delta_weight=delta_weight,
        disable_ref_input=args.disable_ref_input,
        use_action_adapter=adapter is not None,
        action_q01=action_q01,
        action_q99=action_q99,
    )
    rng = np.random.default_rng(args.seed)

    metrics_path = output_dir / "metrics.jsonl"
    status_path = output_dir / "status.json"
    best_actor_snapshot_path = output_dir / "best_actor_snapshot.pkl"
    best_critic_snapshot_path = output_dir / "best_critic_snapshot.pkl"
    final_actor_snapshot_path = output_dir / "final_actor_snapshot.pkl"
    final_critic_snapshot_path = output_dir / "final_critic_snapshot.pkl"

    experiment_meta = {
        "task_dir": _relativize_path(task_dir, output_dir / "experiment.json"),
        "replay_path": _relativize_path(replay_path, output_dir / "experiment.json"),
        "phase": args.phase,
        "source": args.source,
        "collection_phase_counts": collection_phase_counts,
        "source_counts": source_counts,
        "num_records": int(len(records)),
        "train_size": int(train_ds["z_rl"].shape[0]),
        "val_size": int(val_ds["z_rl"].shape[0]),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "bc_weight": float(args.bc_weight),
        "q_weight": float(args.q_weight),
        "delta_weight": float(delta_weight),
        "fixed_std": float(rl_config.fixed_std),
        "disable_ref_input": bool(args.disable_ref_input),
        "actor_hidden_dim": int(rl_config.actor_hidden_dim),
        "actor_num_layers": int(rl_config.actor_num_layers),
        "critic_hidden_dim": int(rl_config.critic_hidden_dim),
        "critic_num_layers": int(rl_config.critic_num_layers),
        "published_model_tag": "final",
        "rl_config": _portable_rl_config_dict(rl_config, output_dir / "experiment.json"),
    }
    _atomic_write_json(output_dir / "experiment.json", experiment_meta)
    print(
        "offline_train_from_replay "
        f"replay={replay_path.name} phase={args.phase} source={args.source} "
        f"train={train_ds['z_rl'].shape[0]} val={val_ds['z_rl'].shape[0]} "
        f"steps={args.steps} batch={args.batch_size} bc_weight={args.bc_weight:.3f} q_weight={args.q_weight:.3f} "
        f"fixed_std={rl_config.fixed_std:.4f} disable_ref_input={args.disable_ref_input}"
    )

    best_val = float("inf")
    best_step = 0
    for step in range(1, args.steps + 1):
        batch_np = _sample_batch(train_ds, batch_size=args.batch_size, rng=rng)
        if adapter is not None:
            batch_np = adapter.prepare_training_batch(batch_np)
        batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
        state, raw_metrics = train_step(state, batch)
        metrics = {key: float(value) for key, value in jax.device_get(raw_metrics).items()}

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            actor_params_np = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), state.actor_params)
            train_fit = _evaluate_fit(
                actor, actor_params_np, adapter, train_ds, disable_ref_input=args.disable_ref_input
            )
            val_fit = _evaluate_fit(actor, actor_params_np, adapter, val_ds, disable_ref_input=args.disable_ref_input)
            metrics.update({f"train_{k}": v for k, v in train_fit.items()})
            metrics.update({f"val_{k}": v for k, v in val_fit.items()})
            if val_fit["mean_abs_delta"] < best_val:
                best_val = val_fit["mean_abs_delta"]
                best_step = step
                _save_actor_snapshot(
                    best_actor_snapshot_path, int(metrics["actor_version"]), rl_config, state.actor_params
                )
                _save_critic_snapshot(
                    best_critic_snapshot_path, int(metrics["actor_version"]), rl_config, state.critic_params
                )
                best_tag = " best"
            else:
                best_tag = ""
            _atomic_write_json(
                status_path,
                {
                    "step": step,
                    "best_val_mean_abs_delta": best_val,
                    "best_step": best_step,
                    "latest_metrics": metrics,
                },
            )
            print(
                f"[step {step:>6d}/{args.steps}] "
                f"critic={metrics['critic_loss']:.4f} "
                f"actor={metrics['actor_loss']:.4f} "
                f"q={metrics['actor_q']:.4f} "
                f"bc={metrics['bc_penalty']:.4f} "
                f"delta={metrics['delta_penalty']:.4f} "
                f"train_fit={metrics['train_mean_abs_delta']:.4f} "
                f"val_fit={metrics['val_mean_abs_delta']:.4f} "
                f"val_max={metrics['val_max_abs_delta']:.4f} "
                f"actor_ver={int(metrics['actor_version'])}{best_tag}"
            )

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    _save_actor_snapshot(
        final_actor_snapshot_path, int(jax.device_get(state.actor_version)), rl_config, state.actor_params
    )
    _save_critic_snapshot(
        final_critic_snapshot_path, int(jax.device_get(state.actor_version)), rl_config, state.critic_params
    )
    bundle_replay_path = _write_filtered_replay(output_dir, records)
    checkpoint_path, actor_snapshot_path, learner_status_path = _save_online_checkpoint_bundle(
        output_dir=output_dir,
        rl_config=rl_config,
        state=state,
        replay_size=len(records),
    )
    bundle_config_path = _save_bundle_config(
        output_dir=output_dir,
        run_dir=task_dir,
        rl_config=rl_config,
    )
    _write_bundle_manifest(
        output_dir=output_dir,
        run_dir=task_dir,
        source_replay_path=replay_path,
        bundle_replay_path=bundle_replay_path,
        rl_config=rl_config,
        checkpoint_path=checkpoint_path,
        actor_snapshot_path=actor_snapshot_path,
        learner_status_path=learner_status_path,
        bundle_config_path=bundle_config_path,
        state=state,
        published_model_tag="final",
    )
    experiment_meta["best_step"] = int(best_step)
    experiment_meta["best_val_mean_abs_delta"] = float(best_val)
    _atomic_write_json(output_dir / "experiment.json", experiment_meta)
    print(f"wrote offline training artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
