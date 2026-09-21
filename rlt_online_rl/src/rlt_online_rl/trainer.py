from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import pickle
import time
from typing import Any

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.action_representation import jax_denormalize_to_abs_chunk
from rlt_online_rl.config import LearnerServiceConfig
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.config import relativize_rl_config_paths
from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.networks import PyTree
from rlt_online_rl.networks import TwinCritic
from rlt_online_rl.networks import apply_reference_dropout
from rlt_online_rl.networks import compute_critic_loss
from rlt_online_rl.replay import COLLECTION_PHASE_ONLINE
from rlt_online_rl.replay import COLLECTION_PHASE_WARMUP
from rlt_online_rl.replay import ReplayBatchSource
from rlt_online_rl.replay import TransitionSource
from rlt_online_rl.runtime_logging import append_jsonl

logger = logging.getLogger(__name__)


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _portable_rl_config_dict(rl_config: RLTOnlineRLConfig, anchor_path: str) -> dict[str, Any]:
    return dataclasses.asdict(relativize_rl_config_paths(rl_config, anchor_path))


@struct.dataclass
class RLTTrainState:
    actor_params: PyTree
    target_actor_params: PyTree
    critic_params: PyTree
    target_critic_params: PyTree
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    rng: jax.Array
    global_step: jax.Array
    actor_version: jax.Array

    actor_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_tx: optax.GradientTransformation = struct.field(pytree_node=False)


def _make_networks(rl_config: RLTOnlineRLConfig) -> tuple[ChunkActor, TwinCritic]:
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
    )
    return actor, critic


def init_train_state(
    rl_config: RLTOnlineRLConfig,
    *,
    rng: jax.Array,
) -> tuple[RLTTrainState, ChunkActor, TwinCritic]:
    actor, critic = _make_networks(rl_config)
    actor_key, critic_key, state_key = jax.random.split(rng, 3)
    actor_params = actor.init_params(actor_key)
    critic_params = critic.init_params(critic_key)
    actor_tx = optax.adam(rl_config.actor_lr)
    critic_tx = optax.adam(rl_config.critic_lr)
    state = RLTTrainState(
        actor_params=actor_params,
        target_actor_params=actor_params,
        critic_params=critic_params,
        target_critic_params=critic_params,
        actor_opt_state=actor_tx.init(actor_params),
        critic_opt_state=critic_tx.init(critic_params),
        rng=state_key,
        global_step=jnp.array(0, dtype=jnp.int32),
        actor_version=jnp.array(0, dtype=jnp.int32),
        actor_tx=actor_tx,
        critic_tx=critic_tx,
    )
    return state, actor, critic


def soft_update_targets(target_params: PyTree, source_params: PyTree, tau: float) -> PyTree:
    return jax.tree_util.tree_map(
        lambda target, source: (1.0 - tau) * target + tau * source, target_params, source_params
    )


def _resolve_actor_loss_weights(
    rl_config: RLTOnlineRLConfig,
    progress: dict[str, int | bool],
) -> tuple[float, float]:
    warmup_required_updates = int(progress["warmup_required_updates"])
    global_step = int(progress["global_step"])
    if warmup_required_updates > 0 and global_step < warmup_required_updates:
        return float(rl_config.warmup_bc_weight), float(rl_config.warmup_q_weight)
    return float(rl_config.online_bc_weight), float(rl_config.online_q_weight)


def update_critic(
    state: RLTTrainState,
    batch: dict[str, jax.Array],
    actor: ChunkActor,
    critic: TwinCritic,
    rl_config: RLTOnlineRLConfig,
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
            batch["next_ref_chunk"],
            rl_config.gamma,
            critic_rng,
            target_actor_deterministic=rl_config.target_actor_deterministic,
        )

    (critic_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.critic_params)
    updates, critic_opt_state = state.critic_tx.update(grads, state.critic_opt_state, state.critic_params)
    critic_params = optax.apply_updates(state.critic_params, updates)
    new_state = state.replace(
        critic_params=critic_params,
        critic_opt_state=critic_opt_state,
        rng=next_rng,
    )
    metrics = {
        **metrics,
        "critic_loss": critic_loss,
    }
    return new_state, metrics


def update_actor(
    state: RLTTrainState,
    batch: dict[str, jax.Array],
    actor: ChunkActor,
    critic: TwinCritic,
    rl_config: RLTOnlineRLConfig,
    *,
    bc_weight: float,
    q_weight: float,
    delta_weight: float = 0.0,
    use_action_adapter: bool = False,
    action_q01: jax.Array | None = None,
    action_q99: jax.Array | None = None,
) -> tuple[RLTTrainState, dict[str, jax.Array]]:
    actor_rng, next_rng = jax.random.split(state.rng)

    def loss_fn(actor_params: PyTree) -> tuple[jax.Array, dict[str, jax.Array]]:
        dropout_rng, sample_rng = jax.random.split(actor_rng)
        dropped_ref = apply_reference_dropout(
            dropout_rng,
            batch["ref_chunk"],
            rl_config.reference_dropout_prob,
        )
        # Explore for the Q term, while supervising the deterministic policy used at deployment.
        action_mean, action_std = actor.actor_dist(actor_params, batch["z_rl"], batch["proprio"], dropped_ref)
        sampled_action = action_mean + action_std * jax.random.normal(
            sample_rng, action_mean.shape, dtype=action_mean.dtype
        )
        q1, q2 = critic.q_values(
            state.critic_params,
            batch["z_rl"],
            batch["proprio"],
            sampled_action,
        )
        source_chunk = batch["source_chunk"]
        human_mask = jnp.logical_or(
            source_chunk == int(TransitionSource.HUMAN),
            source_chunk == int(TransitionSource.MIXED),
        )
        human_mask_f = human_mask.astype(jnp.float32)
        policy_mask_f = 1.0 - human_mask_f
        bc_target = jnp.where(human_mask[..., None], batch["action_chunk"], batch["ref_chunk"])
        bc_error = jnp.mean(jnp.square(action_mean - bc_target), axis=-1)
        ref_error = jnp.mean(jnp.square(action_mean - batch["ref_chunk"]), axis=-1)
        human_error = jnp.mean(jnp.square(action_mean - batch["action_chunk"]), axis=-1)
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
                batch["proprio"],
                action_q01,
                action_q99,
                action_representation=rl_config.action_representation,
                delta_action_dims=rl_config.delta_action_dims,
            )
            target_abs_chunk = jax_denormalize_to_abs_chunk(
                bc_target,
                batch["proprio"],
                action_q01,
                action_q99,
                action_representation=rl_config.action_representation,
                delta_action_dims=rl_config.delta_action_dims,
            )
        dims = rl_config.delta_action_dims
        pred_step_delta = pred_abs_chunk[:, 1:, :dims] - pred_abs_chunk[:, :-1, :dims]
        target_step_delta = target_abs_chunk[:, 1:, :dims] - target_abs_chunk[:, :-1, :dims]
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
        }
        return actor_loss, metrics

    (actor_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.actor_params)
    updates, actor_opt_state = state.actor_tx.update(grads, state.actor_opt_state, state.actor_params)
    actor_params = optax.apply_updates(state.actor_params, updates)
    new_state = state.replace(
        actor_params=actor_params,
        actor_opt_state=actor_opt_state,
        rng=next_rng,
    )
    metrics = {
        **metrics,
        "actor_loss": actor_loss,
    }
    return new_state, metrics


@functools.partial(jax.jit, static_argnames=("actor", "critic", "rl_config", "use_action_adapter"))
def train_step(
    state: RLTTrainState,
    batch: dict[str, jax.Array],
    *,
    actor: ChunkActor,
    critic: TwinCritic,
    rl_config: RLTOnlineRLConfig,
    bc_weight: float = 1.0,
    q_weight: float = 1.0,
    delta_weight: float = 0.0,
    use_action_adapter: bool = False,
    action_q01: jax.Array | None = None,
    action_q99: jax.Array | None = None,
) -> tuple[RLTTrainState, dict[str, jax.Array]]:
    state, critic_metrics = update_critic(state, batch, actor, critic, rl_config)
    should_update_actor = ((state.global_step + 1) % rl_config.actor_update_period) == 0

    zero_metrics = {
        "actor_loss": jnp.array(0.0, dtype=jnp.float32),
        "actor_q": jnp.array(0.0, dtype=jnp.float32),
        "bc_penalty": jnp.array(0.0, dtype=jnp.float32),
        "bc_ref_penalty": jnp.array(0.0, dtype=jnp.float32),
        "bc_human_penalty": jnp.array(0.0, dtype=jnp.float32),
        "human_mask_ratio": jnp.array(0.0, dtype=jnp.float32),
        "policy_mask_ratio": jnp.array(0.0, dtype=jnp.float32),
        "delta_penalty": jnp.array(0.0, dtype=jnp.float32),
        "weighted_bc": jnp.array(0.0, dtype=jnp.float32),
        "weighted_delta": jnp.array(0.0, dtype=jnp.float32),
        "weighted_q": jnp.array(0.0, dtype=jnp.float32),
    }

    def do_actor_update(train_state: RLTTrainState) -> tuple[RLTTrainState, dict[str, jax.Array]]:
        updated_state, actor_metrics = update_actor(
            train_state,
            batch,
            actor,
            critic,
            rl_config,
            bc_weight=bc_weight,
            q_weight=q_weight,
            delta_weight=delta_weight,
            use_action_adapter=use_action_adapter,
            action_q01=action_q01,
            action_q99=action_q99,
        )
        target_actor = soft_update_targets(
            updated_state.target_actor_params, updated_state.actor_params, rl_config.target_tau
        )
        target_critic = soft_update_targets(
            updated_state.target_critic_params, updated_state.critic_params, rl_config.target_tau
        )
        updated_state = updated_state.replace(
            target_actor_params=target_actor,
            target_critic_params=target_critic,
            actor_version=updated_state.actor_version + 1,
        )
        return updated_state, actor_metrics

    state, actor_metrics = jax.lax.cond(
        should_update_actor,
        do_actor_update,
        lambda train_state: (train_state, zero_metrics),
        state,
    )
    state = state.replace(global_step=state.global_step + 1)
    metrics = {
        **critic_metrics,
        **actor_metrics,
        "did_actor_update": should_update_actor.astype(jnp.float32),
        "global_step": state.global_step.astype(jnp.float32),
        "actor_version": state.actor_version.astype(jnp.float32),
        "bc_weight": jnp.asarray(bc_weight, dtype=jnp.float32),
        "q_weight": jnp.asarray(q_weight, dtype=jnp.float32),
        "delta_weight": jnp.asarray(delta_weight, dtype=jnp.float32),
    }
    return state, metrics


def _tree_to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def _tree_to_jax(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, tree)


def _ensure_source_chunk(batch_np: dict[str, np.ndarray], chunk_len: int) -> dict[str, np.ndarray]:
    if "source_chunk" in batch_np:
        return batch_np
    batch_np = dict(batch_np)
    source = np.asarray(batch_np["source"], dtype=np.uint8).reshape(-1, 1)
    batch_np["source_chunk"] = np.repeat(source, int(chunk_len), axis=1)
    return batch_np


def _sample_composition_metrics(
    batch_np: dict[str, np.ndarray],
    *,
    recent_start_episode_id: int,
) -> dict[str, float]:
    source = np.asarray(batch_np.get("source", []), dtype=np.uint8).reshape(-1)
    if source.size == 0:
        return {}
    phase = np.asarray(batch_np.get("collection_phase_id", np.zeros_like(source)), dtype=np.uint8).reshape(-1)
    intervention = np.asarray(batch_np.get("intervention_flag", np.zeros_like(source)), dtype=np.bool_).reshape(-1)
    success = np.asarray(batch_np.get("success", np.zeros_like(source)), dtype=np.float32).reshape(-1)
    episode_id = np.asarray(batch_np.get("episode_id", np.zeros_like(source)), dtype=np.int32).reshape(-1)
    step_id = np.asarray(batch_np.get("step_id", np.zeros_like(source)), dtype=np.int32).reshape(-1)
    source_chunk = np.asarray(batch_np.get("source_chunk", source[:, None]), dtype=np.uint8)
    human_chunk = np.logical_or(
        source_chunk == int(TransitionSource.HUMAN),
        source_chunk == int(TransitionSource.MIXED),
    )
    human_intervention = (
        intervention
        | (source == int(TransitionSource.HUMAN))
        | (source == int(TransitionSource.MIXED))
        | np.any(human_chunk, axis=1)
    )
    recent_online = (phase == COLLECTION_PHASE_ONLINE) & (episode_id >= recent_start_episode_id)
    return {
        "sample_recent_online_ratio": float(np.mean(recent_online)),
        "sample_warmup_demo_ratio": float(np.mean(phase == COLLECTION_PHASE_WARMUP)),
        "sample_human_intervention_ratio": float(np.mean(human_intervention)),
        "sample_success_ratio": float(np.mean(success)),
        "sample_source_base_ratio": float(np.mean(source == int(TransitionSource.BASE))),
        "sample_source_rl_ratio": float(np.mean(source == int(TransitionSource.RL))),
        "sample_source_human_ratio": float(np.mean(source == int(TransitionSource.HUMAN))),
        "sample_source_mixed_ratio": float(np.mean(source == int(TransitionSource.MIXED))),
        "sample_episode_id_min": float(np.min(episode_id)),
        "sample_episode_id_max": float(np.max(episode_id)),
        "sample_episode_id_mean": float(np.mean(episode_id)),
        "sample_step_id_mean": float(np.mean(step_id)),
    }


class LearnerService:
    """GPU training outer shell for the online RL learner."""

    def __init__(
        self,
        rl_config: RLTOnlineRLConfig,
        service_config: LearnerServiceConfig,
        replay_source: ReplayBatchSource,
        *,
        rng: jax.Array | None = None,
        metrics_path: str | None = None,
    ) -> None:
        self._rl_config = rl_config
        self._service_config = service_config
        self._replay_source = replay_source
        self._checkpoint_dir = service_config.checkpoint_dir
        self._snapshot_path = service_config.actor_snapshot_path
        self._metrics_path = metrics_path
        if metrics_path is not None:
            self._status_path = os.path.join(os.path.dirname(metrics_path), "learner_status.json")
        else:
            run_dir = os.path.dirname(os.path.abspath(self._checkpoint_dir))
            self._status_path = os.path.join(run_dir, "metrics", "learner_status.json")
        self._last_warmup_log_time = 0.0
        self._last_budget_idle_log_time = 0.0
        self._action_adapter = ActionRepresentationAdapter.from_config(rl_config)
        if self._action_adapter is None:
            self._action_q01 = None
            self._action_q99 = None
        else:
            self._action_q01 = jnp.asarray(self._action_adapter.stats.q01, dtype=jnp.float32)
            self._action_q99 = jnp.asarray(self._action_adapter.stats.q99, dtype=jnp.float32)
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._snapshot_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self._status_path) or ".", exist_ok=True)

        rng = rng if rng is not None else jax.random.PRNGKey(0)
        self._state, self._actor, self._critic = init_train_state(rl_config, rng=rng)
        self._warmup_ready_adds_total: int | None = None
        self._pending_update_budget = 0
        self._freeze_logged = False
        restored = self._load_latest_checkpoint()
        if restored is not None:
            self._state = restored
        self._refresh_progress(self._replay_source.stats())
        self.export_actor_snapshot(force=True)
        logger.debug(
            "LearnerService initialized checkpoint_dir=%s snapshot_path=%s sample_batch_size=%s warmup_min_size=%s",
            self._checkpoint_dir,
            self._snapshot_path,
            self._service_config.sample_batch_size,
            self._rl_config.warmup_min_size,
        )

    @property
    def state(self) -> RLTTrainState:
        return self._state

    def train_once(self, *, stop_event: Any | None = None) -> dict[str, float] | None:
        stats = self._replay_source.stats()
        progress = self._refresh_progress(stats)
        if progress["replay_size"] < self._rl_config.warmup_min_size:
            now = time.time()
            if now - self._last_warmup_log_time >= 5.0:
                logger.info(
                    "Waiting for warmup replay_size=%s required=%s",
                    progress["replay_size"],
                    self._rl_config.warmup_min_size,
                )
                self._last_warmup_log_time = now
            return None
        if self._rl_config.freeze_after_warmup and bool(progress["ready_for_online"]):
            if not self._freeze_logged:
                self.export_actor_snapshot(force=True)
                logger.info(
                    "Warmup complete and freeze_after_warmup enabled; learner updates disabled at global_step=%s",
                    progress["global_step"],
                )
                self._freeze_logged = True
            return None
        if self._pending_update_budget <= 0:
            now = time.time()
            if now - self._last_budget_idle_log_time >= 5.0:
                logger.info(
                    "Learner caught up replay_size=%s adds_total=%s global_step=%s ratio=%s",
                    progress["replay_size"],
                    progress["adds_total"],
                    progress["global_step"],
                    self._rl_config.grad_updates_per_cycle,
                )
                self._last_budget_idle_log_time = now
            return None

        if stop_event is not None and stop_event.is_set():
            return None

        batch_np = _ensure_source_chunk(
            self._replay_source.sample_batch(self._service_config.sample_batch_size),
            self._rl_config.chunk_len,
        )
        max_episode_id = int(stats.get("max_episode_id", -1))
        recent_window = int(stats.get("recent_episode_window", 20))
        recent_start_episode_id = max_episode_id - max(recent_window, 1) + 1
        sample_metrics = _sample_composition_metrics(
            batch_np,
            recent_start_episode_id=recent_start_episode_id,
        )
        if self._action_adapter is not None:
            batch_np = self._action_adapter.prepare_training_batch(batch_np)
        batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
        bc_weight, q_weight = _resolve_actor_loss_weights(self._rl_config, progress)
        self._state, raw_metrics = train_step(
            self._state,
            batch,
            actor=self._actor,
            critic=self._critic,
            rl_config=self._rl_config,
            bc_weight=bc_weight,
            q_weight=q_weight,
            delta_weight=self._rl_config.delta_weight,
            use_action_adapter=self._action_adapter is not None,
            action_q01=self._action_q01,
            action_q99=self._action_q99,
        )
        progress = self._refresh_progress(stats)
        metrics = {key: float(value) for key, value in jax.device_get(raw_metrics).items()}
        metrics["replay_size"] = float(progress["replay_size"])
        metrics["adds_total"] = float(progress["adds_total"])
        metrics["pending_update_budget"] = float(progress["pending_update_budget"])
        metrics["warmup_required_updates"] = float(progress["warmup_required_updates"])
        metrics["ready_for_online"] = float(progress["ready_for_online"])
        metrics.update(sample_metrics)
        if self._metrics_path is not None:
            append_jsonl(self._metrics_path, metrics)
        step = int(metrics["global_step"])
        if step <= 3 or step % 50 == 0:
            logger.info(
                "Learner step=%s actor_version=%s critic_loss=%.4f actor_loss=%.4f replay_size=%s budget=%s bc_weight=%.3f q_weight=%.3f delta_weight=%.3f",
                step,
                int(metrics["actor_version"]),
                metrics["critic_loss"],
                metrics["actor_loss"],
                progress["replay_size"],
                progress["pending_update_budget"],
                metrics["bc_weight"],
                metrics["q_weight"],
                metrics["delta_weight"],
            )
        self._maybe_save_checkpoint()
        self._maybe_export_actor_snapshot()
        return metrics

    def run_forever(self, *, stop_event: Any | None = None) -> None:
        logger.debug("LearnerService entering training loop.")
        while stop_event is None or not stop_event.is_set():
            metrics = self.train_once(stop_event=stop_event)
            if metrics is None:
                time.sleep(self._service_config.poll_interval_sec)
                continue
        logger.debug("LearnerService training loop stopped.")

    def export_actor_snapshot(self, *, force: bool = False) -> None:
        version = int(jax.device_get(self._state.actor_version))
        if not force and version == 0:
            return
        global_step = int(jax.device_get(self._state.global_step))
        payload = {
            "version": version,
            "global_step": global_step,
            "rl_config": _portable_rl_config_dict(self._rl_config, self._snapshot_path),
            "actor_params": _tree_to_numpy(self._state.actor_params),
        }
        self._write_actor_snapshot(self._snapshot_path, payload)
        if version > 0:
            self._write_actor_snapshot(self._actor_history_path(version), payload)
        logger.debug("Exported actor snapshot version=%s path=%s", version, self._snapshot_path)

    def _actor_history_path(self, version: int) -> str:
        snapshot_dir = os.path.dirname(self._snapshot_path) or "."
        return os.path.join(snapshot_dir, "history", f"actor_v{int(version):06d}.pkl")

    @staticmethod
    def _write_actor_snapshot(path: str, payload: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def get_actor_snapshot(self) -> dict[str, Any]:
        return {
            "version": int(jax.device_get(self._state.actor_version)),
            "actor_params": _tree_to_numpy(self._state.actor_params),
        }

    def _maybe_export_actor_snapshot(self) -> None:
        step = int(jax.device_get(self._state.global_step))
        if step % self._service_config.push_actor_interval_steps == 0:
            self.export_actor_snapshot(force=True)

    def _maybe_save_checkpoint(self) -> None:
        step = int(jax.device_get(self._state.global_step))
        if step % self._service_config.checkpoint_interval_steps == 0:
            self.save_checkpoint()

    def save_checkpoint(self) -> str:
        step = int(jax.device_get(self._state.global_step))
        path = os.path.join(self._checkpoint_dir, "latest.pkl")
        payload = {
            "rl_config": _portable_rl_config_dict(self._rl_config, path),
            "state": {
                "actor_params": _tree_to_numpy(self._state.actor_params),
                "target_actor_params": _tree_to_numpy(self._state.target_actor_params),
                "critic_params": _tree_to_numpy(self._state.critic_params),
                "target_critic_params": _tree_to_numpy(self._state.target_critic_params),
                "actor_opt_state": _tree_to_numpy(self._state.actor_opt_state),
                "critic_opt_state": _tree_to_numpy(self._state.critic_opt_state),
                "rng": _tree_to_numpy(self._state.rng),
                "global_step": int(jax.device_get(self._state.global_step)),
                "actor_version": int(jax.device_get(self._state.actor_version)),
            },
            "progress": {
                "warmup_ready_adds_total": self._warmup_ready_adds_total,
            },
        }
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        step_path = os.path.join(self._checkpoint_dir, f"step_{step}.pkl")
        with open(step_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved checkpoint step=%s path=%s", step, path)
        return path

    def _load_latest_checkpoint(self) -> RLTTrainState | None:
        path = os.path.join(self._checkpoint_dir, "latest.pkl")
        if not os.path.exists(path):
            logger.debug("No existing checkpoint found at %s", path)
            return None
        with open(path, "rb") as f:
            payload = pickle.load(f)
        state_payload = payload["state"]
        logger.debug(
            "Restored checkpoint from %s step=%s actor_version=%s",
            path,
            state_payload["global_step"],
            state_payload["actor_version"],
        )
        progress_payload = payload.get("progress", {})
        warmup_ready_adds_total = progress_payload.get("warmup_ready_adds_total")
        self._warmup_ready_adds_total = None if warmup_ready_adds_total is None else int(warmup_ready_adds_total)
        return RLTTrainState(
            actor_params=_tree_to_jax(state_payload["actor_params"]),
            target_actor_params=_tree_to_jax(state_payload["target_actor_params"]),
            critic_params=_tree_to_jax(state_payload["critic_params"]),
            target_critic_params=_tree_to_jax(state_payload["target_critic_params"]),
            actor_opt_state=_tree_to_jax(state_payload["actor_opt_state"]),
            critic_opt_state=_tree_to_jax(state_payload["critic_opt_state"]),
            rng=_tree_to_jax(state_payload["rng"]),
            global_step=jnp.asarray(state_payload["global_step"], dtype=jnp.int32),
            actor_version=jnp.asarray(state_payload["actor_version"], dtype=jnp.int32),
            actor_tx=optax.adam(self._rl_config.actor_lr),
            critic_tx=optax.adam(self._rl_config.critic_lr),
        )

    def flush_artifacts(self) -> None:
        logger.debug("Flushing learner artifacts before shutdown.")
        self._refresh_progress(self._replay_source.stats())
        step = int(jax.device_get(self._state.global_step))
        if step > 0:
            self.save_checkpoint()
        self.export_actor_snapshot(force=True)

    def _warmup_required_updates(self) -> int:
        if self._warmup_ready_adds_total is None:
            return 0
        if self._rl_config.warmup_post_collect_updates is not None:
            return int(self._rl_config.warmup_post_collect_updates)
        return self._warmup_ready_adds_total * self._rl_config.grad_updates_per_cycle

    def _desired_total_updates(self, replay_size: int, adds_total: int) -> int:
        if replay_size < self._rl_config.warmup_min_size or self._warmup_ready_adds_total is None:
            return 0
        warmup_required_updates = self._warmup_required_updates()
        if self._rl_config.warmup_post_collect_updates is None:
            return adds_total * self._rl_config.grad_updates_per_cycle
        online_adds_total = max(adds_total - self._warmup_ready_adds_total, 0)
        return warmup_required_updates + online_adds_total * self._rl_config.grad_updates_per_cycle

    def _refresh_progress(self, stats: dict[str, Any]) -> dict[str, int | bool]:
        replay_size = int(stats["size"])
        adds_total = int(stats.get("adds_total", replay_size))
        global_step = int(jax.device_get(self._state.global_step))
        actor_version = int(jax.device_get(self._state.actor_version))

        if replay_size >= self._rl_config.warmup_min_size and self._warmup_ready_adds_total is None:
            self._warmup_ready_adds_total = adds_total
            warmup_required_updates = self._warmup_required_updates()
            logger.info(
                "Warmup training ready latched replay_size=%s adds_total=%s required_updates=%s online_ratio=%s",
                replay_size,
                adds_total,
                warmup_required_updates,
                self._rl_config.grad_updates_per_cycle,
            )

        warmup_required_updates = self._warmup_required_updates()
        desired_total_updates = self._desired_total_updates(replay_size, adds_total)
        self._pending_update_budget = max(desired_total_updates - global_step, 0)
        ready_for_online = replay_size >= self._rl_config.warmup_min_size and global_step >= warmup_required_updates
        progress = {
            "replay_size": replay_size,
            "adds_total": adds_total,
            "global_step": global_step,
            "actor_version": actor_version,
            "pending_update_budget": self._pending_update_budget,
            "warmup_required_updates": warmup_required_updates,
            "ready_for_online": ready_for_online,
        }
        self._write_status(progress)
        return progress

    def _write_status(self, progress: dict[str, int | bool]) -> None:
        _atomic_write_json(
            self._status_path,
            {
                "replay_size": int(progress["replay_size"]),
                "adds_total": int(progress["adds_total"]),
                "global_step": int(progress["global_step"]),
                "actor_version": int(progress["actor_version"]),
                "pending_update_budget": int(progress["pending_update_budget"]),
                "warmup_ready_adds_total": self._warmup_ready_adds_total,
                "warmup_required_updates": int(progress["warmup_required_updates"]),
                "warmup_post_collect_updates": self._rl_config.warmup_post_collect_updates,
                "training_frozen": bool(self._rl_config.freeze_after_warmup and progress["ready_for_online"]),
                "ready_for_online": bool(progress["ready_for_online"]),
                "update_ratio": int(self._rl_config.grad_updates_per_cycle),
                "timestamp": time.time(),
            },
        )
