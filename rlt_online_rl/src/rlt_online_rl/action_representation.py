from __future__ import annotations

import dataclasses
import json

import jax.numpy as jnp
import numpy as np

from rlt_online_rl.config import RLTOnlineRLConfig


@dataclasses.dataclass(frozen=True)
class QuantileStats:
    q01: np.ndarray
    q99: np.ndarray


def _load_quantile_stats(path: str, action_dim: int) -> QuantileStats:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    stats = payload["norm_stats"]["actions"]
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    if q01.shape[-1] < action_dim or q99.shape[-1] < action_dim:
        raise ValueError(f"Action stats at {path} have fewer than {action_dim} dimensions")
    q01 = q01[..., :action_dim]
    q99 = q99[..., :action_dim]
    return QuantileStats(q01=q01, q99=q99)


def _quantile_normalize(x: np.ndarray, stats: QuantileStats) -> np.ndarray:
    q01 = stats.q01.astype(np.float32, copy=False)
    q99 = stats.q99.astype(np.float32, copy=False)
    scale = q99 - q01 + 1e-6
    return (x - q01) / scale * 2.0 - 1.0


def _quantile_denormalize(x: np.ndarray, stats: QuantileStats) -> np.ndarray:
    q01 = stats.q01.astype(np.float32, copy=False)
    q99 = stats.q99.astype(np.float32, copy=False)
    scale = q99 - q01 + 1e-6
    return (x + 1.0) * 0.5 * scale + q01


def _zero_row_mask(chunk: np.ndarray) -> np.ndarray:
    return np.all(np.isclose(chunk, 0.0), axis=-1, keepdims=True)


def _broadcast_state0(state0: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    state0 = np.asarray(state0, dtype=np.float32)
    while state0.ndim < chunk.ndim:
        state0 = np.expand_dims(state0, axis=-2)
    return state0


def _broadcast_state0_jax(state0: jnp.ndarray, chunk: jnp.ndarray) -> jnp.ndarray:
    state0 = jnp.asarray(state0, dtype=jnp.float32)
    while state0.ndim < chunk.ndim:
        state0 = jnp.expand_dims(state0, axis=-2)
    return state0


def jax_quantile_denormalize(x: jnp.ndarray, q01: jnp.ndarray, q99: jnp.ndarray) -> jnp.ndarray:
    q01 = jnp.asarray(q01, dtype=jnp.float32)
    q99 = jnp.asarray(q99, dtype=jnp.float32)
    scale = q99 - q01 + 1e-6
    return (jnp.asarray(x, dtype=jnp.float32) + 1.0) * 0.5 * scale + q01


def jax_delta_to_abs_chunk(chunk_delta: jnp.ndarray, state0: jnp.ndarray, delta_action_dims: int = 6) -> jnp.ndarray:
    chunk_delta = jnp.asarray(chunk_delta, dtype=jnp.float32)
    state0 = _broadcast_state0_jax(state0, chunk_delta)
    return chunk_delta.at[..., :delta_action_dims].add(state0[..., :delta_action_dims])


def jax_denormalize_to_abs_chunk(
    chunk_norm: jnp.ndarray,
    state0: jnp.ndarray,
    q01: jnp.ndarray,
    q99: jnp.ndarray,
    *,
    action_representation: str,
    delta_action_dims: int = 6,
) -> jnp.ndarray:
    chunk_repr = jax_quantile_denormalize(chunk_norm, q01, q99)
    if action_representation == "abs":
        return chunk_repr
    return jax_delta_to_abs_chunk(chunk_repr, state0, delta_action_dims)


@dataclasses.dataclass(frozen=True)
class ActionRepresentationAdapter:
    rl_config: RLTOnlineRLConfig
    stats: QuantileStats

    @classmethod
    def from_config(cls, rl_config: RLTOnlineRLConfig) -> ActionRepresentationAdapter | None:
        if rl_config.action_norm_stats_path is None:
            return None
        return cls(
            rl_config=rl_config,
            stats=_load_quantile_stats(rl_config.action_norm_stats_path, rl_config.action_dim),
        )

    def _abs_to_delta_chunk(self, chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        chunk_abs = np.asarray(chunk_abs, dtype=np.float32)
        state0 = _broadcast_state0(state0, chunk_abs)
        chunk_delta = chunk_abs.copy()
        zero_mask = _zero_row_mask(chunk_abs)
        dims = self.rl_config.delta_action_dims
        chunk_delta[..., :dims] = chunk_abs[..., :dims] - state0[..., :dims]
        return np.where(zero_mask, 0.0, chunk_delta)

    def _delta_to_abs_chunk(self, chunk_delta: np.ndarray, state0: np.ndarray) -> np.ndarray:
        chunk_delta = np.asarray(chunk_delta, dtype=np.float32)
        state0 = _broadcast_state0(state0, chunk_delta)
        chunk_abs = chunk_delta.copy()
        dims = self.rl_config.delta_action_dims
        chunk_abs[..., :dims] = chunk_delta[..., :dims] + state0[..., :dims]
        return chunk_abs

    def _to_representation(self, chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        if self.rl_config.action_representation == "abs":
            return np.asarray(chunk_abs, dtype=np.float32)
        return self._abs_to_delta_chunk(chunk_abs, state0)

    def _from_representation(self, chunk_repr: np.ndarray, state0: np.ndarray) -> np.ndarray:
        if self.rl_config.action_representation == "abs":
            return np.asarray(chunk_repr, dtype=np.float32)
        return self._delta_to_abs_chunk(chunk_repr, state0)

    def normalize_chunk(self, chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        chunk_repr = self._to_representation(chunk_abs, state0)
        normalized = _quantile_normalize(chunk_repr, self.stats)
        return np.where(_zero_row_mask(chunk_abs), 0.0, normalized).astype(np.float32, copy=False)

    def denormalize_to_abs_chunk(self, chunk_norm: np.ndarray, state0: np.ndarray) -> np.ndarray:
        chunk_repr = _quantile_denormalize(np.asarray(chunk_norm, dtype=np.float32), self.stats)
        return self._from_representation(chunk_repr, state0).astype(np.float32, copy=False)

    def normalize_ref_chunk(self, ref_chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        return self.normalize_chunk(ref_chunk_abs, state0)

    def prepare_training_batch(self, batch_np: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        proprio = np.asarray(batch_np["proprio"], dtype=np.float32)
        next_proprio = np.asarray(batch_np["next_proprio"], dtype=np.float32)
        transformed = dict(batch_np)
        transformed["ref_chunk"] = self.normalize_ref_chunk(batch_np["ref_chunk"], proprio)
        transformed["action_chunk"] = self.normalize_chunk(batch_np["action_chunk"], proprio)
        transformed["next_ref_chunk"] = self.normalize_ref_chunk(batch_np["next_ref_chunk"], next_proprio)
        transformed["proprio"] = proprio
        transformed["next_proprio"] = next_proprio
        return transformed
