from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

PyTree = Any


def _build_hidden_dims(hidden_dim: int, num_layers: int) -> tuple[int, ...]:
    return tuple(hidden_dim for _ in range(num_layers))


def _init_linear_params(rng: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    limit = jnp.sqrt(6.0 / float(in_dim + out_dim))
    w_key, _ = jax.random.split(rng)
    return {
        "w": jax.random.uniform(w_key, (in_dim, out_dim), minval=-limit, maxval=limit),
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def _layer_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps)


def _init_mlp_params(
    rng: jax.Array,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
) -> dict[str, tuple[dict[str, jax.Array], ...]]:
    dims = (input_dim, *hidden_dims, output_dim)
    keys = jax.random.split(rng, len(dims) - 1)
    layers = tuple(_init_linear_params(k, dims[i], dims[i + 1]) for i, k in enumerate(keys))
    return {"layers": layers}


def _mlp_forward(params: PyTree, x: jax.Array) -> jax.Array:
    hidden = x
    for layer in params["layers"][:-1]:
        hidden = hidden @ layer["w"] + layer["b"]
        hidden = _layer_norm(hidden)
        hidden = jax.nn.gelu(hidden)
    last = params["layers"][-1]
    return hidden @ last["w"] + last["b"]


@dataclasses.dataclass(frozen=True)
class ChunkActor:
    z_dim: int
    proprio_dim: int
    chunk_len: int
    action_dim: int
    hidden_dim: int
    num_layers: int
    fixed_std: float
    residual_scale: float = 0.0

    def init_params(self, rng: jax.Array) -> PyTree:
        z_key, proprio_key, ref_key, trunk_key = jax.random.split(rng, 4)
        input_dim = 256 + 64 + 256
        output_dim = self.chunk_len * self.action_dim
        params = {
            "z_proj": _init_linear_params(z_key, self.z_dim, 256),
            "proprio_proj": _init_linear_params(proprio_key, self.proprio_dim, 64),
            "ref_proj": _init_linear_params(ref_key, self.chunk_len * self.action_dim, 256),
            "trunk": _init_mlp_params(
                trunk_key, input_dim, _build_hidden_dims(self.hidden_dim, self.num_layers), output_dim
            ),
        }
        if self.residual_scale > 0.0:
            last = params["trunk"]["layers"][-1]
            params["trunk"]["layers"] = (
                *params["trunk"]["layers"][:-1],
                {"w": jnp.zeros_like(last["w"]), "b": jnp.zeros_like(last["b"])},
            )
        return params

    def _encode_inputs(
        self,
        params: PyTree,
        z_rl: jax.Array,
        proprio: jax.Array,
        ref_chunk: jax.Array,
    ) -> jax.Array:
        batch_size = z_rl.shape[0]
        ref_flat = ref_chunk.reshape(batch_size, self.chunk_len * self.action_dim)
        z_feat = _layer_norm(z_rl @ params["z_proj"]["w"] + params["z_proj"]["b"])
        proprio_feat = jnp.tanh(_layer_norm(proprio @ params["proprio_proj"]["w"] + params["proprio_proj"]["b"]))
        ref_feat = jnp.tanh(_layer_norm(ref_flat @ params["ref_proj"]["w"] + params["ref_proj"]["b"]))
        return jnp.concatenate([z_feat, proprio_feat, ref_feat], axis=-1)

    def actor_mean(
        self,
        params: PyTree,
        z_rl: jax.Array,
        proprio: jax.Array,
        ref_chunk: jax.Array,
    ) -> jax.Array:
        batch_size = z_rl.shape[0]
        features = self._encode_inputs(params, z_rl, proprio, ref_chunk)
        mu = _mlp_forward(params["trunk"], features).reshape(batch_size, self.chunk_len, self.action_dim)
        if self.residual_scale > 0.0:
            return ref_chunk + self.residual_scale * jnp.tanh(mu)
        return mu

    def actor_dist(
        self,
        params: PyTree,
        z_rl: jax.Array,
        proprio: jax.Array,
        ref_chunk: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        mu = self.actor_mean(params, z_rl, proprio, ref_chunk)
        std = jnp.full_like(mu, self.fixed_std)
        return mu, std

    def sample_action(
        self,
        params: PyTree,
        rng: jax.Array,
        z_rl: jax.Array,
        proprio: jax.Array,
        ref_chunk: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        mu, std = self.actor_dist(params, z_rl, proprio, ref_chunk)
        if deterministic:
            return mu
        noise = jax.random.normal(rng, mu.shape, dtype=mu.dtype)
        return mu + std * noise


@dataclasses.dataclass(frozen=True)
class QNetwork:
    z_dim: int
    proprio_dim: int
    chunk_len: int
    action_dim: int
    hidden_dim: int
    num_layers: int

    def init_params(self, rng: jax.Array) -> PyTree:
        z_key, proprio_key, action_key, trunk_key = jax.random.split(rng, 4)
        input_dim = 256 + 64 + 256
        return {
            "z_proj": _init_linear_params(z_key, self.z_dim, 256),
            "proprio_proj": _init_linear_params(proprio_key, self.proprio_dim, 64),
            "action_proj": _init_linear_params(action_key, self.chunk_len * self.action_dim, 256),
            "trunk": _init_mlp_params(trunk_key, input_dim, _build_hidden_dims(self.hidden_dim, self.num_layers), 1),
        }

    def apply(
        self,
        params: PyTree,
        z_rl: jax.Array,
        proprio: jax.Array,
        action_chunk: jax.Array,
    ) -> jax.Array:
        batch_size = z_rl.shape[0]
        action_flat = action_chunk.reshape(batch_size, self.chunk_len * self.action_dim)
        z_feat = _layer_norm(z_rl @ params["z_proj"]["w"] + params["z_proj"]["b"])
        proprio_feat = jnp.tanh(_layer_norm(proprio @ params["proprio_proj"]["w"] + params["proprio_proj"]["b"]))
        action_feat = jnp.tanh(_layer_norm(action_flat @ params["action_proj"]["w"] + params["action_proj"]["b"]))
        features = jnp.concatenate([z_feat, proprio_feat, action_feat], axis=-1)
        q_value = _mlp_forward(params["trunk"], features)
        return q_value.squeeze(-1)


@dataclasses.dataclass(frozen=True)
class TwinCritic:
    z_dim: int
    proprio_dim: int
    chunk_len: int
    action_dim: int
    hidden_dim: int
    num_layers: int

    def init_params(self, rng: jax.Array) -> PyTree:
        q1_key, q2_key = jax.random.split(rng)
        q_network = QNetwork(
            z_dim=self.z_dim,
            proprio_dim=self.proprio_dim,
            chunk_len=self.chunk_len,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        return {
            "q1": q_network.init_params(q1_key),
            "q2": q_network.init_params(q2_key),
        }

    def q_values(
        self,
        params: PyTree,
        z_rl: jax.Array,
        proprio: jax.Array,
        action_chunk: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        q_network = QNetwork(
            z_dim=self.z_dim,
            proprio_dim=self.proprio_dim,
            chunk_len=self.chunk_len,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        q1 = q_network.apply(params["q1"], z_rl, proprio, action_chunk)
        q2 = q_network.apply(params["q2"], z_rl, proprio, action_chunk)
        return q1, q2


def apply_reference_dropout(
    rng: jax.Array,
    ref_chunk: jax.Array,
    dropout_prob: float,
) -> jax.Array:
    if dropout_prob <= 0.0:
        return ref_chunk
    keep_mask = jax.random.bernoulli(rng, 1.0 - dropout_prob, (ref_chunk.shape[0], 1, 1))
    return ref_chunk * keep_mask.astype(ref_chunk.dtype)


def _discounted_chunk_rewards(rewards: jax.Array, gamma: float) -> jax.Array:
    discounts = jnp.power(gamma, jnp.arange(rewards.shape[-1], dtype=rewards.dtype))
    return jnp.sum(rewards * discounts[None, :], axis=-1)


def build_td_target(
    target_actor: ChunkActor,
    target_actor_params: PyTree,
    target_critic: TwinCritic,
    target_critic_params: PyTree,
    next_z_rl: jax.Array,
    next_proprio: jax.Array,
    next_ref_chunk: jax.Array,
    rewards: jax.Array,
    done: jax.Array,
    gamma: float,
    rng: jax.Array,
    *,
    target_actor_deterministic: bool = False,
) -> jax.Array:
    next_action = target_actor.sample_action(
        target_actor_params,
        rng,
        next_z_rl,
        next_proprio,
        next_ref_chunk,
        deterministic=target_actor_deterministic,
    )
    next_q1, next_q2 = target_critic.q_values(target_critic_params, next_z_rl, next_proprio, next_action)
    bootstrap = (1.0 - done.astype(rewards.dtype)) * (gamma ** rewards.shape[-1]) * jnp.minimum(next_q1, next_q2)
    return _discounted_chunk_rewards(rewards, gamma) + bootstrap


def compute_actor_loss(
    actor: ChunkActor,
    actor_params: PyTree,
    critic: TwinCritic,
    critic_params: PyTree,
    z_rl: jax.Array,
    proprio: jax.Array,
    ref_chunk: jax.Array,
    beta: float,
    reference_dropout_prob: float,
    rng: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    dropout_rng, sample_rng = jax.random.split(rng)
    dropped_ref = apply_reference_dropout(dropout_rng, ref_chunk, reference_dropout_prob)
    # Explore for the Q term, while supervising the deterministic policy used at deployment.
    action_mean, action_std = actor.actor_dist(actor_params, z_rl, proprio, dropped_ref)
    sampled_action = action_mean + action_std * jax.random.normal(
        sample_rng, action_mean.shape, dtype=action_mean.dtype
    )
    q1, q2 = critic.q_values(critic_params, z_rl, proprio, sampled_action)
    actor_q = jnp.mean(jnp.minimum(q1, q2))
    bc_penalty = jnp.mean(jnp.square(action_mean - ref_chunk))
    actor_loss = -actor_q + beta * bc_penalty
    metrics = {
        "actor_loss": actor_loss,
        "actor_q": actor_q,
        "bc_penalty": bc_penalty,
    }
    return actor_loss, metrics


def compute_critic_loss(
    critic: TwinCritic,
    critic_params: PyTree,
    actor: ChunkActor,
    target_actor_params: PyTree,
    target_critic_params: PyTree,
    z_rl: jax.Array,
    proprio: jax.Array,
    action_chunk: jax.Array,
    rewards: jax.Array,
    done: jax.Array,
    next_z_rl: jax.Array,
    next_proprio: jax.Array,
    next_ref_chunk: jax.Array,
    gamma: float,
    rng: jax.Array,
    *,
    target_actor_deterministic: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    q1, q2 = critic.q_values(critic_params, z_rl, proprio, action_chunk)
    target_q = build_td_target(
        actor,
        target_actor_params,
        critic,
        target_critic_params,
        next_z_rl,
        next_proprio,
        next_ref_chunk,
        rewards,
        done,
        gamma,
        rng,
        target_actor_deterministic=target_actor_deterministic,
    )
    critic_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
    metrics = {
        "critic_loss": critic_loss,
        "q1_mean": jnp.mean(q1),
        "q2_mean": jnp.mean(q2),
        "target_q_mean": jnp.mean(target_q),
    }
    return critic_loss, metrics
