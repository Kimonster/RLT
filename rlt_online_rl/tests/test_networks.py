from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.networks import TwinCritic
from rlt_online_rl.networks import apply_reference_dropout
from rlt_online_rl.networks import build_td_target
from rlt_online_rl.networks import compute_actor_loss
from rlt_online_rl.networks import compute_critic_loss


def _config() -> RLTOnlineRLConfig:
    return RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=4,
        z_dim=5,
        proprio_dim=2,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        actor_num_layers=2,
        critic_num_layers=2,
    )


def test_actor_output_shape() -> None:
    cfg = _config()
    actor = ChunkActor(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2, cfg.fixed_std)
    params = actor.init_params(jax.random.PRNGKey(0))
    z = jnp.ones((6, cfg.z_dim))
    proprio = jnp.ones((6, cfg.proprio_dim))
    ref = jnp.ones((6, cfg.chunk_len, cfg.action_dim))
    mu = actor.actor_mean(params, z, proprio, ref)
    assert mu.shape == (6, cfg.chunk_len, cfg.action_dim)


def test_residual_actor_starts_as_exact_reference_and_is_bounded() -> None:
    cfg = _config()
    scale = 0.1
    actor = ChunkActor(
        cfg.z_dim,
        cfg.proprio_dim,
        cfg.chunk_len,
        cfg.action_dim,
        32,
        2,
        cfg.fixed_std,
        residual_scale=scale,
    )
    params = actor.init_params(jax.random.PRNGKey(11))
    z = jax.random.normal(jax.random.PRNGKey(12), (6, cfg.z_dim))
    proprio = jax.random.normal(jax.random.PRNGKey(13), (6, cfg.proprio_dim))
    ref = jax.random.normal(jax.random.PRNGKey(14), (6, cfg.chunk_len, cfg.action_dim))

    initial = actor.actor_mean(params, z, proprio, ref)
    assert jnp.array_equal(initial, ref)

    last = params["trunk"]["layers"][-1]
    changed_last = {"w": jnp.ones_like(last["w"]), "b": jnp.ones_like(last["b"])}
    changed_params = {
        **params,
        "trunk": {"layers": (*params["trunk"]["layers"][:-1], changed_last)},
    }
    changed = actor.actor_mean(changed_params, z, proprio, ref)
    assert jnp.max(jnp.abs(changed - ref)) <= scale + 1e-6


def test_twin_critic_output_shape() -> None:
    cfg = _config()
    critic = TwinCritic(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2)
    params = critic.init_params(jax.random.PRNGKey(1))
    z = jnp.ones((6, cfg.z_dim))
    proprio = jnp.ones((6, cfg.proprio_dim))
    action = jnp.ones((6, cfg.chunk_len, cfg.action_dim))
    q1, q2 = critic.q_values(params, z, proprio, action)
    assert q1.shape == (6,)
    assert q2.shape == (6,)


def test_reference_dropout_zeroes_entire_chunk() -> None:
    ref_chunk = jnp.ones((8, 4, 3))
    dropped = apply_reference_dropout(jax.random.PRNGKey(2), ref_chunk, 1.0)
    assert jnp.allclose(dropped, 0.0)


def test_actor_and_critic_losses_are_scalars() -> None:
    cfg = _config()
    actor = ChunkActor(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2, cfg.fixed_std)
    critic = TwinCritic(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2)
    actor_params = actor.init_params(jax.random.PRNGKey(3))
    critic_params = critic.init_params(jax.random.PRNGKey(4))
    batch_size = 5
    z = jnp.ones((batch_size, cfg.z_dim))
    proprio = jnp.ones((batch_size, cfg.proprio_dim))
    ref = jnp.ones((batch_size, cfg.chunk_len, cfg.action_dim))
    action = jnp.ones((batch_size, cfg.chunk_len, cfg.action_dim))
    rewards = jnp.ones((batch_size, cfg.chunk_len))
    done = jnp.zeros((batch_size,))
    actor_loss, _ = compute_actor_loss(
        actor,
        actor_params,
        critic,
        critic_params,
        z,
        proprio,
        ref,
        1.0,
        cfg.reference_dropout_prob,
        jax.random.PRNGKey(5),
    )
    critic_loss, _ = compute_critic_loss(
        critic,
        critic_params,
        actor,
        actor_params,
        critic_params,
        z,
        proprio,
        action,
        rewards,
        done,
        z,
        proprio,
        ref,
        cfg.gamma,
        jax.random.PRNGKey(6),
    )
    assert actor_loss.shape == ()
    assert critic_loss.shape == ()


def test_actor_bc_penalty_uses_deterministic_mean() -> None:
    cfg = _config()
    actor = ChunkActor(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2, cfg.fixed_std)
    critic = TwinCritic(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2)
    actor_params = actor.init_params(jax.random.PRNGKey(7))
    critic_params = critic.init_params(jax.random.PRNGKey(8))
    z = jnp.ones((5, cfg.z_dim))
    proprio = jnp.ones((5, cfg.proprio_dim))
    ref = jnp.ones((5, cfg.chunk_len, cfg.action_dim))

    _, metrics_a = compute_actor_loss(
        actor,
        actor_params,
        critic,
        critic_params,
        z,
        proprio,
        ref,
        1.0,
        0.0,
        jax.random.PRNGKey(9),
    )
    _, metrics_b = compute_actor_loss(
        actor,
        actor_params,
        critic,
        critic_params,
        z,
        proprio,
        ref,
        1.0,
        0.0,
        jax.random.PRNGKey(10),
    )

    assert jnp.allclose(metrics_a["bc_penalty"], metrics_b["bc_penalty"])


def test_actor_objective_uses_conservative_twin_value() -> None:
    class FixedTwinCritic:
        def q_values(self, _params, z_rl, _proprio, _action):
            batch_size = z_rl.shape[0]
            return jnp.full((batch_size,), 2.0), jnp.full((batch_size,), -3.0)

    cfg = _config()
    actor = ChunkActor(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2, cfg.fixed_std)
    actor_params = actor.init_params(jax.random.PRNGKey(30))
    z = jnp.ones((5, cfg.z_dim))
    proprio = jnp.ones((5, cfg.proprio_dim))
    ref = jnp.ones((5, cfg.chunk_len, cfg.action_dim))

    _, metrics = compute_actor_loss(
        actor,
        actor_params,
        FixedTwinCritic(),
        None,
        z,
        proprio,
        ref,
        1.0,
        0.0,
        jax.random.PRNGKey(31),
    )

    assert float(metrics["actor_q"]) == -3.0


def test_deterministic_td_target_does_not_depend_on_rng() -> None:
    cfg = _config()
    actor = ChunkActor(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2, cfg.fixed_std)
    critic = TwinCritic(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2)
    actor_params = actor.init_params(jax.random.PRNGKey(20))
    critic_params = critic.init_params(jax.random.PRNGKey(21))
    batch_size = 5
    z = jnp.ones((batch_size, cfg.z_dim))
    proprio = jnp.ones((batch_size, cfg.proprio_dim))
    ref = jnp.ones((batch_size, cfg.chunk_len, cfg.action_dim))
    rewards = jnp.ones((batch_size, cfg.chunk_len))
    done = jnp.zeros((batch_size,))

    target_a = build_td_target(
        actor,
        actor_params,
        critic,
        critic_params,
        z,
        proprio,
        ref,
        rewards,
        done,
        cfg.gamma,
        jax.random.PRNGKey(22),
        target_actor_deterministic=True,
    )
    target_b = build_td_target(
        actor,
        actor_params,
        critic,
        critic_params,
        z,
        proprio,
        ref,
        rewards,
        done,
        cfg.gamma,
        jax.random.PRNGKey(23),
        target_actor_deterministic=True,
    )

    assert jnp.array_equal(target_a, target_b)
