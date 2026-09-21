from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
from _common import PHASE_CHOICES
from _common import SOURCE_CHOICES
from _common import ActionRepresentationAdapter
from _common import build_model_ref_chunk
from _common import default_filter_suffix
from _common import filter_replay_records
from _common import infer_task_dir_from_replay_path
from _common import load_critic_snapshot
from _common import load_replay_journal
from _common import load_snapshot
from _common import resolve_default_actor_checkpoint_path
from _common import resolve_default_critic_snapshot_path
import matplotlib.pyplot as plt
import numpy as np

from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.networks import TwinCritic

"""
Analyze critic Q trajectories for one or more replay episodes.

Main reported comparisons:

- `q_data`: critic value for the replay `action_chunk`
- `q_ref`: critic value for `ref_chunk`
- `q_actor_mean`: critic value for deterministic actor mean
- `q_actor_sample`: optional value for actor samples when `actor-mode=sample`
- `chunk_reward_sum`: discounted reward sum inside the chunk

Replay filters match the current replay semantics:

- `phase`: `all / warmup / online / unknown`
- `source`: `all / base / rl / human / mixed`
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate critic Q trajectories on one or more replay episodes.")
    parser.add_argument(
        "--replay-path",
        type=Path,
        required=True,
        help="Replay journal to analyze, usually runs/<task>/replay/replay_journal.pkl",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory used to auto-resolve actor/critic artifacts and default output locations. Defaults to <task-dir>/offline_train_bcq",
    )
    parser.add_argument(
        "--episode-ids",
        type=str,
        nargs="+",
        required=True,
        help="One or more replay episode ids to evaluate. Supports both discrete ids (4 12 18) and inclusive ranges (1-20 35 40-45).",
    )
    parser.add_argument(
        "--actor-path",
        type=Path,
        default=None,
        help="Optional explicit actor snapshot/checkpoint path. Overrides --model-dir auto resolution.",
    )
    parser.add_argument(
        "--critic-path",
        type=Path,
        default=None,
        help="Optional explicit critic snapshot/checkpoint path. Overrides --model-dir auto resolution.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--disable-ref-input", action="store_true", help="Evaluate actor Q with ref_chunk input replaced by zeros."
    )
    parser.add_argument(
        "--actor-mode",
        choices=("mean", "sample"),
        default="mean",
        help="mean: evaluate actor mean without std sampling; sample: additionally evaluate q_actor_sample using the actor's fixed_std.",
    )
    parser.add_argument("--actor-seed", type=int, default=0, help="Random seed used only when actor-mode=sample.")
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    parser.add_argument("--source", choices=tuple(SOURCE_CHOICES), default="all")
    return parser.parse_args()


def _expand_episode_id_token(token: str) -> list[int]:
    token = token.strip()
    if not token:
        raise argparse.ArgumentTypeError("episode id token cannot be empty")
    if token.count("-") == 1:
        start_str, end_str = token.split("-", 1)
        if start_str.isdigit() and end_str.isdigit():
            start = int(start_str)
            end = int(end_str)
            if start > end:
                raise argparse.ArgumentTypeError(f"invalid episode id range '{token}': start must be <= end")
            return list(range(start, end + 1))
    try:
        return [int(token)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid episode id token '{token}': expected an integer or range like 1-20"
        ) from exc


def _resolve_episode_ids(episode_ids: list[str]) -> list[int]:
    resolved: list[int] = []
    for episode_id in episode_ids:
        resolved.extend(_expand_episode_id_token(episode_id))
    return sorted(set(resolved))


def _relativize_path(path_value: Path, anchor_path: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    resolved = path_value.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return os.path.relpath(str(resolved), start=str(anchor_path.resolve().parent))
    return os.path.relpath(str(resolved), start=str(root))


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
    )
    return actor, critic


def _normalize_action_chunk(
    adapter: ActionRepresentationAdapter | None,
    action_chunk_abs: np.ndarray,
    proprio: np.ndarray,
) -> np.ndarray:
    if adapter is None:
        return np.asarray(action_chunk_abs, dtype=np.float32)
    return adapter.normalize_chunk(action_chunk_abs, proprio)


def _discounted_chunk_reward(rewards: np.ndarray, gamma: float) -> float:
    rewards = np.asarray(rewards, dtype=np.float32)
    discounts = np.power(np.float32(gamma), np.arange(rewards.shape[0], dtype=np.float32))
    return float(np.sum(rewards * discounts))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _adjacent_rise_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1:
        return float("nan")
    return float(np.mean(values[1:] >= values[:-1]))


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.size <= 1 or np.allclose(x, x[0]):
        return float("nan")
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def _select_episode_records(records: list[dict[str, Any]], episode_id: int) -> list[dict[str, Any]]:
    selected = [record for record in records if int(record["episode_id"]) == int(episode_id)]
    if not selected:
        raise RuntimeError(f"No replay records found for episode_id={episode_id}")
    selected.sort(key=lambda record: (int(record["step_id"]), bool(record["done"])))
    return selected


def _annotate_episode(records: list[dict[str, Any]], gamma: float) -> dict[str, np.ndarray]:
    num_records = len(records)
    chunk_index = np.arange(num_records, dtype=np.int32)
    distance_to_terminal = (num_records - 1 - chunk_index).astype(np.int32, copy=False)
    chunk_reward_sum = np.asarray(
        [_discounted_chunk_reward(np.asarray(record["rewards"], dtype=np.float32), gamma) for record in records],
        dtype=np.float32,
    )
    done = np.asarray([bool(record["done"]) for record in records], dtype=np.bool_)
    transition_success = np.asarray([int(record["success"]) for record in records], dtype=np.int32)
    step_id = np.asarray([int(record["step_id"]) for record in records], dtype=np.int32)
    return {
        "chunk_index": chunk_index,
        "distance_to_terminal": distance_to_terminal,
        "chunk_reward_sum": chunk_reward_sum,
        "done": done,
        "transition_success": transition_success,
        "step_id": step_id,
    }


def _predict_q_arrays(
    actor: ChunkActor,
    critic: TwinCritic,
    actor_params: Any,
    critic_params: Any,
    adapter: ActionRepresentationAdapter | None,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    disable_ref_input: bool,
    actor_mode: str,
    actor_seed: int,
) -> dict[str, np.ndarray]:
    z_rl = np.stack([np.asarray(record["z_rl"], dtype=np.float32) for record in records], axis=0)
    proprio = np.stack([np.asarray(record["proprio"], dtype=np.float32) for record in records], axis=0)
    ref_chunk_abs = np.stack([np.asarray(record["ref_chunk"], dtype=np.float32) for record in records], axis=0)
    action_chunk_abs = np.stack([np.asarray(record["action_chunk"], dtype=np.float32) for record in records], axis=0)

    q_data = []
    q_ref = []
    q_actor_mean = []
    q_actor_sample = [] if actor_mode == "sample" else None
    q1_data = []
    q2_data = []
    q_gap_data = []
    base_rng = jax.random.PRNGKey(actor_seed)

    for batch_index, start in enumerate(range(0, len(records), batch_size)):
        end = min(start + batch_size, len(records))
        z_batch = z_rl[start:end]
        proprio_batch = proprio[start:end]
        ref_abs_batch = ref_chunk_abs[start:end]
        action_abs_batch = action_chunk_abs[start:end]
        ref_model_batch = build_model_ref_chunk(adapter, ref_abs_batch, proprio_batch, disable_ref_input=False)
        action_model_batch = _normalize_action_chunk(adapter, action_abs_batch, proprio_batch)
        actor_ref_model_batch = build_model_ref_chunk(
            adapter, ref_abs_batch, proprio_batch, disable_ref_input=disable_ref_input
        )

        actor_mean_batch = actor.actor_mean(
            actor_params,
            jnp.asarray(z_batch),
            jnp.asarray(proprio_batch),
            jnp.asarray(actor_ref_model_batch),
        )
        data_q1, data_q2 = critic.q_values(
            critic_params,
            jnp.asarray(z_batch),
            jnp.asarray(proprio_batch),
            jnp.asarray(action_model_batch),
        )
        ref_q1, ref_q2 = critic.q_values(
            critic_params,
            jnp.asarray(z_batch),
            jnp.asarray(proprio_batch),
            jnp.asarray(ref_model_batch),
        )
        actor_q1, actor_q2 = critic.q_values(
            critic_params,
            jnp.asarray(z_batch),
            jnp.asarray(proprio_batch),
            actor_mean_batch,
        )

        data_q1_np = np.asarray(jax.device_get(data_q1), dtype=np.float32)
        data_q2_np = np.asarray(jax.device_get(data_q2), dtype=np.float32)
        ref_q1_np = np.asarray(jax.device_get(ref_q1), dtype=np.float32)
        ref_q2_np = np.asarray(jax.device_get(ref_q2), dtype=np.float32)
        actor_q1_np = np.asarray(jax.device_get(actor_q1), dtype=np.float32)
        actor_q2_np = np.asarray(jax.device_get(actor_q2), dtype=np.float32)

        q1_data.append(data_q1_np)
        q2_data.append(data_q2_np)
        q_gap_data.append(np.abs(data_q1_np - data_q2_np))
        q_data.append(np.minimum(data_q1_np, data_q2_np))
        q_ref.append(np.minimum(ref_q1_np, ref_q2_np))
        q_actor_mean.append(np.minimum(actor_q1_np, actor_q2_np))

        if actor_mode == "sample":
            assert q_actor_sample is not None
            sample_rng = jax.random.fold_in(base_rng, batch_index)
            actor_sample_batch = actor.sample_action(
                actor_params,
                sample_rng,
                jnp.asarray(z_batch),
                jnp.asarray(proprio_batch),
                jnp.asarray(actor_ref_model_batch),
                deterministic=False,
            )
            actor_sample_q1, actor_sample_q2 = critic.q_values(
                critic_params,
                jnp.asarray(z_batch),
                jnp.asarray(proprio_batch),
                actor_sample_batch,
            )
            actor_sample_q1_np = np.asarray(jax.device_get(actor_sample_q1), dtype=np.float32)
            actor_sample_q2_np = np.asarray(jax.device_get(actor_sample_q2), dtype=np.float32)
            q_actor_sample.append(np.minimum(actor_sample_q1_np, actor_sample_q2_np))

    return {
        "q_data": np.concatenate(q_data, axis=0),
        "q_ref": np.concatenate(q_ref, axis=0),
        "q_actor_mean": np.concatenate(q_actor_mean, axis=0),
        "q_actor_sample": None if q_actor_sample is None else np.concatenate(q_actor_sample, axis=0),
        "q1_data": np.concatenate(q1_data, axis=0),
        "q2_data": np.concatenate(q2_data, axis=0),
        "q_gap_data": np.concatenate(q_gap_data, axis=0),
    }


def _distance_table(
    distance_to_terminal: np.ndarray,
    q_data: np.ndarray,
    q_ref: np.ndarray,
    q_actor_mean: np.ndarray,
    q_actor_sample: np.ndarray | None,
    chunk_reward_sum: np.ndarray,
    q_gap_data: np.ndarray,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for distance in np.unique(distance_to_terminal):
        mask = distance_to_terminal == distance
        rows.append(
            {
                "distance_to_terminal": float(distance),
                "count": float(mask.sum()),
                "mean_q_data": float(q_data[mask].mean()),
                "mean_q_ref": float(q_ref[mask].mean()),
                "mean_q_actor_mean": float(q_actor_mean[mask].mean()),
                "mean_chunk_reward_sum": float(chunk_reward_sum[mask].mean()),
                "mean_q_gap_data": float(q_gap_data[mask].mean()),
            }
        )
        if q_actor_sample is not None:
            rows[-1]["mean_q_actor_sample"] = float(q_actor_sample[mask].mean())
    rows.sort(key=lambda row: row["distance_to_terminal"])
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_episode_q_profile(
    path: Path,
    chunk_index: np.ndarray,
    q_data: np.ndarray,
    q_ref: np.ndarray,
    q_actor_mean: np.ndarray,
    q_actor_sample: np.ndarray | None,
    chunk_reward_sum: np.ndarray,
    done: np.ndarray,
    transition_success: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(chunk_index, q_data, marker="o", linewidth=2.0, label="Q_data")
    ax.plot(chunk_index, q_ref, marker="o", linewidth=2.0, label="Q_ref")
    ax.plot(chunk_index, q_actor_mean, marker="o", linewidth=2.0, label="Q_actor_mean")
    if q_actor_sample is not None:
        ax.plot(chunk_index, q_actor_sample, marker="o", linewidth=2.0, linestyle="--", label="Q_actor_sample")
    ax.plot(
        chunk_index,
        chunk_reward_sum,
        marker="s",
        linewidth=1.8,
        linestyle="--",
        color="tab:gray",
        label="chunk_reward_sum",
    )
    ax.set_title("Episode Q Profile by Chunk Order")
    ax.set_xlabel("chunk index (chronological)")
    ax.set_ylabel("Q / discounted chunk reward")
    ax.grid(True, alpha=0.25)

    terminal_idx = np.where(done)[0]
    if terminal_idx.size > 0:
        ax.axvline(x=chunk_index[terminal_idx[-1]], color="tab:red", linestyle="--", alpha=0.6, label="terminal chunk")
    success_idx = np.where(transition_success > 0)[0]
    if success_idx.size > 0:
        ax.scatter(
            chunk_index[success_idx],
            q_data[success_idx],
            color="tab:red",
            s=50,
            zorder=5,
            label="success chunk (Q_data)",
        )

    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_q_by_distance(path: Path, distance_rows: list[dict[str, float]]) -> None:
    xs = np.asarray([row["distance_to_terminal"] for row in distance_rows], dtype=np.float32)
    q_data = np.asarray([row["mean_q_data"] for row in distance_rows], dtype=np.float32)
    q_ref = np.asarray([row["mean_q_ref"] for row in distance_rows], dtype=np.float32)
    q_actor_mean = np.asarray([row["mean_q_actor_mean"] for row in distance_rows], dtype=np.float32)
    q_actor_sample = (
        np.asarray([row["mean_q_actor_sample"] for row in distance_rows], dtype=np.float32)
        if "mean_q_actor_sample" in distance_rows[0]
        else None
    )
    chunk_reward = np.asarray([row["mean_chunk_reward_sum"] for row in distance_rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(xs, q_data, marker="o", linewidth=2.0, label="Q_data")
    ax.plot(xs, q_ref, marker="o", linewidth=2.0, label="Q_ref")
    ax.plot(xs, q_actor_mean, marker="o", linewidth=2.0, label="Q_actor_mean")
    if q_actor_sample is not None:
        ax.plot(xs, q_actor_sample, marker="o", linewidth=2.0, linestyle="--", label="Q_actor_sample")
    ax.plot(xs, chunk_reward, marker="s", linewidth=1.8, linestyle="--", color="tab:gray", label="chunk_reward_sum")
    ax.set_title("Episode Q vs Distance to Terminal Chunk")
    ax.set_xlabel("distance to terminal chunk (0 = nearest)")
    ax.set_ylabel("Q / discounted chunk reward")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _summary_payload(
    *,
    output_dir: Path,
    actor_path: Path,
    critic_path: Path,
    replay_path: Path,
    episode_id: int,
    records: list[dict[str, Any]],
    annotations: dict[str, np.ndarray],
    q_payload: dict[str, np.ndarray],
    actor_mode: str,
    actor_seed: int,
    fixed_std: float,
) -> dict[str, Any]:
    chunk_index = annotations["chunk_index"]
    distance_to_terminal = annotations["distance_to_terminal"]
    q_data = q_payload["q_data"]
    q_ref = q_payload["q_ref"]
    q_actor_mean = q_payload["q_actor_mean"]
    q_actor_sample = q_payload["q_actor_sample"]
    q_gap_data = q_payload["q_gap_data"]
    chunk_reward_sum = annotations["chunk_reward_sum"]

    near_count = max(1, min(3, len(records) // 3 if len(records) >= 3 else 1))
    far_slice = slice(0, near_count)
    near_slice = slice(len(records) - near_count, len(records))

    payload = {
        "actor_path": _relativize_path(actor_path, output_dir / "summary.json"),
        "critic_path": _relativize_path(critic_path, output_dir / "summary.json"),
        "replay_path": _relativize_path(replay_path, output_dir / "summary.json"),
        "actor_mode": actor_mode,
        "actor_seed": int(actor_seed),
        "fixed_std": float(fixed_std),
        "episode_id": int(episode_id),
        "collection_phase_counts": {
            phase: int(sum(str(record.get("collection_phase", "unknown")) == phase for record in records))
            for phase in sorted({str(record.get("collection_phase", "unknown")) for record in records})
        },
        "num_chunks": int(len(records)),
        "episode_success": int(np.max(annotations["transition_success"])),
        "num_done_chunks": int(np.sum(annotations["done"])),
        "terminal_step_id": int(annotations["step_id"][-1]),
        "q_data": _stats(q_data),
        "q_ref": _stats(q_ref),
        "q_actor_mean": _stats(q_actor_mean),
        "q_gap_data": _stats(q_gap_data),
        "chunk_reward_sum": _stats(chunk_reward_sum),
        "trend": {
            "corr_q_data_vs_chunk_index": _safe_corr(q_data, chunk_index),
            "corr_q_ref_vs_chunk_index": _safe_corr(q_ref, chunk_index),
            "corr_q_actor_mean_vs_chunk_index": _safe_corr(q_actor_mean, chunk_index),
            "corr_q_data_vs_distance_to_terminal": _safe_corr(q_data, distance_to_terminal),
            "corr_q_ref_vs_distance_to_terminal": _safe_corr(q_ref, distance_to_terminal),
            "corr_q_actor_mean_vs_distance_to_terminal": _safe_corr(q_actor_mean, distance_to_terminal),
            "q_data_adjacent_rise_fraction": _adjacent_rise_fraction(q_data),
            "q_ref_adjacent_rise_fraction": _adjacent_rise_fraction(q_ref),
            "q_actor_mean_adjacent_rise_fraction": _adjacent_rise_fraction(q_actor_mean),
            "q_data_slope_vs_chunk_index": _linear_slope(chunk_index, q_data),
            "q_ref_slope_vs_chunk_index": _linear_slope(chunk_index, q_ref),
            "q_actor_mean_slope_vs_chunk_index": _linear_slope(chunk_index, q_actor_mean),
            "q_data_slope_vs_distance_to_terminal": _linear_slope(distance_to_terminal, q_data),
            "q_ref_slope_vs_distance_to_terminal": _linear_slope(distance_to_terminal, q_ref),
            "q_actor_mean_slope_vs_distance_to_terminal": _linear_slope(distance_to_terminal, q_actor_mean),
        },
        "near_vs_far": {
            "window_size": int(near_count),
            "far_mean_q_data": float(q_data[far_slice].mean()),
            "near_mean_q_data": float(q_data[near_slice].mean()),
            "far_mean_q_ref": float(q_ref[far_slice].mean()),
            "near_mean_q_ref": float(q_ref[near_slice].mean()),
            "far_mean_q_actor_mean": float(q_actor_mean[far_slice].mean()),
            "near_mean_q_actor_mean": float(q_actor_mean[near_slice].mean()),
            "far_mean_chunk_reward_sum": float(chunk_reward_sum[far_slice].mean()),
            "near_mean_chunk_reward_sum": float(chunk_reward_sum[near_slice].mean()),
        },
    }
    if q_actor_sample is not None:
        payload["q_actor_sample"] = _stats(q_actor_sample)
        payload["trend"].update(
            {
                "corr_q_actor_sample_vs_chunk_index": _safe_corr(q_actor_sample, chunk_index),
                "corr_q_actor_sample_vs_distance_to_terminal": _safe_corr(q_actor_sample, distance_to_terminal),
                "q_actor_sample_adjacent_rise_fraction": _adjacent_rise_fraction(q_actor_sample),
                "q_actor_sample_slope_vs_chunk_index": _linear_slope(chunk_index, q_actor_sample),
                "q_actor_sample_slope_vs_distance_to_terminal": _linear_slope(distance_to_terminal, q_actor_sample),
            }
        )
        payload["near_vs_far"].update(
            {
                "far_mean_q_actor_sample": float(q_actor_sample[far_slice].mean()),
                "near_mean_q_actor_sample": float(q_actor_sample[near_slice].mean()),
            }
        )
    return payload


def _evaluate_episode(
    *,
    episode_id: int,
    replay_records: list[dict[str, Any]],
    actor_cfg: Any,
    actor_params: Any,
    critic_cfg: Any,
    critic_params: Any,
    actor_path: Path,
    critic_path: Path,
    replay_path: Path,
    batch_size: int,
    disable_ref_input: bool,
    actor_mode: str,
    actor_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    episode_records = _select_episode_records(replay_records, episode_id)
    if dataclasses.asdict(actor_cfg) != dataclasses.asdict(critic_cfg):
        raise ValueError("Actor and critic snapshots were saved with different rl_config values.")

    rl_config = actor_cfg
    adapter = ActionRepresentationAdapter.from_config(rl_config)
    actor, critic = _build_models(rl_config)
    annotations = _annotate_episode(episode_records, rl_config.gamma)
    q_payload = _predict_q_arrays(
        actor,
        critic,
        actor_params,
        critic_params,
        adapter,
        episode_records,
        batch_size=batch_size,
        disable_ref_input=disable_ref_input,
        actor_mode=actor_mode,
        actor_seed=actor_seed,
    )

    chunk_rows: list[dict[str, float | int | str]] = []
    for idx, record in enumerate(episode_records):
        q_data = float(q_payload["q_data"][idx])
        q_ref = float(q_payload["q_ref"][idx])
        q_actor_mean = float(q_payload["q_actor_mean"][idx])
        chunk_reward_sum = float(annotations["chunk_reward_sum"][idx])
        row = {
            "episode_id": int(record["episode_id"]),
            "chunk_index": int(annotations["chunk_index"][idx]),
            "step_id": int(annotations["step_id"][idx]),
            "collection_phase": str(record.get("collection_phase", "unknown")),
            "distance_to_terminal": int(annotations["distance_to_terminal"][idx]),
            "done": int(annotations["done"][idx]),
            "transition_success": int(annotations["transition_success"][idx]),
            "chunk_reward_sum": chunk_reward_sum,
            "q_data": q_data,
            "q_ref": q_ref,
            "q_actor_mean": q_actor_mean,
            "abs_err_q_data_vs_chunk_reward_sum": abs(q_data - chunk_reward_sum),
            "abs_err_q_ref_vs_chunk_reward_sum": abs(q_ref - chunk_reward_sum),
            "abs_err_q_actor_mean_vs_chunk_reward_sum": abs(q_actor_mean - chunk_reward_sum),
            "q1_data": float(q_payload["q1_data"][idx]),
            "q2_data": float(q_payload["q2_data"][idx]),
            "q_gap_data": float(q_payload["q_gap_data"][idx]),
        }
        if q_payload["q_actor_sample"] is not None:
            q_actor_sample = float(q_payload["q_actor_sample"][idx])
            row["q_actor_sample"] = q_actor_sample
            row["abs_err_q_actor_sample_vs_chunk_reward_sum"] = abs(q_actor_sample - chunk_reward_sum)
        chunk_rows.append(row)

    distance_rows = _distance_table(
        annotations["distance_to_terminal"],
        q_payload["q_data"],
        q_payload["q_ref"],
        q_payload["q_actor_mean"],
        q_payload["q_actor_sample"],
        annotations["chunk_reward_sum"],
        q_payload["q_gap_data"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "episode_q_table.csv", chunk_rows)
    _write_csv(output_dir / "distance_q_table.csv", distance_rows)
    _plot_episode_q_profile(
        output_dir / "episode_q_profile.png",
        annotations["chunk_index"],
        q_payload["q_data"],
        q_payload["q_ref"],
        q_payload["q_actor_mean"],
        q_payload["q_actor_sample"],
        annotations["chunk_reward_sum"],
        annotations["done"],
        annotations["transition_success"],
    )
    _plot_q_by_distance(output_dir / "episode_q_by_distance.png", distance_rows)

    summary = _summary_payload(
        output_dir=output_dir,
        actor_path=actor_path,
        critic_path=critic_path,
        replay_path=replay_path,
        episode_id=episode_id,
        records=episode_records,
        annotations=annotations,
        q_payload=q_payload,
        actor_mode=actor_mode,
        actor_seed=actor_seed,
        fixed_std=rl_config.fixed_std,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _default_output_name(args: argparse.Namespace) -> str:
    base = "eval_episode_q"
    if args.disable_ref_input:
        base += "_noref"
    if args.actor_mode == "sample":
        base += "_sample"
    return base + default_filter_suffix(phase=args.phase, source=args.source)


def main() -> None:
    args = _parse_args()
    replay_path = args.replay_path.resolve()
    task_dir = infer_task_dir_from_replay_path(replay_path)
    model_dir = (args.model_dir or (task_dir / "offline_train_bcq")).resolve()
    actor_path = (
        args.actor_path.resolve() if args.actor_path is not None else resolve_default_actor_checkpoint_path(model_dir)
    )
    critic_path = (
        args.critic_path.resolve() if args.critic_path is not None else resolve_default_critic_snapshot_path(model_dir)
    )
    replay_records = filter_replay_records(
        load_replay_journal(replay_path),
        phase=args.phase,
        source=args.source,
    )
    if not replay_records:
        raise RuntimeError(f"No replay samples left after filtering: replay={replay_path}")

    actor_cfg, actor_params = load_snapshot(actor_path, task_dir)
    critic_cfg, critic_params = load_critic_snapshot(critic_path, task_dir)

    output_root = (args.output_dir or (model_dir / _default_output_name(args))).resolve()
    episode_ids = _resolve_episode_ids(args.episode_ids)

    summaries = []
    for episode_id in episode_ids:
        episode_output_dir = output_root / f"episode_{episode_id}"
        summary = _evaluate_episode(
            episode_id=episode_id,
            replay_records=replay_records,
            actor_cfg=actor_cfg,
            actor_params=actor_params,
            critic_cfg=critic_cfg,
            critic_params=critic_params,
            actor_path=actor_path,
            critic_path=critic_path,
            replay_path=replay_path,
            batch_size=args.batch_size,
            disable_ref_input=args.disable_ref_input,
            actor_mode=args.actor_mode,
            actor_seed=args.actor_seed,
            output_dir=episode_output_dir,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        rows = [
            {
                "episode_id": int(summary["episode_id"]),
                "num_chunks": int(summary["num_chunks"]),
                "episode_success": int(summary["episode_success"]),
                "q_data_mean": float(summary["q_data"]["mean"]),
                "q_ref_mean": float(summary["q_ref"]["mean"]),
                "q_actor_mean": float(summary["q_actor_mean"]["mean"]),
                "chunk_reward_sum_mean": float(summary["chunk_reward_sum"]["mean"]),
                "corr_q_data_vs_distance_to_terminal": float(summary["trend"]["corr_q_data_vs_distance_to_terminal"]),
                "corr_q_actor_mean_vs_distance_to_terminal": float(
                    summary["trend"]["corr_q_actor_mean_vs_distance_to_terminal"]
                ),
            }
            for summary in summaries
        ]
        if args.actor_mode == "sample":
            for row, summary in zip(rows, summaries, strict=True):
                row["q_actor_sample_mean"] = float(summary["q_actor_sample"]["mean"])
                row["corr_q_actor_sample_vs_distance_to_terminal"] = float(
                    summary["trend"]["corr_q_actor_sample_vs_distance_to_terminal"]
                )
        output_root.mkdir(parents=True, exist_ok=True)
        _write_csv(output_root / "episodes_summary.csv", rows)

    print(f"wrote episode Q analysis to: {output_root}")


if __name__ == "__main__":
    main()
