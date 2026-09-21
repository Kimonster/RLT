#!/usr/bin/env python3
"""Evaluate and conservatively select a Stage-1 RL Token checkpoint."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import openpi.models.model as model_lib
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as train_config
import openpi.training.data_loader as data_loader
from scripts.geniesim.plan_utils import DEFAULT_CONFIG_NAME
from scripts.geniesim.plan_utils import DEFAULT_PLAN_ROOT
from scripts.geniesim.plan_utils import atomic_write_json
from scripts.geniesim.plan_utils import init_wandb
from scripts.geniesim.plan_utils import load_stage1_split
from scripts.geniesim.plan_utils import log_artifact


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", nargs="*", type=int, default=[5000, 10000, 15000, 20000])
    parser.add_argument("--wandb-run-name", default="stage1_checkpoint_evaluation")
    return parser.parse_args()


def _load_episode_lengths(dataset_root: Path) -> dict[int, int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Episode metadata does not exist: {episodes_path}")
    rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(row["episode_index"]): int(row["length"]) for row in rows}


def _uniform_episode_indices(episodes: list[int], lengths: dict[int, int], max_samples: int) -> tuple[list[int], int]:
    if max_samples <= 0:
        raise ValueError("--max-val-samples must be positive")
    total_frames = sum(lengths[episode] for episode in episodes)
    sample_count = min(max_samples, total_frames)
    base, remainder = divmod(sample_count, len(episodes))
    indices: list[int] = []
    cursor = 0
    covered_episodes = 0
    for episode_index, episode in enumerate(episodes):
        length = lengths[episode]
        count = base + int(episode_index < remainder)
        count = min(count, length)
        if count:
            offsets = np.floor((np.arange(count, dtype=np.float64) + 0.5) * length / count).astype(np.int64)
            indices.extend((cursor + offsets).tolist())
            covered_episodes += 1
        cursor += length
    return indices, covered_episodes


def _load_eval_batches(
    config: train_config.TrainConfig,
    episodes: list[int],
    max_samples: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int, int]:
    eval_config = dataclasses.replace(
        config,
        batch_size=batch_size,
        data=dataclasses.replace(config.data, dataset_episodes=tuple(episodes)),
    )
    data_config = eval_config.data.create(eval_config.assets_dirs, eval_config.model)
    dataset = data_loader.create_torch_dataset(data_config, eval_config.model.action_horizon, eval_config.model)
    dataset = data_loader.transform_dataset(dataset, data_config)
    lengths = _load_episode_lengths(Path(data_config.dataset_root))
    indices, covered_episodes = _uniform_episode_indices(episodes, lengths, max_samples)
    if sum(lengths[episode] for episode in episodes) != len(dataset):
        raise RuntimeError("Validation episode lengths do not match the mapped dataset")
    batches: list[dict[str, Any]] = []
    for start in range(0, len(indices), batch_size):
        samples = [dataset[index] for index in indices[start : start + batch_size]]
        batch = jax.tree.map(lambda *values: np.stack([np.asarray(value) for value in values]), *samples)
        batches.append({key: value for key, value in batch.items() if key != "actions"})
    if not batches:
        raise RuntimeError("Validation loader produced no batches")
    return batches, len(indices), covered_episodes


def _evaluate_checkpoint(
    config: train_config.TrainConfig,
    checkpoint_path: Path,
    batches: list[dict[str, Any]],
) -> dict[str, Any]:
    from scripts.serve_rlt_policy import load_rlt_model

    model = load_rlt_model(config, str(checkpoint_path))
    # The composite model exposes fixed-argument methods so bool/string kwargs
    # remain Python values under ``module_jit`` and satisfy the model type checks.
    extract_fn = nnx_utils.module_jit(model.extract_image_prefix)
    encode_fn = nnx_utils.module_jit(model.encode_rl_tokens)
    decode_fn = nnx_utils.module_jit(model.decode_rl_tokens)
    total_sse = 0.0
    total_count = 0
    target_values: list[np.ndarray] = []
    recon_values: list[np.ndarray] = []
    z_values: list[np.ndarray] = []
    cosine_values: list[float] = []
    frame_count = 0
    rng = jax.random.key(12345)
    for obs_dict in batches:
        observation = model_lib.Observation.from_dict(obs_dict)
        rng, batch_rng = jax.random.split(rng)
        prefix, mask = extract_fn(batch_rng, observation)
        prefix = prefix.astype(jax.numpy.float32)
        tokens = encode_fn(prefix.astype(jax.numpy.float32))
        reconstruction = decode_fn(tokens)
        prefix_np = np.asarray(jax.device_get(prefix), dtype=np.float32)
        reconstruction_np = np.asarray(jax.device_get(reconstruction), dtype=np.float32)
        mask_np = np.asarray(jax.device_get(mask), dtype=bool)
        diff = reconstruction_np - prefix_np
        total_sse += float(np.square(diff[mask_np]).sum())
        total_count += int(mask_np.sum() * prefix_np.shape[-1])
        target_values.append(prefix_np[mask_np].reshape(-1, prefix_np.shape[-1]))
        recon_values.append(reconstruction_np[mask_np].reshape(-1, reconstruction_np.shape[-1]))
        z_values.append(np.asarray(jax.device_get(tokens), dtype=np.float32).reshape(tokens.shape[0], -1))
        frame_count += int(prefix_np.shape[0])
        for row in range(prefix_np.shape[0]):
            row_valid = mask_np[row]
            target_row = prefix_np[row, row_valid].reshape(-1)
            recon_row = reconstruction_np[row, row_valid].reshape(-1)
            denom = float(np.linalg.norm(target_row) * np.linalg.norm(recon_row))
            cosine_values.append(float(np.dot(target_row, recon_row) / denom) if denom > 1e-12 else float("nan"))
    target = np.concatenate(target_values, axis=0)
    reconstruction = np.concatenate(recon_values, axis=0)
    z = np.concatenate(z_values, axis=0)
    embedding_variance = float(np.var(target))
    val_mse = total_sse / max(total_count, 1)
    sst = float(np.square(target - target.mean()).sum())
    val_r2 = float(1.0 - total_sse / sst) if sst > 1e-12 else float("nan")
    cos = np.asarray(cosine_values, dtype=np.float64)
    z_norm = np.linalg.norm(z, axis=1)
    finite = bool(
        np.isfinite(val_mse)
        and np.isfinite(embedding_variance)
        and np.isfinite(val_r2)
        and np.isfinite(cos).all()
        and np.isfinite(z).all()
    )
    mean_per_dim_std = float(np.mean(np.std(z, axis=0)))
    collapse = bool(embedding_variance <= 1e-8 or mean_per_dim_std <= 1e-6)
    norm_explosion = bool(float(np.mean(z_norm)) > 1e4 or float(np.std(z_norm)) > 1e4)
    valid_checkpoint = bool(finite and not collapse and not norm_explosion and embedding_variance > 1e-12)
    result = {
        "step": int(checkpoint_path.name),
        "checkpoint": str(checkpoint_path),
        "train_mse": float("nan"),
        "val_mse": float(val_mse),
        "embedding_variance": embedding_variance,
        "val_nmse": float(val_mse / embedding_variance) if embedding_variance > 1e-12 else float("nan"),
        "val_r2": val_r2,
        "val_cosine_mean": float(np.nanmean(cos)),
        "val_cosine_std": float(np.nanstd(cos)),
        "val_cosine_p10": float(np.nanpercentile(cos, 10)),
        "val_cosine_p50": float(np.nanpercentile(cos, 50)),
        "val_cosine_p90": float(np.nanpercentile(cos, 90)),
        "z_norm_mean": float(np.mean(z_norm)),
        "z_norm_std": float(np.std(z_norm)),
        "mean_per_dimension_std": mean_per_dim_std,
        "finite": finite,
        "representation_collapse": collapse,
        "norm_explosion": norm_explosion,
        "valid": valid_checkpoint,
        "sample_count": frame_count,
        "valid_token_count": int(target.shape[0]),
    }
    del model, extract_fn, encode_fn, decode_fn
    gc.collect()
    jax.clear_caches()
    return result


def _read_train_mse(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            values[int(row["step"])] = float(row["train_reconstruction_mse"])
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return values


def _select(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    valid = [row for row in rows if row["valid"]]
    if not valid:
        return None, "all Stage-1 checkpoints were invalid (NaN/Inf, collapse, norm explosion, or uncomputable metric)"
    best_nmse = min(row["val_nmse"] for row in valid)
    nmse_candidates = [row for row in valid if row["val_nmse"] <= best_nmse * 1.03]
    best_r2 = max(row["val_r2"] for row in valid)
    best_cos = max(row["val_cosine_mean"] for row in valid)
    quality_candidates = [
        row
        for row in nmse_candidates
        if row["val_r2"] >= best_r2 - 0.01 and row["val_cosine_mean"] >= best_cos - 0.005
    ]
    if not quality_candidates:
        # Fallback keeps the documented metric priority before the earlier-step
        # preference used for the normal near-equivalent candidate set.
        selected = sorted(
            nmse_candidates,
            key=lambda row: (row["val_nmse"], -row["val_r2"], -row["val_cosine_mean"], row["step"]),
        )[0]
        reason = "lowest validation NMSE; tie-break by higher R2, higher cosine, then earlier checkpoint"
        return selected, reason
    reason = "earliest valid checkpoint within 3% of best NMSE and within R2/cosine quality thresholds"
    selected = sorted(quality_candidates, key=lambda row: (row["step"], row["val_nmse"], -row["val_r2"], -row["val_cosine_mean"]))[0]
    return selected, reason


def _write_plots(rows: list[dict[str, Any]], analysis_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    # Missing/corrupt checkpoints remain visible in the CSV, but should not
    # prevent the diagnostic plots from being written for valid rows.
    plot_rows = [row for row in rows if "val_mse" in row]
    if not plot_rows:
        return []
    steps = [row["step"] for row in plot_rows]
    plots: list[tuple[str, str, str]] = [
        ("stage1_train_val_mse.png", "Stage 1 reconstruction MSE", "val_mse"),
        ("stage1_nmse.png", "Stage 1 validation NMSE", "val_nmse"),
        ("stage1_r2.png", "Stage 1 validation R2", "val_r2"),
        ("stage1_cosine.png", "Stage 1 validation cosine similarity", "val_cosine_mean"),
        ("stage1_z_norm.png", "Stage 1 RL-token norm", "z_norm_mean"),
    ]
    paths: list[Path] = []
    for filename, title, key in plots:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if filename == "stage1_train_val_mse.png":
            ax.plot(steps, [row.get("train_mse", float("nan")) for row in plot_rows], marker="o", label="train")
            ax.plot(steps, [row.get("val_mse", float("nan")) for row in plot_rows], marker="o", label="validation")
            ax.legend()
        else:
            ax.plot(steps, [row.get(key, float("nan")) for row in plot_rows], marker="o")
        ax.set_title(title)
        ax.set_xlabel("checkpoint step")
        ax.grid(visible=True, alpha=0.25)
        fig.tight_layout()
        path = analysis_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    args = _args()
    output_root = args.output_root.expanduser().resolve()
    analysis_dir = (
        args.analysis_dir.expanduser().resolve() if args.analysis_dir is not None else output_root / "analysis"
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    split = load_stage1_split(output_root)
    config = train_config.get_config(args.config)
    config = dataclasses.replace(
        config,
        checkpoint_base_dir=str(output_root / "checkpoints"),
        assets_base_dir=str(output_root / "assets"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data=dataclasses.replace(config.data, dataset_episodes=tuple(split["validation_episodes"])),
    )
    batches, sample_count, covered_episode_count = _load_eval_batches(
        config, split["validation_episodes"], args.max_val_samples, args.batch_size
    )
    checkpoint_root = config.checkpoint_dir
    rows: list[dict[str, Any]] = []
    train_metrics_path = analysis_dir / "stage1_training_metrics.jsonl"
    if not train_metrics_path.is_file():
        train_metrics_path = output_root / "analysis" / "stage1_training_metrics.jsonl"
    train_mse = _read_train_mse(train_metrics_path)
    for step in sorted(set(args.steps)):
        checkpoint_path = checkpoint_root / str(step)
        if not checkpoint_path.is_dir():
            rows.append({"step": step, "checkpoint": str(checkpoint_path), "valid": False, "error": "checkpoint_missing"})
            continue
        try:
            row = _evaluate_checkpoint(config, checkpoint_path, batches)
        except Exception as exc:  # keep evaluating the remaining checkpoints
            error = f"{type(exc).__name__}: {exc}"
            rows.append({"step": step, "checkpoint": str(checkpoint_path), "valid": False, "error": error})
            with (analysis_dir / "stage1_evaluation_errors.log").open("a", encoding="utf-8") as stream:
                stream.write(f"step={step} checkpoint={checkpoint_path}\n{traceback.format_exc()}\n")
            print(json.dumps({"step": step, "checkpoint": str(checkpoint_path), "valid": False, "error": error}))
            continue
        row["train_mse"] = train_mse.get(step, float("nan"))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    csv_path = analysis_dir / "stage1_checkpoint_metrics.csv"
    fields = [
        "step", "checkpoint", "train_mse", "val_mse", "embedding_variance", "val_nmse", "val_r2",
        "val_cosine_mean", "val_cosine_std", "val_cosine_p10", "val_cosine_p50", "val_cosine_p90",
        "z_norm_mean", "z_norm_std", "mean_per_dimension_std", "finite", "representation_collapse",
        "norm_explosion", "valid", "sample_count", "valid_token_count", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    selected, reason = _select(rows)
    if selected is None:
        (analysis_dir / "FAILED_STAGE1.md").write_text(
            "# Stage-1 failed\n\n" + reason + "\n\nSee `analysis/stage1_checkpoint_metrics.csv`.\n",
            encoding="utf-8",
        )
        raise RuntimeError(reason)
    selection = {
        "step": int(selected["step"]),
        "checkpoint": selected["checkpoint"],
        "val_mse": selected["val_mse"],
        "val_nmse": selected["val_nmse"],
        "val_r2": selected["val_r2"],
        "val_cosine_mean": selected["val_cosine_mean"],
        "reason": reason,
        "validation_sample_count": sample_count,
        "validation_episode_count": covered_episode_count,
        "validation_sampling": "deterministic_uniform_within_each_episode",
    }
    atomic_write_json(analysis_dir / "selected_stage1_checkpoint.json", selection)
    (analysis_dir / "selected_stage1_checkpoint.txt").write_text(
        f"step={selection['step']}\ncheckpoint={selection['checkpoint']}\nval_mse={selection['val_mse']:.9g}\n"
        f"val_nmse={selection['val_nmse']:.9g}\nval_r2={selection['val_r2']:.9g}\n"
        f"val_cosine_mean={selection['val_cosine_mean']:.9g}\nreason={reason}\n",
        encoding="utf-8",
    )
    plot_paths = _write_plots(rows, analysis_dir)

    run = init_wandb(
        project=config.project_name,
        name=args.wandb_run_name,
        config={
            "config_name": args.config,
            "validation_sample_count": sample_count,
            "validation_episode_count": covered_episode_count,
            "validation_sampling": "deterministic_uniform_within_each_episode",
            "selection": selection,
        },
        output_dir=output_root,
    )
    for row in rows:
        run.log(
            {key: value for key, value in row.items() if isinstance(value, int | float) and np.isfinite(value)},
            step=int(row["step"]),
        )
    for path in plot_paths:
        run.log({path.stem: __import__("wandb").Image(str(path))})
    log_artifact(
        run,
        name="rlt-stage1-evaluation",
        artifact_type="evaluation",
        files=[
            csv_path,
            analysis_dir / "selected_stage1_checkpoint.txt",
            analysis_dir / "stage1_training_config.json",
            train_metrics_path,
            *plot_paths,
        ],
        metadata=selection,
    )
    run.summary.update({"selected_stage1_step": selection["step"], "selected_stage1_nmse": selection["val_nmse"]})
    run.finish()
    print(json.dumps({"selected": selection, "plots": [str(path) for path in plot_paths]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
