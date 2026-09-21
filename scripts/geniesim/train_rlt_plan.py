#!/usr/bin/env python3
"""Train the frozen-VLA RL Token stage for the plan run."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
from pathlib import Path
import platform
import sys
from typing import Any

from flax import traverse_util
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import openpi.training.checkpoints as checkpoints
import openpi.training.config as train_config
import openpi.training.data_loader as data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from scripts import train_rlt as rlt_train
from scripts.geniesim.plan_utils import DEFAULT_CONFIG_NAME
from scripts.geniesim.plan_utils import DEFAULT_PLAN_ROOT
from scripts.geniesim.plan_utils import atomic_write_json
from scripts.geniesim.plan_utils import dataset_provenance
from scripts.geniesim.plan_utils import init_wandb
from scripts.geniesim.plan_utils import load_stage1_split
from scripts.geniesim.plan_utils import log_artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--analysis-subdir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--keep-period", type=int, default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def _count_params(tree: Any) -> int:
    total = 0
    for value in traverse_util.flatten_dict(tree).values():
        if hasattr(value, "shape"):
            total += int(np.prod(value.shape, dtype=np.int64))
    return total


def _parameter_audit(state: Any, *, alpha: float, output_path: Path) -> dict[str, Any]:
    pure = state.params.to_pure_dict()
    flat = traverse_util.flatten_dict(pure)
    total = _count_params(pure)
    trainable_paths = ["/".join(str(part) for part in path) for path in flat if "rlt_module" in "/".join(map(str, path))]
    trainable_count = sum(int(np.prod(flat[path].shape, dtype=np.int64)) for path in flat if "rlt_module" in "/".join(map(str, path)))
    frozen_count = total - trainable_count
    audit = {
        "rlt_alpha": float(alpha),
        "total_parameter_count": total,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": frozen_count,
        "trainable_path_count": len(trainable_paths),
        "trainable_paths": trainable_paths,
        "trainable_filter_expectation": "rlt_module_only" if alpha == 0.0 else "all_parameters",
        "verified": bool(alpha != 0.0 or all("rlt_module" in path for path in trainable_paths)),
    }
    if not audit["verified"]:
        raise RuntimeError("rlt_alpha=0 but a non-RLT parameter appeared in the trainable tree")
    atomic_write_json(output_path, audit)
    return audit


def _wandb_config(config: train_config.TrainConfig, split: dict[str, Any], output_root: Path) -> dict[str, Any]:
    model = config.model
    data = config.data
    return {
        "config_name": config.name,
        "experiment_name": config.exp_name,
        "vla_config": dataclasses.asdict(model),
        "sft_checkpoint": str(train_config.GENIESIM_RLT_BASE_CKPT),
        "dataset": str(data.dataset_root),
        "data_config": repr(data),
        "state_normalization": "SFT checkpoint assets/norm_stats.json (first 16 dims, padded to 32)",
        "action_normalization": "SFT checkpoint assets/norm_stats.json (first 16 dims, delta 14 + absolute 2)",
        "image_transforms": "ResizeImages(224,224), frozen pi0.5 preprocessing; training augmentation enabled",
        "action_horizon": int(model.action_horizon),
        "model_action_dim": int(model.action_dim),
        "batch_size": int(config.batch_size),
        "learning_rate_schedule": repr(config.lr_schedule),
        "optimizer": repr(config.optimizer),
        "rlt_num_tokens": int(config.rlt_num_tokens or 0),
        "rlt_num_layers": int(config.rlt_num_layers or 0),
        "rlt_embed_dim": int(config.rlt_embed_dim or 0),
        "rlt_input_dim": int(config.rlt_input_dim or 0),
        "rlt_alpha": float(config.rlt_alpha or 0.0),
        "seed": int(config.seed),
        "train_episode_count": int(split["train_episode_count"]),
        "validation_episode_count": int(split["validation_episode_count"]),
        "output_root": str(output_root),
    }


def _write_effective_config(
    config: train_config.TrainConfig,
    split: dict[str, Any],
    output_root: Path,
    analysis_dir: Path,
) -> dict[str, Any]:
    """Persist the resolved Stage-1 inputs for post-run audit and reporting."""
    resolved_data_config = config.data.create(config.assets_dirs, config.model)
    payload = {
        "config_name": config.name,
        "vla_config": dataclasses.asdict(config.model),
        "sft_checkpoint": str(train_config.GENIESIM_RLT_BASE_CKPT),
        "dataset": str(config.data.dataset_root),
        "data_config": repr(resolved_data_config),
        "data_config_summary": {
            "repo_id": resolved_data_config.repo_id,
            "dataset_root": resolved_data_config.dataset_root,
            "train_episode_count": len(config.data.dataset_episodes or ()),
            "norm_stats_asset": resolved_data_config.asset_id,
            "use_quantile_norm": bool(resolved_data_config.use_quantile_norm),
        },
        "state_normalization": "SFT checkpoint assets/norm_stats.json (32D padded state; first 16 task dimensions)",
        "action_normalization": "SFT checkpoint assets/norm_stats.json; quantile normalization enabled",
        "image_transforms": "ResizeImages(224,224), frozen pi0.5 preprocessing; training augmentation enabled",
        "action_horizon": int(config.model.action_horizon),
        "action_dimension": int(config.model.action_dim),
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "seed": int(config.seed),
        "num_train_steps": int(config.num_train_steps),
        "save_interval": int(config.save_interval),
        "keep_period": config.keep_period,
        "learning_rate_schedule": repr(config.lr_schedule),
        "optimizer": repr(config.optimizer),
        "rlt_num_tokens": int(config.rlt_num_tokens or 0),
        "rlt_num_layers": int(config.rlt_num_layers or 0),
        "rlt_embed_dim": int(config.rlt_embed_dim or 0),
        "rlt_input_dim": int(config.rlt_input_dim or 0),
        "rlt_alpha": float(config.rlt_alpha or 0.0),
        "train_episode_count": int(split["train_episode_count"]),
        "validation_episode_count": int(split["validation_episode_count"]),
        "output_root": str(output_root),
    }
    atomic_write_json(analysis_dir / "stage1_training_config.json", payload)
    return payload


def _image_for_wandb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 3:
        if image.min() < 0:
            image = (image + 1.0) * 127.5
        return np.clip(image, 0, 255).astype(np.uint8)
    raise ValueError(f"Unexpected image shape {image.shape}")


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    analysis_dir = output_root / "analysis"
    if args.analysis_subdir:
        analysis_dir = analysis_dir / args.analysis_subdir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    split = load_stage1_split(output_root)
    config = train_config.get_config(args.config)
    config = dataclasses.replace(
        config,
        checkpoint_base_dir=str(output_root / "checkpoints"),
        assets_base_dir=str(output_root / "assets"),
        resume=args.resume,
        overwrite=args.overwrite,
        num_train_steps=int(args.num_train_steps or config.num_train_steps),
        num_workers=int(args.num_workers if args.num_workers is not None else config.num_workers),
        save_interval=int(args.save_interval if args.save_interval is not None else config.save_interval),
        keep_period=args.keep_period if args.keep_period is not None else config.keep_period,
        data=dataclasses.replace(config.data, dataset_episodes=tuple(split["train_episodes"])),
    )
    if config.save_interval <= 0:
        raise ValueError("--save-interval must be positive")
    if config.keep_period is not None and config.keep_period <= 0:
        raise ValueError("--keep-period must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.info("Running Stage-1 RL Token training on %s", platform.node())
    logging.info("Stage-1 train episodes=%d validation episodes=%d", len(split["train_episodes"]), len(split["validation_episodes"]))

    if config.rlt_num_tokens is None or config.rlt_alpha != 0.0:
        raise ValueError("Plan Stage-1 requires RLT fields and rlt_alpha=0")
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size={config.batch_size} must be divisible by device_count={jax.device_count()}")

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    run_config = _wandb_config(config, split, output_root)
    run_config["analysis_dir"] = str(analysis_dir)
    _write_effective_config(config, split, output_root, analysis_dir)
    run_name = args.wandb_run_name or (config.exp_name + ("_resume" if resuming else ""))
    run = init_wandb(project=config.project_name, name=run_name, config=run_config, output_dir=output_root)
    (analysis_dir / "wandb_stage1_run.json").write_text(
        json.dumps(
            {"id": run.id, "name": run.name, "project": run.project, "entity": run.entity, "url": run.url},
            indent=2,
        ),
        encoding="utf-8",
    )
    log_artifact(
        run,
        name="rlt-stage1-demonstration-data",
        artifact_type="dataset",
        files=[
            output_root / "analysis" / "stage1_dataset_manifest.json",
            output_root / "analysis" / "stage1_episode_split.json",
            analysis_dir / "stage1_training_config.json",
            output_root / "analysis" / "stage1_dataset_samples",
        ],
        metadata={"source": "original_sft_demonstration_dataset", **dataset_provenance(Path(config.data.dataset_root))},
    )

    loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    iterator = iter(loader)
    batch = next(iterator)
    logging.info("Initialized data loader:\n%s", training_utils.array_tree_to_info(batch))
    sample_images = []
    for index in range(min(4, len(next(iter(batch[0].images.values()))))):
        joined = np.concatenate([_image_for_wandb(np.asarray(image[index])) for image in batch[0].images.values()], axis=1)
        sample_images.append(wandb.Image(joined, caption=f"stage1 train sample {index}"))
    run.log({"stage1_camera_views": sample_images}, step=0)

    train_state, train_state_sharding = rlt_train.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    audit = _parameter_audit(train_state, alpha=float(config.rlt_alpha or 0.0), output_path=analysis_dir / "stage1_parameter_audit.json")
    run.config.update(audit, allow_val_change=True)
    if resuming:
        train_state = checkpoints.restore_state(checkpoint_manager, train_state, loader)

    ptrain_step = jax.jit(
        functools.partial(rlt_train.train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    metrics_path = analysis_dir / "stage1_training_metrics.jsonl"
    status_path = analysis_dir / "stage1_training_status.json"
    start_step = int(jax.device_get(train_state.step))
    if start_step >= config.num_train_steps:
        raise ValueError(f"Checkpoint step {start_step} is already at or beyond target {config.num_train_steps}")
    run.config.update(
        {
            "resumed": bool(resuming),
            "resume_checkpoint_step": start_step,
            "target_step": int(config.num_train_steps),
            "save_interval": int(config.save_interval),
            "keep_period": config.keep_period,
        },
        allow_val_change=True,
    )
    lr_schedule = config.lr_schedule.create()
    infos: list[dict[str, Any]] = []
    for _ in range(start_step, config.num_train_steps):
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        actual_step = int(jax.device_get(train_state.step))
        if actual_step % config.log_interval == 0 or actual_step == config.num_train_steps:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            metrics = {"step": actual_step}
            metrics.update({
                "train_reconstruction_mse": float(reduced["mse"]),
                "rlt_loss": float(reduced["rlt_loss"]),
                "loss": float(reduced["loss"]),
                "grad_norm": float(reduced["grad_norm"]),
                "param_norm": float(reduced["param_norm"]),
                "learning_rate": float(jax.device_get(lr_schedule(max(actual_step - 1, 0)))),
            })
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics) + "\n")
            run.log({key: value for key, value in metrics.items() if key != "step"}, step=actual_step)
            logging.info("Stage-1 step=%d train_reconstruction_mse=%.7f rlt_loss=%.7f", actual_step, metrics["train_reconstruction_mse"], metrics["rlt_loss"])
            atomic_write_json(status_path, {"step": actual_step, "num_train_steps": config.num_train_steps, "latest_metrics": metrics, "parameter_audit": audit})
            infos = []
        batch = next(iterator)
        if actual_step % config.save_interval == 0 or actual_step == config.num_train_steps:
            checkpoints.save_state(checkpoint_manager, train_state, loader, actual_step)

    logging.info("Waiting for Stage-1 checkpoint manager")
    checkpoint_manager.wait_until_finished()
    saved_steps = sorted(int(step) for step in checkpoint_manager.all_steps())
    atomic_write_json(analysis_dir / "stage1_saved_steps.json", {"steps": saved_steps})
    run.summary.update({"stage1_final_step": int(config.num_train_steps), "stage1_saved_steps": saved_steps})
    run.finish()


if __name__ == "__main__":
    main()
