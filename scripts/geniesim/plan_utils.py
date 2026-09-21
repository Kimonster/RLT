"""Shared paths, split, provenance, and W&B helpers for the plan run."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

import openpi.training.config as train_config


DEFAULT_PLAN_ROOT = Path(
    os.environ.get("GENIESIM_RLT_PLAN_ROOT", "/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan")
).expanduser()
DEFAULT_DEMO_ROOT = Path(
    os.environ.get("GENIESIM_RLT_DEMO_ROOT", "/mnt/pfs/kk/kk/data/data/geniesim/stack_three_blocks")
).expanduser()
DEFAULT_ROLLOUT_ROOT = Path(
    os.environ.get("GENIESIM_RLT_ROLLOUT_ROOT", "/mnt/pfs/kk/kk/data/data/geniesim/rollout/stack_three_blocks")
).expanduser()
DEFAULT_BASE_CHECKPOINT = Path(
    os.environ.get("GENIESIM_RLT_BASE_CKPT", "/mnt/pfs/kk/kk/ckpt/geniesim3/spatialpi05")
).expanduser()
DEFAULT_CONFIG_NAME = "rlt_pi05_geniesim_stack_three_blocks_plan"
DEFAULT_SEED = 42


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def make_stage1_split(
    dataset_root: Path = DEFAULT_DEMO_ROOT,
    output_root: Path = DEFAULT_PLAN_ROOT,
    *,
    seed: int = DEFAULT_SEED,
    validation_ratio: float = 0.10,
) -> dict[str, Any]:
    """Create or verify an immutable episode-level 90/10 split."""
    output_path = output_root / "analysis" / "stage1_episode_split.json"
    if output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        expected = _build_split(dataset_root, seed=seed, validation_ratio=validation_ratio)
        if payload.get("dataset_root") != str(dataset_root.resolve()) or payload.get("seed") != seed:
            raise RuntimeError(f"Existing Stage-1 split metadata does not match requested data/seed: {output_path}")
        if payload.get("train_episodes") != expected["train_episodes"] or payload.get("validation_episodes") != expected[
            "validation_episodes"
        ]:
            raise RuntimeError(f"Existing Stage-1 split is not reproducible: {output_path}")
        return payload
    payload = _build_split(dataset_root, seed=seed, validation_ratio=validation_ratio)
    atomic_write_json(output_path, payload)
    return payload


def _build_split(dataset_root: Path, *, seed: int, validation_ratio: float) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"Invalid LeRobot dataset: {dataset_root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total = int(info["total_episodes"])
    episode_rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    episode_ids = [int(row["episode_index"]) for row in episode_rows]
    if episode_ids != list(range(total)):
        raise ValueError("Expected contiguous episode indices for the fixed split")
    shuffled = episode_ids.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(total * validation_ratio)))
    validation = sorted(shuffled[:val_count])
    train = sorted(shuffled[val_count:])
    return {
        "schema_version": 1,
        "dataset_root": str(dataset_root.resolve()),
        "dataset_info_sha256": sha256_file(info_path),
        "episodes_sha256": sha256_file(episodes_path),
        "seed": int(seed),
        "validation_ratio": float(validation_ratio),
        "total_episodes": total,
        "train_episodes": train,
        "validation_episodes": validation,
        "train_episode_count": len(train),
        "validation_episode_count": len(validation),
    }


def load_stage1_split(output_root: Path = DEFAULT_PLAN_ROOT) -> dict[str, Any]:
    path = output_root / "analysis" / "stage1_episode_split.json"
    if not path.is_file():
        return make_stage1_split(output_root=output_root)
    return json.loads(path.read_text(encoding="utf-8"))


def plan_config(
    *,
    train_episodes: list[int] | None = None,
    exp_name: str | None = None,
) -> train_config.TrainConfig:
    config = train_config.get_config(DEFAULT_CONFIG_NAME)
    if train_episodes is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, dataset_episodes=tuple(train_episodes)))
    if exp_name is not None:
        config = dataclasses.replace(config, exp_name=exp_name)
    return config


def dataset_provenance(dataset_root: Path = DEFAULT_DEMO_ROOT) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    return {
        "dataset_root": str(dataset_root.resolve()),
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "features": sorted(info["features"]),
        "info_sha256": sha256_file(info_path),
        "episodes_sha256": sha256_file(episodes_path),
        "tasks_sha256": sha256_file(tasks_path) if tasks_path.is_file() else None,
        "parquet_count": len(parquet_files),
        "parquet_bytes": sum(path.stat().st_size for path in parquet_files),
        "video_bytes": sum(path.stat().st_size for path in (dataset_root / "videos").rglob("*.mp4")),
        "raw_state_shape": info["features"]["observation.state"]["shape"],
        "raw_action_shape": info["features"]["action"]["shape"],
    }


def init_wandb(*, project: str, name: str, config: dict[str, Any], output_dir: Path):
    """Initialize an online run using credentials supplied by the process environment."""
    import wandb

    output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY"),
        name=name,
        config=config,
        mode=os.environ.get("WANDB_MODE", "online"),
        dir=str(output_dir / "wandb"),
    )
    return run


def log_artifact(run, *, name: str, artifact_type: str, files: list[Path], metadata: dict[str, Any]) -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
    for path in files:
        if path.is_file():
            artifact.add_file(str(path), name=path.name)
        elif path.is_dir():
            artifact.add_dir(str(path), name=path.name)
    run.log_artifact(artifact)

