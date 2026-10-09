#!/usr/bin/env python3
"""Offline multi-checkpoint analysis for GenieSim RL-token representations."""

# ruff: noqa: E402, PERF401, RUF001

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
import csv
import dataclasses
from datetime import UTC
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from flax import traverse_util
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml

import openpi.models.model as model_lib
from openpi.models.rl_token import RLTokenModel
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as train_config
from openpi.training.geniesim_rollout_dataset import GenieSimRolloutRecord
from openpi.training.geniesim_rollout_dataset import discover_rollout_records
from openpi.training.geniesim_rollout_dataset import load_rollout_sample
import openpi.transforms as transforms
from scripts.serve_rlt_policy import RLTInferenceModel
from scripts.serve_rlt_policy import _create_rlt_config
from scripts.serve_rlt_policy import _infer_prefix_seq_len

DEFAULT_ROLLOUT_ROOT = Path("/mnt/pfs/kk/kk/data/data/geniesim/rollout/stack_three_blocks")
DEFAULT_CACHE_ROOT = Path("/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/cache_224")
DEFAULT_PLAN_ROOT = Path("/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan")
DEFAULT_CHECKPOINT_ROOT = (
    DEFAULT_PLAN_ROOT / "checkpoints/rlt_pi05_geniesim_stack_three_blocks_plan/stage1_rl_token_20k"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_PLAN_ROOT / "rlt_token_analysis_endpoint_v3"
DEFAULT_STEPS = (10_000, 30_000, 50_000, 70_000, 90_000, 110_000, 130_000, 150_000, 170_000)
MODEL_CONFIG = "rlt_pi05_geniesim_stack_three_blocks_plan"
ROLLOUT_CONFIG = "rlt_pi05_geniesim_stack_three_blocks"
TIME_BINS = 5
LOG_TIMESTAMP = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3}),\d{3}\s+INFO\s+"
    r".*CoRobotPolicy: calling model infer\s*$"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path | np.generic):
        return str(value) if isinstance(value, Path) else value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _step_label(step: int) -> str:
    return f"{step // 1000}k"


def _episode_id(category: str, trajectory_id: int) -> str:
    return f"{category}/trajectory_{trajectory_id:06d}"


def _uniform_indices(length: int, maximum: int) -> list[int]:
    count = min(length, maximum)
    if count <= 0:
        return []
    return np.floor((np.arange(count, dtype=np.float64) + 0.5) * length / count).astype(np.int64).tolist()


def _endpoint_uniform_indices(length: int, maximum: int) -> list[int]:
    """Select nearest available calls for an endpoint-inclusive normalized grid."""
    count = min(length, maximum)
    if count <= 0:
        return []
    if count == 1:
        return [0]
    indices = np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
    indices[0] = 0
    indices[-1] = length - 1
    if len(np.unique(indices)) != count:
        raise ValueError(f"Endpoint sampling produced duplicate indices for length={length}, count={count}")
    return indices.tolist()


def _parse_call_timestamps(log_path: Path) -> list[datetime]:
    timestamps: list[datetime] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_TIMESTAMP.match(line)
        if match is None:
            continue
        value = datetime.strptime(f"{match.group('date')}.{match.group('ms')}", "%Y-%m-%d %H:%M:%S.%f").replace(
            tzinfo=UTC
        )
        timestamps.append(value)
    return timestamps


def _load_episode_catalog(
    rollout_root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[GenieSimRolloutRecord]]]:
    grouped: dict[tuple[str, int], list[GenieSimRolloutRecord]] = defaultdict(list)
    for record in discover_rollout_records(rollout_root):
        grouped[(record.category, record.trajectory_id)].append(record)
    catalog: list[dict[str, Any]] = []
    for (category, trajectory_id), records in sorted(grouped.items()):
        records.sort(key=lambda item: item.step_id)
        trajectory_dir = rollout_root / category / f"trajectory_{trajectory_id:06d}"
        metadata_path = trajectory_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        timestamps = _parse_call_timestamps(trajectory_dir / "rollout.log")
        if len(timestamps) != len(records):
            raise ValueError(
                f"Timestamp/record mismatch for {trajectory_dir}: timestamps={len(timestamps)} records={len(records)}"
            )
        expected_calls = int(metadata.get("recorded_model_calls", len(records)))
        if expected_calls != len(records):
            raise ValueError(
                f"Metadata/record mismatch for {trajectory_dir}: expected={expected_calls} records={len(records)}"
            )
        catalog.append(
            {
                "category": category,
                "trajectory_id": trajectory_id,
                "episode_id": _episode_id(category, trajectory_id),
                "success": int(category == "successful"),
                "failure": int(category == "failed"),
                "seed": int(metadata["seed"]),
                "instance_id": int(metadata["instance_id"]),
                "attempt_id": int(metadata["attempt_id"]),
                "frame_count_available": len(records),
                "metadata_path": str(metadata_path),
                "timestamps": timestamps,
            }
        )
    return catalog, grouped


def _select_matched_episodes(catalog: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    success_seeds = [int(row["seed"]) for row in catalog if row["success"]]
    failure_seeds = [int(row["seed"]) for row in catalog if row["failure"]]
    shared_seed_min = max(min(success_seeds), min(failure_seeds))
    shared_seed_max = min(max(success_seeds), max(failure_seeds))
    eligible = [
        row
        for row in catalog
        if shared_seed_min <= int(row["seed"]) <= shared_seed_max and int(row["frame_count_available"]) >= 16
    ]
    by_outcome_instance: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_outcome_instance[(row["success"], row["instance_id"])].append(row)
    common_instances = sorted(
        instance_id
        for instance_id in {row["instance_id"] for row in catalog}
        if by_outcome_instance[(1, instance_id)] and by_outcome_instance[(0, instance_id)]
    )
    if len(common_instances) < count:
        raise ValueError(f"Need {count} shared instance IDs, found {len(common_instances)}")
    eligible_common_instance_count = len(common_instances)
    rng = random.Random(seed)
    rng.shuffle(common_instances)
    selected_instances = sorted(common_instances[:count])
    selected: list[dict[str, Any]] = []
    for pair_index, instance_id in enumerate(selected_instances):
        successful = by_outcome_instance[(1, instance_id)]
        failed = by_outcome_instance[(0, instance_id)]
        candidates = [
            (abs(int(success["seed"]) - int(failure["seed"])), success["seed"], failure["seed"], success, failure)
            for success in successful
            for failure in failed
        ]
        _, _, _, success, failure = min(
            candidates, key=lambda item: (item[0], item[1], item[2], item[3]["trajectory_id"], item[4]["trajectory_id"])
        )
        pair_id = f"pair_{pair_index:02d}_instance_{instance_id:02d}"
        for row in (success, failure):
            copied = {key: value for key, value in row.items() if key != "timestamps"}
            copied["pair_id"] = pair_id
            copied["selection_seed"] = seed
            copied["selection_strategy"] = (
                f"shared seed range and >=16 calls; seeded {count}-of-"
                f"{eligible_common_instance_count} shared instance IDs; within-instance minimum seed distance"
            )
            selected.append(copied)
    return sorted(selected, key=lambda row: (row["category"], row["trajectory_id"]))


def _make_contact_sheet(frame_rows: list[dict[str, Any]], output_path: Path) -> None:
    episode_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        episode_rows[row["episode_id"]].append(row)
    chosen_episodes: list[str] = []
    for success in (1, 0):
        candidates = sorted(key for key, rows in episode_rows.items() if int(rows[0]["success"]) == success)
        chosen_episodes.extend(candidates[:2])
    fig, axes = plt.subplots(len(chosen_episodes), 3, figsize=(12, 3 * len(chosen_episodes)), squeeze=False)
    for row_index, episode in enumerate(chosen_episodes):
        rows = sorted(episode_rows[episode], key=lambda row: int(row["frame_order"]))
        sample = rows[len(rows) // 2]
        with np.load(sample["cache_path"], allow_pickle=False) as payload:
            for column, camera in enumerate(("top_head", "hand_left", "hand_right")):
                axes[row_index, column].imshow(payload[camera])
                axes[row_index, column].axis("off")
                axes[row_index, column].set_title(f"{episode} | {camera}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def prepare_manifests(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    catalog, grouped = _load_episode_catalog(args.rollout_root.resolve())
    selected = _select_matched_episodes(catalog, args.episodes_per_outcome, args.seed)

    episode_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for episode in selected:
        key = (episode["category"], episode["trajectory_id"])
        records = sorted(grouped[key], key=lambda item: item.step_id)
        trajectory_dir = (
            args.rollout_root.resolve() / episode["category"] / f"trajectory_{episode['trajectory_id']:06d}"
        )
        timestamps = _parse_call_timestamps(trajectory_dir / "rollout.log")
        chosen_indices = _endpoint_uniform_indices(len(records), args.frames_per_episode)
        frame_ids = [records[index].step_id for index in chosen_indices]
        episode_rows.append(
            {
                **episode,
                "selected_frame_count": len(chosen_indices),
                "frame_indices": json.dumps(frame_ids),
                "local_frame_indices": json.dumps(chosen_indices),
            }
        )
        first_timestamp = timestamps[0]
        selected_count = len(chosen_indices)
        for selected_order, local_index in enumerate(chosen_indices):
            record = records[local_index]
            timestamp = timestamps[local_index]
            # Keep the analysis coordinate on the same [0, 1] grid for every
            # episode; retain the physical call location for auditability.
            normalized_time = selected_order / max(selected_count - 1, 1)
            actual_normalized_time = local_index / max(len(records) - 1, 1)
            cache_path = (
                args.cache_root.resolve()
                / "records"
                / record.category
                / f"trajectory_{record.trajectory_id:06d}"
                / f"step_{record.step_id}.npz"
            )
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing rollout cache record: {cache_path}")
            frame_rows.append(
                {
                    "row_index": len(frame_rows),
                    "sample_id": f"{episode['episode_id']}/call_{local_index:03d}_step_{record.step_id}",
                    "episode_id": episode["episode_id"],
                    "pair_id": episode["pair_id"],
                    "category": episode["category"],
                    "trajectory_id": episode["trajectory_id"],
                    "frame_id": record.step_id,
                    "frame_order": local_index,
                    "selected_order": selected_order,
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "elapsed_seconds": (timestamp - first_timestamp).total_seconds(),
                    "normalized_time": normalized_time,
                    "actual_normalized_time": actual_normalized_time,
                    "time_bin": min(int(normalized_time * TIME_BINS), TIME_BINS - 1),
                    "success": episode["success"],
                    "failure": episode["failure"],
                    "seed": episode["seed"],
                    "instance_id": episode["instance_id"],
                    "source_path": str(record.source_path),
                    "cache_path": str(cache_path),
                }
            )

    episode_fields = [
        "episode_id",
        "pair_id",
        "category",
        "trajectory_id",
        "success",
        "failure",
        "seed",
        "instance_id",
        "attempt_id",
        "frame_count_available",
        "selected_frame_count",
        "frame_indices",
        "local_frame_indices",
        "selection_seed",
        "selection_strategy",
        "metadata_path",
    ]
    frame_fields = [
        "row_index",
        "sample_id",
        "episode_id",
        "pair_id",
        "category",
        "trajectory_id",
        "frame_id",
        "frame_order",
        "selected_order",
        "timestamp",
        "elapsed_seconds",
        "normalized_time",
        "actual_normalized_time",
        "time_bin",
        "success",
        "failure",
        "seed",
        "instance_id",
        "source_path",
        "cache_path",
    ]
    _write_csv(output_root / "episodes_manifest.csv", episode_rows, episode_fields)
    _write_csv(output_root / "episodes.csv", episode_rows, episode_fields)
    _write_csv(output_root / "frames.csv", frame_rows, frame_fields)
    _make_contact_sheet(frame_rows, output_root / "dataset_contact_sheet.png")

    model_config = train_config.get_config(MODEL_CONFIG)
    rollout_config = train_config.get_config(ROLLOUT_CONFIG)
    if dataclasses.asdict(model_config.model) != dataclasses.asdict(rollout_config.model):
        raise RuntimeError("Plan and rollout model architectures differ")
    checkpoints = {str(step): str((args.checkpoint_root / str(step)).resolve()) for step in args.steps}
    for step, checkpoint in checkpoints.items():
        if not (Path(checkpoint) / "params/_METADATA").is_file():
            raise FileNotFoundError(f"Checkpoint {step} is incomplete: {checkpoint}")
    parameter_audit_path = DEFAULT_PLAN_ROOT / "analysis/stage1_parameter_audit.json"
    parameter_audit = json.loads(parameter_audit_path.read_text(encoding="utf-8"))
    if not parameter_audit.get("verified"):
        raise RuntimeError(f"Frozen-backbone parameter audit is not verified: {parameter_audit_path}")
    resolved = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "feature_definition": "flatten(RLTokenModel.encoder(image_only VLA prefix)); shape [N, 1*2048]",
        "base_vla": str(train_config.GENIESIM_RLT_BASE_CKPT),
        "model_config": MODEL_CONFIG,
        "rollout_preprocessing_config": ROLLOUT_CONFIG,
        "rollout_root": str(args.rollout_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "checkpoints": checkpoints,
        "selection": {
            "strategy": selected[0]["selection_strategy"],
            "seed": args.seed,
            "episodes_per_outcome": args.episodes_per_outcome,
            "max_frames_per_episode": args.frames_per_episode,
            "frame_sampling": (
                "deterministic endpoint-inclusive uniform selection of exactly 16 policy calls per eligible episode; "
                "the first and last available policy calls are always included, and interior targets are mapped "
                "to their nearest available policy call"
            ),
            "semantic_phase_available": False,
            "time_proxy": (
                "five equal bins over the shared endpoint-inclusive normalized-time grid; normalized_time is the "
                "target grid coordinate and actual_normalized_time records the selected call's physical location"
            ),
            "timestamp_source": "UTC timestamps of CoRobotPolicy infer calls parsed from each rollout.log",
        },
        "model": dataclasses.asdict(model_config.model),
        "rlt": {
            "num_tokens": model_config.rlt_num_tokens,
            "num_layers": model_config.rlt_num_layers,
            "embed_dim": model_config.rlt_embed_dim,
            "input_dim": model_config.rlt_input_dim,
            "training_alpha": model_config.rlt_alpha,
        },
        "parameter_audit": parameter_audit,
        "analysis": {
            "pca_fit": "independently per checkpoint on all selected samples",
            "tsne": {"random_state": 42, "init": "pca", "perplexity": 30, "max_iter": 1000},
            "bootstrap_replicates": args.bootstrap_replicates,
            "permutation_replicates": args.permutation_replicates,
            "checkpoint_comparison": "unbiased centered linear CKA and pairwise-distance Spearman on fixed samples",
        },
        "software": {
            "python": platform.python_version(),
            "jax": importlib.metadata.version("jax"),
            "flax": importlib.metadata.version("flax"),
            "orbax-checkpoint": importlib.metadata.version("orbax-checkpoint"),
            "numpy": np.__version__,
            "scikit-learn": importlib.metadata.version("scikit-learn"),
        },
    }
    resolved["manifest_sha256"] = _sha256_file(output_root / "frames.csv")
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(json.loads(json.dumps(resolved, default=_json_default)), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _atomic_json(
        output_root / "dataset_audit.json",
        {
            "available_success_episodes": sum(row["success"] for row in catalog),
            "available_failure_episodes": sum(row["failure"] for row in catalog),
            "selected_success_episodes": sum(row["success"] for row in selected),
            "selected_failure_episodes": sum(row["failure"] for row in selected),
            "selected_frames": len(frame_rows),
            "short_selected_episodes": [
                row["episode_id"] for row in episode_rows if row["selected_frame_count"] < args.frames_per_episode
            ],
            "excluded_short_failure_episodes": sum(
                row["failure"] and row["frame_count_available"] < 16 for row in catalog
            ),
            "sample_ids_unique": len({row["sample_id"] for row in frame_rows}) == len(frame_rows),
            "all_cache_records_exist": True,
            "manifest_sha256": resolved["manifest_sha256"],
            "frame_sampling": resolved["selection"]["frame_sampling"],
            "time_proxy": resolved["selection"]["time_proxy"],
        },
    )
    logging.info("Prepared %d episodes and %d fixed frames in %s", len(selected), len(frame_rows), output_root)


def _input_transform(config: train_config.TrainConfig):
    data_config = config.data.create(config.assets_dirs, config.model)
    return transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )


def _stack(items: list[dict[str, Any]]) -> dict[str, Any]:
    return jax.tree.map(lambda *values: np.stack([np.asarray(value) for value in values]), *items)


def _flatten_array_tree(tree: dict[str, Any]) -> dict[str, Any]:
    flat = traverse_util.flatten_dict(tree)
    return {"/".join(str(part) for part in path): value for path, value in flat.items() if hasattr(value, "shape")}


def _tree_fingerprint(flat: dict[str, Any], *, include_rlt: bool) -> dict[str, Any]:
    selected = {path: value for path, value in flat.items() if ("rlt_module" in path) == include_rlt}
    schema = hashlib.sha256()
    for path, value in sorted(selected.items()):
        schema.update(f"{path}|{tuple(value.shape)}|{value.dtype}\n".encode())
    small = [(path, value) for path, value in sorted(selected.items()) if 0 < int(np.prod(value.shape)) <= 8192]
    if len(small) > 48:
        positions = _uniform_indices(len(small), 48)
        small = [small[index] for index in positions]
    values = hashlib.sha256()
    sampled_paths: list[str] = []
    for path, value in small:
        host = np.asarray(jax.device_get(value))
        values.update(path.encode())
        values.update(host.tobytes())
        sampled_paths.append(path)
    return {
        "leaf_count": len(selected),
        "schema_sha256": schema.hexdigest(),
        "sampled_value_sha256": values.hexdigest(),
        "sampled_leaf_count": len(small),
        "sampled_paths": sampled_paths,
    }


def _load_model_strict(
    config: train_config.TrainConfig, checkpoint_path: Path
) -> tuple[RLTInferenceModel, dict[str, Any]]:
    rlt_config = _create_rlt_config(config)
    prefix_seq_len = _infer_prefix_seq_len(config.model)
    vla_model = nnx.eval_shape(config.model.create, jax.random.key(0))
    model = RLTInferenceModel(
        vla_model,
        rlt_config,
        rngs=nnx.Rngs(jax.random.key(0)),
        prefix_seq_len=prefix_seq_len,
    )
    graphdef, state = nnx.split(model)
    expected = state.to_pure_dict()
    loaded = model_lib.restore_params(checkpoint_path / "params", dtype=None)
    expected_flat = _flatten_array_tree(expected)
    loaded_flat = _flatten_array_tree(loaded)
    missing = sorted(set(expected_flat) - set(loaded_flat))
    unexpected = sorted(set(loaded_flat) - set(expected_flat))
    shape_mismatches = sorted(
        path
        for path in set(expected_flat) & set(loaded_flat)
        if tuple(expected_flat[path].shape) != tuple(loaded_flat[path].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f"Strict checkpoint restore failed: missing={missing[:8]} unexpected={unexpected[:8]} "
            f"shape_mismatches={shape_mismatches[:8]}"
        )
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    audit = {
        "checkpoint": str(checkpoint_path),
        "expected_leaf_count": len(expected_flat),
        "loaded_leaf_count": len(loaded_flat),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "strict_restore": True,
        "restore_dtype": "stored dtype (dtype=None)",
        "prefix_shape": [prefix_seq_len, rlt_config.input_dim],
        "token_shape": [rlt_config.num_rl_tokens, rlt_config.embed_dim],
        "backbone_fingerprint": _tree_fingerprint(loaded_flat, include_rlt=False),
        "rlt_fingerprint": _tree_fingerprint(loaded_flat, include_rlt=True),
    }
    return model, audit


def _audit_frozen_backbones(checkpoint_root: Path, steps: Iterable[int]) -> dict[str, Any]:
    step_list = list(steps)
    schemas: dict[str, str] = {}
    metadata_by_step: dict[int, dict[tuple[Any, ...], Any]] = {}
    for step in step_list:
        with ocp.PyTreeCheckpointer() as checkpointer:
            metadata = checkpointer.metadata(checkpoint_root / str(step) / "params")
        flat = traverse_util.flatten_dict(metadata["params"])
        metadata_by_step[step] = flat
        schema = hashlib.sha256()
        for path, value in sorted(flat.items(), key=lambda item: tuple(str(part) for part in item[0])):
            if not path or path[0] != "vla":
                continue
            schema.update(f"{'/'.join(map(str, path))}|{tuple(value.shape)}|{value.dtype}\n".encode())
        schemas[str(step)] = schema.hexdigest()
    if len(set(schemas.values())) != 1:
        raise RuntimeError(f"VLA parameter schemas differ across checkpoints: {schemas}")

    first_flat = metadata_by_step[step_list[0]]
    vla_paths = [
        path
        for path, value in sorted(first_flat.items(), key=lambda item: tuple(str(part) for part in item[0]))
        if path and path[0] == "vla" and int(np.prod(value.shape)) > 0
    ]
    value_fingerprints: dict[str, str] = {}
    hashed_elements = 0
    for step in step_list:
        flat = metadata_by_step[step]
        target: dict[str, np.ndarray] = {}
        restore_args: dict[str, ocp.ArrayRestoreArgs] = {}
        restore_transforms: dict[str, ocp.Transform] = {}
        for index, path in enumerate(vla_paths):
            key = f"vla_leaf_{index:03d}"
            value = flat[path]
            target[key] = np.empty(value.shape, dtype=value.dtype)
            restore_args[key] = ocp.ArrayRestoreArgs(restore_type=np.ndarray)
            original = "params/" + "/".join(map(str, path))
            restore_transforms[key] = ocp.Transform(original_key=original)
        with ocp.PyTreeCheckpointer() as checkpointer:
            restored = checkpointer.restore(
                checkpoint_root / str(step) / "params",
                ocp.args.PyTreeRestore(
                    item=target,
                    restore_args=restore_args,
                    transforms=restore_transforms,
                    transforms_default_to_original=False,
                ),
            )
        digest = hashlib.sha256()
        for key in sorted(restored):
            value = np.asarray(restored[key])
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value).view(np.uint8))
            if step == step_list[0]:
                hashed_elements += value.size
        value_fingerprints[str(step)] = digest.hexdigest()
        del restored, target
        gc.collect()
        logging.info("Verified all frozen VLA values at checkpoint %s", step)
    if len(set(value_fingerprints.values())) != 1:
        raise RuntimeError(f"VLA values differ across checkpoints: {value_fingerprints}")
    return {
        "consistent": True,
        "method": "all-leaf path/shape/dtype schema plus bitwise hash of every value in every VLA leaf",
        "checkpoint_steps": step_list,
        "schema_sha256": schemas,
        "full_value_sha256": value_fingerprints,
        "hashed_leaf_paths": ["/".join(map(str, path)) for path in vla_paths],
        "hashed_leaf_count": len(vla_paths),
        "hashed_element_count": hashed_elements,
        "training_parameter_audit": str(DEFAULT_PLAN_ROOT / "analysis/stage1_parameter_audit.json"),
    }


def cache_prefixes(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    frame_rows = _read_csv(output_root / "frames.csv")
    checkpoint_root = args.checkpoint_root.resolve()
    checkpoint_path = checkpoint_root / str(args.reference_step)
    backbone_audit = _audit_frozen_backbones(checkpoint_root, args.steps)
    _atomic_json(output_root / "backbone_audit.json", backbone_audit)
    config = train_config.get_config(MODEL_CONFIG)
    rollout_config = train_config.get_config(ROLLOUT_CONFIG)
    model, audit = _load_model_strict(config, checkpoint_path)
    transform = _input_transform(rollout_config)
    extract_fn = nnx_utils.module_jit(model.extract_image_prefix)
    prefix_shape = tuple(audit["prefix_shape"])
    prefix_path = output_root / "prefix_embeddings.npy"
    mask_path = output_root / "prefix_mask.npy"
    prefix = np.lib.format.open_memmap(prefix_path, mode="w+", dtype=np.float32, shape=(len(frame_rows), *prefix_shape))
    mask = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.bool_, shape=(len(frame_rows), prefix_shape[0]))
    content_digest = hashlib.sha256()
    rng = jax.random.key(config.seed)
    for start in range(0, len(frame_rows), args.batch_size):
        real_rows = frame_rows[start : start + args.batch_size]
        samples: list[dict[str, Any]] = []
        for row in real_rows:
            record = GenieSimRolloutRecord(
                category=row["category"],
                trajectory_id=int(row["trajectory_id"]),
                step_id=int(row["frame_id"]),
                source_path=Path(row["source_path"]),
                cache_path=Path(row["cache_path"]),
            )
            samples.append(transform(load_rollout_sample(record)))
        while len(samples) < args.batch_size:
            samples.append(samples[-1])
        observation = model_lib.Observation.from_dict(_stack(samples))
        rng, batch_rng = jax.random.split(rng)
        batch_prefix, batch_mask = extract_fn(batch_rng, observation)
        count = len(real_rows)
        host_prefix = np.asarray(jax.device_get(batch_prefix[:count]), dtype=np.float32)
        host_mask = np.asarray(jax.device_get(batch_mask[:count]), dtype=bool)
        prefix[start : start + count] = host_prefix
        mask[start : start + count] = host_mask
        content_digest.update(host_prefix.tobytes())
        content_digest.update(host_mask.tobytes())
        logging.info("Cached frozen VLA prefixes %d/%d", start + count, len(frame_rows))
    prefix.flush()
    mask.flush()
    true_fraction = float(np.asarray(mask).mean())
    if true_fraction != 1.0:
        raise RuntimeError(f"Expected all image-prefix mask entries to be true, got fraction={true_fraction}")
    prefix_file_sha256 = _sha256_file(prefix_path)
    mask_file_sha256 = _sha256_file(mask_path)
    _atomic_json(output_root / "prefix_load_audit.json", audit)
    _atomic_json(
        output_root / "prefix_cache_manifest.json",
        {
            "reference_checkpoint": str(checkpoint_path),
            "feature_source": "RLTInferenceModel.extract_image_prefix(image_only=True, train=False)",
            "prefix_path": str(prefix_path),
            "mask_path": str(mask_path),
            "shape": list(prefix.shape),
            "dtype": str(prefix.dtype),
            "mask_shape": list(mask.shape),
            "mask_dtype": str(mask.dtype),
            "mask_true_fraction": true_fraction,
            "content_sha256": content_digest.hexdigest(),
            "prefix_file_sha256": prefix_file_sha256,
            "mask_file_sha256": mask_file_sha256,
            "prefix_file_size": prefix_path.stat().st_size,
            "mask_file_size": mask_path.stat().st_size,
            "frames_manifest_sha256": _sha256_file(output_root / "frames.csv"),
        },
    )
    logging.info("Saved frozen prefix cache %s with shape %s", prefix_path, prefix.shape)


class _RLTAnalysisModule(nnx.Module):
    def __init__(self, config: train_config.TrainConfig, prefix_seq_len: int):
        rlt_config = _create_rlt_config(config)
        self.rlt_module = nnx_bridge.ToNNX(RLTokenModel(config=rlt_config))
        self.rlt_module.lazy_init(
            jnp.zeros((1, prefix_seq_len, rlt_config.input_dim), dtype=jnp.float32),
            jnp.ones((1, prefix_seq_len), dtype=jnp.bool_),
            rngs=nnx.Rngs(jax.random.key(0)),
        )
        self.prefix_seq_len = prefix_seq_len

    def encode(self, prefix: jax.Array) -> jax.Array:
        return self.rlt_module(prefix, None, method="encode", train=False)

    def decode(self, tokens: jax.Array) -> jax.Array:
        return self.rlt_module(tokens, self.prefix_seq_len, method="decode", train=False)


def _load_rlt_only_strict(config: train_config.TrainConfig, checkpoint_path: Path, prefix_seq_len: int):
    module = _RLTAnalysisModule(config, prefix_seq_len)
    graphdef, state = nnx.split(module)
    expected = state.to_pure_dict()
    restore_args = jax.tree.map(
        lambda _: ocp.ArrayRestoreArgs(
            sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
            restore_type=jax.Array,
        ),
        expected,
    )
    transforms_map = {r"rlt_module/(.*)": ocp.Transform(original_key=r"params/rlt_module/\1/value")}
    with ocp.PyTreeCheckpointer() as checkpointer:
        loaded = checkpointer.restore(
            checkpoint_path / "params",
            ocp.args.PyTreeRestore(
                item=expected,
                restore_args=restore_args,
                transforms=transforms_map,
                transforms_default_to_original=False,
            ),
        )
    expected_flat = _flatten_array_tree(expected)
    loaded_flat = _flatten_array_tree(loaded)
    missing = sorted(set(expected_flat) - set(loaded_flat))
    unexpected = sorted(set(loaded_flat) - set(expected_flat))
    shape_mismatches = sorted(
        path for path in set(expected_flat) & set(loaded_flat) if expected_flat[path].shape != loaded_flat[path].shape
    )
    dtype_mismatches = sorted(
        path for path in set(expected_flat) & set(loaded_flat) if expected_flat[path].dtype != loaded_flat[path].dtype
    )
    if missing or unexpected or shape_mismatches or dtype_mismatches:
        raise RuntimeError(
            "Strict RLT-only restore failed: "
            f"missing={missing[:8]} unexpected={unexpected[:8]} shape={shape_mismatches[:8]} dtype={dtype_mismatches[:8]}"
        )
    state.replace_by_pure_dict(loaded)
    module = nnx.merge(graphdef, state)
    audit = {
        "checkpoint": str(checkpoint_path),
        "strict_restore": True,
        "restore_scope": "params/rlt_module only",
        "restore_dtype": "stored dtype (float32)",
        "expected_leaf_count": len(expected_flat),
        "loaded_leaf_count": len(loaded_flat),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "rlt_fingerprint": _tree_fingerprint(loaded_flat, include_rlt=True),
    }
    return module, audit


def _episode_groups(values: np.ndarray, episode_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(episode_ids)
    means = np.stack([values[episode_ids == episode].mean(axis=0) for episode in unique])
    return unique, means


def _gap_statistics(vectors: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    success = vectors[labels == 1]
    failure = vectors[labels == 0]
    if len(success) < 2 or len(failure) < 2:
        return {"raw": float("nan"), "scaled": float("nan"), "debiased_squared": float("nan")}
    success_mean = success.mean(axis=0)
    failure_mean = failure.mean(axis=0)
    raw = float(np.linalg.norm(success_mean - failure_mean))
    success_ss = float(np.square(success - success_mean).sum())
    failure_ss = float(np.square(failure - failure_mean).sum())
    pooled_rms = math.sqrt((success_ss / len(success) + failure_ss / len(failure)) / 2.0)
    success_cov_trace = success_ss / (len(success) - 1)
    failure_cov_trace = failure_ss / (len(failure) - 1)
    return {
        "raw": raw,
        "scaled": raw / pooled_rms if pooled_rms > 1e-12 else float("nan"),
        "debiased_squared": raw**2 - success_cov_trace / len(success) - failure_cov_trace / len(failure),
    }


def _paired_gap_inference(
    episode_vectors: np.ndarray,
    episode_labels: np.ndarray,
    episode_pairs: np.ndarray,
    *,
    bootstrap_replicates: int,
    permutation_replicates: int,
) -> dict[str, float]:
    pair_names = np.unique(episode_pairs)
    pair_rows: list[tuple[np.ndarray, np.ndarray]] = []
    for pair in pair_names:
        rows = np.flatnonzero(episode_pairs == pair)
        success_rows = rows[episode_labels[rows] == 1]
        failure_rows = rows[episode_labels[rows] == 0]
        if len(success_rows) != 1 or len(failure_rows) != 1:
            raise ValueError(f"Pair {pair!r} does not contain exactly one success and one failure episode")
        pair_rows.append((episode_vectors[success_rows[0]], episode_vectors[failure_rows[0]]))
    rng = np.random.default_rng(42)
    bootstrap: list[float] = []
    for _ in range(bootstrap_replicates):
        chosen = rng.integers(0, len(pair_rows), size=len(pair_rows))
        vectors = np.stack([value for index in chosen for value in pair_rows[index]])
        labels = np.tile(np.array([1, 0], dtype=np.int8), len(chosen))
        bootstrap.append(_gap_statistics(vectors, labels)["scaled"])
    observed_delta = np.mean([success - failure for success, failure in pair_rows], axis=0)
    observed = float(np.linalg.norm(observed_delta))
    exceed = 0
    deltas = np.stack([success - failure for success, failure in pair_rows])
    for _ in range(permutation_replicates):
        signs = rng.choice(np.array([-1.0, 1.0]), size=(len(deltas), 1))
        permuted = float(np.linalg.norm(np.mean(deltas * signs, axis=0)))
        exceed += int(permuted >= observed)
    finite_bootstrap = np.asarray([value for value in bootstrap if np.isfinite(value)], dtype=np.float64)
    if finite_bootstrap.size == 0:
        raise RuntimeError("Paired bootstrap produced no finite scaled-gap estimates")
    return {
        "success_gap_scaled_ci_low": float(np.percentile(finite_bootstrap, 2.5)),
        "success_gap_scaled_ci_high": float(np.percentile(finite_bootstrap, 97.5)),
        "success_permutation_p": (exceed + 1) / (permutation_replicates + 1),
    }


def _episode_equal_scalar(values: np.ndarray, episode_ids: np.ndarray) -> tuple[float, float, np.ndarray]:
    _, episode_values = _episode_groups(values[:, None], episode_ids)
    episode_values = episode_values[:, 0]
    return float(episode_values.mean()), float(episode_values.std()), episode_values


def _success_probe_auc(
    z: np.ndarray, labels: np.ndarray, episode_ids: np.ndarray, pair_ids: np.ndarray
) -> tuple[float, float]:
    n_splits = min(5, len(np.unique(pair_ids)))
    if n_splits < 2:
        return float("nan"), float("nan")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    episode_counts = {episode: int(np.sum(episode_ids == episode)) for episode in np.unique(episode_ids)}
    sample_weights = np.asarray([1.0 / episode_counts[episode] for episode in episode_ids], dtype=np.float64)
    scores: list[float] = []
    for train_indices, test_indices in splitter.split(z, labels, groups=pair_ids):
        if set(episode_ids[train_indices]) & set(episode_ids[test_indices]):
            raise RuntimeError("Episode leakage detected in success probe")
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=42),
        )
        classifier.fit(
            z[train_indices],
            labels[train_indices],
            logisticregression__sample_weight=sample_weights[train_indices],
        )
        probability = classifier.predict_proba(z[test_indices])[:, 1]
        scores.append(
            float(roc_auc_score(labels[test_indices], probability, sample_weight=sample_weights[test_indices]))
        )
    return float(np.mean(scores)), float(np.std(scores))


def _compute_metrics(
    z: np.ndarray,
    metadata: dict[str, np.ndarray],
    reconstruction: dict[str, np.ndarray | float],
    *,
    bootstrap_replicates: int,
    permutation_replicates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    episode_ids = metadata["episode_id"]
    pair_ids = metadata["pair_id"]
    labels = metadata["success"].astype(np.int8)
    unique_episodes, episode_z = _episode_groups(z, episode_ids)
    episode_labels = np.asarray([labels[np.flatnonzero(episode_ids == episode)[0]] for episode in unique_episodes])
    episode_pairs = np.asarray([pair_ids[np.flatnonzero(episode_ids == episode)[0]] for episode in unique_episodes])
    gap = _gap_statistics(episode_z, episode_labels)
    paired = _paired_gap_inference(
        episode_z,
        episode_labels,
        episode_pairs,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
    )

    global_center = episode_z.mean(axis=0)
    centered_radii = [
        float(np.mean(np.square(z[episode_ids == episode] - global_center))) for episode in unique_episodes
    ]
    centered_rms = math.sqrt(float(np.mean(centered_radii)) * z.shape[1])

    adjacent_episode: list[float] = []
    adjacent_cos_episode: list[float] = []
    nonadjacent_episode: list[float] = []
    for episode in unique_episodes:
        indices = np.flatnonzero(episode_ids == episode)
        order = np.argsort(metadata["frame_order"][indices])
        values = z[indices[order]]
        distances = np.linalg.norm(np.diff(values, axis=0), axis=1)
        adjacent_episode.append(float(distances.mean()))
        dots = np.sum(values[:-1] * values[1:], axis=1)
        denoms = np.linalg.norm(values[:-1], axis=1) * np.linalg.norm(values[1:], axis=1)
        adjacent_cos_episode.append(
            float(np.mean(np.divide(dots, denoms, out=np.zeros_like(dots), where=denoms > 1e-12)))
        )
        far = [
            np.linalg.norm(values[right] - values[left])
            for left in range(len(values))
            for right in range(left + 2, len(values))
        ]
        nonadjacent_episode.append(float(np.mean(far)) if far else float("nan"))
    adjacent = np.asarray(adjacent_episode)
    nonadjacent = np.asarray(nonadjacent_episode)

    pca_components = min(50, len(z) - 1, z.shape[1])
    pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=42)
    pca_values = pca.fit_transform(z)
    eigenvalues = pca.explained_variance_
    eigen_weights = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    effective_rank = float(np.exp(-np.sum(eigen_weights * np.log(np.clip(eigen_weights, 1e-12, None)))))
    tsne_values = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42,
        learning_rate="auto",
        max_iter=1000,
    ).fit_transform(pca_values)

    time_rows: list[dict[str, Any]] = []
    episode_bin_vectors: list[np.ndarray] = []
    episode_bin_numbers: list[int] = []
    episode_bin_episodes: list[str] = []
    for time_bin in range(TIME_BINS):
        bin_episode_vectors: list[np.ndarray] = []
        bin_labels: list[int] = []
        for episode in unique_episodes:
            rows = (episode_ids == episode) & (metadata["time_bin"] == time_bin)
            if not np.any(rows):
                continue
            vector = z[rows].mean(axis=0)
            bin_episode_vectors.append(vector)
            bin_labels.append(int(labels[np.flatnonzero(rows)[0]]))
            episode_bin_vectors.append(vector)
            episode_bin_numbers.append(time_bin)
            episode_bin_episodes.append(episode)
        bin_gap = _gap_statistics(np.stack(bin_episode_vectors), np.asarray(bin_labels))
        time_rows.append(
            {
                "time_bin": time_bin,
                "normalized_time_start": time_bin / TIME_BINS,
                "normalized_time_end": (time_bin + 1) / TIME_BINS,
                "n_episodes": len(bin_episode_vectors),
                "n_success_episodes": sum(bin_labels),
                "n_failure_episodes": len(bin_labels) - sum(bin_labels),
                "success_gap_raw": bin_gap["raw"],
                "success_gap_scaled": bin_gap["scaled"],
                "success_gap2_debiased": bin_gap["debiased_squared"],
            }
        )
    bin_vectors = np.stack(episode_bin_vectors)
    bin_numbers = np.asarray(episode_bin_numbers)
    bin_episodes = np.asarray(episode_bin_episodes)
    residuals = np.empty_like(bin_vectors)
    for episode in np.unique(bin_episodes):
        rows = bin_episodes == episode
        residuals[rows] = bin_vectors[rows] - bin_vectors[rows].mean(axis=0)
    overall_residual = residuals.mean(axis=0)
    ss_total = float(np.square(residuals - overall_residual).sum())
    ss_time = 0.0
    for time_bin in range(TIME_BINS):
        rows = bin_numbers == time_bin
        if np.any(rows):
            ss_time += int(rows.sum()) * float(np.square(residuals[rows].mean(axis=0) - overall_residual).sum())
    time_eta2 = ss_time / ss_total if ss_total > 1e-12 else float("nan")

    norm_mean, norm_std, episode_norms = _episode_equal_scalar(np.linalg.norm(z, axis=1), episode_ids)
    recon_mse_mean, recon_mse_std, recon_episode_mse = _episode_equal_scalar(
        np.asarray(reconstruction["mse_all_per_sample"]), episode_ids
    )
    recon_cos_mean, recon_cos_std, recon_episode_cos = _episode_equal_scalar(
        np.asarray(reconstruction["cosine_all_per_sample"]), episode_ids
    )
    probe_mean, probe_std = _success_probe_auc(z, labels, episode_ids, pair_ids)
    finite = bool(
        np.isfinite(z).all()
        and all(np.isfinite(value) for value in adjacent_episode)
        and all(np.isfinite(np.asarray(value)).all() for value in reconstruction.values())
    )
    metrics: dict[str, Any] = {
        "D": z.shape[1],
        "N": len(z),
        "N_episodes": len(unique_episodes),
        "N_success_episodes": int(episode_labels.sum()),
        "N_failure_episodes": int(len(episode_labels) - episode_labels.sum()),
        "PCA_var1": float(pca.explained_variance_ratio_[0]),
        "PCA_var2": float(pca.explained_variance_ratio_[1]),
        "PCA_var12": float(pca.explained_variance_ratio_[:2].sum()),
        "PCA_effective_rank_50": effective_rank,
        "success_gap": gap["raw"],
        "success_gap_scaled": gap["scaled"],
        "success_gap2_debiased": gap["debiased_squared"],
        **paired,
        "success_cv_auc_mean": probe_mean,
        "success_cv_auc_std": probe_std,
        "smoothness": float(adjacent.mean()),
        "smoothness_std": float(adjacent.std()),
        "smoothness_scaled": float(adjacent.mean() / centered_rms) if centered_rms > 1e-12 else float("nan"),
        "temporal_cosine_mean": float(np.mean(adjacent_cos_episode)),
        "continuity_ratio": float(np.nanmean(adjacent / nonadjacent)),
        "time_eta2": time_eta2,
        "reconstruction": recon_mse_mean,
        "reconstruction_std": recon_mse_std,
        "reconstruction_global_mse": float(reconstruction["global_mse_all"]),
        "reconstruction_nmse": float(reconstruction["global_nmse_all"]),
        "reconstruction_r2": 1.0 - float(reconstruction["global_nmse_all"]),
        "cosine": recon_cos_mean,
        "cosine_std": recon_cos_std,
        "cosine_p10": float(np.percentile(recon_episode_cos, 10)),
        "cosine_p50": float(np.percentile(recon_episode_cos, 50)),
        "cosine_p90": float(np.percentile(recon_episode_cos, 90)),
        "masked_reconstruction": float(reconstruction["global_mse_masked"]),
        "masked_cosine": float(np.mean(np.asarray(reconstruction["cosine_masked_per_sample"]))),
        "prefix_mask_true_fraction": float(reconstruction["mask_true_fraction"]),
        "token_norm_mean": norm_mean,
        "token_norm_std": norm_std,
        "token_norm_p10": float(np.percentile(episode_norms, 10)),
        "token_norm_p50": float(np.percentile(episode_norms, 50)),
        "token_norm_p90": float(np.percentile(episode_norms, 90)),
        "token_norm_cv": norm_std / norm_mean if norm_mean > 1e-12 else float("nan"),
        "token_centered_rms": centered_rms,
        "token_rms_per_dim_std": centered_rms / math.sqrt(z.shape[1]),
        "finite": finite,
        "representation_collapse": bool(centered_rms <= 1e-6),
    }
    return metrics, time_rows, pca_values[:, :2], tsne_values


def _plot_embeddings(
    path: Path,
    values: np.ndarray,
    metadata: dict[str, np.ndarray],
    *,
    title: str,
    trajectories: bool,
) -> None:
    fig, axes = plt.subplots(1, 3 if trajectories else 2, figsize=(15 if trajectories else 10, 4.6))
    labels = metadata["success"].astype(bool)
    axes[0].scatter(values[~labels, 0], values[~labels, 1], c="#c84c4c", s=15, alpha=0.65, label="failure")
    axes[0].scatter(values[labels, 0], values[labels, 1], c="#248f61", s=15, alpha=0.65, label="success")
    axes[0].set_title("Outcome")
    axes[0].legend(frameon=False)
    color = axes[1].scatter(values[:, 0], values[:, 1], c=metadata["normalized_time"], cmap="viridis", s=15, alpha=0.75)
    axes[1].set_title("Normalized episode time")
    fig.colorbar(color, ax=axes[1], fraction=0.046, pad=0.04)
    if trajectories:
        for episode in np.unique(metadata["episode_id"]):
            indices = np.flatnonzero(metadata["episode_id"] == episode)
            indices = indices[np.argsort(metadata["frame_order"][indices])]
            line_color = "#248f61" if metadata["success"][indices[0]] else "#c84c4c"
            axes[2].plot(values[indices, 0], values[indices, 1], color=line_color, alpha=0.28, linewidth=0.8)
        axes[2].set_title("Within-episode trajectories")
    for axis in axes:
        axis.set_xlabel("component 1")
        axis.set_ylabel("component 2")
        axis.grid(alpha=0.15)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_checkpoint(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    checkpoint_path = args.checkpoint_root.resolve() / str(args.step)
    target_dir = (
        args.target_dir.resolve() if args.target_dir is not None else output_root / f"ckpt_{_step_label(args.step)}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    all_rows = _read_csv(output_root / "frames.csv")
    selected_indices = np.arange(len(all_rows), dtype=np.int64)
    if args.smoke_episodes_per_outcome:
        pair_keep = sorted({row["pair_id"] for row in all_rows})[: args.smoke_episodes_per_outcome]
        episode_keep = {
            row["episode_id"] for row in all_rows if row["pair_id"] in pair_keep
        }
        expected_episodes = 2 * len(pair_keep)
        if len(episode_keep) != expected_episodes:
            raise RuntimeError(
                f"Smoke selection expected {expected_episodes} episodes from complete pairs, "
                f"found {len(episode_keep)}"
            )
        selected_indices = np.asarray(
            [index for index, row in enumerate(all_rows) if row["episode_id"] in episode_keep], dtype=np.int64
        )
    rows = [all_rows[index] for index in selected_indices]
    prefix_manifest = json.loads((output_root / "prefix_cache_manifest.json").read_text(encoding="utf-8"))
    if prefix_manifest["frames_manifest_sha256"] != _sha256_file(output_root / "frames.csv"):
        raise RuntimeError("Prefix cache does not match frames.csv")
    prefix_all = np.load(output_root / "prefix_embeddings.npy", mmap_mode="r")
    mask_all = np.load(output_root / "prefix_mask.npy", mmap_mode="r")
    if list(prefix_all.shape) != prefix_manifest["shape"] or str(prefix_all.dtype) != prefix_manifest["dtype"]:
        raise RuntimeError(
            f"Prefix cache shape/dtype mismatch: got {prefix_all.shape}/{prefix_all.dtype}, "
            f"expected {prefix_manifest['shape']}/{prefix_manifest['dtype']}"
        )
    if list(mask_all.shape) != prefix_manifest["mask_shape"] or str(mask_all.dtype) != prefix_manifest["mask_dtype"]:
        raise RuntimeError(
            f"Prefix mask shape/dtype mismatch: got {mask_all.shape}/{mask_all.dtype}, "
            f"expected {prefix_manifest['mask_shape']}/{prefix_manifest['mask_dtype']}"
        )
    if len(prefix_all) != len(all_rows) or len(mask_all) != len(all_rows):
        raise RuntimeError("Prefix cache sample count does not match frames.csv")
    prefix_path = output_root / "prefix_embeddings.npy"
    mask_path = output_root / "prefix_mask.npy"
    if (
        prefix_path.stat().st_size != prefix_manifest["prefix_file_size"]
        or mask_path.stat().st_size != prefix_manifest["mask_file_size"]
    ):
        raise RuntimeError("Prefix cache file size differs from its manifest")
    config = train_config.get_config(MODEL_CONFIG)
    module, load_audit = _load_rlt_only_strict(config, checkpoint_path, prefix_all.shape[1])
    encode_fn = nnx_utils.module_jit(module.encode)
    decode_fn = nnx_utils.module_jit(module.decode)
    z_parts: list[np.ndarray] = []
    mse_all: list[np.ndarray] = []
    mse_masked: list[np.ndarray] = []
    cosine_all: list[np.ndarray] = []
    cosine_masked: list[np.ndarray] = []
    total_sse_all = 0.0
    total_count_all = 0
    total_sse_masked = 0.0
    total_count_masked = 0
    coordinate_sum = np.zeros(prefix_all.shape[1:], dtype=np.float64)
    coordinate_sumsq = np.zeros(prefix_all.shape[1:], dtype=np.float64)
    coordinate_sample_count = 0
    mask_true = mask_count = 0
    for start in range(0, len(selected_indices), args.batch_size):
        real_indices = selected_indices[start : start + args.batch_size]
        batch_indices = real_indices.tolist()
        while len(batch_indices) < args.batch_size:
            batch_indices.append(batch_indices[-1])
        prefix = jnp.asarray(np.asarray(prefix_all[batch_indices]), dtype=jnp.float32)
        tokens = encode_fn(prefix)
        decoded = decode_fn(tokens)
        count = len(real_indices)
        prefix_np = np.asarray(prefix[:count], dtype=np.float32)
        decoded_np = np.asarray(jax.device_get(decoded[:count]), dtype=np.float32)
        token_np = np.asarray(jax.device_get(tokens[:count]), dtype=np.float32).reshape(count, -1)
        mask_np = np.asarray(mask_all[real_indices], dtype=bool)
        difference = decoded_np - prefix_np
        z_parts.append(token_np)
        mse_all.append(np.mean(np.square(difference), axis=(1, 2)))
        per_masked_mse: list[float] = []
        per_masked_cosine: list[float] = []
        target_norm = np.linalg.norm(prefix_np, axis=-1)
        decoded_norm = np.linalg.norm(decoded_np, axis=-1)
        token_cosine = np.divide(
            np.sum(prefix_np * decoded_np, axis=-1),
            target_norm * decoded_norm,
            out=np.zeros_like(target_norm),
            where=(target_norm * decoded_norm) > 1e-12,
        )
        cosine_all.append(token_cosine.mean(axis=1))
        for batch_index in range(count):
            valid = mask_np[batch_index]
            per_masked_mse.append(float(np.mean(np.square(difference[batch_index, valid]))))
            per_masked_cosine.append(float(np.mean(token_cosine[batch_index, valid])))
            masked_target = prefix_np[batch_index, valid]
            masked_diff = difference[batch_index, valid]
            total_sse_masked += float(np.square(masked_diff, dtype=np.float64).sum(dtype=np.float64))
            total_count_masked += masked_target.size
        mse_masked.append(np.asarray(per_masked_mse))
        cosine_masked.append(np.asarray(per_masked_cosine))
        total_sse_all += float(np.square(difference, dtype=np.float64).sum(dtype=np.float64))
        total_count_all += prefix_np.size
        coordinate_sum += prefix_np.sum(axis=0, dtype=np.float64)
        coordinate_sumsq += np.square(prefix_np, dtype=np.float64).sum(axis=0, dtype=np.float64)
        coordinate_sample_count += count
        mask_true += int(mask_np.sum())
        mask_count += mask_np.size
        logging.info("Checkpoint %s encoded %d/%d samples", args.step, start + count, len(selected_indices))
    z = np.concatenate(z_parts)
    mse_all_values = np.concatenate(mse_all)
    mse_masked_values = np.concatenate(mse_masked)
    cosine_all_values = np.concatenate(cosine_all)
    cosine_masked_values = np.concatenate(cosine_masked)
    if mask_true != mask_count:
        raise RuntimeError("Unexpected false entries in image-prefix mask")
    sst_all = float(
        np.maximum(coordinate_sumsq - np.square(coordinate_sum) / coordinate_sample_count, 0.0).sum()
    )
    if not np.isfinite(sst_all) or sst_all <= 1e-12:
        raise RuntimeError(f"Invalid coordinate-centered reconstruction denominator: {sst_all}")
    sst_masked = sst_all
    reconstruction: dict[str, Any] = {
        "mse_all_per_sample": mse_all_values,
        "mse_masked_per_sample": mse_masked_values,
        "cosine_all_per_sample": cosine_all_values,
        "cosine_masked_per_sample": cosine_masked_values,
        "global_mse_all": total_sse_all / total_count_all,
        "global_nmse_all": total_sse_all / sst_all,
        "global_mse_masked": total_sse_masked / total_count_masked,
        "global_nmse_masked": total_sse_masked / sst_masked,
        "mask_true_fraction": mask_true / mask_count,
    }
    metadata = {
        "sample_id": np.asarray([row["sample_id"] for row in rows]),
        "episode_id": np.asarray([row["episode_id"] for row in rows]),
        "pair_id": np.asarray([row["pair_id"] for row in rows]),
        "frame_id": np.asarray([int(row["frame_id"]) for row in rows], dtype=np.int64),
        "frame_order": np.asarray([int(row["frame_order"]) for row in rows], dtype=np.int64),
        "selected_order": np.asarray([int(row["selected_order"]) for row in rows], dtype=np.int64),
        "timestamp": np.asarray([row["timestamp"] for row in rows]),
        "elapsed_seconds": np.asarray([float(row["elapsed_seconds"]) for row in rows], dtype=np.float32),
        "normalized_time": np.asarray([float(row["normalized_time"]) for row in rows], dtype=np.float32),
        "actual_normalized_time": np.asarray(
            [float(row["actual_normalized_time"]) for row in rows], dtype=np.float32
        ),
        "time_bin": np.asarray([int(row["time_bin"]) for row in rows], dtype=np.int8),
        "success": np.asarray([int(row["success"]) for row in rows], dtype=np.int8),
        "seed": np.asarray([int(row["seed"]) for row in rows], dtype=np.int64),
        "instance_id": np.asarray([int(row["instance_id"]) for row in rows], dtype=np.int64),
    }
    metrics, time_rows, pca_values, tsne_values = _compute_metrics(
        z,
        metadata,
        reconstruction,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
    )
    metrics.update(
        {
            "checkpoint": _step_label(args.step),
            "step": args.step,
            "checkpoint_path": str(checkpoint_path),
            "manifest_sha256": prefix_manifest["frames_manifest_sha256"],
            "prefix_content_sha256": prefix_manifest["content_sha256"],
            "smoke_test": bool(args.smoke_episodes_per_outcome),
        }
    )
    nonfinite_metrics = sorted(
        key
        for key, value in metrics.items()
        if isinstance(value, int | float | np.number) and not isinstance(value, bool) and not np.isfinite(value)
    )
    if nonfinite_metrics:
        raise RuntimeError(f"Checkpoint {args.step} produced non-finite metrics: {nonfinite_metrics}")
    if not metrics["finite"] or metrics["representation_collapse"]:
        raise RuntimeError(f"Checkpoint {args.step} produced invalid/collapsed RL-token features")
    if metrics["prefix_mask_true_fraction"] != 1.0:
        raise RuntimeError("Unexpected false entries in image-prefix mask")
    np.savez_compressed(
        target_dir / "features.npz",
        z=z.astype(np.float32),
        **metadata,
        pca=pca_values.astype(np.float32),
        tsne=tsne_values.astype(np.float32),
        reconstruction_mse=mse_all_values.astype(np.float32),
        reconstruction_cosine=cosine_all_values.astype(np.float32),
    )
    _atomic_json(target_dir / "load_audit.json", load_audit)
    _atomic_json(target_dir / "metrics.json", metrics)
    for row in time_rows:
        row["checkpoint"] = _step_label(args.step)
        row["step"] = args.step
    _write_csv(target_dir / "time_bin_metrics.csv", time_rows, list(time_rows[0]))
    _plot_embeddings(
        target_dir / "pca.png",
        pca_values,
        metadata,
        title=f"Checkpoint {_step_label(args.step)} PCA",
        trajectories=True,
    )
    _plot_embeddings(
        target_dir / "tsne.png",
        tsne_values,
        metadata,
        title=f"Checkpoint {_step_label(args.step)} t-SNE",
        trajectories=False,
    )
    _atomic_json(
        target_dir / "completed.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": _step_label(args.step),
            "sample_count": len(z),
            "feature_shape": list(z.shape),
            "finite": True,
        },
    )
    logging.info("Checkpoint %s analysis complete: %s", args.step, target_dir)


def _unbiased_hsic_from_grams(gram_x: np.ndarray, gram_y: np.ndarray) -> float:
    """Unbiased HSIC estimate for symmetric Gram matrices with zero diagonals."""
    if gram_x.shape != gram_y.shape or gram_x.ndim != 2 or gram_x.shape[0] != gram_x.shape[1]:
        raise ValueError(f"HSIC Gram matrices must have equal square shapes, got {gram_x.shape}/{gram_y.shape}")
    n = gram_x.shape[0]
    if n < 4:
        raise ValueError(f"Unbiased HSIC requires at least four samples, got {n}")
    cross = float(np.sum(gram_x * gram_y, dtype=np.float64))
    total_product = float(gram_x.sum(dtype=np.float64) * gram_y.sum(dtype=np.float64))
    row_product = float(np.dot(gram_x.sum(axis=1), gram_y.sum(axis=1)))
    return (cross + total_product / ((n - 1) * (n - 2)) - 2.0 * row_product / (n - 2)) / (n * (n - 3))


def _zero_diagonal_linear_gram(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0, keepdims=True)
    gram = centered @ centered.T
    np.fill_diagonal(gram, 0.0)
    return gram


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError(f"CKA inputs must have the same shape, got {x.shape} and {y.shape}")
    gram_x = _zero_diagonal_linear_gram(x)
    gram_y = _zero_diagonal_linear_gram(y)
    numerator = _unbiased_hsic_from_grams(gram_x, gram_y)
    self_x = _unbiased_hsic_from_grams(gram_x, gram_x)
    self_y = _unbiased_hsic_from_grams(gram_y, gram_y)
    denominator = math.sqrt(self_x * self_y) if self_x > 0.0 and self_y > 0.0 else 0.0
    return numerator / denominator if denominator > 1e-20 else float("nan")


def _distance_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError(f"Distance inputs must have the same shape, got {x.shape} and {y.shape}")
    if len(x) < 3:
        return float("nan")
    return float(spearmanr(pdist(x, metric="euclidean"), pdist(y, metric="euclidean")).statistic)


def _plot_pairwise(matrix: np.ndarray, labels: list[str], output_path: Path, title: str, *, vmin: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] < (vmin + 1.0) / 2 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    steps = [int(row["step"]) / 1000 for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, [row["reconstruction_nmse"] for row in rows], marker="o", color="#295f98")
    axes[0, 0].set_title("Reconstruction NMSE (lower is better)")
    axes[0, 1].plot(steps, [row["success_gap_scaled"] for row in rows], marker="o", color="#9d3f47")
    axes[0, 1].set_title("Episode-balanced success gap (scaled)")
    axes[1, 0].plot(steps, [row["smoothness_scaled"] for row in rows], marker="o", color="#26785f")
    axes[1, 0].set_title("Temporal smoothness distance (scaled)")
    axes[1, 1].plot(steps, [row["time_eta2"] for row in rows], marker="o", color="#82652d")
    axes[1, 1].set_title("Normalized-time sensitivity (eta squared)")
    for axis in axes.flat:
        axis.set_xlabel("checkpoint (k steps)")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def _spearman_summary(rows: list[dict[str, Any]], left: str, right: str) -> tuple[float, float]:
    result = spearmanr([row[left] for row in rows], [row[right] for row in rows])
    return float(result.statistic), float(result.pvalue)


def _holm_adjusted_pvalues(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * float(values[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _write_ranking(rows: list[dict[str, Any]], output_path: Path) -> None:
    by_reconstruction = sorted(rows, key=lambda row: row["reconstruction_nmse"])
    by_gap = sorted(rows, key=lambda row: row["success_gap_scaled"], reverse=True)
    by_auc = sorted(rows, key=lambda row: row["success_cv_auc_mean"], reverse=True)
    by_time = sorted(rows, key=lambda row: row["time_eta2"], reverse=True)
    by_smooth = sorted(rows, key=lambda row: row["smoothness_scaled"])
    by_change = sorted(rows[1:], key=lambda row: row["representation_change_prev"], reverse=True)
    lines = [
        "# Checkpoint 分项对照",
        "",
        "本页不计算综合总分，也不指定单一 checkpoint。各项只能在当前冻结 rollout 分布下解释。",
        "",
        "## 指标相对突出项",
        "",
        f"- 重建 NMSE 较低：{', '.join(row['checkpoint'] for row in by_reconstruction[:3])}",
        f"- episode 等权成败间隔较大：{', '.join(row['checkpoint'] for row in by_gap[:3])}",
        f"- 分组线性探针 AUC 较高：{', '.join(row['checkpoint'] for row in by_auc[:3])}",
        f"- 归一化时间敏感性较高：{', '.join(row['checkpoint'] for row in by_time[:3])}",
        f"- 归一化相邻采样帧距离较低：{', '.join(row['checkpoint'] for row in by_smooth[:3])}",
        f"- 相邻 checkpoint 表示变化较大：{', '.join(row['checkpoint'] for row in by_change[:3])}",
        "",
        "注意：scaled gap 在高维零效应下也有正基线，需结合配对置换和 held-out AUC；相邻采样帧距离较低只有在 "
        "`token_centered_rms` 未坍缩时才可解释为连续；`1-CKA` 是变化幅度，不是质量。",
        "",
        "## 全量数值",
        "",
        "| checkpoint | recon NMSE | recon cosine | success gap | probe AUC | smooth scaled | time eta2 | CKA(prev) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {_fmt(row['reconstruction_nmse'])} | {_fmt(row['cosine'])} | "
            f"{_fmt(row['success_gap_scaled'])} | {_fmt(row['success_cv_auc_mean'])} | "
            f"{_fmt(row['smoothness_scaled'])} | {_fmt(row['time_eta2'])} | {_fmt(row.get('cka_prev'))} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    rows: list[dict[str, Any]],
    time_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    adjacent = list(rows[1:])
    changed = sorted(adjacent, key=lambda row: row["representation_change_prev"], reverse=True)
    recon_gap_rho, _ = _spearman_summary(rows, "reconstruction_nmse", "success_gap_scaled")
    recon_auc_rho, _ = _spearman_summary(rows, "reconstruction_nmse", "success_cv_auc_mean")
    recon_time_rho, _ = _spearman_summary(rows, "reconstruction_nmse", "time_eta2")
    strongest_gap = sorted(rows, key=lambda row: row["success_gap_scaled"], reverse=True)
    supported = [
        row["checkpoint"]
        for row in rows
        if row["success_permutation_p_holm"] <= 0.05 and row["success_cv_auc_mean"] > 0.5
    ]
    per_bin: list[tuple[int, float]] = []
    for time_bin in range(TIME_BINS):
        values = [float(row["success_gap_scaled"]) for row in time_rows if int(row["time_bin"]) == time_bin]
        per_bin.append((time_bin, float(np.nanmean(values))))
    strongest_bin, strongest_bin_gap = max(per_bin, key=lambda item: item[1])
    reconstruction_top = {row["checkpoint"] for row in sorted(rows, key=lambda row: row["reconstruction_nmse"])[:3]}
    gap_bottom = {row["checkpoint"] for row in sorted(rows, key=lambda row: row["success_gap2_debiased"])[:3]}
    time_bottom = {row["checkpoint"] for row in sorted(rows, key=lambda row: row["time_eta2"])[:3]}
    insensitive_bottom = gap_bottom & time_bottom
    possible_tradeoff = sorted(reconstruction_top & insensitive_bottom)
    dataset_audit = json.loads((output_root / "dataset_audit.json").read_text(encoding="utf-8"))
    backbone_audit = json.loads((output_root / "backbone_audit.json").read_text(encoding="utf-8"))
    lines = [
        "# RL Token 多 checkpoint 离线特征分析报告",
        "",
        f"生成时间：{datetime.now(UTC).isoformat()}",
        "",
        "## 实验范围与有效性",
        "",
        f"- 固定数据：{dataset_audit['selected_success_episodes']} 条成功、"
        f"{dataset_audit['selected_failure_episodes']} 条失败 episode，共 {rows[0]['N']} 个 policy-call 样本；"
        f"每条入选 episode 在共享 `j/15` 归一化时间网格取 {rows[0]['N'] // rows[0]['N_episodes']} 帧，"
        "首末 policy call 必含，内部目标映射到最近的实际 call。",
        "- 成功/失败按 GenieSim `instance_id` 一一配对，并限制在两类共同 seed 范围；所有 checkpoint 使用完全相同的 sample ID 与顺序。",
        f"- 原始失败集中有 {dataset_audit['excluded_short_failure_episodes']} 条少于 16 calls 的真实早停轨迹未纳入，因此结论限定于标准长度 episode。",
        f"- `z_rl` 是 RLToken encoder 对三路图像 VLA prefix 的输出，形状为 `[N, {rows[0]['D']}]`；语言和 robot state 不进入此表示。",
        f"- Backbone 审计通过：{len(backbone_audit['checkpoint_steps'])} 个 checkpoint 的完整 VLA schema 及全部 "
        f"{backbone_audit['hashed_element_count']:,} 个参数值 bitwise 一致。",
        "- 每个 RLT checkpoint 均按原始 FP32 参数严格恢复，missing/unexpected/shape/dtype mismatch 均为 0。",
        "- 每个 checkpoint 独立拟合 PCA 和 t-SNE；二维坐标不用于跨 checkpoint 排名。",
        "",
        "## 核心结果",
        "",
        "### 1. 哪些 checkpoint 的表示变化幅度最大？",
        "",
        f"按同一样本上的 unbiased centered linear CKA，观察到变化幅度最大的相邻转移是 "
        f"{changed[0]['previous_checkpoint']} -> {changed[0]['checkpoint']} "
        f"(CKA={_fmt(changed[0]['cka_prev'])}, 1-CKA={_fmt(changed[0]['representation_change_prev'])})；"
        f"其次是 {changed[1]['previous_checkpoint']} -> {changed[1]['checkpoint']} "
        f"(CKA={_fmt(changed[1]['cka_prev'])})。这些值对正交旋转和全局尺度较稳健，但只表示变化幅度，不表示质量提升。",
        "",
        "### 2. Reconstruction 提升是否对应 representation 更好？",
        "",
        f"{len(rows)} 个点上，重建 NMSE 与 episode 等权成败间隔的 Spearman rho={_fmt(recon_gap_rho)}，"
        f"与分组线性探针 AUC 的 rho={_fmt(recon_auc_rho)}，"
        f"与时间敏感性 eta2 的 rho={_fmt(recon_time_rho)}。"
        "checkpoint 来自同一条有序训练轨迹，并非独立样本，因此这里只报告描述性相关，不做常规 p 值推断或因果解释。",
        "",
        "### 3. 成功/失败是否在 token 空间出现差异？",
        "",
        f"成败间隔最大的三个 checkpoint 为 {', '.join(row['checkpoint'] for row in strongest_gap[:3])}；"
        f"其 scaled gap 分别为 {', '.join(_fmt(row['success_gap_scaled']) for row in strongest_gap[:3])}。"
        f"按同 instance 配对的 episode 级 sign-flip permutation，并对 {len(rows)} 个 checkpoint 做 Holm 校正后，"
        f"同时满足 adjusted p<=0.05 与 held-out AUC>0.5 的 checkpoint 为 "
        f"{', '.join(supported) if supported else '无'}。分组线性探针 AUC 范围为 "
        f"{_fmt(min(row['success_cv_auc_mean'] for row in rows))} 到 {_fmt(max(row['success_cv_auc_mean'] for row in rows))}。",
        "",
        (
            f"因此，在当前冻结 rollout 分布下，{', '.join(supported)} 的 RL token 有统计证据包含可泛化的成败相关信息；"
            "这不证明 RL 策略一定有效。"
            if supported
            else "因此，当前样本未检测到同时经多重校正和 held-out 探针支持的稳定成败差异，不能声称 RL token 已编码可泛化的成败信息。"
        ),
        "scaled centroid gap 在 D 远大于 episode 数时即使零效应也有正基线，故上面的 top-3 仅作描述，证据判断不依赖其绝对大小。",
        "",
        "### 4. 差异发生在哪个任务阶段？",
        "",
        "数据没有 grasp/transport/align/release/retract 标签，`task_progress` 也不足以重建这些动作阶段，因此不做语义阶段声明。"
        f"五个等长归一化时间窗中，跨 checkpoint 平均 scaled gap 最大的是 T{strongest_bin + 1} "
        f"([{strongest_bin / TIME_BINS:.1f}, {(strongest_bin + 1) / TIME_BINS:.1f}])，均值 {_fmt(strongest_bin_gap)}。"
        "这只定位到轨迹相对时间，不等价于某个动作语义。",
        "",
        "### 5. 是否有 reconstruction 较好但任务状态不敏感的 checkpoint？",
        "",
        (
            f"按“重建 NMSE 前三”与“debiased 成败间隔后三且时间敏感性后三”的严格交集，候选为 {', '.join(possible_tradeoff)}。"
            if possible_tradeoff
            else "按“重建 NMSE 前三”与“debiased 成败间隔后三且时间敏感性后三”的严格交集，本次没有同时满足三项的 checkpoint。"
        ),
        "该筛选只是显式的 trade-off 检查，不构成 checkpoint 选择结论。",
        "",
        "## Checkpoint 数值表",
        "",
        "| ckpt | PCA var1/2 | success gap scaled | probe AUC | smooth scaled | recon NMSE | recon cosine | time eta2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {_fmt(row['PCA_var1'])}/{_fmt(row['PCA_var2'])} | "
            f"{_fmt(row['success_gap_scaled'])} | {_fmt(row['success_cv_auc_mean'])} | "
            f"{_fmt(row['smoothness_scaled'])} | {_fmt(row['reconstruction_nmse'])} | "
            f"{_fmt(row['cosine'])} | {_fmt(row['time_eta2'])} |"
        )
    lines.extend(
        [
            "",
            "## 限制与后续验证",
            "",
            "- t-SNE 仅作局部结构辅助观察，不据此判断 checkpoint 质量。",
            "- rollout 是观测数据，success/failure 仍可能携带未完全控制的场景和轨迹长度差异。",
            "- 时间平滑度是相邻均匀抽样帧之间的距离，不等同于每个连续 policy call；较低值也可能来自表示坍缩。",
            "- 归一化时间窗的最大 gap 是探索性定位，未对时间窗选择做额外多重校正，也不对应动作语义。",
            "- 研究者仍需结合 Actor-Critic 指标、独立 rollout success rate 和 downstream RL 验证后续输入选择。",
            "",
            "详细中间结果见 `checkpoint_summary.csv`、`checkpoint_pairwise.csv`、`checkpoint_timebins.csv` 及各 `ckpt_*` 目录。",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_analysis(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    sample_ids: np.ndarray | None = None
    time_rows: list[dict[str, Any]] = []
    backbone_audit = json.loads((output_root / "backbone_audit.json").read_text(encoding="utf-8"))
    if not backbone_audit.get("consistent"):
        raise RuntimeError("Frozen backbone audit did not pass")
    reference_rlt_schema: str | None = None
    reference_manifest_sha256: str | None = None
    reference_prefix_sha256: str | None = None
    for step in args.steps:
        checkpoint_dir = output_root / f"ckpt_{_step_label(step)}"
        if not (checkpoint_dir / "completed.json").is_file():
            raise FileNotFoundError(f"Checkpoint analysis is incomplete: {checkpoint_dir}")
        metrics = json.loads((checkpoint_dir / "metrics.json").read_text(encoding="utf-8"))
        load_audit = json.loads((checkpoint_dir / "load_audit.json").read_text(encoding="utf-8"))
        schema = load_audit["rlt_fingerprint"]["schema_sha256"]
        reference_rlt_schema = reference_rlt_schema or schema
        if schema != reference_rlt_schema:
            raise RuntimeError(f"RLT schemas differ at checkpoint {step}")
        with np.load(checkpoint_dir / "features.npz", allow_pickle=False) as payload:
            current_ids = payload["sample_id"]
            current_z = payload["z"].astype(np.float64)
        if sample_ids is None:
            sample_ids = current_ids
        elif not np.array_equal(sample_ids, current_ids):
            raise RuntimeError(f"Sample IDs/order differ at checkpoint {step}")
        if not np.isfinite(current_z).all():
            raise RuntimeError(f"Non-finite feature values at checkpoint {step}")
        reference_manifest_sha256 = reference_manifest_sha256 or metrics["manifest_sha256"]
        reference_prefix_sha256 = reference_prefix_sha256 or metrics["prefix_content_sha256"]
        if metrics["manifest_sha256"] != reference_manifest_sha256:
            raise RuntimeError(f"Frame manifest hash differs at checkpoint {step}")
        if metrics["prefix_content_sha256"] != reference_prefix_sha256:
            raise RuntimeError(f"Prefix content hash differs at checkpoint {step}")
        rows.append(metrics)
        features.append(current_z)
        time_rows.extend(_read_csv(checkpoint_dir / "time_bin_metrics.csv"))
    assert sample_ids is not None
    if len(sample_ids) > args.cka_max_samples:
        comparison_indices = np.asarray(_uniform_indices(len(sample_ids), args.cka_max_samples), dtype=np.int64)
    else:
        comparison_indices = np.arange(len(sample_ids))
    labels = [row["checkpoint"] for row in rows]
    comparison_features = [feature[comparison_indices] for feature in features]
    grams = [_zero_diagonal_linear_gram(feature) for feature in comparison_features]
    self_hsic = [_unbiased_hsic_from_grams(gram, gram) for gram in grams]
    pairwise_distances = [pdist(feature, metric="euclidean") for feature in comparison_features]
    cka_matrix = np.eye(len(rows), dtype=np.float64)
    distance_matrix = np.eye(len(rows), dtype=np.float64)
    pair_rows: list[dict[str, Any]] = []
    for left in range(len(rows)):
        for right in range(left, len(rows)):
            denominator = (
                math.sqrt(self_hsic[left] * self_hsic[right])
                if self_hsic[left] > 0.0 and self_hsic[right] > 0.0
                else 0.0
            )
            cka = (
                _unbiased_hsic_from_grams(grams[left], grams[right]) / denominator
                if denominator > 1e-20
                else float("nan")
            )
            distance_rho = float(spearmanr(pairwise_distances[left], pairwise_distances[right]).statistic)
            cka_matrix[left, right] = cka_matrix[right, left] = cka
            distance_matrix[left, right] = distance_matrix[right, left] = distance_rho
            pair_rows.append(
                {
                    "checkpoint_a": labels[left],
                    "checkpoint_b": labels[right],
                    "step_a": rows[left]["step"],
                    "step_b": rows[right]["step"],
                    "linear_cka": cka,
                    "representation_dissimilarity": 1.0 - cka,
                    "distance_spearman": distance_rho,
                    "sample_count": len(comparison_indices),
                }
            )
    if not np.isfinite(cka_matrix).all() or not np.isfinite(distance_matrix).all():
        raise RuntimeError("Cross-checkpoint comparison produced non-finite values")
    adjusted_pvalues = _holm_adjusted_pvalues([float(row["success_permutation_p"]) for row in rows])
    for row, adjusted_pvalue in zip(rows, adjusted_pvalues, strict=True):
        row["success_permutation_p_holm"] = adjusted_pvalue
    for index, row in enumerate(rows):
        row["cka_10k"] = float(cka_matrix[0, index])
        row["distance_spearman_10k"] = float(distance_matrix[0, index])
        if index == 0:
            row["previous_checkpoint"] = ""
            row["cka_prev"] = ""
            row["distance_spearman_prev"] = ""
            row["representation_change_prev"] = ""
        else:
            row["previous_checkpoint"] = rows[index - 1]["checkpoint"]
            row["cka_prev"] = float(cka_matrix[index - 1, index])
            row["distance_spearman_prev"] = float(distance_matrix[index - 1, index])
            row["representation_change_prev"] = 1.0 - row["cka_prev"]
    summary_fields = [
        "checkpoint",
        "step",
        "D",
        "N",
        "N_episodes",
        "N_success_episodes",
        "N_failure_episodes",
        "PCA_var1",
        "PCA_var2",
        "PCA_var12",
        "PCA_effective_rank_50",
        "success_gap",
        "success_gap_scaled",
        "success_gap2_debiased",
        "success_gap_scaled_ci_low",
        "success_gap_scaled_ci_high",
        "success_permutation_p",
        "success_permutation_p_holm",
        "success_cv_auc_mean",
        "success_cv_auc_std",
        "smoothness",
        "smoothness_std",
        "smoothness_scaled",
        "temporal_cosine_mean",
        "continuity_ratio",
        "time_eta2",
        "reconstruction",
        "reconstruction_std",
        "reconstruction_global_mse",
        "reconstruction_nmse",
        "reconstruction_r2",
        "cosine",
        "cosine_std",
        "cosine_p10",
        "cosine_p50",
        "cosine_p90",
        "masked_reconstruction",
        "masked_cosine",
        "token_norm_mean",
        "token_norm_std",
        "token_norm_p10",
        "token_norm_p50",
        "token_norm_p90",
        "token_norm_cv",
        "token_centered_rms",
        "token_rms_per_dim_std",
        "finite",
        "representation_collapse",
        "previous_checkpoint",
        "cka_prev",
        "distance_spearman_prev",
        "representation_change_prev",
        "cka_10k",
        "distance_spearman_10k",
        "manifest_sha256",
        "prefix_content_sha256",
        "checkpoint_path",
    ]
    _write_csv(output_root / "checkpoint_summary.csv", rows, summary_fields)
    time_fields = list(time_rows[0])
    _write_csv(output_root / "checkpoint_timebins.csv", time_rows, time_fields)
    _write_csv(output_root / "checkpoint_pairwise.csv", pair_rows, list(pair_rows[0]))
    _plot_pairwise(
        cka_matrix,
        labels,
        output_root / "checkpoint_cka.png",
        "Unbiased centered linear CKA across checkpoints",
        vmin=max(-1.0, float(np.nanmin(cka_matrix)) - 0.03),
    )
    _plot_pairwise(
        distance_matrix,
        labels,
        output_root / "checkpoint_distance_spearman.png",
        "Pairwise-distance Spearman across checkpoints",
        vmin=max(-1.0, float(np.nanmin(distance_matrix)) - 0.03),
    )
    _plot_summary(rows, output_root / "checkpoint_metrics.png")
    _write_ranking(rows, output_root / "checkpoint_ranking.md")
    _write_report(rows, time_rows, pair_rows, output_root)
    _atomic_json(
        output_root / "analysis_completed.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "checkpoints": labels,
            "sample_count": len(sample_ids),
            "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids.tolist()).encode()).hexdigest(),
            "backbone_consistent": True,
            "all_features_finite": True,
            "outputs": [
                "checkpoint_summary.csv",
                "checkpoint_pairwise.csv",
                "checkpoint_timebins.csv",
                "checkpoint_ranking.md",
                "report.md",
            ],
        },
    )
    logging.info("Cross-checkpoint analysis complete: %s", output_root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create the immutable episode/frame manifests")
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    prepare.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    prepare.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    prepare.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    prepare.add_argument("--episodes-per-outcome", type=int, default=20)
    prepare.add_argument("--frames-per-episode", type=int, default=16)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--bootstrap-replicates", type=int, default=500)
    prepare.add_argument("--permutation-replicates", type=int, default=2000)
    prepare.set_defaults(func=prepare_manifests)

    prefix = subparsers.add_parser("cache-prefix", help="Extract the frozen VLA image prefix once")
    prefix.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prefix.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    prefix.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    prefix.add_argument("--reference-step", type=int, default=10_000)
    prefix.add_argument("--batch-size", type=int, default=8)
    prefix.set_defaults(func=cache_prefixes)

    checkpoint = subparsers.add_parser("checkpoint", help="Analyze one RLT checkpoint from the fixed prefix cache")
    checkpoint.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    checkpoint.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    checkpoint.add_argument("--step", type=int, required=True)
    checkpoint.add_argument("--batch-size", type=int, default=8)
    checkpoint.add_argument("--bootstrap-replicates", type=int, default=500)
    checkpoint.add_argument("--permutation-replicates", type=int, default=2000)
    checkpoint.add_argument("--smoke-episodes-per-outcome", type=int, default=0)
    checkpoint.add_argument("--target-dir", type=Path, default=None)
    checkpoint.set_defaults(func=analyze_checkpoint)

    finalize = subparsers.add_parser("finalize", help="Compare all checkpoint outputs and write the Chinese report")
    finalize.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    finalize.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    finalize.add_argument("--cka-max-samples", type=int, default=512)
    finalize.set_defaults(func=finalize_analysis)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    args.func(args)


if __name__ == "__main__":
    main()
