#!/usr/bin/env python3
"""Offline Stage-2 RL-token quality checks for recorded GenieSim rollouts.

The script never trains or deploys an Actor/Critic.  Reconstruction controls keep
the frozen decoder and VLA prefix fixed and replace only the observation token.
Geometry and event probes are explicitly reported as unavailable when the rollout
archive has no simulator truth.
"""

# ruff: noqa: B023, E402, PERF401, RUF001, SLF001

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from collections.abc import Iterable
import csv
import datetime as dt
import hashlib
from itertools import pairwise
import json
import logging
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import openpi.models.model as model_lib
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as train_config
from openpi.training.geniesim_rollout_dataset import load_cache_index
from openpi.training.geniesim_rollout_dataset import load_rollout_sample
import openpi.transforms as transforms
from scripts.geniesim import analyze_rl_tokens as stage1

ROLLOUT_ROOT = Path("/mnt/pfs/kk/kk/data/data/geniesim/rollout/stack_three_blocks")
CACHE_ROOT = Path("/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/cache_224")
PLAN_ROOT = Path("/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan")
CHECKPOINT_ROOT = PLAN_ROOT / "checkpoints/rlt_pi05_geniesim_stack_three_blocks_plan/stage1_rl_token_20k"
DEFAULT_OUTPUT = PLAN_ROOT / "rlt_token_quality_validation/stage2_full_20260923"
MODEL_CONFIG = "rlt_pi05_geniesim_stack_three_blocks_plan"
ROLLOUT_CONFIG = "rlt_pi05_geniesim_stack_three_blocks"
STEPS = (10_000, 170_000)
PERM_SEEDS = (17, 29, 43)
TIME_BINS = 5
BOOTSTRAP_REPS = 1000
STACK_EVENT = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3}),\d{3}.*Action \[Stack\] evt: (?P<event>[34])$"
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _episode_id(category: str, trajectory_id: int) -> str:
    return f"{category}/trajectory_{trajectory_id:06d}"


def _deranged_sources(
    indices: np.ndarray, episode_labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return a one-to-one source assignment with no same-episode matches."""
    label_groups = [indices[episode_labels[indices] == label].copy() for label in np.unique(episode_labels[indices])]
    if len(label_groups) < 2:
        raise RuntimeError("derangement stratum has fewer than two episodes")
    rng.shuffle(label_groups)
    for group in label_groups:
        rng.shuffle(group)
    targets = np.concatenate(label_groups)
    ordered_labels = episode_labels[targets]
    valid_shifts = np.asarray(
        [shift for shift in range(1, len(targets)) if np.all(ordered_labels != np.roll(ordered_labels, -shift))],
        dtype=np.int64,
    )
    if not len(valid_shifts):
        counts = Counter(ordered_labels.tolist())
        raise RuntimeError(f"no cross-episode derangement exists for stratum counts={dict(counts)}")
    shift = int(rng.choice(valid_shifts))
    return targets, np.roll(targets, -shift)


def _episode_catalog(rollout_root: Path, cache_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load every archived episode and every policy-call record."""
    index = json.loads((cache_root / "index.json").read_text(encoding="utf-8"))
    records = index["records"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[_episode_id(item["category"], int(item["trajectory_id"]))].append(item)

    episodes: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for episode_id in sorted(grouped):
        category, tail = episode_id.split("/", 1)
        trajectory_id = int(tail.rsplit("_", 1)[1])
        trajectory_dir = rollout_root / category / tail
        metadata = json.loads((trajectory_dir / "metadata.json").read_text(encoding="utf-8"))
        timestamps = stage1._parse_call_timestamps(trajectory_dir / "rollout.log")
        episode_items = sorted(grouped[episode_id], key=lambda item: int(item["step_id"]))
        if len(timestamps) != len(episode_items):
            raise RuntimeError(f"timestamp/record mismatch for {episode_id}: {len(timestamps)} vs {len(episode_items)}")
        success = int(category == "successful")
        instance_id = int(metadata["instance_id"])
        seed = int(metadata["seed"])
        episodes.append(
            {
                "episode_id": episode_id,
                "category": category,
                "trajectory_id": trajectory_id,
                "success": success,
                "failure": int(not success),
                "seed": seed,
                "instance_id": instance_id,
                "attempt_id": metadata.get("attempt_id", ""),
                "num_calls": len(episode_items),
                "short_early_stop": int(len(episode_items) < 32 and not success),
                "below_16_calls": int(len(episode_items) < 16 and not success),
                "metadata_path": str(trajectory_dir / "metadata.json"),
                "success_criterion": metadata.get("success_criterion", ""),
            }
        )
        first = timestamps[0]
        for local_index, (item, timestamp) in enumerate(zip(episode_items, timestamps, strict=True)):
            source = Path(item["source_path"])
            raw = np.load(source, allow_pickle=True).item()
            progress = raw.get("inputs/task_progress", [])
            state = np.asarray(raw["inputs/state"], dtype=np.float32)
            actions = np.asarray(raw["outputs/actions"], dtype=np.float32)
            if state.shape != (32,) or actions.shape != (50, 16):
                raise RuntimeError(f"unexpected shapes in {source}: {state.shape}, {actions.shape}")
            normalized = local_index / max(len(episode_items) - 1, 1)
            calls.append(
                {
                    "row_index": len(calls),
                    "sample_id": f"{episode_id}/call_{local_index:03d}_step_{int(item['step_id'])}",
                    "episode_id": episode_id,
                    "category": category,
                    "trajectory_id": trajectory_id,
                    "call_index": local_index,
                    "record_step": int(item["step_id"]),
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "elapsed_seconds": (timestamp - first).total_seconds(),
                    "normalized_time": normalized,
                    "time_bin": min(int(normalized * TIME_BINS), TIME_BINS - 1),
                    "success": success,
                    "seed": seed,
                    "instance_id": instance_id,
                    "episode_done": int(bool(raw.get("inputs/episode_done", False))),
                    "task_name": str(raw.get("inputs/task_name", "")),
                    "prompt": str(raw.get("inputs/prompt", "")),
                    "task_progress": _json(progress),
                    "task_progress_available": int(bool(progress)),
                    "source_path": str(source),
                    "cache_path": str(cache_root / item["cache_path"]),
                    "state": state,
                    "actions": actions,
                }
            )
    if len(calls) != len(records):
        raise RuntimeError(f"call count mismatch: {len(calls)} vs {len(records)}")
    return episodes, calls


def _write_manifests(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    episodes, calls = _episode_catalog(args.rollout_root.resolve(), args.cache_root.resolve())
    call_arrays = out / "call_inputs.npz"
    np.savez_compressed(
        call_arrays,
        state=np.stack([row["state"] for row in calls]).astype(np.float32),
        actions=np.stack([row["actions"] for row in calls]).astype(np.float32),
    )
    episode_fields = [
        "episode_id",
        "category",
        "trajectory_id",
        "success",
        "failure",
        "seed",
        "instance_id",
        "attempt_id",
        "num_calls",
        "short_early_stop",
        "below_16_calls",
        "metadata_path",
        "success_criterion",
    ]
    call_fields = [
        "row_index",
        "sample_id",
        "episode_id",
        "category",
        "trajectory_id",
        "call_index",
        "record_step",
        "timestamp",
        "elapsed_seconds",
        "normalized_time",
        "time_bin",
        "success",
        "seed",
        "instance_id",
        "episode_done",
        "task_name",
        "prompt",
        "task_progress",
        "task_progress_available",
        "source_path",
        "cache_path",
    ]
    _write_csv(out / "episodes.csv", episodes, episode_fields)
    _write_csv(out / "calls.csv", calls, call_fields)

    event_rows = []
    for episode in episodes:
        for event_name, target in (
            ("alignment_geometry", "current_xy_z_geometry"),
            ("retract_collision", "retract_gap_or_contact"),
        ):
            event_rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "event_name": event_name,
                    "target": target,
                    "availability": "NA",
                    "event_tick": "",
                    "pre_event_call_sample_id": "",
                    "label": "",
                    "reason": "rollout archive has no object pose, contact, collision, or simulator tick state",
                }
            )
        episode_calls = [row for row in calls if row["episode_id"] == episode["episode_id"]]
        trajectory_dir = (
            args.rollout_root.resolve() / episode["category"] / f"trajectory_{int(episode['trajectory_id']):06d}"
        )
        terminal_events = []
        for line in (trajectory_dir / "rollout.log").read_text(encoding="utf-8", errors="replace").splitlines():
            match = STACK_EVENT.match(line)
            if match:
                timestamp = dt.datetime.strptime(
                    f"{match.group('date')}.{match.group('ms')}", "%Y-%m-%d %H:%M:%S.%f"
                ).replace(tzinfo=dt.UTC)
                terminal_events.append((timestamp, int(match.group("event"))))
        expected_event = 3 if int(episode["success"]) else 4
        event_matches = [item for item in terminal_events if item[1] == expected_event]
        if len(event_matches) != 1:
            raise RuntimeError(
                f"expected exactly one Stack evt={expected_event} in {trajectory_dir}, got {event_matches}"
            )
        event_timestamp, event_code = event_matches[0]
        preceding = [
            row
            for row in episode_calls
            if dt.datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")) <= event_timestamp
        ]
        pre_event = preceding[-1] if preceding else None
        lag_seconds = (
            (event_timestamp - dt.datetime.fromisoformat(pre_event["timestamp"].replace("Z", "+00:00"))).total_seconds()
            if pre_event
            else ""
        )
        within_nominal_window = bool(pre_event is not None and 0.0 <= float(lag_seconds) <= 50 / 30)
        event_rows.append(
            {
                "episode_id": episode["episode_id"],
                "event_name": "terminal_stack_evaluator_event",
                "target": "final_benchmark_success",
                "availability": "observed_between_policy_calls",
                "event_timestamp": event_timestamp.isoformat().replace("+00:00", "Z"),
                "event_code": event_code,
                "event_tick": "",
                "pre_event_call_sample_id": pre_event["sample_id"] if pre_event else "",
                "event_lag_seconds": lag_seconds,
                "within_nominal_50_step_window_inferred": int(within_nominal_window),
                "label": episode["success"],
                "reason": (
                    "terminal Stack evaluator event; nominal-window membership inferred only from wall-clock lag, not "
                    "an observed simulator tick; not geometry/contact truth or a pre-action risk label"
                ),
            }
        )
    _write_csv(
        out / "events.csv",
        event_rows,
        [
            "episode_id",
            "event_name",
            "target",
            "availability",
            "event_timestamp",
            "event_code",
            "event_tick",
            "pre_event_call_sample_id",
            "event_lag_seconds",
            "within_nominal_50_step_window_inferred",
            "label",
            "reason",
        ],
    )

    instance_ids = sorted({int(row["instance_id"]) for row in episodes})
    fold_by_instance = {instance: index % 5 for index, instance in enumerate(instance_ids)}
    split_rows = [
        {
            "episode_id": row["episode_id"],
            "instance_id": row["instance_id"],
            "seed": row["seed"],
            "success": row["success"],
            "group": f"instance_{int(row['instance_id']):02d}",
            "fold": fold_by_instance[int(row["instance_id"])],
            "split_rule": "deterministic instance_id modulo 5; frozen before probe fitting",
        }
        for row in episodes
    ]
    _write_csv(out / "splits.csv", split_rows, list(split_rows[0]))

    episode_by_id = {row["episode_id"]: row for row in episodes}
    bins_by_episode: dict[str, set[int]] = defaultdict(set)
    for row in calls:
        bins_by_episode[row["episode_id"]].add(int(row["time_bin"]))
    split_summary_rows = []
    for fold in range(5):
        fold_episodes = [row for row in split_rows if int(row["fold"]) == fold]
        success_episodes = sum(int(row["success"]) for row in fold_episodes)
        failure_episodes = len(fold_episodes) - success_episodes
        success_units = sum(
            len(bins_by_episode[row["episode_id"]])
            for row in fold_episodes
            if int(episode_by_id[row["episode_id"]]["success"]) == 1
        )
        failure_units = sum(
            len(bins_by_episode[row["episode_id"]])
            for row in fold_episodes
            if int(episode_by_id[row["episode_id"]]["success"]) == 0
        )
        if not success_episodes or not failure_episodes:
            raise RuntimeError(f"fold {fold} does not contain both outcomes")
        split_summary_rows.append(
            {
                "fold": fold,
                "instance_groups": len({int(row["instance_id"]) for row in fold_episodes}),
                "episodes": len(fold_episodes),
                "success_episodes": success_episodes,
                "failure_episodes": failure_episodes,
                "binned_probe_units": success_units + failure_units,
                "positive_probe_units": success_units,
                "negative_probe_units": failure_units,
            }
        )
    _write_csv(out / "split_summary.csv", split_summary_rows, list(split_summary_rows[0]))

    outcome_counts = defaultdict(lambda: {0: 0, 1: 0})
    for row in episodes:
        outcome_counts[int(row["instance_id"])][int(row["success"])] += 1
    counts = {
        "episodes": len(episodes),
        "success_episodes": sum(int(row["success"]) for row in episodes),
        "failure_episodes": sum(int(row["failure"]) for row in episodes),
        "calls": len(calls),
        "short_failure_episodes": sum(int(row["short_early_stop"]) for row in episodes),
        "failure_episodes_below_16_calls": sum(int(row["below_16_calls"]) for row in episodes),
        "instance_groups": len(instance_ids),
        "instances_with_both_outcomes": sum(int(v[0] > 0 and v[1] > 0) for v in outcome_counts.values()),
        "task_progress_nonempty_calls": sum(int(row["task_progress_available"]) for row in calls),
        "object_geometry_available": False,
        "event_truth_available": False,
        "cache_inputs_sha256": _sha256(call_arrays),
        "folds": split_summary_rows,
    }
    interval_by_outcome: dict[str, list[float]] = {"successful": [], "failed": []}
    for episode in episodes:
        episode_calls = [row for row in calls if row["episode_id"] == episode["episode_id"]]
        times = [dt.datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")) for row in episode_calls]
        interval_by_outcome[episode["category"]].extend(
            (right - left).total_seconds() for left, right in pairwise(times)
        )
    counts["call_interval_seconds"] = {
        category: {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.1)),
            "p90": float(np.quantile(values, 0.9)),
        }
        for category, values in interval_by_outcome.items()
    }
    first_calls = [row for row in calls if int(row["call_index"]) == 0]
    first_images: dict[str, np.ndarray] = {}
    for row in first_calls:
        with np.load(row["cache_path"], allow_pickle=False) as payload:
            first_images[row["episode_id"]] = np.asarray(payload["top_head"], dtype=np.int16)
    same_instance_image: list[float] = []
    different_instance_image: list[float] = []
    same_instance_state: list[float] = []
    different_instance_state: list[float] = []
    successful_first = [row for row in first_calls if int(row["success"]) == 1]
    failed_first = [row for row in first_calls if int(row["success"]) == 0]
    for success_row in successful_first:
        for failure_row in failed_first:
            image_mae = float(
                np.mean(np.abs(first_images[success_row["episode_id"]] - first_images[failure_row["episode_id"]]))
            )
            state_l2 = float(np.linalg.norm(success_row["state"][:16] - failure_row["state"][:16]))
            if int(success_row["instance_id"]) == int(failure_row["instance_id"]):
                same_instance_image.append(image_mae)
                same_instance_state.append(state_l2)
            else:
                different_instance_image.append(image_mae)
                different_instance_state.append(state_l2)
    scene_audit = {
        "definition": "cross-outcome first-call comparison using cached 224x224 top-head image and raw first-16 robot state",
        "same_instance_pairs": len(same_instance_image),
        "different_instance_pairs": len(different_instance_image),
        "same_instance_top_head_mae_mean": float(np.mean(same_instance_image)),
        "same_instance_top_head_mae_median": float(np.median(same_instance_image)),
        "different_instance_top_head_mae_mean": float(np.mean(different_instance_image)),
        "different_instance_top_head_mae_median": float(np.median(different_instance_image)),
        "different_to_same_image_mae_ratio": float(np.mean(different_instance_image) / np.mean(same_instance_image)),
        "same_instance_state_l2_median": float(np.median(same_instance_state)),
        "different_instance_state_l2_median": float(np.median(different_instance_state)),
        "interpretation": "supports instance_id as an initial-layout grouping key; it does not imply the seeds are identical",
    }
    _atomic_json(out / "scene_group_audit.json", scene_audit)
    counts["scene_group_audit"] = scene_audit
    _atomic_json(out / "manifest_counts.json", counts)

    n = len(calls)
    episode_ids = np.asarray([row["episode_id"] for row in calls], dtype=object)
    bins = np.asarray([int(row["time_bin"]) for row in calls])
    call_instances = np.asarray([int(row["instance_id"]) for row in calls], dtype=np.int64)
    within = np.full((n, len(PERM_SEEDS)), -1, dtype=np.int64)
    global_map = np.full((n, len(PERM_SEEDS)), -1, dtype=np.int64)
    excluded_strata: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(PERM_SEEDS):
        rng = np.random.default_rng(seed)
        for instance_id in np.unique(call_instances):
            for time_bin in range(TIME_BINS):
                stratum = np.flatnonzero((call_instances == instance_id) & (bins == time_bin))
                if not len(stratum):
                    continue
                stratum_counts = Counter(episode_ids[stratum].tolist())
                largest_episode = max(stratum_counts.values())
                if largest_episode > len(stratum) - largest_episode:
                    if seed_index == 0:
                        excluded_strata.append(
                            {
                                "instance_id": int(instance_id),
                                "time_bin": time_bin,
                                "call_count": len(stratum),
                                "episode_call_counts": dict(stratum_counts),
                                "reason": "no one-to-one cross-episode derangement exists",
                            }
                        )
                    continue
                targets, sources = _deranged_sources(stratum, episode_ids, rng)
                within[targets, seed_index] = sources
        targets, sources = _deranged_sources(np.arange(n, dtype=np.int64), episode_ids, rng)
        global_map[targets, seed_index] = sources
        valid_within = within[:, seed_index] >= 0
        if np.any(global_map[:, seed_index] < 0):
            raise RuntimeError(f"incomplete global control map for seed {seed}")
        if not np.array_equal(np.sort(within[valid_within, seed_index]), np.flatnonzero(valid_within)):
            raise RuntimeError(f"within control is not a one-to-one permutation for seed {seed}")
        if not np.array_equal(np.sort(global_map[:, seed_index]), np.arange(n)):
            raise RuntimeError(f"global control is not a one-to-one permutation for seed {seed}")
        if np.any(episode_ids[within[valid_within, seed_index]] == episode_ids[valid_within]):
            raise RuntimeError(f"within control contains a same-episode pair for seed {seed}")
        if np.any(call_instances[within[valid_within, seed_index]] != call_instances[valid_within]) or np.any(
            bins[within[valid_within, seed_index]] != bins[valid_within]
        ):
            raise RuntimeError(f"within control violates instance/time matching for seed {seed}")
        if np.any(episode_ids[global_map[:, seed_index]] == episode_ids):
            raise RuntimeError(f"global control contains a same-episode pair for seed {seed}")
    np.savez_compressed(
        out / "control_pair_maps.npz", within=within, global_map=global_map, seeds=np.asarray(PERM_SEEDS)
    )
    control_map_audit = {
        "within_definition": "one-to-one cross-episode derangement within each instance_id/time_bin stratum",
        "within_valid_calls_per_seed": int(np.sum(within[:, 0] >= 0)),
        "within_excluded_calls_per_seed": int(np.sum(within[:, 0] < 0)),
        "within_excluded_strata": excluded_strata,
        "within_sources_unique_per_seed": [
            len(np.unique(within[within[:, i] >= 0, i])) for i in range(len(PERM_SEEDS))
        ],
        "global_definition": "one-to-one cross-episode derangement over all calls",
        "global_valid_calls_per_seed": n,
        "global_sources_unique_per_seed": [len(np.unique(global_map[:, i])) for i in range(len(PERM_SEEDS))],
    }
    _atomic_json(out / "control_pair_audit.json", control_map_audit)

    checkpoints = {str(step): str((args.checkpoint_root / str(step)).resolve()) for step in STEPS}
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        git_worktree_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
        git_worktree_dirty = None
    manifest = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "scope": "Stage-1 RL-token quality validation; no Actor/Critic training or rollout",
        "rollout_root": str(args.rollout_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "checkpoints": checkpoints,
        "base_vla": "/mnt/pfs/kk/kk/ckpt/geniesim3/spatialpi05",
        "model_config": MODEL_CONFIG,
        "rollout_config": ROLLOUT_CONFIG,
        "code": {
            "git_commit": git_commit,
            "git_worktree_dirty": git_worktree_dirty,
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "observation_preprocessing": {
            "camera_keys": ["top_head", "hand_left", "hand_right"],
            "cached_image_shape": [224, 224, 3],
            "cached_image_dtype": "uint8",
            "cache_resize": "openpi.shared.image_tools.resize_with_pad(..., 224, 224)",
            "task_names": sorted({row["task_name"] for row in calls}),
            "prompts": sorted({row["prompt"] for row in calls}),
            "input_transform": (
                "rlt_pi05_geniesim_stack_three_blocks repack/data transforms; quantile normalization from "
                "configured state/action norm_stats; model resize/tokenization/padding"
            ),
            "state_shape": [32],
            "recorded_ref_chunk_shape": [50, 16],
            "prefix_position_blocks": (
                "positions 0:256 base_0_rgb/top_head, 256:512 left_wrist_0_rgb/hand_left, 512:768 "
                "right_wrist_0_rgb/hand_right; outputs are transformer-contextualized across cameras"
            ),
        },
        "token_definition": "image-only frozen VLA prefix -> observation-conditioned RLTokenModel.encode; one 2048-D token",
        "decoder_definition": (
            "learned 768-position queries cross-attend only to the supplied RL token; the observed VLA prefix is the "
            "reconstruction target, not a decoder input; no autoregressive teacher forcing or prefix bypass"
        ),
        "control_definition": {
            "matched_z": "same call token",
            "within_phase_shuffled_z": (
                "one-to-one derangement within the same instance_id and normalized-time bin, always a different "
                "episode; strata without a mathematically feasible one-to-one derangement are explicitly excluded; "
                "semantic phase/block unavailable"
            ),
            "global_shuffled_z": "one-to-one cross-episode derangement across all calls",
            "constant_z": "mean token over all calls; debugging control, not a held-out estimate",
            "permutation_seeds": list(PERM_SEEDS),
            "sampling_replacement": False,
            "map_audit": control_map_audit,
        },
        "probe_definition": {
            "unit": "episode x normalized-time bin",
            "split": "five fixed held-out folds grouped by instance_id",
            "auc_aggregation": "positive-negative-pair-weighted mean of within-held-out-fold AUCs",
            "confidence_interval": (
                "instance-cluster bootstrap within fixed folds over fixed OOF predictions; models are not refit in "
                "bootstrap replicates"
            ),
            "z_preprocessing": "training-fold StandardScaler -> 20-component PCA -> StandardScaler",
            "state_ref_z_preprocessing": (
                "retain separately standardized 88-D state+ref baseline; concatenate separately standardized "
                "20-PC z projection"
            ),
            "classifier": "class-balanced L2 logistic regression, fixed C=1",
        },
        "sample_counts": counts,
        "truth_availability": {
            "xy_alignment": False,
            "z_height_support": False,
            "retract_gap_contact": False,
            "next_chunk_alignment_risk": False,
            "next_chunk_retract_risk": False,
            "terminal_success": True,
            "task_progress_status": True,
        },
        "split": "instance_id groups, deterministic five folds; frozen in splits.csv before probes",
        "stage1_dataset_provenance": {
            "dataset_root": "/mnt/pfs/kk/kk/data/data/geniesim/stack_three_blocks",
            "split_manifest": str(PLAN_ROOT / "analysis/stage1_episode_split.json"),
            "training_episodes": 450,
            "validation_episodes": 50,
            "overlap_status": "not identity-verifiable: demonstration episode IDs and policy-rollout seeds are different namespaces; results are exploratory on a source-separated rollout archive",
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__},
    }
    _atomic_json(out / "run_manifest.json", manifest)
    yaml_lines = ["# Generated Stage2 manifest"] + [
        f"{key}: {json.dumps(value, ensure_ascii=False, default=str)}" for key, value in manifest.items()
    ]
    (out / "run_manifest.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    logging.info("prepared %d episodes and %d calls at %s", len(episodes), len(calls), out)


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


def _load_calls(output: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    rows = _read_csv(output / "calls.csv")
    with np.load(output / "call_inputs.npz") as data:
        state = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
    if len(rows) != len(state):
        raise RuntimeError("calls.csv and call_inputs.npz are misaligned")
    return rows, state, actions


def _extract_prefix(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    rows, _, _ = _load_calls(out)
    records = load_cache_index(args.cache_root.resolve())
    if len(records) != len(rows):
        raise RuntimeError(f"cache index and calls.csv differ: {len(records)} vs {len(rows)}")
    config = train_config.get_config(MODEL_CONFIG)
    rollout_config = train_config.get_config(ROLLOUT_CONFIG)
    model, audit = stage1._load_model_strict(config, args.checkpoint_root.resolve() / str(args.reference_step))
    transform = _input_transform(rollout_config)
    extract_fn = nnx_utils.module_jit(model.extract_image_prefix)
    prefix_seq_len, input_dim = audit["prefix_shape"]
    prefix_path = out / "prefix_embeddings.npy"
    mask_path = out / "prefix_mask.npy"
    prefix = np.lib.format.open_memmap(
        prefix_path, mode="w+", dtype=np.float32, shape=(len(rows), prefix_seq_len, input_dim)
    )
    mask = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.bool_, shape=(len(rows), prefix_seq_len))
    rng = jax.random.key(config.seed)
    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        samples = [transform(load_rollout_sample(record)) for record in records[start:end]]
        padded = list(samples)
        while len(padded) < args.batch_size:
            padded.append(samples[-1])
        observation = model_lib.Observation.from_dict(stage1._stack(padded))
        rng, batch_rng = jax.random.split(rng)
        batch_prefix, batch_mask = extract_fn(batch_rng, observation)
        prefix[start:end] = np.asarray(jax.device_get(batch_prefix[: end - start]), dtype=np.float32)
        mask[start:end] = np.asarray(jax.device_get(batch_mask[: end - start]), dtype=bool)
        logging.info("prefix %d/%d", end, len(rows))
    prefix.flush()
    mask.flush()
    if not bool(np.asarray(mask).all()):
        raise RuntimeError("prefix mask contains false entries")
    _atomic_json(out / "prefix_load_audit.json", audit)
    _atomic_json(
        out / "prefix_cache_manifest.json",
        {
            "shape": list(prefix.shape),
            "dtype": str(prefix.dtype),
            "mask_shape": list(mask.shape),
            "mask_true_fraction": float(np.asarray(mask).mean()),
            "reference_step": args.reference_step,
            "prefix_sha256": _sha256(prefix_path),
            "mask_sha256": _sha256(mask_path),
        },
    )
    logging.info("saved prefix cache %s", prefix_path)


def _encode_tokens(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    prefix = np.load(out / "prefix_embeddings.npy", mmap_mode="r")
    config = train_config.get_config(MODEL_CONFIG)
    checkpoint = args.checkpoint_root.resolve() / str(args.step)
    module, audit = stage1._load_rlt_only_strict(config, checkpoint, prefix.shape[1])
    encode_fn = nnx_utils.module_jit(module.encode)
    path = out / f"z_{args.step // 1000}k.npy"
    z = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(prefix), 2048))
    for start in range(0, len(prefix), args.batch_size):
        end = min(start + args.batch_size, len(prefix))
        actual = np.arange(start, end, dtype=np.int64)
        padded = actual.tolist()
        while len(padded) < args.batch_size:
            padded.append(int(actual[-1]))
        tokens = encode_fn(jnp.asarray(np.asarray(prefix[padded]), dtype=jnp.float32))
        z[start:end] = np.asarray(jax.device_get(tokens[: end - start]), dtype=np.float32).reshape(end - start, -1)
        logging.info("tokens step=%s %d/%d", args.step, end, len(prefix))
    z.flush()
    _atomic_json(out / f"z_{args.step // 1000}k_audit.json", {**audit, "shape": list(z.shape), "sha256": _sha256(path)})
    logging.info("saved %s", path)


def _position_slices(seq_len: int) -> dict[str, slice]:
    third = seq_len // 3
    return {
        "all": slice(0, seq_len),
        "early": slice(0, third),
        "middle": slice(third, 2 * third),
        "late": slice(2 * third, seq_len),
    }


def _metric_arrays(
    target: np.ndarray, decoded: np.ndarray, target_center: np.ndarray, slices: dict[str, slice]
) -> dict[str, np.ndarray]:
    difference = decoded - target
    result: dict[str, np.ndarray] = {}
    for name, selection in slices.items():
        diff = difference[:, selection]
        target_part = target[:, selection]
        decoded_part = decoded[:, selection]
        norms = np.linalg.norm(target_part, axis=-1) * np.linalg.norm(decoded_part, axis=-1)
        token_cosine = np.divide(
            np.sum(target_part * decoded_part, axis=-1),
            norms,
            out=np.zeros_like(norms),
            where=norms > 1e-12,
        )
        sse = np.square(diff, dtype=np.float64).sum(axis=(1, 2))
        denominator = np.square(target_part - target_center[selection], dtype=np.float64).sum(axis=(1, 2))
        result[f"mse_{name}"] = np.mean(np.square(diff), axis=(1, 2)).astype(np.float32)
        result[f"nmse_{name}"] = np.divide(
            sse, denominator, out=np.full_like(sse, np.nan), where=denominator > 1e-12
        ).astype(np.float32)
        result[f"cosine_{name}"] = token_cosine.mean(axis=1).astype(np.float32)
    return result


def _metric_arrays_device(
    decoded: jax.Array, target: jax.Array, target_center: jax.Array, slices: dict[str, slice]
) -> dict[str, jax.Array]:
    """Compute controls on device; only small metric vectors cross the host boundary."""
    result: dict[str, jax.Array] = {}
    for name, selection in slices.items():
        diff = decoded[:, selection] - target[:, selection]
        target_part = target[:, selection]
        decoded_part = decoded[:, selection]
        norms = jnp.linalg.norm(target_part, axis=-1) * jnp.linalg.norm(decoded_part, axis=-1)
        numerator = jnp.sum(target_part * decoded_part, axis=-1)
        cosine = jnp.where(norms > 1e-12, numerator / jnp.maximum(norms, 1e-12), 0.0).mean(axis=1)
        sse = jnp.sum(jnp.square(diff), axis=(1, 2), dtype=jnp.float32)
        denominator = jnp.sum(jnp.square(target_part - target_center[selection]), axis=(1, 2), dtype=jnp.float32)
        result[f"mse_{name}"] = jnp.mean(jnp.square(diff), axis=(1, 2))
        result[f"nmse_{name}"] = jnp.where(denominator > 1e-12, sse / denominator, jnp.nan)
        result[f"cosine_{name}"] = cosine
    return result


def _bootstrap_ci(values: np.ndarray, groups: np.ndarray, seed: int, reps: int = BOOTSTRAP_REPS) -> tuple[float, float]:
    finite = np.isfinite(values)
    values = np.asarray(values, dtype=np.float64)[finite]
    groups = np.asarray(groups)[finite]
    unique = np.unique(groups)
    if len(unique) < 2 or len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    by_group = {group: values[groups == group] for group in unique}
    estimates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        estimates[index] = np.mean(np.concatenate([by_group[group] for group in sampled]))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _summarize_reconstruction(control_rows: list[dict[str, Any]], checkpoint: str) -> list[dict[str, Any]]:
    """Reduce call-level controls to episode means and instance-cluster bootstrap CIs."""
    matched: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in control_rows:
        if row["variant"] == "matched_z":
            matched[(row["episode_id"], row["position"], str(row["time_bin"]))].append(row)
            matched[(row["episode_id"], row["position"], "all")].append(row)

    output: list[dict[str, Any]] = []
    for requested_bin in ["all", "0", "1", "2", "3", "4"]:
        selected = (
            control_rows
            if requested_bin == "all"
            else [row for row in control_rows if str(row["time_bin"]) == requested_bin]
        )
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[(row["variant"], int(row["perm_seed"]), row["position"])].append(row)
        for (variant, seed, position), items in grouped.items():
            by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
            instance_by_episode: dict[str, int] = {}
            for item in items:
                by_episode[item["episode_id"]].append(item)
                instance_by_episode[item["episode_id"]] = int(item["instance_id"])
            episode_values: list[tuple[str, float, float, float, float]] = []
            for episode_id, ep_items in by_episode.items():
                mse = float(np.mean([float(item["mse"]) for item in ep_items]))
                cosine = float(np.mean([float(item["cosine"]) for item in ep_items]))
                base_items = matched.get((episode_id, position, requested_bin), [])
                base_mse = float(np.mean([float(item["mse"]) for item in base_items])) if base_items else float("nan")
                base_cosine = (
                    float(np.mean([float(item["cosine"]) for item in base_items])) if base_items else float("nan")
                )
                episode_values.append((episode_id, mse, cosine, mse - base_mse, cosine - base_cosine))
            deltas_mse = np.asarray([value[3] for value in episode_values], dtype=np.float64)
            deltas_cos = np.asarray([value[4] for value in episode_values], dtype=np.float64)
            groups = np.asarray([instance_by_episode[value[0]] for value in episode_values])
            finite_m = np.isfinite(deltas_mse)
            finite_c = np.isfinite(deltas_cos)
            lo_m, hi_m = _bootstrap_ci(deltas_mse, groups, 101 + seed)
            lo_c, hi_c = _bootstrap_ci(deltas_cos, groups, 202 + seed)
            output.append(
                {
                    "checkpoint": checkpoint,
                    "variant": variant,
                    "perm_seed": seed,
                    "position": position,
                    "time_bin": requested_bin,
                    "episode_count": len(episode_values),
                    "instance_count": len(np.unique(groups)) if len(groups) else 0,
                    "mean_mse": float(np.mean([value[1] for value in episode_values]))
                    if episode_values
                    else float("nan"),
                    "mean_cosine": float(np.mean([value[2] for value in episode_values]))
                    if episode_values
                    else float("nan"),
                    "mean_delta_mse": float(np.nanmean(deltas_mse)) if finite_m.any() else float("nan"),
                    "delta_mse_ci_low": lo_m,
                    "delta_mse_ci_high": hi_m,
                    "mean_delta_cosine": float(np.nanmean(deltas_cos)) if finite_c.any() else float("nan"),
                    "delta_cosine_ci_low": lo_c,
                    "delta_cosine_ci_high": hi_c,
                }
            )
    return output


def _compare_reconstruction_checkpoints(control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matched: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in control_rows:
        if row["variant"] == "matched_z":
            matched[(row["checkpoint"], row["sample_id"], row["position"])] = (float(row["mse"]), float(row["cosine"]))
    episode_deltas: dict[tuple[str, str, int, str, str, int], list[tuple[float, float]]] = defaultdict(list)
    for row in control_rows:
        if row["variant"] == "matched_z":
            continue
        base_mse, base_cosine = matched[(row["checkpoint"], row["sample_id"], row["position"])]
        key = (
            row["checkpoint"],
            row["variant"],
            int(row["perm_seed"]),
            row["position"],
            row["episode_id"],
            int(row["instance_id"]),
        )
        episode_deltas[key].append((float(row["mse"]) - base_mse, float(row["cosine"]) - base_cosine))
    episode_means = {
        key: (float(np.mean([value[0] for value in values])), float(np.mean([value[1] for value in values])))
        for key, values in episode_deltas.items()
    }
    results: list[dict[str, Any]] = []
    combinations = sorted({(key[1], key[2], key[3]) for key in episode_means})
    for variant, seed, position in combinations:
        ten = {
            key[4]: (value, key[5])
            for key, value in episode_means.items()
            if key[0] == "10k" and key[1:4] == (variant, seed, position)
        }
        one_seventy = {
            key[4]: (value, key[5])
            for key, value in episode_means.items()
            if key[0] == "170k" and key[1:4] == (variant, seed, position)
        }
        episodes = sorted(set(ten) & set(one_seventy))
        mse_difference = np.asarray([one_seventy[episode][0][0] - ten[episode][0][0] for episode in episodes])
        cosine_difference = np.asarray([one_seventy[episode][0][1] - ten[episode][0][1] for episode in episodes])
        groups = np.asarray([ten[episode][1] for episode in episodes])
        mse_low, mse_high = _bootstrap_ci(mse_difference, groups, 9000 + seed)
        cos_low, cos_high = _bootstrap_ci(cosine_difference, groups, 10000 + seed)
        results.append(
            {
                "variant": variant,
                "perm_seed": seed,
                "position": position,
                "episode_count": len(episodes),
                "instance_count": len(np.unique(groups)),
                "delta_mse_170k_minus_10k": float(np.mean(mse_difference)),
                "delta_mse_ci_low": mse_low,
                "delta_mse_ci_high": mse_high,
                "delta_cosine_170k_minus_10k": float(np.mean(cosine_difference)),
                "delta_cosine_ci_low": cos_low,
                "delta_cosine_ci_high": cos_high,
                "definition": "(control-matched at 170k) minus (control-matched at 10k), episode means, instance bootstrap",
            }
        )
    return results


def _reconstruct(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    calls, _, _ = _load_calls(out)
    target = np.load(out / "prefix_embeddings.npy", mmap_mode="r")
    maps = np.load(out / "control_pair_maps.npz")
    within = np.asarray(maps["within"], dtype=np.int64)
    global_map = np.asarray(maps["global_map"], dtype=np.int64)
    seeds = [int(value) for value in maps["seeds"]]
    # A uniform audit subset is sufficient for the coordinate-centred NMSE
    # denominator and avoids an unnecessary 25 GB full-prefix scan.
    center_indices = np.rint(np.linspace(0, len(calls) - 1, min(128, len(calls)))).astype(np.int64)
    target_center = np.asarray(target[center_indices], dtype=np.float32).mean(axis=0)
    slices = _position_slices(target.shape[1])
    config = train_config.get_config(MODEL_CONFIG)
    control_fields = [
        "checkpoint",
        "variant",
        "perm_seed",
        "target_row",
        "sample_id",
        "episode_id",
        "instance_id",
        "time_bin",
        "source_row",
        "source_sample_id",
        "source_episode_id",
        "position",
        "mse",
        "nmse",
        "cosine",
    ]
    all_control_rows: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []

    for step in STEPS:
        z = np.load(out / f"z_{step // 1000}k.npy", mmap_mode="r")
        module, audit = stage1._load_rlt_only_strict(
            config, args.checkpoint_root.resolve() / str(step), target.shape[1]
        )
        decode_fn = nnx_utils.module_jit(module.decode)
        output_rows: list[dict[str, Any]] = []

        def run_variant(
            variant: str,
            perm_seed: int,
            source_indices: np.ndarray | None,
            *,
            constant: bool = False,
        ) -> None:
            metrics_parts: dict[str, list[np.ndarray]] = defaultdict(list)
            source_parts: list[np.ndarray] = []
            target_indices = (
                np.arange(len(calls), dtype=np.int64)
                if constant
                else np.flatnonzero(np.asarray(source_indices, dtype=np.int64) >= 0)
            )
            if not len(target_indices):
                return
            constant_token = np.asarray(z.mean(axis=0), dtype=np.float32) if constant else None
            for start in range(0, len(target_indices), args.batch_size):
                end = min(start + args.batch_size, len(target_indices))
                actual = target_indices[start:end]
                padded = actual.tolist()
                while len(padded) < args.batch_size:
                    padded.append(int(actual[-1]))
                if constant:
                    token_batch = np.broadcast_to(constant_token, (len(padded), 1, 2048))
                    source = np.full(len(actual), -1, dtype=np.int64)
                else:
                    source = np.asarray(source_indices[actual], dtype=np.int64)
                    token_batch = np.asarray(
                        z[np.asarray([int(item) for item in source_indices[padded]])], dtype=np.float32
                    ).reshape(len(padded), 1, 2048)
                target_np = np.asarray(target[actual], dtype=np.float32)
                decoded = decode_fn(jnp.asarray(token_batch, dtype=jnp.float32))[: len(actual)]
                device_metrics = _metric_arrays_device(
                    decoded,
                    jnp.asarray(target_np, dtype=jnp.float32),
                    jnp.asarray(target_center, dtype=jnp.float32),
                    slices,
                )
                batch_metrics = {
                    name: np.asarray(jax.device_get(value), dtype=np.float32) for name, value in device_metrics.items()
                }
                for name, values in batch_metrics.items():
                    metrics_parts[name].append(values)
                source_parts.append(source)
            metrics = {name: np.concatenate(values) for name, values in metrics_parts.items()}
            sources = np.concatenate(source_parts)
            for result_index, row_index in enumerate(target_indices):
                row = calls[int(row_index)]
                source_row = int(sources[result_index])
                for position in slices:
                    output_rows.append(
                        {
                            "checkpoint": f"{step // 1000}k",
                            "variant": variant,
                            "perm_seed": perm_seed,
                            "target_row": row_index,
                            "sample_id": row["sample_id"],
                            "episode_id": row["episode_id"],
                            "instance_id": row["instance_id"],
                            "time_bin": row["time_bin"],
                            "source_row": source_row,
                            "source_sample_id": "" if source_row < 0 else calls[source_row]["sample_id"],
                            "source_episode_id": "" if source_row < 0 else calls[source_row]["episode_id"],
                            "position": position,
                            "mse": float(metrics[f"mse_{position}"][result_index]),
                            "nmse": float(metrics[f"nmse_{position}"][result_index]),
                            "cosine": float(metrics[f"cosine_{position}"][result_index]),
                        }
                    )

        run_variant("matched_z", 0, np.arange(len(calls), dtype=np.int64))
        for seed_index, seed in enumerate(seeds):
            run_variant("within_phase_shuffled_z", seed, within[:, seed_index])
            run_variant("global_shuffled_z", seed, global_map[:, seed_index])
        run_variant("constant_z", 0, None, constant=True)
        _write_csv(out / f"reconstruction_controls_{step // 1000}k.csv", output_rows, control_fields)
        all_control_rows.extend(output_rows)
        all_summary.extend(_summarize_reconstruction(output_rows, f"{step // 1000}k"))
        _atomic_json(
            out / f"reconstruction_{step // 1000}k_audit.json",
            {
                **audit,
                "num_rows": len(output_rows),
                "analysis_script_sha256": _sha256(Path(__file__).resolve()),
                "control_pair_maps_sha256": _sha256(out / "control_pair_maps.npz"),
            },
        )
        logging.info("reconstruction controls complete for %s", step)

    _write_csv(out / "reconstruction_controls.csv", all_control_rows, control_fields)
    _write_csv(
        out / "reconstruction_summary.csv",
        all_summary,
        [
            "checkpoint",
            "variant",
            "perm_seed",
            "position",
            "time_bin",
            "episode_count",
            "instance_count",
            "mean_mse",
            "mean_cosine",
            "mean_delta_mse",
            "delta_mse_ci_low",
            "delta_mse_ci_high",
            "mean_delta_cosine",
            "delta_cosine_ci_low",
            "delta_cosine_ci_high",
        ],
    )
    checkpoint_comparison = _compare_reconstruction_checkpoints(all_control_rows)
    _write_csv(out / "reconstruction_checkpoint_comparison.csv", checkpoint_comparison, list(checkpoint_comparison[0]))
    _atomic_json(
        out / "reconstruction_run_audit.json",
        {
            "analysis_script_sha256": _sha256(Path(__file__).resolve()),
            "control_pair_maps_sha256": _sha256(out / "control_pair_maps.npz"),
            "reconstruction_controls_10k_sha256": _sha256(out / "reconstruction_controls_10k.csv"),
            "reconstruction_controls_170k_sha256": _sha256(out / "reconstruction_controls_170k.csv"),
            "reconstruction_summary_sha256": _sha256(out / "reconstruction_summary.csv"),
            "reconstruction_checkpoint_comparison_sha256": _sha256(out / "reconstruction_checkpoint_comparison.csv"),
        },
    )


def _resummarize_reconstruction(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    all_control_rows: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    for step in STEPS:
        checkpoint = f"{step // 1000}k"
        rows = _read_csv(out / f"reconstruction_controls_{checkpoint}.csv")
        all_control_rows.extend(rows)
        all_summary.extend(_summarize_reconstruction(rows, checkpoint))
    _write_csv(
        out / "reconstruction_summary.csv",
        all_summary,
        [
            "checkpoint",
            "variant",
            "perm_seed",
            "position",
            "time_bin",
            "episode_count",
            "instance_count",
            "mean_mse",
            "mean_cosine",
            "mean_delta_mse",
            "delta_mse_ci_low",
            "delta_mse_ci_high",
            "mean_delta_cosine",
            "delta_cosine_ci_low",
            "delta_cosine_ci_high",
        ],
    )
    comparison = _compare_reconstruction_checkpoints(all_control_rows)
    _write_csv(out / "reconstruction_checkpoint_comparison.csv", comparison, list(comparison[0]))


def _state_ref_features(state: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The archived action is an absolute joint/gripper proposal, not a delta.
    state_features = state[:, :24]
    first = actions[:, 0]
    last = actions[:, -1]
    mean = actions.mean(axis=1)
    max_abs_delta = np.max(np.abs(np.diff(actions, axis=1)), axis=1)
    ref_features = np.concatenate([first, last, mean, max_abs_delta], axis=1)
    return state_features.astype(np.float32), ref_features.astype(np.float32)


def _fold_auc_rows(y: np.ndarray, score: np.ndarray, folds: np.ndarray) -> list[dict[str, float | int]]:
    """Compute AUC only between examples scored by the same held-out-fold model."""
    output: list[dict[str, float | int]] = []
    for fold in sorted(np.unique(folds)):
        selected = (folds == fold) & np.isfinite(score)
        fold_y = y[selected]
        positives = int(np.sum(fold_y == 1))
        negatives = int(np.sum(fold_y == 0))
        pair_count = positives * negatives
        auc = float(roc_auc_score(fold_y, score[selected])) if pair_count else float("nan")
        output.append(
            {
                "fold": int(fold),
                "positive_units": positives,
                "negative_units": negatives,
                "positive_negative_pairs": pair_count,
                "roc_auc": auc,
            }
        )
    return output


def _fold_conditional_auc(y: np.ndarray, score: np.ndarray, folds: np.ndarray) -> float:
    rows = _fold_auc_rows(y, score, folds)
    valid = [row for row in rows if int(row["positive_negative_pairs"]) > 0 and np.isfinite(row["roc_auc"])]
    if not valid:
        return float("nan")
    weights = np.asarray([int(row["positive_negative_pairs"]) for row in valid], dtype=np.float64)
    values = np.asarray([float(row["roc_auc"]) for row in valid], dtype=np.float64)
    return float(np.average(values, weights=weights))


def _cluster_bootstrap_indices(groups: np.ndarray, folds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample instance groups within each fixed fold, preserving the fold structure."""
    parts: list[np.ndarray] = []
    for fold in sorted(np.unique(folds)):
        fold_groups = np.unique(groups[folds == fold])
        selected_groups = rng.choice(fold_groups, size=len(fold_groups), replace=True)
        by_group = {group: np.flatnonzero((groups == group) & (folds == fold)) for group in fold_groups}
        parts.extend(by_group[group] for group in selected_groups)
    return np.concatenate(parts)


def _cluster_auc_ci(
    y: np.ndarray, score: np.ndarray, groups: np.ndarray, folds: np.ndarray, seed: int
) -> tuple[float, float, float]:
    finite = np.isfinite(score)
    y = np.asarray(y, dtype=np.int8)[finite]
    score = np.asarray(score, dtype=np.float64)[finite]
    groups = np.asarray(groups)[finite]
    folds = np.asarray(folds)[finite]
    auc = _fold_conditional_auc(y, score, folds)
    if not np.isfinite(auc):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(BOOTSTRAP_REPS):
        indices = _cluster_bootstrap_indices(groups, folds, rng)
        estimate = _fold_conditional_auc(y[indices], score[indices], folds[indices])
        if np.isfinite(estimate):
            boot.append(estimate)
    if len(boot) < 20:
        return auc, float("nan"), float("nan")
    return auc, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _cluster_auc_delta_ci(
    y: np.ndarray,
    baseline: np.ndarray,
    full: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    finite = np.isfinite(baseline) & np.isfinite(full)
    y = np.asarray(y, dtype=np.int8)[finite]
    baseline = np.asarray(baseline, dtype=np.float64)[finite]
    full = np.asarray(full, dtype=np.float64)[finite]
    groups = np.asarray(groups)[finite]
    folds = np.asarray(folds)[finite]
    baseline_auc = _fold_conditional_auc(y, baseline, folds)
    full_auc = _fold_conditional_auc(y, full, folds)
    if not np.isfinite(baseline_auc) or not np.isfinite(full_auc):
        return float("nan"), float("nan"), float("nan")
    delta = full_auc - baseline_auc
    rng = np.random.default_rng(seed)
    bootstrap: list[float] = []
    for _ in range(BOOTSTRAP_REPS):
        indices = _cluster_bootstrap_indices(groups, folds, rng)
        baseline_estimate = _fold_conditional_auc(y[indices], baseline[indices], folds[indices])
        full_estimate = _fold_conditional_auc(y[indices], full[indices], folds[indices])
        if np.isfinite(baseline_estimate) and np.isfinite(full_estimate):
            bootstrap.append(full_estimate - baseline_estimate)
    if len(bootstrap) < 20:
        return delta, float("nan"), float("nan")
    return delta, float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def _probe_model(input_name: str, n_samples: int, n_features: int, state_ref_dim: int) -> Pipeline:
    if input_name == "z_rl":
        components = max(1, min(20, n_samples - 1, n_features))
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=components, svd_solver="randomized", random_state=42)),
                ("scale_pca", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=1.0, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=42
                    ),
                ),
            ]
        )
    if input_name == "state_ref_z":
        z_dim = n_features - state_ref_dim
        components = max(1, min(20, n_samples - 1, z_dim))
        z_pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=components, svd_solver="randomized", random_state=42)),
                ("scale_pca", StandardScaler()),
            ]
        )
        features = ColumnTransformer(
            [
                ("state_ref", StandardScaler(), slice(0, state_ref_dim)),
                ("z", z_pipeline, slice(state_ref_dim, n_features)),
            ]
        )
        return Pipeline(
            [
                ("features", features),
                (
                    "logit",
                    LogisticRegression(
                        C=1.0, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=42
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=42),
            ),
        ]
    )


def _run_probes(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    call_rows, state, actions = _load_calls(out)
    split_rows = {row["episode_id"]: row for row in _read_csv(out / "splits.csv")}
    episode_ids = sorted({row["episode_id"] for row in call_rows})
    state_features, ref_features = _state_ref_features(state, actions)

    # Aggregate to episode x normalized-time bin. This prevents long episodes
    # from dominating and respects the fact that calls are 50-tick chunks.
    units: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        indices = [index for index, row in enumerate(call_rows) if row["episode_id"] == episode_id]
        for time_bin in range(TIME_BINS):
            selected = [index for index in indices if int(call_rows[index]["time_bin"]) == time_bin]
            if not selected:
                continue
            split = split_rows[episode_id]
            units.append(
                {
                    "episode_id": episode_id,
                    "time_bin": time_bin,
                    "indices": selected,
                    "instance_id": int(split["instance_id"]),
                    "fold": int(split["fold"]),
                    "y": int(split["success"]),
                }
            )

    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    checkpoint_scores: dict[int, dict[str, np.ndarray]] = {}
    input_names = ("z_rl", "robot_state", "state_ref", "state_ref_z", "time_context")
    for step in STEPS:
        z = np.asarray(np.load(out / f"z_{step // 1000}k.npy", mmap_mode="r"), dtype=np.float32)
        x_by_name = {
            "z_rl": np.stack([z[unit["indices"]].mean(axis=0) for unit in units]),
            "robot_state": np.stack([state_features[unit["indices"]].mean(axis=0) for unit in units]),
            "state_ref": np.stack(
                [
                    np.concatenate(
                        [state_features[unit["indices"]].mean(axis=0), ref_features[unit["indices"]].mean(axis=0)]
                    )
                    for unit in units
                ]
            ),
        }
        x_by_name["state_ref_z"] = np.concatenate([x_by_name["state_ref"], x_by_name["z_rl"]], axis=1)
        x_by_name["time_context"] = np.eye(TIME_BINS, dtype=np.float32)[
            np.asarray([unit["time_bin"] for unit in units])
        ]
        y = np.asarray([unit["y"] for unit in units], dtype=np.int8)
        groups = np.asarray([unit["instance_id"] for unit in units], dtype=np.int64)
        folds = np.asarray([unit["fold"] for unit in units], dtype=np.int8)
        scores_by_input: dict[str, np.ndarray] = {}
        state_ref_dim = x_by_name["state_ref"].shape[1]

        for input_name in input_names:
            x = x_by_name[input_name]
            scores = np.full(len(units), np.nan, dtype=np.float64)
            for fold in sorted(np.unique(folds)):
                train_idx = np.flatnonzero(folds != fold)
                test_idx = np.flatnonzero(folds == fold)
                if len(test_idx) == 0 or len(np.unique(y[train_idx])) < 2:
                    continue
                model = _probe_model(input_name, len(train_idx), x.shape[1], state_ref_dim)
                model.fit(x[train_idx], y[train_idx])
                scores[test_idx] = model.predict_proba(x[test_idx])[:, 1]
            scores_by_input[input_name] = scores
            for index, unit in enumerate(units):
                predictions.append(
                    {
                        "checkpoint": f"{step // 1000}k",
                        "input": input_name,
                        "target": "terminal_success_retrospective",
                        "episode_id": unit["episode_id"],
                        "instance_id": unit["instance_id"],
                        "time_bin": unit["time_bin"],
                        "fold": unit["fold"],
                        "y_true": int(y[index]),
                        "y_score": scores[index],
                    }
                )
            input_fold_rows = _fold_auc_rows(y, scores, folds)
            for fold_row in input_fold_rows:
                fold_metrics.append(
                    {
                        "checkpoint": f"{step // 1000}k",
                        "input": input_name,
                        "target": "terminal_success_retrospective",
                        **fold_row,
                    }
                )
            auc, low, high = _cluster_auc_ci(y, scores, groups, folds, 5000 + step + len(input_name))
            metrics.append(
                {
                    "checkpoint": f"{step // 1000}k",
                    "input": input_name,
                    "target": "terminal_success_retrospective",
                    "metric": "roc_auc",
                    "value": auc,
                    "ci_low": low,
                    "ci_high": high,
                    "n_units": len(units),
                    "n_groups": len(np.unique(groups)),
                    "status": "supplementary",
                    "reason": (
                        "pair-weighted mean of within-held-out-fold AUCs; CI is an instance-cluster bootstrap of "
                        "fixed OOF predictions, not a refit bootstrap; final outcome is retrospective"
                    ),
                }
            )

        baseline = scores_by_input["state_ref"]
        full = scores_by_input["state_ref_z"]
        valid = np.isfinite(baseline) & np.isfinite(full)
        if valid.any():
            delta = full[valid] - baseline[valid]
            low, high = _bootstrap_ci(delta, groups[valid], 7000 + step)
            metrics.append(
                {
                    "checkpoint": f"{step // 1000}k",
                    "input": "state_ref_z_minus_state_ref",
                    "target": "terminal_success_retrospective",
                    "metric": "score_delta_mean",
                    "value": float(np.mean(delta)),
                    "ci_low": low,
                    "ci_high": high,
                    "n_units": int(valid.sum()),
                    "n_groups": len(np.unique(groups[valid])),
                    "status": "supplementary",
                    "reason": "paired score difference; not a causal or pre-event risk gain",
                }
            )
        auc_delta, auc_low, auc_high = _cluster_auc_delta_ci(y, baseline, full, groups, folds, 8000 + step)
        metrics.append(
            {
                "checkpoint": f"{step // 1000}k",
                "input": "state_ref_z_minus_state_ref",
                "target": "terminal_success_retrospective",
                "metric": "roc_auc_delta",
                "value": auc_delta,
                "ci_low": auc_low,
                "ci_high": auc_high,
                "n_units": int(valid.sum()),
                "n_groups": len(np.unique(groups[valid])),
                "status": "supplementary",
                "reason": (
                    "difference of pair-weighted within-fold AUCs; fixed-OOF-prediction instance bootstrap; z is "
                    "PCA-compressed separately while the state+ref baseline representation is retained"
                ),
            }
        )
        checkpoint_scores[step] = scores_by_input

    for input_name in ("z_rl", "state_ref_z"):
        delta, low, high = _cluster_auc_delta_ci(
            y,
            checkpoint_scores[10_000][input_name],
            checkpoint_scores[170_000][input_name],
            groups,
            folds,
            12000 + len(input_name),
        )
        metrics.append(
            {
                "checkpoint": "170k_minus_10k",
                "input": input_name,
                "target": "terminal_success_retrospective",
                "metric": "roc_auc_delta",
                "value": delta,
                "ci_low": low,
                "ci_high": high,
                "n_units": len(units),
                "n_groups": len(np.unique(groups)),
                "status": "supplementary",
                "reason": (
                    "difference of pair-weighted within-fold AUCs; fixed-OOF-prediction instance bootstrap; not "
                    "pre-event risk"
                ),
            }
        )

    for step in STEPS:
        for target_name in (
            "alignment_xy_error",
            "alignment_z_height_error",
            "retract_min_gap",
            "next_chunk_alignment_risk",
            "next_chunk_retract_risk",
        ):
            metrics.append(
                {
                    "checkpoint": f"{step // 1000}k",
                    "input": "all_probe_inputs",
                    "target": target_name,
                    "metric": "NA",
                    "value": "",
                    "ci_low": "",
                    "ci_high": "",
                    "n_units": 0,
                    "n_groups": 0,
                    "status": "insufficient_data",
                    "reason": "archived rollout has no per-call object geometry/contact/event truth",
                }
            )

    _write_csv(
        out / "probe_predictions.csv",
        predictions,
        ["checkpoint", "input", "target", "episode_id", "instance_id", "time_bin", "fold", "y_true", "y_score"],
    )
    _write_csv(
        out / "probe_metrics.csv",
        metrics,
        [
            "checkpoint",
            "input",
            "target",
            "metric",
            "value",
            "ci_low",
            "ci_high",
            "n_units",
            "n_groups",
            "status",
            "reason",
        ],
    )
    _write_csv(
        out / "probe_fold_metrics.csv",
        fold_metrics,
        [
            "checkpoint",
            "input",
            "target",
            "fold",
            "positive_units",
            "negative_units",
            "positive_negative_pairs",
            "roc_auc",
        ],
    )
    _atomic_json(
        out / "probe_run_audit.json",
        {
            "analysis_script_sha256": _sha256(Path(__file__).resolve()),
            "splits_sha256": _sha256(out / "splits.csv"),
            "z_10k_sha256": _sha256(out / "z_10k.npy"),
            "z_170k_sha256": _sha256(out / "z_170k.npy"),
            "probe_predictions_sha256": _sha256(out / "probe_predictions.csv"),
            "probe_metrics_sha256": _sha256(out / "probe_metrics.csv"),
            "probe_fold_metrics_sha256": _sha256(out / "probe_fold_metrics.csv"),
            "auc_aggregation": "positive-negative-pair-weighted mean of within-held-out-fold AUCs",
            "confidence_interval": "fixed-OOF-prediction instance-cluster bootstrap within fixed folds",
        },
    )
    logging.info("probe diagnostics complete: %d episode-time units", len(units))


def _make_figures(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(out / "reconstruction_summary.csv")

    variants = ["within_phase_shuffled_z", "global_shuffled_z", "constant_z"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for axis, position in zip(axes, ("all", "late"), strict=True):
        for x_index, variant in enumerate(variants):
            values10 = [
                float(row["mean_delta_mse"])
                for row in summary
                if row["checkpoint"] == "10k"
                and row["variant"] == variant
                and row["position"] == position
                and row["time_bin"] == "all"
            ]
            values170 = [
                float(row["mean_delta_mse"])
                for row in summary
                if row["checkpoint"] == "170k"
                and row["variant"] == variant
                and row["position"] == position
                and row["time_bin"] == "all"
            ]
            if values10 and values170:
                y10 = float(np.nanmean(values10))
                y170 = float(np.nanmean(values170))
                axis.plot([x_index - 0.12, x_index + 0.12], [y10, y170], marker="o", color="#456b8c")
                axis.text(x_index - 0.12, y10, "10k", fontsize=8, ha="right")
                axis.text(x_index + 0.12, y170, "170k", fontsize=8, ha="left")
        axis.axhline(0, color="black", lw=0.8)
        axis.set_xticks(range(len(variants)), ["within-instance/time\n(strict)", "global", "constant"])
        axis.set_ylabel("episode mean MSE(control) - MSE(matched)")
        axis.set_title(f"decoder target positions: {position}")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Stage 2 reconstruction controls (permutation points average three fixed maps)")
    figure.tight_layout()
    figure.savefig(figure_dir / "reconstruction_paired_deltas.png", dpi=180)
    plt.close(figure)

    control_rows = _read_csv(out / "reconstruction_controls.csv")
    matched_mse = {
        (row["checkpoint"], row["sample_id"]): float(row["mse"])
        for row in control_rows
        if row["variant"] == "matched_z" and row["position"] == "all"
    }
    episode_delta: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in control_rows:
        if (
            row["variant"] == "within_phase_shuffled_z"
            and row["perm_seed"] == str(PERM_SEEDS[0])
            and row["position"] == "all"
        ):
            episode_delta[(row["checkpoint"], row["episode_id"])].append(
                float(row["mse"]) - matched_mse[(row["checkpoint"], row["sample_id"])]
            )
    episode_delta_mean = {key: float(np.mean(values)) for key, values in episode_delta.items()}
    call_rows, _, _ = _load_calls(out)
    success_by_episode = {row["episode_id"]: int(row["success"]) for row in call_rows}
    episodes = sorted({key[1] for key in episode_delta_mean})
    figure, axis = plt.subplots(figsize=(8, 6))
    for episode in episodes:
        values = [episode_delta_mean[(checkpoint, episode)] for checkpoint in ("10k", "170k")]
        color = "#2c7a66" if success_by_episode[episode] else "#b55b52"
        axis.plot([0, 1], values, color=color, alpha=0.15, lw=0.8)
        axis.scatter([0, 1], values, color=color, alpha=0.35, s=9)
    for x_index, checkpoint in enumerate(("10k", "170k")):
        values = np.asarray([episode_delta_mean[(checkpoint, episode)] for episode in episodes])
        axis.scatter([x_index], [values.mean()], color="black", s=55, zorder=5)
        axis.text(x_index + 0.03, values.mean(), f"mean={values.mean():.3f}", va="center", fontsize=9)
    axis.scatter([], [], color="#2c7a66", label="successful episode")
    axis.scatter([], [], color="#b55b52", label="failed episode")
    axis.scatter([], [], color="black", label="all-episode mean")
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xticks([0, 1], ["10k", "170k"])
    axis.set_ylabel("episode mean MSE(within-instance/time swap) - MSE(matched)")
    axis.set_title(f"Each line is one episode; fixed permutation seed {PERM_SEEDS[0]}")
    axis.legend(loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "reconstruction_episode_deltas.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    position_labels = {
        "early": "top-head token block",
        "middle": "left-hand token block",
        "late": "right-hand token block",
    }
    for axis, checkpoint in zip(axes, ("10k", "170k"), strict=True):
        for position, color in zip(("early", "middle", "late"), ("#3b6f8f", "#b56c3b", "#4c8c6a"), strict=True):
            values = []
            for time_bin in range(TIME_BINS):
                selected = [
                    float(row["mean_delta_mse"])
                    for row in summary
                    if row["checkpoint"] == checkpoint
                    and row["variant"] == "within_phase_shuffled_z"
                    and row["position"] == position
                    and row["time_bin"] == str(time_bin)
                ]
                values.append(float(np.mean(selected)))
            axis.plot(range(1, TIME_BINS + 1), values, marker="o", label=position_labels[position], color=color)
        axis.axhline(0, color="black", lw=0.8)
        axis.set_xticks(range(1, TIME_BINS + 1))
        axis.set_xlabel("normalized policy-call time bin")
        axis.set_title(checkpoint)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("episode mean MSE(swap) - MSE(matched)")
    axes[1].legend(title="contextualized token positions")
    figure.suptitle("Within-instance/time token swap by trajectory time and contextualized camera-token block")
    figure.tight_layout()
    figure.savefig(figure_dir / "reconstruction_time_position.png", dpi=180)
    plt.close(figure)

    metrics = _read_csv(out / "probe_metrics.csv")
    outcome_rows = [
        row for row in metrics if row["target"] == "terminal_success_retrospective" and row["metric"] == "roc_auc"
    ]
    names = ["robot_state", "state_ref", "state_ref_z", "z_rl", "time_context"]
    figure, axis = plt.subplots(figsize=(10, 5))
    width = 0.18
    x = np.arange(len(names))
    for checkpoint_index, checkpoint in enumerate(("10k", "170k")):
        values = []
        lows = []
        highs = []
        for name in names:
            match = [row for row in outcome_rows if row["checkpoint"] == checkpoint and row["input"] == name]
            values.append(float(match[0]["value"]) if match and match[0]["value"] not in {"", "nan"} else np.nan)
            lows.append(float(match[0]["ci_low"]) if match and match[0]["ci_low"] not in {"", "nan"} else np.nan)
            highs.append(float(match[0]["ci_high"]) if match and match[0]["ci_high"] not in {"", "nan"} else np.nan)
        values_array = np.asarray(values)
        error = np.vstack((values_array - np.asarray(lows), np.asarray(highs) - values_array))
        axis.bar(
            x + (checkpoint_index - 0.5) * width,
            values_array,
            width,
            yerr=error,
            capsize=3,
            label=checkpoint,
        )
    axis.set_xticks(x, ["state", "state+ref", "state+ref+z", "z only", "time"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("ROC-AUC (retrospective final outcome)")
    axis.axhline(0.5, color="black", lw=0.8, ls="--", label="chance")
    axis.legend()
    axis.set_title("Supplementary grouped outcome probe; not a pre-event risk test")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "retrospective_outcome_probe.png", dpi=180)
    plt.close(figure)

    selected_episodes: list[str] = []
    for category in ("successful", "failed"):
        selected_episodes.extend(sorted({row["episode_id"] for row in call_rows if row["category"] == category})[:3])
    figure, axes = plt.subplots(len(selected_episodes), 3, figsize=(10, 2.6 * len(selected_episodes)), squeeze=False)
    for row_index, episode_id in enumerate(selected_episodes):
        episode_rows = [row for row in call_rows if row["episode_id"] == episode_id]
        for column, call_index in enumerate((0, len(episode_rows) // 2, len(episode_rows) - 1)):
            with np.load(episode_rows[call_index]["cache_path"], allow_pickle=False) as payload:
                axes[row_index, column].imshow(payload["top_head"])
            axes[row_index, column].axis("off")
            axes[row_index, column].set_title(f"{episode_id} call {episode_rows[call_index]['call_index']}", fontsize=8)
    figure.suptitle("Fixed keyframes: first 3 success + first 3 failure episodes")
    figure.tight_layout()
    figure.savefig(figure_dir / "fixed_keyframes.png", dpi=160)
    plt.close(figure)

    cameras = ("top_head", "hand_left", "hand_right")
    figure, axes = plt.subplots(
        len(selected_episodes), len(cameras), figsize=(11, 2.8 * len(selected_episodes)), squeeze=False
    )
    for row_index, episode_id in enumerate(selected_episodes):
        episode_rows = [row for row in call_rows if row["episode_id"] == episode_id]
        selected_call = round(0.75 * (len(episode_rows) - 1))
        with np.load(episode_rows[selected_call]["cache_path"], allow_pickle=False) as payload:
            for column, camera in enumerate(cameras):
                axes[row_index, column].imshow(payload[camera])
                axes[row_index, column].axis("off")
                axes[row_index, column].set_title(f"{episode_id} call {selected_call} | {camera}", fontsize=8)
    figure.suptitle("Fixed 75%-time multi-camera keyframes (geometry/event truth unavailable)")
    figure.tight_layout()
    figure.savefig(figure_dir / "fixed_multicamera_keyframes.png", dpi=160)
    plt.close(figure)

    event_rows = [row for row in _read_csv(out / "events.csv") if row["event_name"] == "terminal_stack_evaluator_event"]
    success_lags = [float(row["event_lag_seconds"]) for row in event_rows if row["label"] == "1"]
    failure_lags = [float(row["event_lag_seconds"]) for row in event_rows if row["label"] == "0"]
    figure, axis = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(success_lags + failure_lags) + 0.05, 18)
    axis.hist(
        success_lags, bins=bins, alpha=0.65, label=f"success Stack evt=3 (n={len(success_lags)})", color="#2c7a66"
    )
    axis.hist(
        failure_lags, bins=bins, alpha=0.65, label=f"failure Stack evt=4 (n={len(failure_lags)})", color="#b55b52"
    )
    axis.set_xlabel("seconds from latest policy call to terminal evaluator event")
    axis.set_ylabel("episodes")
    axis.set_title("Terminal evaluator timing only; not alignment/retract event truth")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "terminal_event_timing_audit.png", dpi=180)
    plt.close(figure)


def _write_report(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    counts = json.loads((out / "manifest_counts.json").read_text())
    summaries = _read_csv(out / "reconstruction_summary.csv")
    checkpoint_comparison = _read_csv(out / "reconstruction_checkpoint_comparison.csv")
    metrics = _read_csv(out / "probe_metrics.csv")
    split_summary = _read_csv(out / "split_summary.csv")
    control_map_audit = json.loads((out / "control_pair_audit.json").read_text())

    matched = {
        row["checkpoint"]: row
        for row in summaries
        if row["variant"] == "matched_z" and row["position"] == "all" and row["time_bin"] == "all"
    }
    matched_improvement = 1.0 - float(matched["170k"]["mean_mse"]) / float(matched["10k"]["mean_mse"])

    def number(value: str) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "NA"

    lines = [
        "# RL Token Stage 2 质量验证报告",
        "",
        "## 结论摘要",
        "",
        "本报告只验证 Stage 1 的观测条件化 RL token；没有训练、部署或在线运行 Actor/Critic。重建控制固定同一个 prefix 重建目标和同一个冻结 decoder，只替换送入 decoder 的 token。",
        "",
        "**当前判定：实验 A 显示明确的 token-target 样本配对依赖，decoder 没有忽略 latent；实验 B 所要求的当前几何和下一 chunk 风险因真值缺失而不可判。补充性的终局成败 probe 有关联，但没有证明 token 在 `state+ref` 之上提供可重复增量。**",
        "",
        f"数据覆盖 **{counts['episodes']} 条 episode / {counts['calls']} 次 policy call**（成功 {counts['success_episodes']}、失败 {counts['failure_episodes']}；少于 32 次调用的早停失败 {counts['short_failure_episodes']} 条，其中少于 16 calls、会被旧版 16 帧筛选排除的有 {counts['failure_episodes_below_16_calls']} 条）。每次 call 保存一个 50-step action proposal；归档没有实际执行 tick 数，且不能把相邻 call 当作连续 30 Hz 图像。",
        "",
        f"实际 infer-call 间隔中位数为成功 {counts['call_interval_seconds']['successful']['median']:.3f}s、失败 {counts['call_interval_seconds']['failed']['median']:.3f}s，而不是固定 1/30s。`instance_id` 的跨成败首帧 top-head MAE：同 instance 均值 {counts['scene_group_audit']['same_instance_top_head_mae_mean']:.3f}，不同 instance 均值 {counts['scene_group_audit']['different_instance_top_head_mae_mean']:.3f}（后者为前者 {counts['scene_group_audit']['different_to_same_image_mae_ratio']:.2f} 倍），支持把 instance 用作初始布局分组，但不代表 seed 相同。",
        "",
        "归档没有物体位姿、支撑关系、夹爪间隙、接触/碰撞或逐 tick 仿真状态。因此文档要求的 XY/Z 当前几何和下一 chunk 对齐/回撤风险无法诚实计算；`probe_metrics.csv` 中保留这些目标的 `insufficient_data/NA` 行，没有用失败 episode 的终局标签冒充事前风险。",
        "",
        "## 实验 A：token 置换重建",
        "",
        f"`matched_z`、同 `instance_id` 且同归一化时间 bin 的跨 episode 一一错排（仅是阶段代理）、全局跨 episode 一一错排和 `constant_z` 均使用同一 decoder。三个 seed 的映射都无放回、每个有效来源恰用一次；严格匹配覆盖 {control_map_audit['within_valid_calls_per_seed']}/{counts['calls']} calls，另有 {control_map_audit['within_excluded_calls_per_seed']} calls 因所在 stratum 不存在一一跨 episode 错排而明确排除，没有降级成任意来源。逐调用结果在 `reconstruction_controls_10k.csv` / `reconstruction_controls_170k.csv`，episode 等权汇总在 `reconstruction_summary.csv`。`mean_delta_mse` 为 control 减 matched；正值表示换 token 后重建变差。",
        "",
        f"各自 encoder-decoder pair 的匹配 token episode 等权 MSE：10k 为 {float(matched['10k']['mean_mse']):.6f}，170k 为 {float(matched['170k']['mean_mse']):.6f}（后者相对低 {matched_improvement:.1%}）；对应 cosine 为 {float(matched['10k']['mean_cosine']):.6f} 和 {float(matched['170k']['mean_cosine']):.6f}。这不是固定 decoder 下只比较 encoder 的实验。",
        "",
    ]
    for checkpoint in ("10k", "170k"):
        lines.extend(
            [
                f"### {checkpoint}",
                "",
                "| control / fixed map | episode-mean ΔMSE | conditional clustered 95% interval | mean Δcosine |",
                "|---|---:|---:|---:|",
            ]
        )
        for variant in ("within_phase_shuffled_z", "global_shuffled_z", "constant_z"):
            selected = [
                row
                for row in summaries
                if row["checkpoint"] == checkpoint
                and row["position"] == "all"
                and row["time_bin"] == "all"
                and row["variant"] == variant
            ]
            if not selected:
                continue
            for row in sorted(selected, key=lambda item: int(item["perm_seed"])):
                label = variant if variant == "constant_z" else f"{variant}, seed={row['perm_seed']}"
                lines.append(
                    f"| {label} | {float(row['mean_delta_mse']):.6f} | "
                    f"[{float(row['delta_mse_ci_low']):.6f}, {float(row['delta_mse_ci_high']):.6f}] | "
                    f"{float(row['mean_delta_cosine']):.6f} |"
                )
        lines.append("")
    within_comparison = [
        row for row in checkpoint_comparison if row["variant"] == "within_phase_shuffled_z" and row["position"] == "all"
    ]
    lines.extend(
        [
            "同 instance/time 错排 penalty 的 checkpoint 配对差（170k pair 减 10k pair）：",
            "",
            "| fixed map seed | ΔΔMSE | conditional clustered 95% interval |",
            "|---:|---:|---:|",
        ]
    )
    for row in sorted(within_comparison, key=lambda item: int(item["perm_seed"])):
        lines.append(
            f"| {row['perm_seed']} | {float(row['delta_mse_170k_minus_10k']):.6f} | "
            f"[{float(row['delta_mse_ci_low']):.6f}, {float(row['delta_mse_ci_high']):.6f}] |"
        )
    lines.extend(
        [
            "",
            "这些区间在固定错排映射下先做 episode 等权点估计，再按 instance 聚类重采样；不同 instance 的 episode 数不同，因此 estimand 是 episode-weighted、instance-clustered。这里逐 seed 报告，不把三个条件区间拼成一个名义 95% 区间。",
            "",
            "本地 decoder 的 768 个 learned queries 只 cross-attend 到输入 token；真实 prefix 只是重建目标，不是 decoder 输入，也不存在 teacher-forcing/bypass 路径。因此错排结果证明 latent 与目标样本身份有关，仍不等同于 token 已编码可用于控制的几何信息。",
            "",
            "768 个 target 位置依次对应 top-head、left-hand、right-hand 的 256-token block；但它们是 VLA transformer 输出后的上下文化表示，因此位置分块曲线不是隔离单相机的 ablation。",
            "",
            "## 实验 B：补充性探针",
            "",
            "在冻结的 instance_id 五折划分上，把相邻调用先聚合为 episode × 五个归一化时间 bin；输入比较 `z_rl`、robot state、robot state + 当前 recorded ref_chunk 摘要、再加 z，以及时间代理。ref_chunk 是记录中的绝对关节/夹爪提案，摘要为首/尾/均值/相邻最大变化。`state+ref+z` 保留与主基线相同的 88 维表示，只对新增 z 在训练折内标准化并压缩为 20 个 PC。",
            "",
            "| held-out fold | instance groups | episodes (成功/失败) | binned probe units (正/负) |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in split_summary:
        lines.append(
            f"| {row['fold']} | {row['instance_groups']} | {row['episodes']} "
            f"({row['success_episodes']}/{row['failure_episodes']}) | {row['binned_probe_units']} "
            f"({row['positive_probe_units']}/{row['negative_probe_units']}) |"
        )
    lines.extend(
        [
            "",
            "下面是终局成功标签的补充性、事后解码结果；AUC 只比较由同一个 held-out-fold 模型打分的正负样本，再按各 fold 的正负 pair 数加权，避免混入 fold 间概率校准差。区间是固定 OOF 预测上的 instance-cluster conditional bootstrap，未在每次 bootstrap 中重拟合模型，不能解释成完整训练流程的不确定性，也不能解释成错误发生前的风险预警：",
            "",
            "| checkpoint | input | fold-conditional AUC | conditional 95% interval |",
            "|---|---|---:|---:|",
        ]
    )
    for row in metrics:
        if row["target"] == "terminal_success_retrospective" and row["metric"] == "roc_auc":
            lines.append(
                f"| {row['checkpoint']} | {row['input']} | {number(row['value'])} | [{number(row['ci_low'])}, {number(row['ci_high'])}] |"
            )
    lines.extend(
        [
            "",
            "主基线之上的配对增量：",
            "",
            "| checkpoint/comparison | quantity | ΔAUC | conditional 95% interval |",
            "|---|---|---:|---:|",
        ]
    )
    for row in metrics:
        if row["target"] == "terminal_success_retrospective" and row["metric"] == "roc_auc_delta":
            lines.append(
                f"| {row['checkpoint']} | {row['input']} | {number(row['value'])} | [{number(row['ci_low'])}, {number(row['ci_high'])}] |"
            )
    lines.extend(
        [
            "",
            "10k 和 170k 的 `state+ref+z - state+ref` AUC 点估计都为正，但 cluster-bootstrap 区间跨 0；170k 相对 10k 的 checkpoint 差异区间也跨 0。因此目前只有事后成败关联，**没有可重复的主基线增量证据，也没有可信的 170k 优于 10k 的探针证据**。",
            "",
            "几何回归（XY、Z、回撤间隙）和下一 chunk 风险（对齐/回撤）均为 NA，原因是归档缺少对应真值；不能从 verification.json 的终局 Stack 分数推回事件 tick。",
            "",
            "## 产物",
            "",
            "- `episodes.csv` / `calls.csv`：全部样本清单，含实际 UTC 调用时间、record step、state/action 来源和短失败。",
            "- `events.csv`：明确列出不可用的几何/接触事件以及仅供事后分析的终局标签。",
            "- `splits.csv` / `split_summary.csv`：冻结的 instance 分组五折及各折样本数，避免同一场景的成功/失败跨侧泄漏。",
            "- `probe_fold_metrics.csv`：每个 checkpoint/input/fold 的正负单元数、pair 数和 fold 内 AUC。",
            "- `reconstruction_run_audit.json` / `probe_run_audit.json`：记录生成数值产物时的精确脚本与输入/输出 SHA256。",
            "- Stage 1 使用独立 demonstration 数据根目录（450 train / 50 validation）；rollout seed 与 demonstration episode ID 不在同一身份命名空间，无法证明逐 episode 不重叠，因此本报告只称 source-separated rollout 上的探索性结果。",
            "- `figures/reconstruction_episode_deltas.png`：每条 episode 的置换配对差；`figures/reconstruction_time_position.png`：时间 bin 与 decoder 位置细分；`figures/retrospective_outcome_probe.png`：事后结果探针。",
            "- `figures/fixed_multicamera_keyframes.png`：固定 75% 时间点的三相机样本；`figures/terminal_event_timing_audit.png`：终局 evaluator 事件相对最新 policy call 的时序审计。",
            "- `commands.txt`：实际执行命令；`run_manifest.yaml`：checkpoint、预处理、token/decoder 定义和数据限制。",
            "",
            "## 最省成本的下一步",
            "",
            "保持当前 policy 配置，为同一 BASE π0.5 rollout 额外导出每个仿真 tick 的实际执行 action、物体 world pose、support/stack 判据、夹爪到保护物体的最小距离、接触对和事件 tick，并保留事件前最近 policy call 的三路图像与 ref_chunk。随后复用本目录的冻结 splits 和 10k/170k probe 流程，才能回答 Stage2_check 的核心事前几何/风险问题。",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_commands(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    commands = [
        "# Successful final commands (run from /root/workspace/rlt/RLT)",
        f"/root/workspace/envs/openpi/bin/python scripts/geniesim/stage2_token_quality.py prepare --output {out}",
        f"/root/workspace/envs/openpi/bin/python scripts/geniesim/stage2_token_quality.py extract-prefix --output {out} --batch-size 16",
        *[
            f"/root/workspace/envs/openpi/bin/python scripts/geniesim/stage2_token_quality.py encode --output {out} --step {step} --batch-size 16"
            for step in STEPS
        ],
        f"/root/workspace/envs/openpi/bin/python scripts/geniesim/stage2_token_quality.py reconstruct --output {out} --batch-size 64",
        f"/root/workspace/envs/openpi/bin/python scripts/geniesim/stage2_token_quality.py summarize-reconstruction --output {out}",
        f"/root/workspace/envs/openpi/bin/python scripts/geniesim/stage2_token_quality.py probes --output {out}",
        f"/root/workspace/envs/openpi/bin/python scripts/geniesim/stage2_token_quality.py figures --output {out}",
        f"/root/workspace/envs/openpi/bin/python scripts/geniesim/stage2_token_quality.py report --output {out}",
        "",
        "# Recovery record",
        "# Initial reconstruction attempts were interrupted during host-transfer performance diagnosis; no partial CSV was accepted.",
        "# The completed reconstruction above aggregates metrics on device before host transfer.",
        "# Final controls were regenerated as one-to-one derangements after an audit found replacement sampling and insufficient instance matching.",
        "# Probe metrics were regenerated with within-fold AUC aggregation and a baseline-preserving state+ref+z pipeline after statistical audit.",
    ]
    (out / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "extract-prefix",
            "encode",
            "reconstruct",
            "summarize-reconstruction",
            "probes",
            "figures",
            "report",
            "all",
        ],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rollout-root", type=Path, default=ROLLOUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--reference-step", type=int, default=10_000)
    parser.add_argument("--step", type=int, choices=list(STEPS), default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "prepare":
        _write_manifests(args)
    elif args.command == "extract-prefix":
        _extract_prefix(args)
    elif args.command == "encode":
        _encode_tokens(args)
    elif args.command == "reconstruct":
        _reconstruct(args)
    elif args.command == "summarize-reconstruction":
        _resummarize_reconstruction(args)
    elif args.command == "probes":
        _run_probes(args)
    elif args.command == "figures":
        _make_figures(args)
    elif args.command == "report":
        _write_report(args)
    elif args.command == "all":
        _write_manifests(args)
        _extract_prefix(args)
        for step in STEPS:
            args.step = step
            _encode_tokens(args)
        _reconstruct(args)
        _run_probes(args)
        _make_figures(args)
        _write_report(args)
        _write_commands(args)
    if args.command != "prepare":
        _write_commands(args)


if __name__ == "__main__":
    main()
