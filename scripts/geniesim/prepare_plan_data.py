#!/usr/bin/env python3
"""Prepare immutable episode splits and small auditable dataset samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.geniesim.plan_utils import DEFAULT_DEMO_ROOT
from scripts.geniesim.plan_utils import DEFAULT_PLAN_ROOT
from scripts.geniesim.plan_utils import atomic_write_json
from scripts.geniesim.plan_utils import dataset_provenance
from scripts.geniesim.plan_utils import load_stage1_split
from scripts.geniesim.plan_utils import make_stage1_split


def _save_image(path: Path, value) -> None:
    from PIL import Image

    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0 if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0 else array, 0, 255)
        array = array.astype(np.uint8)
    Image.fromarray(array).save(path)


def prepare_samples(dataset_root: Path, output_root: Path, split: dict, sample_count: int) -> list[str]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    sample_dir = output_root / "analysis" / "stage1_dataset_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    # Use fixed validation episodes for provenance; these files are tiny compared
    # with the source videos and make the W&B dataset artifact inspectable.
    episodes = split["validation_episodes"][: max(1, min(sample_count, len(split["validation_episodes"])))]
    written: list[str] = []
    for episode_index in episodes:
        # Loading one episode at a time avoids an indexing quirk in older
        # LeRobot releases when a non-contiguous episode subset is supplied.
        dataset = LeRobotDataset(
            "local/stack_three_blocks",
            root=str(dataset_root),
            episodes=[int(episode_index)],
            delta_timestamps=None,
        )
        row = 0
        item = dataset[row]
        stem = f"episode_{int(episode_index):06d}_frame_{int(item['frame_index']):06d}"
        for key, short_name in (
            ("observation.images.top_head", "top_head"),
            ("observation.images.hand_left", "hand_left"),
            ("observation.images.hand_right", "hand_right"),
        ):
            path = sample_dir / f"{stem}_{short_name}.png"
            _save_image(path, item[key])
            written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=8)
    args = parser.parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split = make_stage1_split(dataset_root, output_root, seed=args.seed)
    samples = prepare_samples(dataset_root, output_root, split, args.sample_count)
    manifest = {
        "schema_version": 1,
        "source": "original_sft_demonstration_dataset",
        "provenance": dataset_provenance(dataset_root),
        "episode_split": str(output_root / "analysis" / "stage1_episode_split.json"),
        "sample_files": samples,
        "sample_count": len(samples),
        "control_vector": "14 joint positions [state 30:44/action 16:30] + 2 grippers [0:2], then pad to 32",
    }
    atomic_write_json(output_root / "analysis" / "stage1_dataset_manifest.json", manifest)
    print(json.dumps({"split": split, "manifest": manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
