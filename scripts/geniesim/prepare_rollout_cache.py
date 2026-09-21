#!/usr/bin/env python3
"""Build a compact 224x224 cache from recorded GenieSim policy calls."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from openpi.training.geniesim_rollout_dataset import GenieSimRolloutRecord
from openpi.training.geniesim_rollout_dataset import discover_rollout_records
from openpi.training.geniesim_rollout_dataset import load_rollout_sample

DEFAULT_ROLLOUT_ROOT = Path("/mnt/pfs/kk/kk/data/data/geniesim/rollout/stack_three_blocks")
DEFAULT_CACHE_ROOT = Path("/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks/cache_224")


def _resize_image(image: np.ndarray) -> np.ndarray:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from openpi.shared.image_tools import resize_with_pad

    return np.asarray(resize_with_pad(image, 224, 224), dtype=np.uint8)


def _cache_relative_path(record: GenieSimRolloutRecord) -> Path:
    return Path("records") / record.category / f"trajectory_{record.trajectory_id:06d}" / f"step_{record.step_id}.npz"


def _prepare_one(args: tuple[GenieSimRolloutRecord, Path]) -> dict[str, Any]:
    record, cache_root = args
    relative_path = _cache_relative_path(record)
    output_path = cache_root / relative_path
    if not output_path.is_file():
        sample = load_rollout_sample(record)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(f"{output_path.suffix}.{os.getpid()}.tmp")
        with temporary_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                top_head=_resize_image(sample["images"]["top_head"]),
                hand_left=_resize_image(sample["images"]["hand_left"]),
                hand_right=_resize_image(sample["images"]["hand_right"]),
                state=np.asarray(sample["state"], dtype=np.float32),
                actions=np.asarray(sample["actions"], dtype=np.float32),
                prompt=np.asarray(sample["prompt"]),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)

    return {
        "category": record.category,
        "trajectory_id": record.trajectory_id,
        "step_id": record.step_id,
        "source_path": str(record.source_path),
        "cache_path": str(relative_path),
    }


def _write_index(cache_root: Path, rollout_root: Path, records: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "source_root": str(rollout_root),
        "image_size": [224, 224],
        "records": records,
    }
    index_path = cache_root / "index.json"
    temporary_path = index_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, index_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_root = args.rollout_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    records = discover_rollout_records(rollout_root)

    work = [(record, cache_root) for record in records]
    if args.workers == 1:
        entries = [_prepare_one(item) for item in tqdm.tqdm(work, desc="Caching rollout records")]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            entries = list(
                tqdm.tqdm(
                    executor.map(_prepare_one, work, chunksize=4),
                    total=len(work),
                    desc="Caching rollout records",
                )
            )
    _write_index(cache_root, rollout_root, entries)
    print(f"Cached {len(entries)} records at {cache_root}")


if __name__ == "__main__":
    main()
