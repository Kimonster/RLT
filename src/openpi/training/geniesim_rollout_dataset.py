from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re
from typing import Any, SupportsIndex

import numpy as np

_TRAJECTORY_PATTERN = re.compile(r"trajectory_(\d+)$")
_STEP_PATTERN = re.compile(r"step_(\d+)\.npy$")


@dataclasses.dataclass(frozen=True)
class GenieSimRolloutRecord:
    category: str
    trajectory_id: int
    step_id: int
    source_path: Path
    cache_path: Path | None = None


def discover_rollout_records(rollout_root: str | Path) -> list[GenieSimRolloutRecord]:
    root = Path(rollout_root).expanduser().resolve()
    records: list[GenieSimRolloutRecord] = []
    for category in ("successful", "failed"):
        category_root = root / category
        for trajectory_path in category_root.glob("trajectory_*"):
            trajectory_match = _TRAJECTORY_PATTERN.fullmatch(trajectory_path.name)
            if trajectory_match is None:
                continue
            trajectory_id = int(trajectory_match.group(1))
            for source_path in (trajectory_path / "policy_records").glob("step_*.npy"):
                step_match = _STEP_PATTERN.fullmatch(source_path.name)
                if step_match is None:
                    continue
                records.append(
                    GenieSimRolloutRecord(
                        category=category,
                        trajectory_id=trajectory_id,
                        step_id=int(step_match.group(1)),
                        source_path=source_path,
                    )
                )
    records.sort(key=lambda item: (item.category, item.trajectory_id, item.step_id))
    if not records:
        raise FileNotFoundError(f"No GenieSim policy records found under {root}")
    return records


def load_cache_index(cache_root: str | Path) -> list[GenieSimRolloutRecord]:
    root = Path(cache_root).expanduser().resolve()
    index_path = root / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported GenieSim rollout cache schema in {index_path}")

    source_root = Path(payload["source_root"])
    records = [
        GenieSimRolloutRecord(
            category=item["category"],
            trajectory_id=int(item["trajectory_id"]),
            step_id=int(item["step_id"]),
            source_path=source_root / item["source_path"],
            cache_path=root / item["cache_path"],
        )
        for item in payload["records"]
    ]
    if not records:
        raise ValueError(f"GenieSim rollout cache is empty: {index_path}")
    return records


def _decode_record(payload: dict[str, Any]) -> dict[str, Any]:
    required = (
        "inputs/state",
        "inputs/images/top_head",
        "inputs/images/hand_left",
        "inputs/images/hand_right",
        "outputs/actions",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"GenieSim policy record is missing fields: {missing}")
    prompt = payload.get("inputs/prompt", "stack all the building blocks on the middle of the table")
    return {
        "images": {
            "top_head": np.asarray(payload["inputs/images/top_head"], dtype=np.uint8),
            "hand_left": np.asarray(payload["inputs/images/hand_left"], dtype=np.uint8),
            "hand_right": np.asarray(payload["inputs/images/hand_right"], dtype=np.uint8),
        },
        "state": np.asarray(payload["inputs/state"], dtype=np.float32),
        "actions": np.asarray(payload["outputs/actions"], dtype=np.float32),
        "prompt": str(prompt),
    }


def load_rollout_sample(record: GenieSimRolloutRecord) -> dict[str, Any]:
    if record.cache_path is not None:
        with np.load(record.cache_path, allow_pickle=False) as payload:
            return {
                "images": {
                    "top_head": payload["top_head"],
                    "hand_left": payload["hand_left"],
                    "hand_right": payload["hand_right"],
                },
                "state": payload["state"].astype(np.float32, copy=False),
                "actions": payload["actions"].astype(np.float32, copy=False),
                "prompt": str(payload["prompt"].item()),
            }

    raw = np.load(record.source_path, allow_pickle=True)
    if raw.shape != () or raw.dtype != object:
        raise ValueError(f"Expected a scalar object array in {record.source_path}, got {raw.shape} {raw.dtype}")
    payload = raw.item()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary in {record.source_path}, got {type(payload)}")
    return _decode_record(payload)


class GenieSimRolloutDataset:
    """Random-access dataset over recorded GenieSim policy calls."""

    def __init__(self, rollout_root: str | Path, *, cache_root: str | Path | None = None):
        if cache_root is not None and (Path(cache_root).expanduser() / "index.json").is_file():
            self.records = load_cache_index(cache_root)
        else:
            self.records = discover_rollout_records(rollout_root)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        return load_rollout_sample(self.records[index.__index__()])

    def __len__(self) -> int:
        return len(self.records)
