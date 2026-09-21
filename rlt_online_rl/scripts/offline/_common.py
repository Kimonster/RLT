from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig-rlt"))

from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.config import resolve_rl_config_paths
from rlt_online_rl.inference import RLTPolicyInferenceWrapper

JOINT_LABELS = [f"joint{i + 1}" for i in range(6)] + ["gripper"]
JOINT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#e377c2", "#7f7f7f"]
PHASE_CHOICES = ("all", "warmup", "online", "unknown")
SOURCE_CHOICES = {
    "all": None,
    "base": {0},
    "rl": {1},
    "human": {2},
    "mixed": {3},
}


def resolve_stats_path(path: str | None, run_dir: Path) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    parts = list(candidate.parts)
    if "rlt_online_rl" in parts:
        anchor = parts.index("rlt_online_rl")
        rebased = ROOT.joinpath(*parts[anchor + 1 :])
        if rebased.exists():
            return str(rebased.resolve())
    for resolved in (
        run_dir / candidate,
        Path.cwd() / candidate,
        ROOT / candidate,
        ROOT / "configs" / "tasks" / run_dir.name / candidate,
    ):
        if resolved.exists():
            return str(resolved.resolve())
    raise FileNotFoundError(f"Could not resolve action_norm_stats_path={path!r}")


def _load_rl_config_from_snapshot(payload: dict[str, Any], run_dir: Path, source_path: Path) -> RLTOnlineRLConfig:
    raw_config = dict(payload["rl_config"])
    allowed_keys = {field.name for field in dataclasses.fields(RLTOnlineRLConfig)}
    filtered_config = {key: value for key, value in raw_config.items() if key in allowed_keys}
    cfg = RLTOnlineRLConfig(**filtered_config)
    cfg = resolve_rl_config_paths(cfg, str(source_path), require_exists=True)
    return dataclasses.replace(cfg, action_norm_stats_path=resolve_stats_path(cfg.action_norm_stats_path, run_dir))


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Snapshot/checkpoint at {path} must contain a dict payload.")
    return payload


def resolve_default_actor_snapshot_path(offline_dir: Path) -> Path:
    for candidate in (
        offline_dir / "actor_snapshot" / "actor_snapshot.pkl",
        offline_dir / "checkpoints" / "latest.pkl",
        offline_dir / "best_actor_snapshot.pkl",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (offline_dir / "best_actor_snapshot.pkl").resolve()


def resolve_default_actor_checkpoint_path(offline_dir: Path) -> Path:
    for candidate in (
        offline_dir / "checkpoints" / "latest.pkl",
        offline_dir / "actor_snapshot" / "actor_snapshot.pkl",
        offline_dir / "best_actor_snapshot.pkl",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (offline_dir / "best_actor_snapshot.pkl").resolve()


def resolve_default_critic_snapshot_path(offline_dir: Path) -> Path:
    for candidate in (
        offline_dir / "checkpoints" / "latest.pkl",
        offline_dir / "best_critic_snapshot.pkl",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (offline_dir / "best_critic_snapshot.pkl").resolve()


def load_snapshot(snapshot_path: Path, run_dir: Path) -> tuple[RLTOnlineRLConfig, Any]:
    payload = _load_payload(snapshot_path)
    cfg = _load_rl_config_from_snapshot(payload, run_dir, snapshot_path)
    if "actor_params" in payload:
        return cfg, payload["actor_params"]
    if "state" in payload and "actor_params" in payload["state"]:
        return cfg, payload["state"]["actor_params"]
    raise KeyError(f"Could not find actor params in {snapshot_path}")


def load_critic_snapshot(snapshot_path: Path, run_dir: Path) -> tuple[RLTOnlineRLConfig, Any]:
    payload = _load_payload(snapshot_path)
    cfg = _load_rl_config_from_snapshot(payload, run_dir, snapshot_path)
    if "critic_params" in payload:
        return cfg, payload["critic_params"]
    if "state" in payload and "critic_params" in payload["state"]:
        return cfg, payload["state"]["critic_params"]
    raise KeyError(f"Could not find critic params in {snapshot_path}")


def load_replay_journal(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as f:
        try:
            while True:
                item = pickle.load(f)
                if isinstance(item, dict):
                    records.append(item)
        except EOFError:
            pass
    return records


def write_replay_journal(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for record in records:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)


def infer_task_dir_from_replay_path(replay_path: Path) -> Path:
    replay_path = replay_path.resolve()
    if replay_path.parent.name != "replay":
        raise ValueError(f"Replay path must live under a replay/ directory: {replay_path}")
    return replay_path.parent.parent


def collection_phase(record: dict[str, Any]) -> str:
    return str(record.get("collection_phase", "unknown"))


def filter_replay_records(
    records: list[dict[str, Any]],
    *,
    phase: str,
    source: str,
) -> list[dict[str, Any]]:
    source_ids = SOURCE_CHOICES[source]
    filtered = []
    for record in records:
        if phase != "all" and collection_phase(record) != phase:
            continue
        if source_ids is not None and int(record["source"]) not in source_ids:
            continue
        filtered.append(record)
    return filtered


def default_filter_suffix(*, phase: str, source: str) -> str:
    parts: list[str] = []
    if phase != "all":
        parts.append(f"phase-{phase}")
    if source != "all":
        parts.append(f"source-{source}")
    return "" if not parts else "_" + "_".join(parts)


def build_model_ref_chunk(
    adapter: ActionRepresentationAdapter | None,
    ref_chunk: np.ndarray,
    proprio: np.ndarray,
    *,
    disable_ref_input: bool = False,
) -> np.ndarray:
    model_ref_chunk = (
        adapter.normalize_ref_chunk(ref_chunk, proprio)
        if adapter is not None
        else np.asarray(ref_chunk, dtype=np.float32)
    )
    if disable_ref_input:
        model_ref_chunk = np.zeros_like(model_ref_chunk, dtype=np.float32)
    return np.asarray(model_ref_chunk, dtype=np.float32)


def predict_refined_chunk(
    wrapper: RLTPolicyInferenceWrapper,
    adapter: ActionRepresentationAdapter | None,
    actor_params: Any,
    record: dict[str, Any],
    *,
    disable_ref_input: bool = False,
    deterministic: bool = True,
    rng: Any | None = None,
) -> np.ndarray:
    z_rl = np.asarray(record["z_rl"], dtype=np.float32)
    proprio = np.asarray(record["proprio"], dtype=np.float32)
    ref_chunk = np.asarray(record["ref_chunk"], dtype=np.float32)
    model_ref_chunk = build_model_ref_chunk(adapter, ref_chunk, proprio, disable_ref_input=disable_ref_input)
    refined = wrapper.infer(actor_params, z_rl, proprio, model_ref_chunk, deterministic=deterministic, rng=rng)
    if adapter is not None:
        refined = adapter.denormalize_to_abs_chunk(refined, proprio)
    return np.asarray(refined, dtype=np.float32)
