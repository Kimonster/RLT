#!/usr/bin/env python3
"""Repair effective RL weights embedded in completed Stage-2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_pickle(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--warmup-bc-weight", type=float, required=True)
    parser.add_argument("--warmup-q-weight", type=float, required=True)
    parser.add_argument("--online-bc-weight", type=float, required=True)
    parser.add_argument("--online-q-weight", type=float, required=True)
    parser.add_argument("--delta-weight", type=float, required=True)
    args = parser.parse_args()

    stage2_root = args.stage2_root.expanduser().resolve()
    overrides = {
        "warmup_bc_weight": args.warmup_bc_weight,
        "warmup_q_weight": args.warmup_q_weight,
        "online_bc_weight": args.online_bc_weight,
        "online_q_weight": args.online_q_weight,
        "delta_weight": args.delta_weight,
    }
    records: list[dict[str, Any]] = []
    for path in sorted((stage2_root / "checkpoints").glob("*/*.pkl")):
        before_sha256 = _sha256(path)
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or not isinstance(payload.get("rl_config"), dict):
            raise TypeError(f"Missing rl_config in {path}")
        before = {key: payload["rl_config"].get(key) for key in overrides}
        payload["rl_config"] = {**payload["rl_config"], **overrides}
        _atomic_pickle(path, payload)
        records.append(
            {
                "path": str(path),
                "before_sha256": before_sha256,
                "after_sha256": _sha256(path),
                "before": before,
                "after": overrides,
            }
        )

    training_config_path = stage2_root / "analysis/stage2_training_config.json"
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    training_config["rl_config"] = {**training_config["rl_config"], **overrides}
    _atomic_json(training_config_path, training_config)
    report = {
        "schema_version": 1,
        "reason": "Persist the effective CLI loss weights used by the completed run for deployment and continuation.",
        "overrides": overrides,
        "files_rewritten": records,
    }
    _atomic_json(stage2_root / "analysis/checkpoint_config_repair.json", report)
    print(json.dumps({"files_rewritten": len(records), "overrides": overrides}, indent=2))


if __name__ == "__main__":
    main()
