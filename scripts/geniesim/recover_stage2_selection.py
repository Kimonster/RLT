#!/usr/bin/env python3
"""Recover Stage-2 selection and diagnostics from completed metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.geniesim.plan_utils import atomic_write_json
from scripts.geniesim.plan_utils import init_wandb
from scripts.geniesim.plan_utils import log_artifact
from scripts.geniesim.train_stage2_plan import _select_checkpoint


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-root", type=Path, required=True)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    args = _args()
    stage2_root = args.stage2_root.expanduser().resolve()
    analysis = stage2_root / "analysis"
    rows = _read_rows(analysis / "stage2_checkpoint_comparison.csv")
    selected, reason = _select_checkpoint(rows)
    if selected is None:
        raise RuntimeError(reason)

    selection = {
        "checkpoint_label": selected["checkpoint_label"],
        "critic_updates": int(selected["critic_updates"]),
        "actor_updates": int(selected["actor_updates"]),
        "checkpoint": str(stage2_root / "checkpoints" / selected["checkpoint_label"]),
        "reason": reason,
        "recovered_from_completed_metrics": True,
    }
    atomic_write_json(analysis / "selected_stage2_checkpoint.json", selection)
    (analysis / "selected_stage2_checkpoint.txt").write_text(
        f"checkpoint_label={selection['checkpoint_label']}\n"
        f"critic_updates={selection['critic_updates']}\n"
        f"actor_updates={selection['actor_updates']}\n"
        f"checkpoint={selection['checkpoint']}\n"
        f"reason={reason}\n",
        encoding="utf-8",
    )
    (stage2_root / "FAILED_STAGE2.md").write_text(
        "# Stage-2 selection recovery\n\n"
        "The 30,000-step Actor/Critic training completed successfully. The initial "
        "selector read the unprefixed `valid_checkpoint` key while CSV evaluation "
        "metrics use `val_valid_checkpoint`; selection was rerun with the corrected "
        "parser. See `analysis/selected_stage2_checkpoint.json`.\n",
        encoding="utf-8",
    )

    run = init_wandb(
        project="rlt-geniesim-stack-three-blocks-reproduction",
        name="stage2_actor_critic_30k_diagnostics",
        config={"selection": selection, "recovered_from_completed_metrics": True},
        output_dir=stage2_root,
    )
    atomic_write_json(
        analysis / "wandb_stage2_diagnostics_run.json",
        {"id": run.id, "name": run.name, "project": run.project, "entity": run.entity, "url": run.url},
    )
    for row in rows:
        step = int(row["critic_updates"])
        metrics = {}
        for key, value in row.items():
            if key in {"checkpoint_label"}:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                metrics[key] = number
        if metrics:
            run.log(metrics, step=step)
    plot_paths = sorted(analysis.glob("*.png"))
    log_artifact(
        run,
        name="rlt-stage2-diagnostics-recovered",
        artifact_type="evaluation",
        files=[
            analysis / "stage2_training_metrics.csv",
            analysis / "stage2_checkpoint_comparison.csv",
            analysis / "selected_stage2_checkpoint.txt",
            *plot_paths,
        ],
        metadata=selection,
    )
    run.summary.update(selection)
    run.finish()
    print(json.dumps({"selection": selection, "wandb_url": run.url}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
