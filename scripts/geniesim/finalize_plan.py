#!/usr/bin/env python3
"""Create the reproducibility summary and a reviewable source diff."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.geniesim.plan_utils import atomic_write_json  # noqa: E402
from scripts.geniesim.plan_utils import init_wandb  # noqa: E402
from scripts.geniesim.plan_utils import log_artifact  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-root", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _num(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "NaN"
    return f"{number:.{digits}g}"


def _bool_text(value: bool, /) -> str:
    return "yes" if value else "no"


def _wandb_url(payload: dict[str, Any]) -> str:
    url = str(payload.get("url") or "").strip()
    if url:
        return url
    entity = str(payload.get("entity") or "").strip()
    project = str(payload.get("project") or "").strip()
    run_id = str(payload.get("id") or "").strip()
    if entity and project and run_id:
        return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
    return "n/a"


def _source_diff(path: Path) -> list[str]:
    diff = subprocess.run(["git", "diff", "--", "."], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    status = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    untracked: list[str] = []
    for line in status.splitlines():
        if line.startswith("?? "):
            candidate = REPO_ROOT / line[3:]
            if candidate.is_file() and candidate.suffix in {".py", ".sh", ".yaml", ".yml", ".md", ".toml"}:
                untracked.append(line[3:])
    sections = ["# Tracked changes\n", diff, "\n# Untracked source files\n"]
    sections.append("\n".join(untracked) + "\n")
    for relative in untracked:
        candidate = REPO_ROOT / relative
        try:
            content = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        sections.extend([f"\n# BEGIN {relative}\n", content, f"\n# END {relative}\n"])
    path.write_text("".join(sections), encoding="utf-8")
    return untracked


def _stage1_diagnosis(rows: list[dict[str, Any]]) -> tuple[bool, bool, bool, str]:
    valid = [row for row in rows if str(row.get("valid", "")).lower() == "true"]
    collapse = any(str(row.get("representation_collapse", "")).lower() == "true" for row in rows)
    if not valid:
        return collapse, False, False, "failed"
    overfit = any(_num(row, "val_mse") > max(2.0 * _num(row, "train_mse"), 1e-6) for row in valid)
    nmse = [_num(row, "val_nmse") for row in valid]
    plateau = len(nmse) >= 2 and abs(nmse[-1] - nmse[-2]) <= max(abs(nmse[-2]) * 0.03, 1e-4)
    health = "suspicious" if collapse or overfit else "healthy"
    return collapse, overfit, plateau, health


def _stage2_diagnosis(rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    valid = [row for row in rows if str(row.get("val_valid_checkpoint", "")).lower() == "true"]
    if not valid:
        return "failed", "unstable", "unstable"
    critic_diverged = any(str(row.get("val_critic_divergence", "")).lower() == "true" for row in rows)
    if critic_diverged:
        critic = "unstable"
    else:
        best_validation_row = max(valid, key=lambda row: _num(row, "val_q_vs_mc_pearson"))
        val_pearson = _num(best_validation_row, "val_q_vs_mc_pearson")
        train_pearson = _num(best_validation_row, "train_q_vs_mc_pearson")
        val_spearman = _num(best_validation_row, "val_q_vs_mc_spearman")
        if train_pearson == train_pearson and val_pearson == val_pearson and train_pearson - val_pearson > 0.25:
            critic = "overfit / weak validation generalization"
        elif not (val_spearman == val_spearman and val_spearman > 0.2):
            critic = "weak validation ranking"
        else:
            critic = "healthy"

    best_actor_row = min(valid, key=lambda row: _num(row, "val_actor_val_action_mse"))
    smoothness_ratio = _num(best_actor_row, "val_actor_step_delta_ratio")
    max_delta = _num(best_actor_row, "val_actor_delta_max")
    q_gain = abs(_num(best_actor_row, "val_q_actor_minus_ref"))
    q_ref = abs(_num(best_actor_row, "val_q_ref_mean"))
    if smoothness_ratio == smoothness_ratio and smoothness_ratio > 2.0:
        actor = "temporally rough"
    elif max_delta == max_delta and max_delta > 0.25:
        actor = "large action outliers"
    elif q_gain == q_gain and q_gain <= max(1e-3, 0.05 * q_ref):
        actor = "no measured improvement over reference"
    else:
        actor = "healthy"
    return "healthy" if critic == "healthy" and actor == "healthy" else "suspicious", critic, actor


def main() -> None:
    args = _parse_args()
    plan_root = args.plan_root.expanduser().resolve()
    analysis = plan_root / "analysis"
    stage2_analysis = plan_root / "stage2" / "analysis"
    stage1_rows = _read_csv(analysis / "stage1_checkpoint_metrics.csv")
    stage2_rows = _read_csv(stage2_analysis / "stage2_checkpoint_comparison.csv")
    stage1_split = json.loads((analysis / "stage1_episode_split.json").read_text(encoding="utf-8"))
    stage2_split = json.loads((stage2_analysis / "stage2_episode_split.json").read_text(encoding="utf-8"))
    stage1_selection = json.loads((analysis / "selected_stage1_checkpoint.json").read_text(encoding="utf-8"))
    stage2_selection = json.loads((stage2_analysis / "selected_stage2_checkpoint.json").read_text(encoding="utf-8"))
    replay_manifest = json.loads((stage2_analysis / "stage2_replay_manifest.json").read_text(encoding="utf-8"))
    stage1_config = json.loads((analysis / "stage1_training_status.json").read_text(encoding="utf-8"))
    effective_stage1_path = analysis / "stage1_training_config.json"
    effective_stage1 = json.loads(effective_stage1_path.read_text(encoding="utf-8")) if effective_stage1_path.is_file() else {}
    stage2_config = json.loads((stage2_analysis / "stage2_training_config.json").read_text(encoding="utf-8"))
    stage1_wandb = json.loads((analysis / "wandb_stage1_run.json").read_text(encoding="utf-8")) if (analysis / "wandb_stage1_run.json").is_file() else {}
    stage2_wandb = json.loads((stage2_analysis / "wandb_stage2_run.json").read_text(encoding="utf-8")) if (stage2_analysis / "wandb_stage2_run.json").is_file() else {}
    stage2_diagnostics_wandb = (
        json.loads((stage2_analysis / "wandb_stage2_diagnostics_run.json").read_text(encoding="utf-8"))
        if (stage2_analysis / "wandb_stage2_diagnostics_run.json").is_file()
        else {}
    )
    collapse, overfit, plateau, stage1_health = _stage1_diagnosis(stage1_rows)
    stage2_health, critic_health, actor_health = _stage2_diagnosis(stage2_rows)
    modified_files = _source_diff(analysis / "code_changes.diff")

    stage1_lines = []
    for row in stage1_rows:
        stage1_lines.append(
            f"{row.get('step')}: val MSE={_fmt(_num(row, 'val_mse'))}, NMSE={_fmt(_num(row, 'val_nmse'))}, "
            f"R2={_fmt(_num(row, 'val_r2'))}, cosine={_fmt(_num(row, 'val_cosine_mean'))}"
        )
    stage2_lines = []
    for row in stage2_rows:
        stage2_lines.append(
            f"{row.get('checkpoint_label')}: TD loss={_fmt(_num(row, 'val_critic_val_td_loss'))}, "
            f"Q-MC Pearson train/val={_fmt(_num(row, 'train_q_vs_mc_pearson'))}/"
            f"{_fmt(_num(row, 'val_q_vs_mc_pearson'))}, "
            f"Q-MC Spearman={_fmt(_num(row, 'val_q_vs_mc_spearman'))}, "
            f"Q success={_fmt(_num(row, 'val_q_success_mean'))}, Q failure={_fmt(_num(row, 'val_q_failure_mean'))}, "
            f"separation={_fmt(_num(row, 'val_q_success_minus_failure'))}, "
            f"Twin-Q gap={_fmt(_num(row, 'val_q1_q2_gap_mean'))}, actor MSE={_fmt(_num(row, 'val_actor_val_action_mse'))}, "
            f"actor delta p90={_fmt(_num(row, 'val_actor_delta_p90'))}, "
            f"actor/ref temporal ratio={_fmt(_num(row, 'val_actor_step_delta_ratio'))}, "
            f"stochastic/ref temporal ratio={_fmt(_num(row, 'val_actor_stochastic_step_delta_ratio'))}, "
            f"Q actor={_fmt(_num(row, 'val_q_actor_mean'))}, Q ref={_fmt(_num(row, 'val_q_ref_mean'))}"
        )

    sft_checkpoint = "/mnt/pfs/kk/kk/ckpt/geniesim3/spatialpi05"
    dataset = "/mnt/pfs/kk/kk/data/data/geniesim/stack_three_blocks"
    summary = f"""# RLT reproduction summary

=============================
STAGE 1
=============================

SFT checkpoint: `{sft_checkpoint}`
dataset: `{dataset}`
train episodes: {stage1_split['train_episode_count']}
validation episodes: {stage1_split['validation_episode_count']} (fixed seed {stage1_split['seed']})
W&B run: {_wandb_url(stage1_wandb)}
Dataset artifact: provenance manifest, fixed split, effective config, and 12 sample frames are uploaded with the Stage-1 run; raw source videos remain at the mounted dataset path and are not duplicated to W&B.

VLA: pi0.5, action horizon={effective_stage1.get('action_horizon', 50)}, model action dimension={effective_stage1.get('action_dimension', 32)}
DataConfig: `{effective_stage1.get('data_config_summary', 'see training log')}`
State normalization: `{effective_stage1.get('state_normalization', 'SFT checkpoint assets/norm_stats.json')}`
Image transforms: {effective_stage1.get('image_transforms', 'ResizeImages(224,224), frozen pi0.5 preprocessing; training augmentation enabled')}
Batch size: {effective_stage1.get('batch_size', 'n/a')}; learning-rate schedule: `{effective_stage1.get('learning_rate_schedule', 'n/a')}`

RL Token architecture: 1 token, 2 layers, input/embed dimension 2048, `rlt_alpha=0` (VLA frozen)
training steps: {stage1_config.get('num_train_steps', 20000)}

""" + "\n".join(stage1_lines) + f"""

Selected Stage-1 checkpoint: `{stage1_selection['checkpoint']}` (step {stage1_selection['step']})
Selection reason: {stage1_selection['reason']}

Representation collapse: {_bool_text(collapse)}
Overfitting: {_bool_text(overfit)}
Plateau: {_bool_text(plateau)}
Stage-1 diagnosis: **{stage1_health}**

=============================
STAGE 2
=============================

Replay episodes: {replay_manifest['episode_count']}
success = {replay_manifest['success_episode_count']}
failure = {replay_manifest['failure_episode_count']}
Replay transitions: {replay_manifest['transition_count']}
W&B run: {_wandb_url(stage2_wandb)}
Stage-2 diagnostics/selection run: {_wandb_url(stage2_diagnostics_wandb)}

Train episodes: {stage2_split['train_episode_count']} ({stage2_split['train_success_episode_count']} success + {stage2_split['train_failure_episode_count']} failure)
Validation episodes: {stage2_split['validation_episode_count']} ({stage2_split['validation_success_episode_count']} success + {stage2_split['validation_failure_episode_count']} failure)

Actor config: hidden={stage2_config['rl_config']['actor_hidden_dim']}x{stage2_config['rl_config']['actor_num_layers']}, lr={stage2_config['rl_config']['actor_lr']}, fixed_std={stage2_config['rl_config']['fixed_std']}, update period={stage2_config['rl_config']['actor_update_period']}
Critic config: hidden={stage2_config['rl_config']['critic_hidden_dim']}x{stage2_config['rl_config']['critic_num_layers']}, lr={stage2_config['rl_config']['critic_lr']}, gamma={stage2_config['rl_config']['gamma']}, target tau={stage2_config['rl_config']['target_tau']}
BC weight: {stage2_config.get('bc_weight', stage2_config['rl_config']['warmup_bc_weight'])}; Q weight: {stage2_config.get('q_weight', stage2_config['rl_config']['warmup_q_weight'])}; delta weight: {stage2_config['rl_config']['delta_weight']}; batch size: {stage2_config['batch_size']}

""" + "\n".join(stage2_lines) + f"""

Recommended Stage-2 checkpoint: `{stage2_selection['checkpoint']}`
Reason: {stage2_selection['reason']}
Stage-2 diagnosis: **{stage2_health}**

=============================
DIAGNOSIS
=============================

Stage 1: {stage1_health}
Critic: {critic_health}
Actor: {actor_health}

Recommended next action: use the selected Stage-2 checkpoint for a separately supervised policy evaluation or an explicitly attended rollout review. This run stopped after offline training and did not start a robot, online collector, or hyperparameter sweep.

Modified files recorded in `analysis/code_changes.diff`:
{chr(10).join(f'- `{item}`' for item in modified_files)}
"""
    (analysis / "final_summary.md").write_text(summary, encoding="utf-8")
    atomic_write_json(
        analysis / "finalization_manifest.json",
        {
            "stage1_health": stage1_health,
            "stage2_health": stage2_health,
            "critic_health": critic_health,
            "actor_health": actor_health,
            "selected_stage1": stage1_selection,
            "selected_stage2": stage2_selection,
            "modified_files": modified_files,
        },
    )
    try:
        run = init_wandb(
            project="rlt-geniesim-stack-three-blocks-reproduction",
            name="rlt_reproduction_final_summary",
            config={"stage1_selection": stage1_selection, "stage2_selection": stage2_selection},
            output_dir=plan_root,
        )
        log_artifact(
            run,
            name="rlt-reproduction-final-summary",
            artifact_type="report",
            files=[
                analysis / "final_summary.md",
                analysis / "code_changes.diff",
                analysis / "finalization_manifest.json",
                analysis / "stage1_training_config.json",
                analysis / "stage1_episode_split.json",
                analysis / "stage1_dataset_manifest.json",
                analysis / "stage1_dataset_samples",
                stage2_analysis / "stage2_training_config.json",
                stage2_analysis / "stage2_episode_split.json",
                stage2_analysis / "wandb_stage2_diagnostics_run.json",
            ],
            metadata={"stage1_health": stage1_health, "stage2_health": stage2_health},
        )
        run.finish()
    except Exception as exc:  # report generation must remain usable if W&B is temporarily unavailable
        (analysis / "wandb_finalization_error.txt").write_text(str(exc), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
