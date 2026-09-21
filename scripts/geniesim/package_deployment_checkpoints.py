#!/usr/bin/env python3
"""Package the selected GenieSim RLT artifacts for deployment.

The serving process restores the Stage-1 ``params`` and ``assets`` trees.  The
Stage-1 optimizer/train state is intentionally left out of this deployment
bundle; it is only needed to resume Stage-1 training.  The Stage-2 bundle keeps
the actor, critic, learner state, and replay journal so the online-RL runtime
can be started from the same point after extraction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
from typing import Any

import yaml


DEFAULT_PLAN_ROOT = Path("/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan")
DEFAULT_OUTPUT_ROOT = DEFAULT_PLAN_ROOT / "deploy"
DEFAULT_STAGE1_SELECTION = DEFAULT_PLAN_ROOT / "analysis/selected_stage1_checkpoint.json"
DEFAULT_STAGE2_SELECTION = DEFAULT_PLAN_ROOT / "stage2/analysis/selected_stage2_checkpoint.json"
DEFAULT_ONLINE_CONFIG = Path("rlt_online_rl/configs/tasks/geniesim_stack_three_blocks/online_rl.yaml")


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _link_or_copy(source: Path, target: Path) -> None:
    """Populate the staging tree without duplicating large checkpoint shards."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        target.symlink_to(os.readlink(source))
        return
    if source.is_dir():
        target.mkdir(exist_ok=True)
        for child in sorted(source.iterdir()):
            _link_or_copy(child, target / child.name)
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_pickle_with_portable_stats(source: Path, target: Path, stats_path: str) -> None:
    """Copy a Stage-2 pickle while making its embedded config relocatable."""
    with source.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary payload in {source}")
    config = payload.get("rl_config")
    if isinstance(config, dict):
        config = dict(config)
        config["action_norm_stats_path"] = stats_path
        payload = dict(payload)
        payload["rl_config"] = config
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _git_info(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "revision": run("rev-parse", "HEAD"),
        "status": run("status", "--short"),
    }


def _write_portable_online_config(
    config_source: Path,
    target: Path,
    *,
    selected_rl_config: dict[str, Any],
) -> None:
    with config_source.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping in {config_source}")

    # The source config uses the grouped experiment/runtime schema.  The file
    # lives in stage2/checkpoints/, so all paths below are relative to that
    # directory and the extracted bundle can be moved as one unit.
    experiment = config.setdefault("experiment", {})
    runtime = config.setdefault("runtime", {})
    experiment["rl"] = {**experiment.get("rl", {}), **selected_rl_config}
    experiment["rl"]["action_norm_stats_path"] = "../../stage1_rlt_checkpoint/assets/assets/norm_stats.json"
    runtime["role"] = "all"
    runtime.setdefault("actor_service", {})["snapshot_path"] = "../actor_snapshot/actor_snapshot.pkl"
    runtime.setdefault("learner_service", {})["checkpoint_dir"] = "."
    runtime.setdefault("learner_service", {})["actor_snapshot_path"] = "../actor_snapshot/actor_snapshot.pkl"
    runtime.setdefault("replay", {})["journal_path"] = "../replay/replay_journal.pkl"
    runtime.setdefault("monitoring", {})["wandb_dir"] = "../wandb"
    env_driver = runtime.setdefault("env_driver", {})
    env_driver["actor_deterministic"] = True
    env_driver["chunk_exec_horizon"] = 10

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def _iter_files(root: Path) -> list[Path]:
    return sorted(item for item in root.rglob("*") if item.is_file())


def _write_checksums(root: Path) -> Path:
    checksum_path = root / "checksums.sha256"
    lines: list[str] = []
    for path in _iter_files(root):
        if path == checksum_path:
            continue
        lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def _write_readme(root: Path, *, stage1: dict[str, Any], stage2: dict[str, Any]) -> None:
    text = textwrap.dedent(
        f"""\
        # GenieSim Stack-Three-Blocks RLT deployment bundle

        This bundle contains the selected Stage-1 RLT/VLA inference checkpoint
        and the selected Stage-2 actor/critic artifacts.

        ## Contents

        - `stage1_rlt_checkpoint/`: inference checkpoint (`params/` and
          `assets/`).  The Stage-1 optimizer `train_state/` is intentionally
          omitted because the serving path does not read it.
        - `stage2/actor_snapshot/actor_snapshot.pkl`: actor snapshot consumed
          by `rlt_online_rl` actor service.
        - `stage2/checkpoints/latest.pkl`: learner-compatible state restored by
          the online-RL learner service.
        - `stage2/critic_snapshot/critic_snapshot.pkl`: selected critic
          snapshot for diagnostics/evaluation.
        - `stage2/replay/replay_journal.pkl`: replay journal used to continue
          online training.
        - `stage2/checkpoints/online_rl_config.yaml`: portable config with
          paths relative to this bundle.
        - `checksums.sha256`: per-file checksums after extraction.

        Selected checkpoints:

        - Stage-1 step `{stage1.get("step")}`
        - Stage-2 `{stage2.get("checkpoint_label")}`

        ## Start the RLT feature/reference server

        Run this from the repository root with the same openpi environment used
        for training:

        ```bash
        python scripts/serve_rlt_policy.py \\
          --config rlt_pi05_geniesim_stack_three_blocks_plan \\
          --checkpoint-dir /path/to/rlt_geniesim_stack_three_blocks_deploy/stage1_rlt_checkpoint \\
          --port 8000
        ```

        The Stage-1 checkpoint already contains the VLA parameters used by the
        server, so the original pi0.5 SFT checkpoint is not required at runtime.
        The repository code, Python/JAX environment, and any robot/simulator
        process are still required.

        ## Start the Stage-2 actor service

        ```bash
        python rlt_online_rl/scripts/run_online_rl.py \\
          --config /path/to/rlt_geniesim_stack_three_blocks_deploy/stage2/checkpoints/online_rl_config.yaml \\
          --system.role actor_service
        ```

        For learner/replay continuation, start the replay manager and learner
        with the same config.  The config points to the bundled `latest.pkl`
        and replay journal.

        ## Validation note

        The selected actor is the {stage2.get("checkpoint_label")} checkpoint.  The accompanying
        analysis found weak Stage-2 critic generalization and no offline evidence
        that the actor improves on the VLA reference.  The bundled deployment
        config therefore uses deterministic actor inference and replans after
        10 actions instead of executing the full 50-action open-loop chunk. Do not enable
        stochastic sampling on a robot: `fixed_std=0.05` is independent at every
        action and timestep and produces visibly rough 50-step chunks.
        """
    )
    (root / "README.md").write_text(text, encoding="utf-8")


def _build_staging(
    *,
    stage_root: Path,
    repo_root: Path,
    stage1_selection_path: Path,
    stage2_selection_path: Path,
) -> dict[str, Any]:
    stage1_selection = _json(stage1_selection_path)
    stage2_selection = _json(stage2_selection_path)
    stage1_source = Path(str(stage1_selection["checkpoint"])).expanduser().resolve()
    stage2_source = Path(str(stage2_selection["checkpoint"])).expanduser().resolve()
    if not (stage1_source / "params").is_dir() or not (stage1_source / "assets").is_dir():
        raise FileNotFoundError(f"Stage-1 serving trees are missing under {stage1_source}")
    for required in ("actor_snapshot.pkl", "critic_snapshot.pkl", "checkpoint.pkl", "checkpoint.json"):
        if not (stage2_source / required).is_file():
            raise FileNotFoundError(f"Stage-2 artifact is missing: {stage2_source / required}")
    with (stage2_source / "actor_snapshot.pkl").open("rb") as stream:
        actor_snapshot = pickle.load(stream)
    selected_rl_config = actor_snapshot.get("rl_config") if isinstance(actor_snapshot, dict) else None
    if not isinstance(selected_rl_config, dict):
        raise TypeError(f"Actor snapshot does not contain rl_config: {stage2_source / 'actor_snapshot.pkl'}")

    stage_root.mkdir(parents=True, exist_ok=False)
    stage1_target = stage_root / "stage1_rlt_checkpoint"
    for name in ("_CHECKPOINT_METADATA", "params", "assets"):
        _link_or_copy(stage1_source / name, stage1_target / name)

    stage2_target = stage_root / "stage2"
    selected_target = stage2_target / "selected_checkpoint" / str(stage2_selection["checkpoint_label"])
    selected_stats_path = "../../../stage1_rlt_checkpoint/assets/assets/norm_stats.json"
    for name in ("actor_snapshot.pkl", "critic_snapshot.pkl", "checkpoint.pkl"):
        _copy_pickle_with_portable_stats(stage2_source / name, selected_target / name, selected_stats_path)
    _link_or_copy(stage2_source / "checkpoint.json", selected_target / "checkpoint.json")

    # Conventional paths consumed by the online-RL runtime.
    service_stats_path = "../../stage1_rlt_checkpoint/assets/assets/norm_stats.json"
    _copy_pickle_with_portable_stats(
        stage2_source / "actor_snapshot.pkl",
        stage2_target / "actor_snapshot/actor_snapshot.pkl",
        service_stats_path,
    )
    _copy_pickle_with_portable_stats(
        stage2_source / "critic_snapshot.pkl",
        stage2_target / "critic_snapshot/critic_snapshot.pkl",
        service_stats_path,
    )
    _copy_pickle_with_portable_stats(
        stage2_source / "checkpoint.pkl",
        stage2_target / "checkpoints/latest.pkl",
        service_stats_path,
    )
    _copy_pickle_with_portable_stats(
        stage2_source / "checkpoint.pkl",
        stage2_target / f"checkpoints/step_{int(stage2_selection['critic_updates'])}.pkl",
        service_stats_path,
    )

    replay_manifest_source = stage2_selection_path.parent / "stage2_replay_manifest.json"
    replay_source = stage2_source.parent.parent / "replay_source/replay/replay_journal.pkl"
    if replay_manifest_source.is_file():
        replay_manifest = _json(replay_manifest_source)
        recorded_replay = replay_manifest.get("replay_path")
        if recorded_replay:
            replay_source = Path(str(recorded_replay)).expanduser().resolve()
    if replay_source.is_file():
        _link_or_copy(replay_source, stage2_target / "replay/replay_journal.pkl")
    if replay_manifest_source.is_file():
        _copy_file(replay_manifest_source, stage2_target / "replay/replay_manifest.json")

    config_source = repo_root / DEFAULT_ONLINE_CONFIG
    if not config_source.is_file():
        raise FileNotFoundError(f"Online-RL source config is missing: {config_source}")
    _write_portable_online_config(
        config_source,
        stage2_target / "checkpoints/online_rl_config.yaml",
        selected_rl_config=selected_rl_config,
    )

    metadata_target = stage_root / "metadata"
    metadata_target.mkdir(parents=True, exist_ok=True)
    for source in (
        stage1_selection_path,
        stage2_selection_path,
        stage1_selection_path.parent / "stage1_training_config.json",
        stage1_selection_path.parent / "final_summary.md",
        stage1_selection_path.parent / "pipeline_completion.json",
        stage2_selection_path.parent / "stage2_training_config.json",
        stage2_selection_path.parent / "stage2_training_status.json",
        stage2_selection_path.parent / "stage2_replay_manifest.json",
    ):
        if source.is_file():
            _copy_file(source, metadata_target / source.name)

    stage1_size = _size(stage1_target)
    stage2_size = _size(stage2_target)
    manifest: dict[str, Any] = {
        "format_version": 1,
        "bundle_type": "rlt_geniesim_stack_three_blocks_deployment",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage1": {
            "step": int(stage1_selection["step"]),
            "source_checkpoint": str(stage1_source),
            "bundle_path": "stage1_rlt_checkpoint",
            "included": ["_CHECKPOINT_METADATA", "params", "assets"],
            "omitted": ["train_state (training-only)"],
            "size_bytes": stage1_size,
            "selection": stage1_selection,
        },
        "stage2": {
            "checkpoint_label": str(stage2_selection["checkpoint_label"]),
            "source_checkpoint": str(stage2_source),
            "bundle_path": "stage2",
            "included": [
                "selected_checkpoint/",
                "actor_snapshot/",
                "critic_snapshot/",
                "checkpoints/latest.pkl",
                f"checkpoints/step_{int(stage2_selection['critic_updates'])}.pkl",
                "checkpoints/online_rl_config.yaml",
                "replay/",
            ],
            "size_bytes": stage2_size,
            "selection": stage2_selection,
        },
        "base_pi05_sft_checkpoint": {
            "source_path": "/mnt/pfs/kk/kk/ckpt/geniesim3/spatialpi05",
            "included": False,
            "reason": "Stage-1 params contains the VLA plus RLT parameters required by serve_rlt_policy.py.",
        },
        "runtime": {
            "rlt_config": "rlt_pi05_geniesim_stack_three_blocks_plan",
            "stage2_config": "stage2/checkpoints/online_rl_config.yaml",
            "repository_revision": _git_info(repo_root),
        },
    }
    (stage_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    _write_readme(stage_root, stage1=stage1_selection, stage2=stage2_selection)
    _write_checksums(stage_root)
    return manifest


def _create_archive(stage_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "tar",
        "--sort=name",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "-C",
        str(stage_root.parent),
        "-I",
        "zstd -T0 -3",
        "-cf",
        str(archive_path),
        stage_root.name,
    ]
    subprocess.run(command, check=True)


def _verify_archive(archive_path: Path, stage_name: str) -> None:
    subprocess.run(["zstd", "-t", str(archive_path)], check=True, stdout=subprocess.DEVNULL)
    listing = subprocess.check_output(["tar", "-I", "zstd -T0 -3", "-tf", str(archive_path)], text=True)
    required = (
        f"{stage_name}/stage1_rlt_checkpoint/params/",
        f"{stage_name}/stage1_rlt_checkpoint/assets/assets/norm_stats.json",
        f"{stage_name}/stage2/actor_snapshot/actor_snapshot.pkl",
        f"{stage_name}/stage2/checkpoints/latest.pkl",
        f"{stage_name}/stage2/checkpoints/online_rl_config.yaml",
        f"{stage_name}/checksums.sha256",
    )
    missing = [item for item in required if item not in listing]
    if missing:
        raise RuntimeError(f"Archive is missing required entries: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage1-selection", type=Path, default=DEFAULT_STAGE1_SELECTION)
    parser.add_argument("--stage2-selection", type=Path, default=DEFAULT_STAGE2_SELECTION)
    parser.add_argument("--name", default=None, help="Archive stem; defaults to a UTC timestamp.")
    parser.add_argument("--keep-expanded", action="store_true", help="Keep the expanded staging directory.")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = args.name or f"rlt_geniesim_stack_three_blocks_deploy_{stamp}"
    archive_path = output_root / f"{stem}.tar.zst"
    outer_manifest_path = output_root / f"{stem}.manifest.json"
    checksum_path = output_root / f"{archive_path.name}.sha256"
    if archive_path.exists():
        raise FileExistsError(f"Archive already exists: {archive_path}")

    stage_parent = Path(tempfile.mkdtemp(prefix=f".{stem}.stage-", dir=output_root))
    stage_root = stage_parent / stem
    try:
        manifest = _build_staging(
            stage_root=stage_root,
            repo_root=args.repo_root.expanduser().resolve(),
            stage1_selection_path=args.stage1_selection.expanduser().resolve(),
            stage2_selection_path=args.stage2_selection.expanduser().resolve(),
        )
        _create_archive(stage_root, archive_path)
        _verify_archive(archive_path, stem)
        archive_sha = _sha256(archive_path)
        archive_size = archive_path.stat().st_size
        checksum_path.write_text(f"{archive_sha}  {archive_path.name}\n", encoding="utf-8")
        outer_manifest = {
            **manifest,
            "archive": {
                "filename": archive_path.name,
                "size_bytes": archive_size,
                "sha256": archive_sha,
                "sha256_file": checksum_path.name,
            },
        }
        outer_manifest_path.write_text(json.dumps(outer_manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(json.dumps({"archive": str(archive_path), "size_bytes": archive_size, "sha256": archive_sha}, indent=2))
    finally:
        if not args.keep_expanded:
            shutil.rmtree(stage_parent, ignore_errors=True)


if __name__ == "__main__":
    main()
