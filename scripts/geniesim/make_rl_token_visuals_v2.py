#!/usr/bin/env python3
"""Create episode-balanced RL-token visualizations from completed analysis features."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/rlt_token_analysis_endpoint_v3"
)
DEFAULT_STEPS = (10_000, 30_000, 50_000, 70_000, 90_000, 110_000, 130_000, 150_000, 170_000)
SUCCESS_COLOR = "#0072B2"
FAILURE_COLOR = "#D55E00"
PAIR_COLOR = "#8A8A8A"
GRID_COLOR = "#D8D8D8"
TIME_COLORS = mpl.colormaps["viridis"](np.linspace(0.08, 0.92, 5))
TIME_BIN_COUNT = 5
BOOTSTRAP_SAMPLES = 4_000
T4_BIN = 3


@dataclass(frozen=True)
class CheckpointFeatures:
    step: int
    z: np.ndarray
    pca: np.ndarray
    tsne: np.ndarray
    episode_id: np.ndarray
    pair_id: np.ndarray
    frame_order: np.ndarray
    time_bin: np.ndarray
    success: np.ndarray


@dataclass(frozen=True)
class EpisodeSummary:
    episode_ids: np.ndarray
    pair_ids: np.ndarray
    success: np.ndarray
    z: np.ndarray
    coords: np.ndarray


def _step_label(step: int) -> str:
    return f"{step // 1000}k"


def _load_checkpoint(output_root: Path, step: int) -> CheckpointFeatures:
    path = output_root / f"ckpt_{_step_label(step)}" / "features.npz"
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "z",
            "pca",
            "tsne",
            "episode_id",
            "pair_id",
            "frame_order",
            "time_bin",
            "success",
        }
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
        feature = CheckpointFeatures(
            step=step,
            z=payload["z"].astype(np.float32),
            pca=payload["pca"].astype(np.float64),
            tsne=payload["tsne"].astype(np.float64),
            episode_id=payload["episode_id"],
            pair_id=payload["pair_id"],
            frame_order=payload["frame_order"].astype(np.int64),
            time_bin=payload["time_bin"].astype(np.int8),
            success=payload["success"].astype(np.int8),
        )
    lengths = {
        len(feature.z),
        len(feature.pca),
        len(feature.tsne),
        len(feature.episode_id),
        len(feature.pair_id),
        len(feature.frame_order),
        len(feature.time_bin),
        len(feature.success),
    }
    if len(lengths) != 1:
        raise ValueError(f"Checkpoint {step} has inconsistent sample counts: {sorted(lengths)}")
    if not all(np.isfinite(value).all() for value in (feature.z, feature.pca, feature.tsne)):
        raise ValueError(f"Checkpoint {step} contains non-finite features")
    return feature


def _episode_summary(feature: CheckpointFeatures, coords: np.ndarray) -> EpisodeSummary:
    episodes = np.unique(feature.episode_id)
    episode_z: list[np.ndarray] = []
    episode_coords: list[np.ndarray] = []
    outcomes: list[int] = []
    pairs: list[str] = []
    for episode in episodes:
        rows = feature.episode_id == episode
        episode_z.append(feature.z[rows].mean(axis=0))
        episode_coords.append(coords[rows].mean(axis=0))
        outcome_values = np.unique(feature.success[rows])
        pair_values = np.unique(feature.pair_id[rows])
        if len(outcome_values) != 1 or len(pair_values) != 1:
            raise ValueError(f"Episode {episode!r} has inconsistent outcome/pair metadata")
        outcomes.append(int(outcome_values[0]))
        pairs.append(str(pair_values[0]))
    return EpisodeSummary(
        episode_ids=episodes,
        pair_ids=np.asarray(pairs),
        success=np.asarray(outcomes, dtype=np.int8),
        z=np.stack(episode_z),
        coords=np.stack(episode_coords),
    )


def _trajectory_signatures(feature: CheckpointFeatures) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_ids = np.unique(feature.episode_id)
    signatures: list[np.ndarray] = []
    outcomes: list[int] = []
    for episode in episode_ids:
        episode_rows = feature.episode_id == episode
        outcome_values = np.unique(feature.success[episode_rows])
        if len(outcome_values) != 1:
            raise ValueError(f"Episode {episode!r} has inconsistent outcomes")
        bins: list[np.ndarray] = []
        for time_bin in range(TIME_BIN_COUNT):
            rows = episode_rows & (feature.time_bin == time_bin)
            if not np.any(rows):
                raise ValueError(f"Episode {episode!r} has no samples in time bin {time_bin}")
            bins.append(feature.z[rows].mean(axis=0, dtype=np.float64))
        signatures.append(np.concatenate(bins))
        outcomes.append(int(outcome_values[0]))
    return episode_ids, np.asarray(outcomes, dtype=np.int8), np.stack(signatures)


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    squared_norms = np.einsum("ij,ij->i", values, values)
    squared = squared_norms[:, None] + squared_norms[None, :] - 2.0 * values @ values.T
    return np.sqrt(np.maximum(squared, 0.0))


def _pam_medoids(distances: np.ndarray, count: int) -> tuple[list[int], list[int]]:
    sample_count = len(distances)
    if distances.shape != (sample_count, sample_count):
        raise ValueError("PAM requires a square distance matrix")
    if not 0 < count <= sample_count:
        raise ValueError(f"Invalid medoid count {count} for {sample_count} samples")

    medoids: list[int] = []
    while len(medoids) < count:
        candidates: list[tuple[float, int]] = []
        for candidate in range(sample_count):
            if candidate in medoids:
                continue
            proposal = [*medoids, candidate]
            cost = float(np.min(distances[:, proposal], axis=1).sum())
            candidates.append((cost, candidate))
        medoids.append(min(candidates)[1])
        medoids.sort()

    current_cost = float(np.min(distances[:, medoids], axis=1).sum())
    while True:
        best_cost = current_cost
        best_medoids = medoids
        non_medoids = [index for index in range(sample_count) if index not in medoids]
        for old in medoids:
            for new in non_medoids:
                proposal = sorted([index for index in medoids if index != old] + [new])
                cost = float(np.min(distances[:, proposal], axis=1).sum())
                if cost < best_cost - 1e-10 or (
                    abs(cost - best_cost) <= 1e-10 and tuple(proposal) < tuple(best_medoids)
                ):
                    best_cost = cost
                    best_medoids = proposal
        if best_medoids == medoids:
            break
        medoids = best_medoids
        current_cost = best_cost

    assignments = np.argmin(distances[:, medoids], axis=1)
    cluster_sizes = [int(np.sum(assignments == cluster)) for cluster in range(count)]
    return medoids, cluster_sizes


def _select_fixed_representatives(
    features: list[CheckpointFeatures], count: int
) -> tuple[dict[int, list[str]], dict[str, Any]]:
    reference_ids, reference_outcomes, _ = _trajectory_signatures(features[0])
    distance_sums = {
        outcome: np.zeros((int(np.sum(reference_outcomes == outcome)),) * 2, dtype=np.float64) for outcome in (0, 1)
    }
    for feature in features:
        episode_ids, outcomes, signatures = _trajectory_signatures(feature)
        if not np.array_equal(episode_ids, reference_ids) or not np.array_equal(outcomes, reference_outcomes):
            raise ValueError(f"Episode metadata differs at checkpoint {feature.step}")
        for outcome in (0, 1):
            values = signatures[outcomes == outcome]
            distances = _pairwise_distances(values)
            off_diagonal = distances[np.triu_indices(len(distances), k=1)]
            scale = float(np.median(off_diagonal))
            if scale <= 0:
                raise ValueError(f"Degenerate trajectory distances at checkpoint {feature.step}")
            distance_sums[outcome] += distances / scale

    selected: dict[int, list[str]] = {}
    details: dict[str, Any] = {}
    for outcome, name in ((0, "failure"), (1, "success")):
        outcome_ids = reference_ids[reference_outcomes == outcome]
        average_distances = distance_sums[outcome] / len(features)
        medoids, cluster_sizes = _pam_medoids(average_distances, count)
        selected[outcome] = [str(outcome_ids[index]) for index in medoids]
        details[name] = {
            "episode_ids": selected[outcome],
            "cluster_sizes": cluster_sizes,
        }
    return selected, details


def _axis_limits(values: np.ndarray, padding: float = 0.07) -> tuple[tuple[float, float], tuple[float, float]]:
    low = np.min(values, axis=0)
    high = np.max(values, axis=0)
    span = np.maximum(high - low, 1e-6)
    return (
        (float(low[0] - span[0] * padding), float(high[0] + span[0] * padding)),
        (float(low[1] - span[1] * padding), float(high[1] + span[1] * padding)),
    )


def _style_axis(axis: plt.Axes, x_label: str, y_label: str) -> None:
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.55)
    axis.set_axisbelow(True)


def _add_confidence_ellipse(
    axis: plt.Axes,
    bootstrap_means: np.ndarray,
    color: str,
    *,
    center: np.ndarray,
    probability: float = 0.95,
    face_alpha: float = 0.12,
    edge_alpha: float = 0.72,
    linewidth: float = 1.6,
) -> None:
    if len(bootstrap_means) < 3:
        return
    covariance = np.cov(bootstrap_means, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    radius = math.sqrt(-2.0 * math.log(1.0 - probability))
    width, height = 2.0 * radius * np.sqrt(eigenvalues)
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    axis.add_patch(
        Ellipse(
            xy=center,
            width=float(width),
            height=float(height),
            angle=angle,
            facecolor=mpl.colors.to_rgba(color, face_alpha),
            edgecolor=mpl.colors.to_rgba(color, edge_alpha),
            linewidth=linewidth,
            zorder=2,
        )
    )


def _draw_direction_arrow(
    axis: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    alpha: float = 1.0,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5, "alpha": alpha, "mutation_scale": 10},
        zorder=5,
    )


def _paired_bootstrap_outcome_means(
    summary: EpisodeSummary,
    *,
    seed: int,
) -> dict[int, np.ndarray]:
    failure_points: list[np.ndarray] = []
    success_points: list[np.ndarray] = []
    for pair in np.unique(summary.pair_ids):
        rows = np.flatnonzero(summary.pair_ids == pair)
        failure = rows[summary.success[rows] == 0]
        success = rows[summary.success[rows] == 1]
        if len(failure) != 1 or len(success) != 1:
            raise ValueError(f"Pair {pair!r} is not one failure plus one success")
        failure_points.append(summary.coords[failure[0]])
        success_points.append(summary.coords[success[0]])
    failure_array = np.stack(failure_points)
    success_array = np.stack(success_points)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(failure_array), size=(BOOTSTRAP_SAMPLES, len(failure_array)))
    return {
        0: failure_array[draws].mean(axis=1),
        1: success_array[draws].mean(axis=1),
    }


def _plot_outcome_panel(
    axis: plt.Axes,
    feature: CheckpointFeatures,
    coords: np.ndarray,
    summary: EpisodeSummary,
) -> None:
    bootstrap_means = _paired_bootstrap_outcome_means(summary, seed=feature.step + 101)
    for outcome, color in ((0, FAILURE_COLOR), (1, SUCCESS_COLOR)):
        rows = feature.success == outcome
        axis.scatter(
            coords[rows, 0],
            coords[rows, 1],
            s=7,
            color=color,
            alpha=0.055,
            linewidths=0,
            zorder=0,
        )
    for pair in np.unique(summary.pair_ids):
        rows = np.flatnonzero(summary.pair_ids == pair)
        failure = rows[summary.success[rows] == 0]
        success = rows[summary.success[rows] == 1]
        if len(failure) == 1 and len(success) == 1:
            pair_coords = summary.coords[[failure[0], success[0]]]
            axis.plot(
                pair_coords[:, 0],
                pair_coords[:, 1],
                color=PAIR_COLOR,
                alpha=0.22,
                linewidth=0.7,
                zorder=1,
            )
    for outcome, color, marker, label in (
        (0, FAILURE_COLOR, "X", "failure episode"),
        (1, SUCCESS_COLOR, "o", "success episode"),
    ):
        points = summary.coords[summary.success == outcome]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=43,
            color=color,
            marker=marker,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.5,
            label=label,
            zorder=3,
        )
        mean = points.mean(axis=0)
        _add_confidence_ellipse(
            axis,
            bootstrap_means[outcome],
            color,
            center=mean,
        )
        axis.scatter(
            mean[0],
            mean[1],
            s=155,
            color=color,
            marker="*",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
    failure_mean = summary.coords[summary.success == 0].mean(axis=0)
    success_mean = summary.coords[summary.success == 1].mean(axis=0)
    _draw_direction_arrow(axis, failure_mean, success_mean, "#222222")
    axis.set_title("Episode-level outcome")
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor=FAILURE_COLOR,
                markeredgecolor="white",
                label="failure episode",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=SUCCESS_COLOR,
                markeredgecolor="white",
                label="success episode",
            ),
            Line2D([0], [0], color=PAIR_COLOR, alpha=0.5, label="matched instance"),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="#555555",
                label="group mean + 95% CI",
            ),
        ],
        frameon=False,
        fontsize=7,
        loc="best",
    )


def _centered_episode_bin_paths(
    feature: CheckpointFeatures,
    coords: np.ndarray,
    outcome: int,
) -> np.ndarray:
    paths: list[np.ndarray] = []
    for episode in np.unique(feature.episode_id[feature.success == outcome]):
        episode_rows = (feature.episode_id == episode) & (feature.success == outcome)
        center = coords[episode_rows].mean(axis=0)
        bins: list[np.ndarray] = []
        for time_bin in range(TIME_BIN_COUNT):
            rows = episode_rows & (feature.time_bin == time_bin)
            if not np.any(rows):
                raise ValueError(f"Episode {episode!r} has no samples in time bin {time_bin}")
            bins.append(coords[rows].mean(axis=0) - center)
        paths.append(np.stack(bins))
    return np.stack(paths)


def _bootstrap_path_means(paths: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(paths), size=(BOOTSTRAP_SAMPLES, len(paths)))
    return paths[draws].mean(axis=1)


def _plot_time_panel(axis: plt.Axes, feature: CheckpointFeatures, coords: np.ndarray) -> None:
    bootstrap_clouds: list[np.ndarray] = []
    for outcome, color, marker, linestyle in (
        (0, FAILURE_COLOR, "X", "--"),
        (1, SUCCESS_COLOR, "o", "-"),
    ):
        episode_paths = _centered_episode_bin_paths(feature, coords, outcome)
        path = episode_paths.mean(axis=0)
        bootstrap_paths = _bootstrap_path_means(
            episode_paths,
            seed=feature.step + 401 + outcome,
        )
        bootstrap_clouds.append(bootstrap_paths.reshape(-1, 2))
        axis.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            alpha=0.94,
            zorder=3,
        )
        for time_bin, point in enumerate(path):
            _add_confidence_ellipse(
                axis,
                bootstrap_paths[:, time_bin],
                color,
                center=point,
                face_alpha=0.07,
                edge_alpha=0.38,
                linewidth=1.0,
            )
            axis.scatter(
                point[0],
                point[1],
                s=73,
                color=TIME_COLORS[time_bin],
                marker=marker,
                edgecolors=color,
                linewidths=1.3,
                zorder=4,
            )
        for left in range(len(path) - 1):
            _draw_direction_arrow(axis, path[left], path[left + 1], color, alpha=0.85)
    confidence_region = np.concatenate([*bootstrap_clouds, np.zeros((1, 2))], axis=0)
    x_limits, y_limits = _axis_limits(confidence_region, padding=0.08)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.axhline(0.0, color="#BBBBBB", linewidth=0.7, zorder=-1)
    axis.axvline(0.0, color="#BBBBBB", linewidth=0.7, zorder=-1)
    axis.set_title("Within-episode time displacement")
    outcome_legend = axis.legend(
        handles=[
            Line2D([0], [0], color=FAILURE_COLOR, linestyle="--", marker="X", label="failure"),
            Line2D([0], [0], color=SUCCESS_COLOR, linestyle="-", marker="o", label="success"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )
    axis.add_artist(outcome_legend)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=TIME_COLORS[time_bin],
                markeredgecolor="none",
                label=f"T{time_bin + 1}",
            )
            for time_bin in range(TIME_BIN_COUNT)
        ],
        frameon=False,
        fontsize=7,
        ncol=3,
        loc="lower left",
        title="time bin",
        title_fontsize=7,
    )


def _short_episode_id(episode: str) -> str:
    category, trajectory = episode.split("/", maxsplit=1)
    prefix = "S" if category == "successful" else "F"
    return f"{prefix}{int(trajectory.rsplit('_', maxsplit=1)[1]):03d}"


def _plot_representative_panel(
    axis: plt.Axes,
    feature: CheckpointFeatures,
    coords: np.ndarray,
    representatives: dict[int, list[str]],
    outcome: int,
) -> None:
    color = FAILURE_COLOR if outcome == 0 else SUCCESS_COLOR
    linestyle = "--" if outcome == 0 else "-"
    for representative_index, episode in enumerate(representatives[outcome]):
        rows = np.flatnonzero(feature.episode_id == episode)
        rows = rows[np.argsort(feature.frame_order[rows])]
        values = coords[rows]
        alpha = max(0.55, 1.0 - 0.20 * representative_index)
        width = max(1.2, 2.0 - 0.35 * representative_index)
        label = _short_episode_id(episode)
        axis.plot(
            values[:, 0],
            values[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=width,
            alpha=alpha,
            marker=".",
            markersize=3.5,
            label=label,
            zorder=2,
        )
        axis.scatter(
            values[0, 0],
            values[0, 1],
            s=55,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=1.3,
            zorder=4,
        )
        axis.scatter(
            values[-1, 0],
            values[-1, 1],
            s=65,
            marker="^",
            color=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )
        _draw_direction_arrow(axis, values[-2], values[-1], color, alpha=alpha)
    outcome_name = "Failure" if outcome == 0 else "Success"
    count = len(representatives[outcome])
    axis.set_title(f"{outcome_name} representatives ({count})")
    axis.legend(
        frameon=False,
        fontsize=7,
        ncol=1,
        loc="best",
        title="o start / ^ end",
        title_fontsize=7,
    )


def _plot_tsne_outcome_panel(
    axis: plt.Axes,
    feature: CheckpointFeatures,
    coords: np.ndarray,
) -> None:
    for outcome, color, marker, label in (
        (0, FAILURE_COLOR, "x", "failure frame"),
        (1, SUCCESS_COLOR, "o", "success frame"),
    ):
        rows = feature.success == outcome
        axis.scatter(
            coords[rows, 0],
            coords[rows, 1],
            s=17 if outcome == 0 else 13,
            color=color,
            marker=marker,
            alpha=0.50 if outcome == 0 else 0.40,
            linewidths=0.65 if outcome == 0 else 0,
            label=label,
            zorder=2 if outcome == 0 else 1,
        )
    axis.set_title("Outcome in local neighborhoods")
    axis.legend(frameon=False, fontsize=8, loc="best")


def _plot_tsne_time_panel(
    axis: plt.Axes,
    feature: CheckpointFeatures,
    coords: np.ndarray,
) -> None:
    for time_bin in range(TIME_BIN_COUNT):
        for outcome, marker in ((0, "x"), (1, "o")):
            rows = (feature.time_bin == time_bin) & (feature.success == outcome)
            axis.scatter(
                coords[rows, 0],
                coords[rows, 1],
                s=15 if outcome == 0 else 12,
                color=TIME_COLORS[time_bin],
                marker=marker,
                alpha=0.52 if outcome == 0 else 0.38,
                linewidths=0.65 if outcome == 0 else 0,
                zorder=2 if outcome == 0 else 1,
            )
    time_legend = axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=TIME_COLORS[time_bin],
                markeredgecolor="none",
                label=f"T{time_bin + 1}",
            )
            for time_bin in range(TIME_BIN_COUNT)
        ],
        frameon=False,
        fontsize=7,
        ncol=3,
        loc="best",
        title="time bin | o success / x failure",
        title_fontsize=7,
    )
    axis.add_artist(time_legend)
    axis.set_title("Time bins in local neighborhoods")


def _plot_checkpoint_embedding(
    feature: CheckpointFeatures,
    coords: np.ndarray,
    method: str,
    output_path: Path,
    representatives: dict[int, list[str]],
    pca_variance: float | None = None,
) -> None:
    figure_height = 10.4 if method == "PCA" else 7.4
    fig, axes_grid = plt.subplots(
        2,
        2,
        figsize=(12.8, figure_height),
        constrained_layout=True,
    )
    axes = axes_grid.ravel()
    x_limits, y_limits = _axis_limits(coords)
    if method == "PCA":
        summary = _episode_summary(feature, coords)
        _plot_outcome_panel(axes[0], feature, coords, summary)
        _plot_time_panel(axes[1], feature, coords)
        _plot_representative_panel(axes[2], feature, coords, representatives, outcome=0)
        _plot_representative_panel(axes[3], feature, coords, representatives, outcome=1)
        for axis in (axes[0], axes[2], axes[3]):
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            _style_axis(axis, "PC 1", "PC 2")
        _style_axis(axes[1], "delta PC 1", "delta PC 2")
        for axis in axes:
            axis.set_aspect("equal", adjustable="box")
        footer = (
            "Outcome means: paired-episode bootstrap 95% CI; gray links match instance, not seed. "
            "Time paths: episode-centered, episode-balanced means."
        )
    else:
        _plot_tsne_outcome_panel(axes[0], feature, coords)
        _plot_tsne_time_panel(axes[1], feature, coords)
        _plot_representative_panel(axes[2], feature, coords, representatives, outcome=0)
        _plot_representative_panel(axes[3], feature, coords, representatives, outcome=1)
        for axis in axes:
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_aspect("equal", adjustable="box")
            _style_axis(axis, "t-SNE 1", "t-SNE 2")
        footer = (
            "t-SNE is local-neighborhood evidence only; centroids and ellipses are intentionally omitted. "
            "Highlighted trajectory lengths are exploratory, not metric evidence."
        )
    variance_text = f" | PC1+PC2 variance={pca_variance:.1%}" if pca_variance is not None else ""
    fig.suptitle(f"Checkpoint {_step_label(feature.step)} | {method} v2{variance_text}", fontsize=14)
    fig.supxlabel(footer, fontsize=8, color="#555555")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_t4_tsne_diagnostic(
    feature: CheckpointFeatures,
    output_path: Path,
    *,
    scaled_gap: float,
    debiased_gap_squared: float,
) -> None:
    selected = feature.time_bin == T4_BIN
    selected_count = int(selected.sum())
    if selected_count <= 30:
        raise ValueError(f"T4 contains too few samples for perplexity 30: {selected_count}")
    t4_coords = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        perplexity=30,
        max_iter=1000,
    ).fit_transform(feature.z[selected])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.24, top=0.82, wspace=0.18)
    axes[0].scatter(
        feature.tsne[~selected, 0],
        feature.tsne[~selected, 1],
        s=8,
        color="#9AA0A6",
        alpha=0.12,
        linewidths=0,
        label="other time bins",
        zorder=0,
    )
    for axis, coords, outcomes, title in (
        (
            axes[0],
            feature.tsne[selected],
            feature.success[selected],
            "T4 highlighted in the all-sample t-SNE",
        ),
        (
            axes[1],
            t4_coords,
            feature.success[selected],
            "t-SNE refit on T4 only",
        ),
    ):
        for outcome, color, marker, label in (
            (0, FAILURE_COLOR, "x", "failure frame"),
            (1, SUCCESS_COLOR, "o", "success frame"),
        ):
            rows = outcomes == outcome
            axis.scatter(
                coords[rows, 0],
                coords[rows, 1],
                s=30 if outcome == 0 else 24,
                color=color,
                marker=marker,
                alpha=0.68 if outcome == 0 else 0.58,
                linewidths=0.9 if outcome == 0 else 0,
                label=label,
                zorder=2 if outcome == 0 else 1,
            )
        axis.set_title(title)
        _style_axis(axis, "t-SNE 1", "t-SNE 2")
        axis.legend(frameon=False, fontsize=8, loc="best")
        axis.set_aspect("equal", adjustable="box")

    episode_count = len(np.unique(feature.episode_id[selected]))
    success_count = int(feature.success[selected].sum())
    fig.suptitle(
        f"Checkpoint {_step_label(feature.step)} | T4 normalized time [0.6, 0.8)",
        fontsize=15,
        y=0.96,
    )
    fig.text(
        0.5,
        0.135,
        f"T4: {selected_count} frames from {episode_count} episodes "
        f"({success_count} success / {selected_count - success_count} failure frames). "
        f"High-dimensional episode-balanced scaled gap={scaled_gap:.3f}; "
        f"debiased gap squared={debiased_gap_squared:.3f}.",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.text(
        0.5,
        0.075,
        "T4 is the point-estimate peak in 2048-D; paired bootstrap uncertainty does not establish T4 > T5.",
        ha="center",
        fontsize=8,
        color="#666666",
    )
    fig.text(
        0.5,
        0.035,
        "Both t-SNE panels show local neighborhoods only; global spacing and apparent islands are not effect sizes.",
        ha="center",
        fontsize=8,
        color="#666666",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _paired_pc_effects(feature: CheckpointFeatures, scores: np.ndarray) -> np.ndarray:
    summary = _episode_summary(feature, scores)
    differences: list[np.ndarray] = []
    for pair in np.unique(summary.pair_ids):
        rows = np.flatnonzero(summary.pair_ids == pair)
        failure = rows[summary.success[rows] == 0]
        success = rows[summary.success[rows] == 1]
        if len(failure) != 1 or len(success) != 1:
            raise ValueError(f"Pair {pair!r} does not contain one episode per outcome")
        differences.append(summary.coords[success[0]] - summary.coords[failure[0]])
    paired_differences = np.stack(differences)
    return np.abs(paired_differences.mean(axis=0)) / np.maximum(
        paired_differences.std(axis=0, ddof=1),
        1e-12,
    )


def _plot_pca_dimension_diagnostic(
    feature: CheckpointFeatures,
    output_path: Path,
) -> dict[str, Any]:
    component_count = min(50, len(feature.z) - 1, feature.z.shape[1])
    pca = PCA(n_components=component_count, svd_solver="randomized", random_state=42)
    scores = pca.fit_transform(feature.z)
    paired_effects = _paired_pc_effects(feature, scores)
    candidate_count = min(10, component_count)
    selected = np.argsort(-paired_effects[:candidate_count])[:2]
    panels = (
        (np.asarray([0, 1]), "Highest-variance PCA pair"),
        (selected, f"Outcome-associated pair among PC1-PC{candidate_count}"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.7))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.19, top=0.84, wspace=0.18)
    for axis, (components, title) in zip(axes, panels, strict=True):
        coords = scores[:, components]
        summary = _episode_summary(feature, coords)
        _plot_outcome_panel(axis, feature, coords, summary)
        x_limits, y_limits = _axis_limits(coords)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{title}\nPC{components[0] + 1} ({pca.explained_variance_ratio_[components[0]]:.2%}) + "
            f"PC{components[1] + 1} ({pca.explained_variance_ratio_[components[1]]:.2%})"
        )
        _style_axis(axis, f"PC {components[0] + 1}", f"PC {components[1] + 1}")

    selected_variance = float(pca.explained_variance_ratio_[selected].sum())
    fig.suptitle(f"Checkpoint {_step_label(feature.step)} | PCA dimension diagnostic", fontsize=15, y=0.97)
    fig.text(
        0.5,
        0.095,
        f"PC1+PC2 explain {pca.explained_variance_ratio_[:2].sum():.2%}; "
        f"PC{selected[0] + 1}+PC{selected[1] + 1} explain {selected_variance:.2%}.",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.text(
        0.5,
        0.035,
        f"The right pair was selected from PC1-PC{candidate_count} using these outcome labels. "
        "Only PC6 survives correction across the searched dimensions; independent rollout validation is required.",
        ha="center",
        fontsize=8,
        color="#666666",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return {
        "component_count_searched": component_count,
        "selection_candidate_count": candidate_count,
        "selected_components_one_based": [int(index + 1) for index in selected],
        "selected_explained_variance_ratio": [float(pca.explained_variance_ratio_[index]) for index in selected],
        "selected_variance_ratio_sum": selected_variance,
        "selected_paired_effect_dz": [float(paired_effects[index]) for index in selected],
        "pc1_pc2_variance_ratio_sum": float(pca.explained_variance_ratio_[:2].sum()),
        "selection_warning": (
            "post hoc label-based selection among PC1-PC10; only PC6 survives multiple-comparison correction"
        ),
        "path": str(output_path),
    }


def _aggregate_episode_scores(
    scores: np.ndarray,
    episode_ids: np.ndarray,
    pair_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episodes = np.unique(episode_ids)
    episode_scores = np.stack([scores[episode_ids == episode].mean(axis=0) for episode in episodes])
    episode_labels = np.asarray(
        [labels[np.flatnonzero(episode_ids == episode)[0]] for episode in episodes],
        dtype=np.int8,
    )
    episode_pairs = np.asarray([pair_ids[np.flatnonzero(episode_ids == episode)[0]] for episode in episodes])
    return episodes, episode_scores, episode_labels, episode_pairs


def _paired_effects_from_episode_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    pair_ids: np.ndarray,
) -> np.ndarray:
    differences: list[np.ndarray] = []
    for pair in np.unique(pair_ids):
        rows = np.flatnonzero(pair_ids == pair)
        failure = rows[labels[rows] == 0]
        success = rows[labels[rows] == 1]
        if len(failure) != 1 or len(success) != 1:
            raise ValueError(f"Pair {pair!r} does not contain one episode per outcome")
        differences.append(scores[success[0]] - scores[failure[0]])
    paired_differences = np.stack(differences)
    return np.abs(paired_differences.mean(axis=0)) / np.maximum(
        paired_differences.std(axis=0, ddof=1),
        1e-12,
    )


def _paired_outcome_arrays(
    pair_ids: np.ndarray,
    labels: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = np.unique(pair_ids)
    failures: list[float] = []
    successes: list[float] = []
    for pair in pairs:
        rows = np.flatnonzero(pair_ids == pair)
        failure = rows[labels[rows] == 0]
        success = rows[labels[rows] == 1]
        if len(failure) != 1 or len(success) != 1:
            raise ValueError(f"Pair {pair!r} does not contain one episode per outcome")
        failures.append(float(values[failure[0]]))
        successes.append(float(values[success[0]]))
    return pairs, np.asarray(failures), np.asarray(successes)


def _paired_auc_bootstrap(
    failure_scores: np.ndarray,
    success_scores: np.ndarray,
    *,
    seed: int,
    replicates: int = 50_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(failure_scores), size=(replicates, len(failure_scores)))
    boot_failure = failure_scores[draws]
    boot_success = success_scores[draws]
    greater = (boot_success[:, :, None] > boot_failure[:, None, :]).mean(axis=(1, 2))
    ties = (boot_success[:, :, None] == boot_failure[:, None, :]).mean(axis=(1, 2))
    auc_values = greater + 0.5 * ties
    return float(np.percentile(auc_values, 2.5)), float(np.percentile(auc_values, 97.5))


def _plot_paired_axis(
    axis: plt.Axes,
    failure_values: np.ndarray,
    success_values: np.ndarray,
    *,
    ylabel: str,
    title: str,
) -> None:
    for failure, success in zip(failure_values, success_values, strict=True):
        correct = success > failure
        axis.plot(
            [0, 1],
            [failure, success],
            color="#737373" if correct else "#CC6677",
            alpha=0.35 if correct else 0.80,
            linewidth=1.0 if correct else 1.4,
            zorder=1,
        )
    axis.scatter(
        np.zeros(len(failure_values)),
        failure_values,
        color=FAILURE_COLOR,
        marker="X",
        s=43,
        edgecolors="white",
        linewidths=0.5,
        label="failure episode",
        zorder=3,
    )
    axis.scatter(
        np.ones(len(success_values)),
        success_values,
        color=SUCCESS_COLOR,
        marker="o",
        s=39,
        edgecolors="white",
        linewidths=0.5,
        label="success episode",
        zorder=3,
    )
    axis.set_xlim(-0.25, 1.25)
    axis.set_xticks([0, 1], ["failure", "success"])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.65)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=8, loc="best")


def _plot_outcome_signal_diagnostic(
    feature: CheckpointFeatures,
    output_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    pca_components = 10
    full_pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=42)
    full_scores = full_pca.fit_transform(feature.z)
    _, episode_scores, episode_labels, episode_pairs = _aggregate_episode_scores(
        full_scores,
        feature.episode_id,
        feature.pair_id,
        feature.success,
    )
    pc6_values = episode_scores[:, 5].copy()
    if pc6_values[episode_labels == 1].mean() < pc6_values[episode_labels == 0].mean():
        pc6_values *= -1.0
    pair_names, pc6_failure, pc6_success = _paired_outcome_arrays(
        episode_pairs,
        episode_labels,
        pc6_values,
    )

    prediction_rows: list[dict[str, Any]] = []
    selected_components: list[int] = []
    for held_pair in np.unique(feature.pair_id):
        train = feature.pair_id != held_pair
        test = ~train
        fold_pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=42)
        train_scores = fold_pca.fit_transform(feature.z[train])
        test_scores = fold_pca.transform(feature.z[test])
        _, train_episode_scores, train_labels, train_pairs = _aggregate_episode_scores(
            train_scores,
            feature.episode_id[train],
            feature.pair_id[train],
            feature.success[train],
        )
        effects = _paired_effects_from_episode_scores(train_episode_scores, train_labels, train_pairs)
        chosen = np.argsort(-effects)[:2]
        selected_components.extend(int(index + 1) for index in chosen)
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=42,
            ),
        )
        classifier.fit(train_episode_scores[:, chosen], train_labels)
        test_episodes, test_episode_scores, test_labels, test_pairs = _aggregate_episode_scores(
            test_scores,
            feature.episode_id[test],
            feature.pair_id[test],
            feature.success[test],
        )
        probabilities = classifier.predict_proba(test_episode_scores[:, chosen])[:, 1]
        for episode, pair, label, probability in zip(
            test_episodes,
            test_pairs,
            test_labels,
            probabilities,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "pair_id": str(pair),
                    "episode_id": str(episode),
                    "success": int(label),
                    "oof_success_score": float(probability),
                    "selected_pc_1": int(chosen[0] + 1),
                    "selected_pc_2": int(chosen[1] + 1),
                }
            )
    prediction_rows.sort(key=lambda row: (row["pair_id"], row["success"]))
    oof_pairs = np.asarray([row["pair_id"] for row in prediction_rows])
    oof_labels = np.asarray([row["success"] for row in prediction_rows], dtype=np.int8)
    oof_scores = np.asarray([row["oof_success_score"] for row in prediction_rows], dtype=np.float64)
    _, oof_failure, oof_success = _paired_outcome_arrays(oof_pairs, oof_labels, oof_scores)
    oof_auc = float(roc_auc_score(oof_labels, oof_scores))
    auc_ci_low, auc_ci_high = _paired_auc_bootstrap(oof_failure, oof_success, seed=42)
    false_positive_rate, true_positive_rate, _ = roc_curve(oof_labels, oof_scores)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.2))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.27, top=0.80, wspace=0.28)
    _plot_paired_axis(
        axes[0],
        pc6_failure,
        pc6_success,
        ylabel="episode-mean PC6 score (sign oriented)",
        title=f"Descriptive PC6 direction (post hoc)\n{int(np.sum(pc6_success > pc6_failure))}/20 pairs increase",
    )
    _plot_paired_axis(
        axes[1],
        oof_failure,
        oof_success,
        ylabel="OOF classifier score (uncalibrated)",
        title=f"Leave-one-pair-out score\n{int(np.sum(oof_success > oof_failure))}/20 pairs correctly ordered",
    )
    axes[1].set_ylim(-0.03, 1.03)
    axes[2].plot(false_positive_rate, true_positive_rate, color="#009E73", linewidth=2.2)
    axes[2].fill_between(false_positive_rate, true_positive_rate, false_positive_rate, color="#009E73", alpha=0.10)
    axes[2].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel("false-positive rate")
    axes[2].set_ylabel("true-positive rate")
    axes[2].set_title(f"Out-of-fold ROC\nAUC={oof_auc:.3f} [{auc_ci_low:.3f}, {auc_ci_high:.3f}]")
    axes[2].grid(color=GRID_COLOR, linewidth=0.6, alpha=0.65)
    axes[2].set_axisbelow(True)
    fig.suptitle(
        f"Checkpoint {_step_label(feature.step)} | episode-level outcome signal on matched instances",
        fontsize=15,
        y=0.95,
    )
    fig.text(
        0.5,
        0.135,
        "For each OOF pair, PCA fitting, two-PC selection, scaling, and logistic fitting use only the other 19 pairs.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.text(
        0.5,
        0.085,
        "OOF scores are uncalibrated. Whole-episode means test retrospective outcome information, not online perception.",
        ha="center",
        fontsize=8,
        color="#666666",
    )
    fig.text(
        0.5,
        0.035,
        "The interval is a paired bootstrap conditional on these OOF predictions. "
        "Only 20 pairs are available; independent rollout validation is still required.",
        ha="center",
        fontsize=8,
        color="#666666",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    selection_counts = {
        f"PC{component}": int(selected_components.count(component)) for component in sorted(set(selected_components))
    }
    return {
        "step": feature.step,
        "unit": "episode mean; 20 held-out matched-instance pairs",
        "pc6_explained_variance_ratio": float(full_pca.explained_variance_ratio_[5]),
        "pc6_pair_direction_count": int(np.sum(pc6_success > pc6_failure)),
        "oof_auc": oof_auc,
        "oof_auc_paired_bootstrap_ci": [auc_ci_low, auc_ci_high],
        "oof_pair_order_count": int(np.sum(oof_success > oof_failure)),
        "outer_folds": len(pair_names),
        "selected_component_counts": selection_counts,
        "pipeline": (
            "leave one pair_id out; fit 10-PC PCA on training frames; select two PCs by training-pair "
            "episode-mean effect; fit standardized episode-level logistic regression; predict held pair"
        ),
        "figure": str(output_path),
        "predictions": str(predictions_path),
    }


def _align_independent_pca(
    features: list[CheckpointFeatures],
) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    reference = features[0].pca - features[0].pca.mean(axis=0)
    reference_norm = float(np.linalg.norm(reference))
    aligned: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    for feature in features:
        centered = feature.pca - feature.pca.mean(axis=0)
        left, _, right_transpose = np.linalg.svd(centered.T @ reference)
        rotation = left @ right_transpose
        transformed = centered @ rotation
        aligned.append(transformed)
        diagnostics.append(
            {
                "step": feature.step,
                "orthogonal_determinant": float(np.linalg.det(rotation)),
                "relative_residual_to_10k": float(np.linalg.norm(transformed - reference) / reference_norm),
            }
        )
    return aligned, diagnostics


def _plot_aligned_pca_small_multiples(
    features: list[CheckpointFeatures],
    aligned_coords: list[np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.07, top=0.90, hspace=0.24, wspace=0.12)
    combined = np.concatenate(aligned_coords, axis=0)
    x_limits, y_limits = _axis_limits(combined, padding=0.04)
    for index, (feature, coords, axis) in enumerate(zip(features, aligned_coords, axes.flat, strict=True)):
        summary = _episode_summary(feature, coords)
        for outcome, color, marker in ((0, FAILURE_COLOR, "X"), (1, SUCCESS_COLOR, "o")):
            frame_rows = feature.success == outcome
            episode_rows = summary.success == outcome
            axis.scatter(
                coords[frame_rows, 0],
                coords[frame_rows, 1],
                s=5,
                color=color,
                alpha=0.045,
                linewidths=0,
            )
            axis.scatter(
                summary.coords[episode_rows, 0],
                summary.coords[episode_rows, 1],
                s=30,
                color=color,
                marker=marker,
                alpha=0.84,
                edgecolors="white",
                linewidths=0.4,
            )
        axis.set_title(_step_label(feature.step), fontsize=11, fontweight="bold")
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.45)
        axis.set_axisbelow(True)
        if index // 3 == 2:
            axis.set_xlabel("aligned local PC 1")
        if index % 3 == 0:
            axis.set_ylabel("aligned local PC 2")
    fig.suptitle(
        "Independent PCA at each checkpoint, orthogonally aligned to 10k",
        fontsize=15,
        y=0.985,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor=FAILURE_COLOR,
                label="failure episode",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=SUCCESS_COLOR,
                label="success episode",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.text(
        0.5,
        0.018,
        "No pooled PCA fit: fixed samples are used only to resolve 2D rotation/reflection. "
        "This compares layout, not checkpoint quality or latent mean drift.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _write_gallery(
    output_root: Path,
    steps: tuple[int, ...],
    representatives: dict[int, list[str]],
    t4_detail_step: int,
) -> Path:
    dataset_audit = json.loads((output_root / "dataset_audit.json").read_text(encoding="utf-8"))
    sampling_description = dataset_audit.get("frame_sampling")
    endpoint_aligned = bool(sampling_description and "endpoint-inclusive" in sampling_description)
    path = output_root / (
        "pca_tsne_gallery_endpoint_v3.md" if endpoint_aligned else "pca_tsne_gallery_v2.md"
    )
    lines = [
        "# RL Token PCA / t-SNE - endpoint-aligned resampling" if endpoint_aligned else "# RL Token PCA / t-SNE v2",
        "",
        (
            "本版对所有入选 episode 使用包含首尾的统一归一化时间网格，重新提取全部 checkpoint 特征；"
            "可视化以 40 个 episode 为统计单位。"
            if endpoint_aligned
            else "这版保留原图不动，重点减少轨迹堆叠，并把统计单位从 640 个相关帧改为 40 个 episode。"
        ),
        "",
        f"> 抽样口径：{sampling_description}" if sampling_description else "> 抽样口径见同目录 `resolved_config.yaml`。",
        "> `normalized_time` 使用所有 episode 共用的端点包含归一化时间网格；如需核对离散 call 的实际位置，可查看 `frames.csv` 的 `actual_normalized_time`。",
        "",
        "## 读图方式",
        "",
        "- PCA 左图：episode 质心、同 instance 的成败连线、组均值和 paired-bootstrap 95% 均值置信椭圆。",
        "- PCA 中图：每个 episode 自身中心化后，按 episode 等权汇总的五时间窗位移；阴影椭圆为均值的 95% bootstrap CI。",
        "- PCA / t-SNE 下排：成功与失败分面显示，九个 checkpoint 固定使用同一组 3+3 高维代表轨迹；圆点是起点，三角是终点。",
        "- t-SNE 只读局部邻域；这里刻意不画全局质心、椭圆或平均方向，轨迹长度也不作为距离证据。",
        "",
        "> 各 checkpoint 的 PCA 仍然单独拟合，符合原实验约束。图中的蓝/圆表示成功，橙/X 表示失败。",
        "",
        "## 固定代表 episode",
        "",
        "**Failure:** " + ", ".join(f"`{item}`" for item in representatives[0]),
        "",
        "**Success:** " + ", ".join(f"`{item}`" for item in representatives[1]),
        "",
        "代表样本由九个 checkpoint 的 5-bin 高维轨迹距离共同做确定性 k-medoids 选出，不按二维图形效果挑选。",
        "",
        "## 跨 checkpoint 布局对照",
        "",
        "![Aligned independent PCA](visuals_v2/aligned_independent_pca.png)",
        "",
        "> 上图先对每个 checkpoint 独立 PCA，再利用完全相同的 640 个样本做二维正交对齐；没有混合拟合 PCA。"
        "它只便于比较布局，不表示 latent 均值漂移或 checkpoint 优劣。数值结构对照见下方 CKA。",
        "",
        "![Checkpoint CKA](checkpoint_cka.png)",
        "",
        f"## T4 单独诊断（{_step_label(t4_detail_step)}）",
        "",
        f"![{_step_label(t4_detail_step)} T4 t-SNE](visuals_v2/ckpt_{_step_label(t4_detail_step)}/tsne_t4_diagnostic.png)",
        "",
        "> 左图保留全样本 t-SNE 坐标并只高亮 T4；右图只用 T4 样本重新拟合。"
        "T4 是 2048 维 episode 等权间隔的点估计峰值，不是按二维视觉分离程度选出的；"
        "当前配对 bootstrap 不能证明 T4 显著高于 T5。",
        "",
        f"## PCA 维度诊断（{_step_label(t4_detail_step)}）",
        "",
        f"![{_step_label(t4_detail_step)} PCA dimensions](visuals_v2/ckpt_{_step_label(t4_detail_step)}/pca_dimension_diagnostic.png)",
        "",
        "> 左图是方差最大的 PC1+PC2；右图从 PC1-PC10 中按当前成败标签事后选取。"
        "右图只用于确认成败信号可能位于低方差方向；只有 PC6 通过多重比较校正，"
        "第二维和整体图形仍需独立 rollout 验证。",
        "",
        f"## 配对样本外成败信号（{_step_label(t4_detail_step)}）",
        "",
        f"![{_step_label(t4_detail_step)} paired OOF outcome signal](visuals_v2/ckpt_{_step_label(t4_detail_step)}/outcome_signal_oof.png)",
        "",
        "> 左图显示 PC6 上同 instance 成败 episode 的配对变化；中图和右图来自严格 leave-one-pair-out 预测。"
        "被展示的一对不会参与其 PCA、维度选择、标准化或分类器训练。该图比 t-SNE 更直接，但仍只有 20 对样本。",
        "> 这里对整条 episode 的 16 个时间点求均值，因此证明的是回顾性 outcome 信息可解码，"
        "不是在线逐帧环境感知，也不证明 token 会提升策略。",
        "",
    ]
    for step in steps:
        label = _step_label(step)
        lines.extend(
            [
                f"## {label}",
                "",
                "### Enhanced PCA",
                "",
                f"![{label} PCA v2](visuals_v2/ckpt_{label}/pca_v2.png)",
                "",
                "### Enhanced t-SNE",
                "",
                f"![{label} t-SNE v2](visuals_v2/ckpt_{label}/tsne_v2.png)",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    steps = tuple(args.steps)
    features = [_load_checkpoint(output_root, step) for step in steps]
    reference_ids = features[0].episode_id
    for feature in features[1:]:
        if not np.array_equal(reference_ids, feature.episode_id):
            raise ValueError(f"Episode/sample order differs at checkpoint {feature.step}")
    visuals_root = output_root / "visuals_v2"
    representatives, representative_details = _select_fixed_representatives(
        features,
        args.representatives_per_outcome,
    )
    for feature in features:
        label = _step_label(feature.step)
        metrics = json.loads((output_root / f"ckpt_{label}" / "metrics.json").read_text(encoding="utf-8"))
        checkpoint_root = visuals_root / f"ckpt_{label}"
        _plot_checkpoint_embedding(
            feature,
            feature.pca,
            "PCA",
            checkpoint_root / "pca_v2.png",
            representatives,
            pca_variance=float(metrics["PCA_var12"]),
        )
        _plot_checkpoint_embedding(
            feature,
            feature.tsne,
            "t-SNE",
            checkpoint_root / "tsne_v2.png",
            representatives,
        )
    aligned_coords, alignment_diagnostics = _align_independent_pca(features)
    _plot_aligned_pca_small_multiples(
        features,
        aligned_coords,
        visuals_root / "aligned_independent_pca.png",
    )
    t4_feature = features[-1]
    t4_label = _step_label(t4_feature.step)
    with (output_root / f"ckpt_{t4_label}" / "time_bin_metrics.csv").open(encoding="utf-8") as stream:
        t4_metrics = next(row for row in csv.DictReader(stream) if int(row["time_bin"]) == T4_BIN)
    t4_output = visuals_root / f"ckpt_{t4_label}" / "tsne_t4_diagnostic.png"
    _plot_t4_tsne_diagnostic(
        t4_feature,
        t4_output,
        scaled_gap=float(t4_metrics["success_gap_scaled"]),
        debiased_gap_squared=float(t4_metrics["success_gap2_debiased"]),
    )
    pca_dimension_output = visuals_root / f"ckpt_{t4_label}" / "pca_dimension_diagnostic.png"
    pca_dimension_diagnostic = _plot_pca_dimension_diagnostic(t4_feature, pca_dimension_output)
    outcome_signal_output = visuals_root / f"ckpt_{t4_label}" / "outcome_signal_oof.png"
    outcome_predictions_output = visuals_root / f"ckpt_{t4_label}" / "outcome_signal_oof_predictions.csv"
    outcome_signal_diagnostic = _plot_outcome_signal_diagnostic(
        t4_feature,
        outcome_signal_output,
        outcome_predictions_output,
    )
    gallery = _write_gallery(output_root, steps, representatives, t4_feature.step)
    dataset_audit = json.loads((output_root / "dataset_audit.json").read_text(encoding="utf-8"))
    sampling_description = dataset_audit.get("frame_sampling")
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "source": (
            "fresh per-checkpoint features.npz extracted for this analysis manifest; "
            "the visualization stage performs no additional model inference"
        ),
        "sampling": sampling_description or "see resolved_config.yaml",
        "steps": list(steps),
        "sample_count_per_checkpoint": len(features[0].z),
        "episode_count_per_checkpoint": len(np.unique(features[0].episode_id)),
        "representative_selection": (
            "Fixed across checkpoints. For each outcome/checkpoint, concatenate five episode-bin means in full "
            "latent space, compute pairwise Euclidean distances, divide by the off-diagonal median, average the "
            "nine distance matrices, then run deterministic PAM k-medoids with episode-id tie breaking."
        ),
        "representatives": representative_details,
        "pca_outcome_ci": (
            "Gaussian-equivalent 95% covariance ellipse of 4000 paired-instance bootstrap outcome means; "
            "pairing is same instance, not same seed"
        ),
        "pca_time_path": (
            "Within each episode, average each time bin and subtract the episode frame centroid; then average "
            "episodes equally within outcome. Ellipses are 4000-draw episode-bootstrap 95% mean CIs."
        ),
        "tsne_policy": (
            "Local-neighborhood display only; no global centroid, covariance ellipse, or aggregate direction. "
            "Representative trajectory links are explicitly exploratory."
        ),
        "t4_diagnostic": {
            "step": t4_feature.step,
            "time_bin": T4_BIN,
            "normalized_time": "[0.6, 0.8)",
            "sample_count": int((t4_feature.time_bin == T4_BIN).sum()),
            "scaled_gap_2048d": float(t4_metrics["success_gap_scaled"]),
            "debiased_gap_squared_2048d": float(t4_metrics["success_gap2_debiased"]),
            "path": str(t4_output),
            "panels": "all-sample t-SNE coordinates with T4 highlighted; t-SNE refit on T4 only",
        },
        "pca_dimension_diagnostic": pca_dimension_diagnostic,
        "outcome_signal_diagnostic": outcome_signal_diagnostic,
        "cross_checkpoint_pca": (
            "PCA remains independently fit at each checkpoint. 2D scores are orthogonally aligned to 10k "
            "using the fixed 640 samples; no pooled PCA is fit."
        ),
        "alignment_diagnostics": alignment_diagnostics,
        "gallery": str(gallery),
    }
    (visuals_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Created v2 visualization gallery: {gallery}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--representatives-per-outcome", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    generate(_parse_args())
