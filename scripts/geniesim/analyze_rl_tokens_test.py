# ruff: noqa: PERF401, SLF001

from __future__ import annotations

import numpy as np

from scripts.geniesim import analyze_rl_tokens as analysis


def test_uniform_indices_are_unique_and_cover_short_episode() -> None:
    assert analysis._uniform_indices(4, 16) == [0, 1, 2, 3]
    values = analysis._uniform_indices(32, 16)
    assert len(values) == len(set(values)) == 16
    assert values[0] == 1
    assert values[-1] == 31


def test_endpoint_uniform_indices_use_shared_grid_and_include_endpoints() -> None:
    assert analysis._endpoint_uniform_indices(0, 16) == []
    assert analysis._endpoint_uniform_indices(1, 16) == [0]
    assert analysis._endpoint_uniform_indices(16, 16) == list(range(16))
    assert analysis._endpoint_uniform_indices(32, 16) == [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        17,
        19,
        21,
        23,
        25,
        27,
        29,
        31,
    ]
    for length in (18, 20, 32):
        values = analysis._endpoint_uniform_indices(length, 16)
        assert len(values) == len(set(values)) == 16
        assert values[0] == 0
        assert values[-1] == length - 1
        assert values == sorted(values)


def test_scaled_gap_is_rigid_transform_and_scale_invariant() -> None:
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(40, 12))
    labels = np.repeat([0, 1], 20)
    vectors[labels == 1, 0] += 0.8
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    baseline = analysis._gap_statistics(vectors, labels)
    transformed = analysis._gap_statistics(4.2 * (vectors @ q) + 7.0, labels)
    np.testing.assert_allclose(baseline["scaled"], transformed["scaled"], rtol=1e-10)
    np.testing.assert_allclose(transformed["raw"], 4.2 * baseline["raw"], rtol=1e-10)


def test_linear_cka_is_rotation_scale_and_translation_invariant() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(48, 16))
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    y = 2.5 * (x @ q) + 3.0
    assert analysis._linear_cka(x, y) > 1.0 - 1e-10
    shuffled = y[rng.permutation(len(y))]
    assert analysis._linear_cka(x, shuffled) < 0.8


def test_unbiased_linear_cka_has_no_high_dimensional_similarity_floor() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(128, 512))
    y = rng.normal(size=(128, 512))
    assert abs(analysis._linear_cka(x, y)) < 0.1
    assert analysis._linear_cka(x, x) > 1.0 - 1e-12


def test_holm_adjustment_is_monotone_in_sorted_pvalues() -> None:
    raw = [0.04, 0.001, 0.02, 0.5]
    adjusted = analysis._holm_adjusted_pvalues(raw)
    assert adjusted == [0.08, 0.004, 0.06, 0.5]


def test_episode_equal_scalar_is_unchanged_by_repeating_whole_episode() -> None:
    values = np.asarray([1.0, 3.0, 10.0, 14.0])
    episodes = np.asarray(["a", "a", "b", "b"])
    mean, _, _ = analysis._episode_equal_scalar(values, episodes)
    repeated_values = np.asarray([1.0, 3.0, 1.0, 3.0, 10.0, 14.0])
    repeated_episodes = np.asarray(["a", "a", "a", "a", "b", "b"])
    repeated_mean, _, _ = analysis._episode_equal_scalar(repeated_values, repeated_episodes)
    assert mean == repeated_mean == 7.0


def test_matched_selection_has_one_outcome_per_instance() -> None:
    catalog = []
    for instance_id in range(4):
        for success in (0, 1):
            catalog.append(
                {
                    "category": "successful" if success else "failed",
                    "trajectory_id": instance_id + 1,
                    "episode_id": f"{success}/{instance_id}",
                    "success": success,
                    "failure": 1 - success,
                    "seed": 10 + instance_id,
                    "instance_id": instance_id,
                    "attempt_id": instance_id,
                    "frame_count_available": 20,
                    "metadata_path": "unused",
                }
            )
    selected = analysis._select_matched_episodes(catalog, count=3, seed=42)
    assert len(selected) == 6
    for pair_id in {row["pair_id"] for row in selected}:
        rows = [row for row in selected if row["pair_id"] == pair_id]
        assert sorted(row["success"] for row in rows) == [0, 1]
        assert len({row["instance_id"] for row in rows}) == 1
