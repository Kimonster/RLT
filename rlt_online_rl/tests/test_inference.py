from __future__ import annotations

import dataclasses
from pathlib import Path
import pickle
import sys
import time

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.config import ActorServiceConfig
from rlt_online_rl.config import EnvDriverConfig
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.inference import ActorClient
from rlt_online_rl.inference import ActorRequest
from rlt_online_rl.inference import ActorService
from rlt_online_rl.inference import EnvDriver
from rlt_online_rl.inference import RLTPolicyInferenceWrapper
from rlt_online_rl.inference import maybe_refine_chunk
from rlt_online_rl.inference import normalize_feature_payload
from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.replay import RawEpisodeChunk
from rlt_online_rl.replay import RawEpisodeStep
from rlt_online_rl.replay import RawEpisodeTrace
from rlt_online_rl.replay import TransitionSource


def _config() -> RLTOnlineRLConfig:
    return RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=4,
        z_dim=5,
        proprio_dim=2,
        actor_hidden_dim=32,
        actor_num_layers=2,
    )


def _replay_config() -> RLTOnlineRLConfig:
    return RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=10,
        z_dim=5,
        proprio_dim=3,
        actor_hidden_dim=32,
        actor_num_layers=2,
    )


def _make_feature_anchor(cfg: RLTOnlineRLConfig, observation_idx: int) -> dict[str, np.ndarray]:
    value = float(observation_idx)
    return {
        "z_rl": np.full((cfg.z_dim,), value, dtype=np.float32),
        "proprio": np.full((cfg.proprio_dim,), value, dtype=np.float32),
        "ref_chunk": np.full((cfg.chunk_len, cfg.action_dim), value, dtype=np.float32),
    }


def _make_observation(cfg: RLTOnlineRLConfig, observation_idx: int) -> dict[str, np.ndarray | str | dict]:
    return {
        "state": np.full((cfg.proprio_dim,), float(observation_idx), dtype=np.float32),
        "images": {},
        "prompt": "test",
    }


def _make_raw_episode(
    cfg: RLTOnlineRLConfig,
    *,
    total_steps: int,
    policy_anchor_steps: list[int] | None = None,
    cached_policy_anchor_steps: list[int] | None = None,
) -> RawEpisodeTrace:
    policy_anchor_steps = [] if policy_anchor_steps is None else list(policy_anchor_steps)
    cached_policy_anchor_steps = [] if cached_policy_anchor_steps is None else list(cached_policy_anchor_steps)
    observations = [_make_observation(cfg, idx) for idx in range(total_steps + 1)]
    steps = [
        RawEpisodeStep(
            observation_idx=idx,
            next_observation_idx=idx + 1,
            action=np.full((cfg.action_dim,), idx + 1.0, dtype=np.float32),
            ref_action=np.full((cfg.action_dim,), idx + 0.5, dtype=np.float32),
            reward=float(idx),
            done=idx == total_steps - 1,
            source=int(TransitionSource.RL),
            collection_phase="online",
            success=int(idx == total_steps - 1),
            episode_id=11,
            step_id=idx,
        )
        for idx in range(total_steps)
    ]
    chunks = []
    for chunk_start in range(0, total_steps, cfg.chunk_len):
        anchor = _make_feature_anchor(cfg, chunk_start)
        chunks.append(
            RawEpisodeChunk(
                episode_id=11,
                chunk_step_id=chunk_start // cfg.chunk_len,
                observation_idx=chunk_start,
                step_start=chunk_start,
                step_stop=min(chunk_start + cfg.chunk_len, total_steps),
                source=int(TransitionSource.RL),
                collection_phase="online",
                done=chunk_start + cfg.chunk_len >= total_steps,
                success=int(chunk_start + cfg.chunk_len >= total_steps),
                drop_transition=False,
                start_z_rl=anchor["z_rl"],
                start_proprio=anchor["proprio"],
                start_ref_chunk=anchor["ref_chunk"],
            )
        )
    summary: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    if cached_policy_anchor_steps:
        summary["feature_anchors"] = {
            int(step_idx): _make_feature_anchor(cfg, int(step_idx)) for step_idx in cached_policy_anchor_steps
        }
    return RawEpisodeTrace(
        episode_id=11,
        chunk_len=cfg.chunk_len,
        observations=observations,
        steps=steps,
        chunks=chunks,
        policy_start_steps=policy_anchor_steps,
        summary=summary,
    )


class _CountingFeatureProvider:
    def __init__(self, cfg: RLTOnlineRLConfig):
        self._cfg = cfg
        self.calls: list[int] = []

    def get_features(self, observation: dict[str, np.ndarray | str | dict]) -> dict[str, np.ndarray]:
        observation_idx = int(np.asarray(observation["state"], dtype=np.float32)[0])
        self.calls.append(observation_idx)
        anchor = _make_feature_anchor(self._cfg, observation_idx)
        return {
            "z_rl": anchor["z_rl"],
            "ref_chunk": anchor["ref_chunk"],
        }


class _BatchCountingFeatureProvider(_CountingFeatureProvider):
    def __init__(self, cfg: RLTOnlineRLConfig):
        super().__init__(cfg)
        self.batch_calls: list[list[int]] = []

    def get_features_batch(self, observations: list[dict[str, np.ndarray | str | dict]]) -> list[dict[str, np.ndarray]]:
        observation_indices = [int(np.asarray(obs["state"], dtype=np.float32)[0]) for obs in observations]
        self.batch_calls.append(observation_indices)
        return [self.get_features(obs) for obs in observations]


def _make_replay_driver(cfg: RLTOnlineRLConfig, provider: _CountingFeatureProvider, *, stride: int) -> EnvDriver:
    return EnvDriver(
        env=object(),
        feature_provider=provider,
        actor_client=object(),
        replay_client=object(),
        rl_config=cfg,
        env_config=EnvDriverConfig(step_trace_stride=stride),
    )


class _StatsReplayClient:
    def __init__(self, max_episode_id: int):
        self._max_episode_id = max_episode_id

    def stats(self) -> dict[str, int]:
        return {"max_episode_id": self._max_episode_id}


def _write_snapshot(path: str, version: int, cfg: RLTOnlineRLConfig, seed: int) -> dict:
    actor = ChunkActor(
        cfg.z_dim,
        cfg.proprio_dim,
        cfg.chunk_len,
        cfg.action_dim,
        cfg.actor_hidden_dim,
        cfg.actor_num_layers,
        cfg.fixed_std,
    )
    params = actor.init_params(jax.random.PRNGKey(seed))
    payload = {
        "version": version,
        "rl_config": dataclasses.asdict(cfg),
        "actor_params": jax.tree.map(np.asarray, params),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return payload


def test_actor_service_returns_refined_chunk(tmp_path) -> None:
    cfg = _config()
    snapshot_path = tmp_path / "actor.pkl"
    payload = _write_snapshot(str(snapshot_path), 1, cfg, seed=0)
    service = ActorService(cfg, ActorServiceConfig(snapshot_path=str(snapshot_path)))
    time.sleep(0.2)
    request = ActorRequest(
        z_rl=np.ones((cfg.z_dim,), dtype=np.float32),
        proprio=np.ones((cfg.proprio_dim,), dtype=np.float32),
        ref_chunk=np.ones((cfg.chunk_len, cfg.action_dim), dtype=np.float32),
        request_id="req-1",
        episode_id=1,
        step_id=2,
        deterministic=True,
    )
    response = service.infer(request)
    wrapper = RLTPolicyInferenceWrapper(cfg)
    expected = wrapper.infer(
        payload["actor_params"], request.z_rl, request.proprio, request.ref_chunk, deterministic=True
    )
    assert response.refined_chunk.shape == (cfg.chunk_len, cfg.action_dim)
    assert np.allclose(response.refined_chunk, expected)


def test_actor_param_version_hot_update(tmp_path) -> None:
    cfg = _config()
    snapshot_path = tmp_path / "actor.pkl"
    _write_snapshot(str(snapshot_path), 1, cfg, seed=0)
    service = ActorService(cfg, ActorServiceConfig(snapshot_path=str(snapshot_path)))
    time.sleep(0.2)
    assert service.actor_param_version == 1
    _write_snapshot(str(snapshot_path), 2, cfg, seed=1)
    time.sleep(0.4)
    assert service.actor_param_version == 2


def test_inference_default_uses_actor_mean_without_dropout(tmp_path) -> None:
    cfg = _config()
    snapshot_path = tmp_path / "actor.pkl"
    payload = _write_snapshot(str(snapshot_path), 3, cfg, seed=0)
    service = ActorService(cfg, ActorServiceConfig(snapshot_path=str(snapshot_path)))
    time.sleep(0.3)
    request = ActorRequest(
        z_rl=np.ones((cfg.z_dim,), dtype=np.float32),
        proprio=np.ones((cfg.proprio_dim,), dtype=np.float32),
        ref_chunk=np.ones((cfg.chunk_len, cfg.action_dim), dtype=np.float32),
        request_id="req-2",
        episode_id=1,
        step_id=0,
    )
    assert request.deterministic
    response = service.infer(request)
    wrapper = RLTPolicyInferenceWrapper(cfg)
    expected = wrapper.infer(
        payload["actor_params"], request.z_rl, request.proprio, request.ref_chunk, deterministic=True
    )
    assert np.allclose(response.refined_chunk, expected)


def test_client_timeout_fallback_logic() -> None:
    cfg = _config()
    ref_chunk = np.ones((cfg.chunk_len, cfg.action_dim), dtype=np.float32)
    client = ActorClient("http://127.0.0.1:65530", timeout_sec=0.01, max_retries=0)
    result = maybe_refine_chunk(
        client,
        z_rl=np.ones((cfg.z_dim,), dtype=np.float32),
        proprio=np.ones((cfg.proprio_dim,), dtype=np.float32),
        ref_chunk=ref_chunk,
        request_id="fallback",
        episode_id=0,
        step_id=0,
        on_error_fallback=True,
    )
    assert result.used_fallback
    assert np.allclose(result.refined_chunk, ref_chunk)


def test_feature_payload_normalizes_singleton_rl_token_shapes() -> None:
    cfg = _config()
    payload = normalize_feature_payload(
        {
            "z_rl": np.ones((1, cfg.z_dim), dtype=np.float32),
            "ref_chunk": np.ones((1, cfg.chunk_len + 40, cfg.action_dim + 2), dtype=np.float32),
        },
        cfg,
        observation={"state": np.full((cfg.proprio_dim,), 3.0, dtype=np.float32)},
    )
    assert payload["z_rl"].shape == (cfg.z_dim,)
    assert payload["proprio"].shape == (cfg.proprio_dim,)
    assert np.allclose(payload["proprio"], 3.0)
    assert payload["ref_chunk"].shape == (cfg.chunk_len, cfg.action_dim)


def test_trace_records_keep_actor_version_for_raw_episode() -> None:
    cfg = _replay_config()
    provider = _CountingFeatureProvider(cfg)
    driver = _make_replay_driver(cfg, provider, stride=0)
    observation = _make_observation(cfg, 0)
    next_observation = _make_observation(cfg, 1)
    records = driver._build_trace_records(
        [
            {
                "observation": observation,
                "next_observation": next_observation,
                "action": np.ones((cfg.action_dim,), dtype=np.float32),
                "ref_action": np.zeros((cfg.action_dim,), dtype=np.float32),
                "reward": 1.0,
                "source": int(TransitionSource.RL),
                "actor_param_version": 12,
                "human_controlled": False,
                "done": True,
            }
        ],
        episode_id=3,
        start_env_step_id=5,
        chunk_success=1,
        collection_phase="online",
    )
    raw_episode = RawEpisodeTrace(
        episode_id=3,
        chunk_len=cfg.chunk_len,
        observations=[observation],
        steps=[],
        chunks=[],
    )

    driver._append_raw_chunk(
        raw_episode,
        observation_idx=0,
        trace_records=records,
        chunk_step_id=0,
        chunk_source=int(TransitionSource.RL),
        collection_phase="online",
        done=True,
        success=1,
        drop_transition=False,
        start_features=None,
        policy_anchor_offsets=[],
        policy_anchor_features=[],
    )

    assert records[0].actor_param_version == 12
    assert raw_episode.steps[0].actor_param_version == 12


def test_rollout_trace_summary_counts_sources_and_actor_versions() -> None:
    source_counts = EnvDriver._new_source_counts()
    actor_versions: list[int] = []
    EnvDriver._accumulate_rollout_trace(
        [
            {"source": int(TransitionSource.BASE), "actor_param_version": -1},
            {"source": int(TransitionSource.RL), "actor_param_version": 7},
            {"source": int(TransitionSource.RL), "actor_param_version": 9},
            {"source": int(TransitionSource.HUMAN), "actor_param_version": -1},
        ],
        source_counts,
        actor_versions,
    )
    summary = EnvDriver._summarize_rollout_trace(source_counts, actor_versions)

    assert summary["actor_version_start"] == 7
    assert summary["actor_version_end"] == 9
    assert summary["actor_version_min"] == 7
    assert summary["actor_version_max"] == 9
    assert summary["actor_version_unique_count"] == 2
    assert summary["base_steps"] == 1
    assert summary["rl_steps"] == 2
    assert summary["human_steps"] == 1
    assert summary["mixed_steps"] == 0


def test_replay_uses_vla_ref_payload_for_human_steps() -> None:
    cfg = _replay_config()
    provider = _CountingFeatureProvider(cfg)
    driver = _make_replay_driver(cfg, provider, stride=0)
    raw_episode = _make_raw_episode(cfg, total_steps=10)
    raw_episode.steps[0].source = int(TransitionSource.HUMAN)
    raw_episode.steps[0].intervention_flag = True
    raw_episode.steps[0].action = np.full((cfg.action_dim,), 7.0, dtype=np.float32)
    raw_episode.steps[0].ref_action = np.full((cfg.action_dim,), 99.0, dtype=np.float32)

    transitions, stats = driver._build_episode_replay(raw_episode)

    assert len(transitions) == 1
    transition = transitions[0]
    assert np.allclose(transition.ref_chunk, 0.0)
    assert np.allclose(transition.action_chunk[0], 7.0)
    assert int(transition.source_chunk[0]) == int(TransitionSource.HUMAN)
    assert np.allclose(transition.next_ref_chunk, 10.0)
    assert stats["fetched_anchor_count"] == 1


def test_replay_crops_cached_feature_ref_horizon() -> None:
    cfg = _replay_config()
    provider = _CountingFeatureProvider(cfg)
    driver = _make_replay_driver(cfg, provider, stride=0)
    raw_episode = _make_raw_episode(cfg, total_steps=20)
    for chunk in raw_episode.chunks:
        chunk.start_ref_chunk = np.full(
            (cfg.chunk_len + 40, cfg.action_dim + 2),
            float(chunk.observation_idx),
            dtype=np.float32,
        )

    transitions, stats = driver._build_episode_replay(raw_episode)

    assert transitions[0].ref_chunk.shape == (cfg.chunk_len, cfg.action_dim)
    assert transitions[0].next_ref_chunk.shape == (cfg.chunk_len, cfg.action_dim)
    assert np.allclose(transitions[0].ref_chunk, 0.0)
    assert np.allclose(transitions[0].next_ref_chunk, 10.0)
    assert stats["fetched_anchor_count"] == 1


def test_stride_zero_reuses_chunk_starts_and_fetches_only_terminal_tail() -> None:
    cfg = _replay_config()
    provider = _CountingFeatureProvider(cfg)
    driver = _make_replay_driver(cfg, provider, stride=0)
    transitions, stats = driver._build_episode_replay(_make_raw_episode(cfg, total_steps=20))

    assert [transition.step_id for transition in transitions] == [0, 10]
    assert stats["replay_mode"] == "chunk"
    assert stats["replay_window_count"] == 2
    assert stats["cached_anchor_count"] == 2
    assert stats["fetched_anchor_count"] == 1
    assert provider.calls == [20]


def test_stride_zero_uses_on_demand_fetch_even_when_batch_is_available() -> None:
    cfg = _replay_config()
    provider = _BatchCountingFeatureProvider(cfg)
    driver = _make_replay_driver(cfg, provider, stride=0)
    transitions, stats = driver._build_episode_replay(_make_raw_episode(cfg, total_steps=20))

    assert [transition.step_id for transition in transitions] == [0, 10]
    assert stats["feature_prefetch_mode"] == "on_demand_single"
    assert stats["fetched_anchor_count"] == 1
    assert provider.calls == [20]
    assert provider.batch_calls == []


def test_dense_stride_fetches_only_stride_window_anchors() -> None:
    cfg = _replay_config()
    provider = _CountingFeatureProvider(cfg)
    driver = _make_replay_driver(cfg, provider, stride=2)
    transitions, stats = driver._build_episode_replay(_make_raw_episode(cfg, total_steps=20))

    assert [transition.step_id for transition in transitions] == [0, 2, 4, 6, 8, 10]
    assert stats["replay_mode"] == "dense"
    assert stats["replay_window_count"] == 6
    assert stats["cached_anchor_count"] == 2
    assert stats["fetched_anchor_count"] == 9
    assert stats["feature_prefetch_mode"] == "micro_batch"
    assert sorted(provider.calls) == [2, 4, 6, 8, 12, 14, 16, 18, 20]


def test_dense_stride_batch_prefetch_uses_configured_micro_batches() -> None:
    cfg = _replay_config()
    provider = _BatchCountingFeatureProvider(cfg)
    driver = EnvDriver(
        env=object(),
        feature_provider=provider,
        actor_client=object(),
        replay_client=object(),
        rl_config=cfg,
        env_config=EnvDriverConfig(step_trace_stride=2, replay_feature_batch_size=4),
    )
    transitions, stats = driver._build_episode_replay(_make_raw_episode(cfg, total_steps=20))

    assert len(transitions) == 6
    assert stats["fetched_anchor_count"] == 9
    assert stats["batch_prefetch_count"] == 9
    assert stats["batch_prefetch_num_requests"] == 3
    assert stats["batch_prefetch_micro_batch_size"] == 4
    assert provider.batch_calls == [[2, 4, 6, 8], [12, 14, 16, 18], [20]]


def test_chunk_replay_reuses_cached_policy_restart_features() -> None:
    cfg = _replay_config()
    provider = _CountingFeatureProvider(cfg)
    driver = _make_replay_driver(cfg, provider, stride=0)
    raw_episode = _make_raw_episode(
        cfg,
        total_steps=20,
        policy_anchor_steps=[5],
        cached_policy_anchor_steps=[5],
    )
    transitions, stats = driver._build_episode_replay(raw_episode)

    assert [transition.step_id for transition in transitions] == [0, 5, 10]
    assert stats["cached_anchor_count"] == 3
    assert stats["fetched_anchor_count"] == 2
    assert sorted(provider.calls) == [15, 20]


def test_next_episode_id_uses_replay_stats() -> None:
    cfg = _config()
    driver = EnvDriver(
        env=object(),
        feature_provider=object(),
        actor_client=object(),
        replay_client=_StatsReplayClient(7),
        rl_config=cfg,
        env_config=EnvDriverConfig(),
    )

    assert driver._next_episode_id() == 8


def test_run_forever_resumes_episode_numbering_and_stops_after_session_count() -> None:
    cfg = _config()
    driver = EnvDriver(
        env=object(),
        feature_provider=object(),
        actor_client=object(),
        replay_client=_StatsReplayClient(2),
        rl_config=cfg,
        env_config=EnvDriverConfig(),
    )
    calls: list[int] = []
    driver.run_episode = lambda episode_id: calls.append(episode_id) or {}

    driver.run_forever(num_episodes=3)

    assert calls == [3, 4, 5]
