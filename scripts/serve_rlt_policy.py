"""Serve RLT policy via WebSocket.

Supports both single and batch inference.
- Single: client sends one observation dict → returns one result dict
- Batch:  client sends {"batch": [obs1, obs2, ...]} → returns {"batch_results": [r1, r2, ...]}

Usage:
    python scripts/serve_rlt_policy.py \
        --config rlt_pi05_agilexbag_image \
        --checkpoint-dir checkpoints/rlt_pi05_agilexbag_image/exp_A_frozen_finetuned/1000 \
        --port 8000
"""

from collections.abc import Sequence
import dataclasses
import logging
import pathlib
import socket
import time
from typing import Any, ClassVar

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override
import tyro

import openpi.models.model as _model
from openpi.models.rl_token import RLTokenConfig
from openpi.models.rl_token import RLTokenModel
from openpi.serving import websocket_policy_server
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.transforms as _transforms


def _create_rlt_config(config: _config.TrainConfig) -> RLTokenConfig:
    kwargs = {}
    if config.rlt_num_tokens is not None:
        kwargs["num_rl_tokens"] = config.rlt_num_tokens
    if config.rlt_num_layers is not None:
        kwargs["num_layers"] = config.rlt_num_layers
    if config.rlt_embed_dim is not None:
        kwargs["embed_dim"] = config.rlt_embed_dim
    if config.rlt_input_dim is not None:
        kwargs["input_dim"] = config.rlt_input_dim
    return RLTokenConfig(**kwargs)


def _infer_prefix_seq_len(model_config: _model.BaseModelConfig) -> int:
    def _get_len(rng):
        model = model_config.create(rng)
        obs = model_config.fake_obs(batch_size=1)
        embs, _ = model.extract_prefix_embeddings(rng, obs, image_only=True)
        return embs

    embs_shape = jax.eval_shape(_get_len, jax.random.key(0))
    return embs_shape.shape[1]


class RLTInferenceModel(nnx.Module):
    """Combined VLA + RLT model for inference. Outputs both actions and RL tokens."""

    def __init__(
        self,
        vla_model: _model.BaseModel,
        rlt_config: RLTokenConfig,
        rngs: nnx.Rngs,
        prefix_seq_len: int = 768,
        *,
        shared_prefix_inference: bool = False,
    ):
        self.vla = vla_model
        self.shared_prefix_inference = shared_prefix_inference
        linen_rlt = RLTokenModel(config=rlt_config)
        self.rlt_module = nnx_bridge.ToNNX(linen_rlt)
        dummy_prefix = jnp.zeros((1, prefix_seq_len, rlt_config.input_dim))
        dummy_mask = jnp.ones((1, prefix_seq_len), dtype=jnp.bool_)
        self.rlt_module.lazy_init(dummy_prefix, dummy_mask, rngs=rngs)
        self._prefix_seq_len = int(prefix_seq_len)
        self.deterministic = True

    def extract_image_prefix(self, rng: at.KeyArrayLike, observation: _model.Observation):
        """Extract the fixed image-only VLA representation used by the RL token."""
        return self.vla.extract_prefix_embeddings(rng, observation, train=False, image_only=True)

    def encode_rl_tokens(self, prefix_embs: jnp.ndarray) -> jnp.ndarray:
        """Encode prefix embeddings with static inference arguments."""
        return self.rlt_module(prefix_embs, None, method="encode", train=False)

    def decode_rl_tokens(self, rl_tokens: jnp.ndarray) -> jnp.ndarray:
        """Decode to the model's fixed image-prefix sequence length."""
        return self.rlt_module(rl_tokens, self._prefix_seq_len, method="decode", train=False)

    def infer(self, rng: at.KeyArrayLike, observation: _model.Observation) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run inference. Supports any batch size.

        Returns:
            actions: [batch, action_horizon, action_dim] VLA action chunks
            rl_token: [batch, num_rl_tokens, embed_dim] compressed RL token
        """
        if self.shared_prefix_inference:
            return self._infer_shared_prefix(rng, observation)
        return self._infer_legacy(rng, observation)

    def encode_rl_token(self, rng: at.KeyArrayLike, observation: _model.Observation) -> jnp.ndarray:
        prefix_embs, _ = self.extract_image_prefix(rng, observation)
        return self.encode_rl_tokens(prefix_embs.astype(jnp.float32))

    def _infer_legacy(self, rng: at.KeyArrayLike, observation: _model.Observation) -> tuple[jnp.ndarray, jnp.ndarray]:
        prefix_embs, _ = self.vla.extract_prefix_embeddings(rng, observation, train=False, image_only=True)
        prefix_f32 = prefix_embs.astype(jnp.float32)
        rl_token = self.rlt_module(prefix_f32, None, method="encode", train=False)
        actions = self.vla.sample_actions(rng, observation)
        return actions, rl_token

    def _infer_shared_prefix(
        self, rng: at.KeyArrayLike, observation: _model.Observation
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        prefix_cache = self.vla.prepare_prefix_for_inference(observation)
        prefix_f32 = prefix_cache.image_prefix_out.astype(jnp.float32)
        rl_token = self.rlt_module(prefix_f32, None, method="encode", train=False)
        actions = self.vla.sample_actions_from_prefix_cache(rng, prefix_cache)
        return actions, rl_token


# Constants for output format
PROPRIO_DIM = 7
CHUNK_LEN = 50
ACTION_DIM = 7


class RLTPolicy(_base_policy.BasePolicy):
    """Policy that returns both VLA actions and RL tokens via WebSocket.

    Supports single and batch inference:
    - Single: obs dict → result dict
    - Batch:  {"batch": [obs1, obs2, ...]} → {"batch_results": [r1, r2, ...]}
    """

    def __init__(
        self,
        model: RLTInferenceModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        metadata: dict[str, Any] | None = None,
        proprio_dim: int = PROPRIO_DIM,
        chunk_len: int = CHUNK_LEN,
        action_dim: int = ACTION_DIM,
    ):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._metadata = metadata or {}
        self._proprio_dim = int(proprio_dim)
        self._chunk_len = int(chunk_len)
        self._action_dim = int(action_dim)
        self._rng = rng or jax.random.key(0)
        self._infer_fn = nnx_utils.module_jit(model.infer)

    @override
    def infer(self, obs: dict) -> dict:
        if "batch" in obs:
            return self._infer_batch(obs["batch"])
        return self._infer_single(obs)

    def _infer_single(self, obs: dict) -> dict:
        """Single observation inference (original behavior)."""
        raw_proprio = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions, rl_token = self._infer_fn(sample_rng, observation)
        infer_time = time.monotonic() - start_time

        return self._build_single_result(
            actions=np.asarray(actions[0]),
            rl_token=np.asarray(rl_token[0]),
            state=np.asarray(inputs["state"])[0],
            raw_proprio=raw_proprio,
            infer_time=infer_time,
        )

    # Pre-compiled batch sizes used for padding and startup warmup.
    COMPILED_BATCH_SIZES: ClassVar[list[int]] = [1, 2, 4, 6, 8, 10, 12, 16]

    @staticmethod
    def _pad_to_compiled_size(n: int) -> int:
        """Find the smallest pre-compiled batch size >= n."""
        for size in RLTPolicy.COMPILED_BATCH_SIZES:
            if size >= n:
                return size
        # If larger than all pre-compiled sizes, use as-is (will trigger JIT)
        return n

    def _infer_batch(self, obs_list: list[dict]) -> dict:
        """Batch inference: process multiple observations in one model forward pass.

        Pads to nearest pre-compiled batch size to avoid JIT recompilation.
        """
        real_batch_size = len(obs_list)
        if real_batch_size == 0:
            return {"batch_results": []}

        start_time = time.monotonic()

        # Step 1: Apply input_transform to each observation individually
        transformed_list = []
        raw_proprios = []
        for obs in obs_list:
            raw_proprios.append(np.asarray(obs["state"], dtype=np.float32).reshape(-1))
            inp = jax.tree.map(lambda x: x, obs)
            inp = self._input_transform(inp)
            transformed_list.append(inp)

        # Step 2: Pad to nearest pre-compiled batch size by repeating last observation
        padded_size = self._pad_to_compiled_size(real_batch_size)
        while len(transformed_list) < padded_size:
            transformed_list.append(transformed_list[-1])  # duplicate last

        # Step 3: Stack into batch tensors
        batch_inputs = jax.tree.map(
            lambda *vals: jnp.stack([jnp.asarray(v) for v in vals], axis=0),
            *transformed_list,
        )

        # Step 4: One model forward pass (padded batch)
        self._rng, sample_rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(batch_inputs)
        actions, rl_token = self._infer_fn(sample_rng, observation)

        infer_time = time.monotonic() - start_time

        # Step 5: Take only the real results (discard padding)
        actions_np = np.asarray(actions)[:real_batch_size]
        rl_token_np = np.asarray(rl_token)[:real_batch_size]
        states_np = np.asarray(batch_inputs["state"])[:real_batch_size]

        results = []
        for i in range(real_batch_size):
            result = self._build_single_result(
                actions=actions_np[i],
                rl_token=rl_token_np[i],
                state=states_np[i],
                raw_proprio=raw_proprios[i],
                infer_time=infer_time / real_batch_size,
            )
            results.append(result)

        return {
            "batch_results": results,
            "batch_size": real_batch_size,
            "padded_size": padded_size,
            "total_infer_ms": infer_time * 1000,
            "per_sample_infer_ms": infer_time / real_batch_size * 1000,
        }

    def _build_single_result(
        self,
        actions: np.ndarray,
        rl_token: np.ndarray,
        state: np.ndarray,
        raw_proprio: np.ndarray,
        infer_time: float,
    ) -> dict:
        """Build output dict for one sample."""
        # Save rl_token before output_transform
        rl_token_flat = rl_token.reshape(-1).astype(np.float32)
        raw_state = state
        if raw_state.ndim > 1:
            raw_state = raw_state[0]

        outputs = {
            "state": np.array(raw_state, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
        }
        outputs = self._output_transform(outputs)

        # z_rl
        z_rl = rl_token_flat

        # proprio
        proprio = np.zeros(self._proprio_dim, dtype=np.float32)
        n = min(self._proprio_dim, raw_proprio.shape[0])
        proprio[:n] = raw_proprio[:n].astype(np.float32)

        # ref_chunk
        vla_actions = outputs["actions"]
        ref_chunk = vla_actions[: self._chunk_len, : self._action_dim].astype(np.float32)

        return {
            "z_rl": z_rl,
            "proprio": proprio,
            "ref_chunk": ref_chunk,
            "policy_timing": {"infer_ms": infer_time * 1000},
            "_raw_actions": outputs["actions"],
            "_raw_rl_token": rl_token,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def load_rlt_model(
    config: _config.TrainConfig,
    checkpoint_dir: str,
    *,
    shared_prefix_inference: bool = False,
) -> RLTInferenceModel:
    """Load RLT model from checkpoint."""
    checkpoint_path = pathlib.Path(checkpoint_dir)
    params_path = checkpoint_path / "params"

    rlt_config = _create_rlt_config(config)
    prefix_seq_len = _infer_prefix_seq_len(config.model)

    logging.info(f"RLT config: {rlt_config}, prefix_seq_len: {prefix_seq_len}")

    vla_model = nnx.eval_shape(config.model.create, jax.random.key(0))
    model = RLTInferenceModel(
        vla_model,
        rlt_config,
        rngs=nnx.Rngs(jax.random.key(0)),
        prefix_seq_len=prefix_seq_len,
        shared_prefix_inference=shared_prefix_inference,
    )

    logging.info(f"Loading params from {params_path}")
    loaded_params = _model.restore_params(params_path, dtype=jnp.bfloat16)

    graphdef, state = nnx.split(model)
    import orbax.checkpoint as ocp

    loaded_params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), loaded_params)
    state.replace_by_pure_dict(loaded_params)
    model = nnx.merge(graphdef, state)

    logging.info("RLT model loaded successfully")
    return model


def _create_checkpoint_data_config(config: _config.TrainConfig, checkpoint_path: pathlib.Path):
    data_factory = config.data
    asset_id = data_factory.assets.asset_id
    checkpoint_assets_dir = checkpoint_path / "assets"
    if asset_id is not None and (checkpoint_assets_dir / asset_id).is_dir():
        data_factory = dataclasses.replace(
            data_factory,
            assets=dataclasses.replace(data_factory.assets, assets_dir=str(checkpoint_assets_dir)),
        )
    return data_factory.create(config.assets_dirs, config.model)


@dataclasses.dataclass
class Args:
    config: str = "rlt_pi05_agilexbag_image"
    checkpoint_dir: str = ""
    port: int = 8000
    default_prompt: str | None = None
    shared_prefix_inference: bool = False


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    config = _config.get_config(args.config)
    if config.rlt_num_tokens is None:
        raise ValueError("Config must have RLT fields set.")

    model = load_rlt_model(
        config,
        args.checkpoint_dir,
        shared_prefix_inference=args.shared_prefix_inference,
    )

    checkpoint_path = pathlib.Path(args.checkpoint_dir)
    data_config = _create_checkpoint_data_config(config, checkpoint_path)
    policy_metadata = config.policy_metadata or {}
    proprio_dim = int(policy_metadata.get("proprio_dim", PROPRIO_DIM))
    chunk_len = int(policy_metadata.get("chunk_len", CHUNK_LEN))
    action_dim = int(policy_metadata.get("action_dim", ACTION_DIM))
    norm_stats = None
    if data_config.asset_id is not None:
        try:
            norm_stats = _checkpoints.load_norm_stats(checkpoint_path / "assets", data_config.asset_id)
        except Exception as e:
            logging.warning(f"Could not load norm stats: {e}")

    transforms_list = [
        _transforms.InjectDefaultPrompt(args.default_prompt),
        *data_config.data_transforms.inputs,
    ]
    if norm_stats:
        transforms_list.append(_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
    transforms_list.extend(data_config.model_transforms.inputs)

    output_transforms = list(data_config.model_transforms.outputs)
    if norm_stats:
        output_transforms.append(_transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
    output_transforms.extend(data_config.data_transforms.outputs)

    policy = RLTPolicy(
        model,
        transforms=transforms_list,
        output_transforms=output_transforms,
        metadata={
            **policy_metadata,
            "has_rl_token": True,
            "z_dim": (config.rlt_num_tokens or 1) * (config.rlt_embed_dim or 2048),
            "proprio_dim": proprio_dim,
            "chunk_len": chunk_len,
            "action_dim": action_dim,
            "supports_batch": True,
            "shared_prefix_inference": args.shared_prefix_inference,
        },
        proprio_dim=proprio_dim,
        chunk_len=chunk_len,
        action_dim=action_dim,
    )

    # Test single inference
    logging.info("Testing single inference...")
    fake_obs = config.model.fake_obs(batch_size=1)
    fake_dict = {
        "images": {k: np.asarray(v[0]) for k, v in fake_obs.images.items()},
        "state": np.zeros(proprio_dim, dtype=np.float32),
        "prompt": "test prompt",
    }
    try:
        result = policy.infer(fake_dict)
        logging.info(f"Single inference OK: z_rl={result['z_rl'].shape}, ref_chunk={result['ref_chunk'].shape}")
    except Exception as e:
        logging.warning(f"Single inference test failed: {e}")

    # Test and warmup batch inference with various batch sizes
    # This triggers JIT compilation for common batch sizes so clients don't time out
    for warmup_bs in RLTPolicy.COMPILED_BATCH_SIZES:
        logging.info(f"Warmup batch inference (batch_size={warmup_bs})...")
        try:
            batch_obs = {"batch": [fake_dict] * warmup_bs}
            batch_result = policy.infer(batch_obs)
            total_ms = batch_result["total_infer_ms"]
            per_ms = batch_result["per_sample_infer_ms"]
            logging.info(f"  Batch {warmup_bs} OK: total={total_ms:.1f}ms, per_sample={per_ms:.1f}ms")
        except Exception as e:
            logging.warning(f"  Batch {warmup_bs} failed: {e}")

    # Serve
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Creating RLT server (host: {hostname}, ip: {local_ip}, port: {args.port})")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
