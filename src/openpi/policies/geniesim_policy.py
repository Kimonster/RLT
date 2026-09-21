"""Transforms for the recorded three-camera GenieSim policy interface."""

from __future__ import annotations

import dataclasses

import numpy as np

import openpi.transforms as transforms


def _select_control_vector(values: np.ndarray, *, action: bool) -> np.ndarray:
    """Map native GenieSim vectors to [14 arm joints, 2 grippers].

    The demonstration LeRobot dataset stores 109D state/38D action vectors,
    while recorded rollout policy calls already contain the compact 16D task
    vector (sometimes padded to the model's 32D state contract).
    """
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] >= (38 if action else 109):
        arm_slice = slice(16, 30) if action else slice(30, 44)
        return np.concatenate((values[..., arm_slice], values[..., 0:2]), axis=-1).astype(np.float32)
    if values.shape[-1] < 16:
        raise ValueError(f"Expected at least 16 GenieSim control dimensions, got {values.shape}")
    return values[..., :16].astype(np.float32, copy=False)


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class GenieSimInputs(transforms.DataTransformFn):
    """Map GenieSim state/images/actions into the OpenPI representation."""

    def __call__(self, data: dict) -> dict:
        images = data.get("images")
        if not isinstance(images, dict):
            raise ValueError("Expected GenieSim input key 'images' to contain a dictionary")

        aliases = {
            "base_0_rgb": ("top_head", "base_0_rgb"),
            "left_wrist_0_rgb": ("hand_left", "left_wrist_0_rgb"),
            "right_wrist_0_rgb": ("hand_right", "right_wrist_0_rgb"),
        }
        parsed_images: dict[str, np.ndarray] = {}
        for output_key, candidates in aliases.items():
            input_key = next((key for key in candidates if key in images), None)
            if input_key is None:
                raise ValueError(f"Missing GenieSim camera for {output_key}; expected one of {candidates}")
            parsed_images[output_key] = _parse_image(images[input_key])

        output = {
            "image": parsed_images,
            "image_mask": dict.fromkeys(parsed_images, np.True_),
            "state": _select_control_vector(data["state"], action=False),
        }
        if "actions" in data:
            output["actions"] = _select_control_vector(data["actions"], action=True)
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        elif "task" in data:
            output["prompt"] = data["task"]
        return output


@dataclasses.dataclass(frozen=True)
class GenieSimOutputs(transforms.DataTransformFn):
    action_dim: int = 16

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}
