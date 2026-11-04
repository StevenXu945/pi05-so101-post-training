"""Input and output transforms for RoboTwin's 14-dimensional joint actions."""

import dataclasses

import numpy as np

from openpi import transforms


def _image_to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D RGB image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return image


@dataclasses.dataclass(frozen=True)
class RoboTwinInputs(transforms.DataTransformFn):
    """Map LeRobot training samples or inference requests to pi0.5 inputs."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"Expected 14 state values, got {state.shape}")
        result = {
            "state": state,
            "image": {
                "base_0_rgb": _image_to_hwc_uint8(data["observation/image"]),
                "left_wrist_0_rgb": _image_to_hwc_uint8(data["observation/left_wrist_image"]),
                "right_wrist_0_rgb": _image_to_hwc_uint8(data["observation/right_wrist_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 14:
                raise ValueError(f"Expected 14 action values, got {actions.shape}")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class RoboTwinOutputs(transforms.DataTransformFn):
    """Return the 14 RoboTwin joint targets from the padded model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :14]}
