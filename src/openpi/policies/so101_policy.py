"""Input and output transforms for a six-joint SO-101 follower arm."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (image * 255).clip(0, 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        front = _parse_image(data["observation/image"])
        wrist = _parse_image(data["observation/wrist_image"])
        result = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(front),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        if "adv_ind" in data:
            result["adv_ind"] = data["adv_ind"]
        return result


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :6])}
