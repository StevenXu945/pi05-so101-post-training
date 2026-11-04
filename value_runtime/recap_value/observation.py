"""Small value-only observation PyTree; no policy/PyTorch imports."""
import flax.struct


@flax.struct.dataclass
class Observation:
    images: dict
    image_masks: dict
    state: object
    tokenized_prompt: object
    tokenized_prompt_mask: object


def preprocess_observation(rng, observation, *, train=False, image_keys=None):
    # ValueDataset supplies padded, normalized 224x224 RGB. No policy augmentation.
    del rng, train
    for key in image_keys or observation.images:
        if observation.images[key].shape[1:] != (224, 224, 3):
            raise ValueError("Value images must be 224x224 RGB")
    return observation
