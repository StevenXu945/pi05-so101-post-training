"""RTC request validation without loading a checkpoint."""

# ruff: noqa: SLF001

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models import pi0
from openpi.policies import policy
from openpi.policies import so101_policy


def test_rtc_rejects_pytorch_before_model_call() -> None:
    instance = object.__new__(policy.Policy)
    instance._is_pytorch_model = True
    with pytest.raises(NotImplementedError, match="JAX"):
        instance.infer({"_rtc": {"enabled": True}})


def test_rtc_requires_previous_actions() -> None:
    instance = object.__new__(policy.Policy)
    instance._is_pytorch_model = False
    instance._model = object.__new__(policy._pi0.Pi0)
    with pytest.raises(ValueError, match="prev_chunk_left_over"):
        instance.infer({"_rtc": {"enabled": True}})


def test_rtc_rejects_nonfinite_actions() -> None:
    instance = object.__new__(policy.Policy)
    instance._is_pytorch_model = False
    model = object.__new__(policy._pi0.Pi0)
    model.__dict__["action_horizon"] = 16
    instance._model = model
    with pytest.raises(ValueError, match="finite"):
        instance.infer({"_rtc": {
            "enabled": True, "prev_chunk_left_over": np.full((5, 6), np.nan),
            "inference_delay": 3, "execution_horizon": 6,
        }})


def test_so101_rtc_actions_follow_the_input_action_transforms(monkeypatch) -> None:
    class FakePi0:
        action_horizon = 16
        action_dim = 32

    monkeypatch.setattr(policy._pi0, "Pi0", FakePi0)
    monkeypatch.setattr(policy._model.Observation, "from_dict", classmethod(lambda cls, data: data))
    instance = object.__new__(policy.Policy)
    instance._is_pytorch_model = False
    instance._model = FakePi0()
    instance._rng = jax.random.key(0)
    captured = {}

    def transform(data):
        data = so101_policy.SO101Inputs(model_type=policy._model.ModelType.PI05)(data)
        data = transforms.DeltaActions(transforms.make_bool_mask(5, -1))(data)
        return transforms.PadStatesAndActions(32)(data)

    def sample(key, observation, **kwargs):
        captured.update(kwargs)
        return jnp.zeros((1, 16, 32), dtype=jnp.float32)

    instance._input_transform = transform
    instance._output_transform = lambda data: data
    instance._sample_kwargs = {}
    instance._sample_actions = sample
    state = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    previous = np.tile(state + 1, (10, 1))
    instance.infer({
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": state,
        "_rtc": {
            "enabled": True, "prev_chunk_left_over": previous,
            "inference_delay": 5, "execution_horizon": 6,
        },
    })
    model_actions = np.asarray(captured["prev_chunk_left_over"])
    assert model_actions.shape == (1, 10, 32)
    np.testing.assert_allclose(model_actions[0, :, :6], np.tile([1, 1, 1, 1, 1, 7], (10, 1)))
    assert captured["inference_delay"] == 5
    assert captured["execution_horizon"] == 6


def test_rtc_prefix_weights_fade_after_delay() -> None:
    weights = np.asarray(pi0.get_prefix_weights(5, 8, 16))
    np.testing.assert_allclose(weights[:5], 1.0)
    assert weights[5] > weights[6] > weights[7] > 0
    np.testing.assert_allclose(weights[8:], 0.0)
