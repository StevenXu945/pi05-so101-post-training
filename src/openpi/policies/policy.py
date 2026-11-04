from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        rtc = inputs.pop("_rtc", None)
        rtc_actions = None
        if rtc is not None:
            if not isinstance(rtc, dict):
                raise ValueError("RTC request must be a dictionary")
            if rtc.get("enabled", False):
                if self._is_pytorch_model:
                    raise NotImplementedError("RTC guidance is supported only by the JAX Pi0/Pi0.5 model")
                if not isinstance(self._model, _pi0.Pi0):
                    raise NotImplementedError("RTC guidance requires a JAX Pi0/Pi0.5 model")
                raw_actions = rtc.get("prev_chunk_left_over")
                if raw_actions is None:
                    raise ValueError("RTC requires prev_chunk_left_over")
                raw_actions = np.asarray(raw_actions)
                if raw_actions.ndim != 2 or not 1 <= raw_actions.shape[0] <= self._model.action_horizon:
                    raise ValueError("RTC prev_chunk_left_over must have shape (K, action_dim), with 1 <= K <= horizon")
                if raw_actions.shape[1] != 6:
                    raise ValueError("RTC SO-101 actions must have six joints")
                if not np.issubdtype(raw_actions.dtype, np.number) or not np.isrealobj(raw_actions):
                    raise ValueError("RTC actions must be real-valued")
                if not np.isfinite(raw_actions).all():
                    raise ValueError("RTC actions must be finite")
                rtc_delay = rtc.get("inference_delay")
                rtc_horizon = rtc.get("execution_horizon")
                if type(rtc_delay) is not int or type(rtc_horizon) is not int:
                    raise ValueError("RTC delay and execution horizon must be integers")
                if not 0 <= rtc_delay <= rtc_horizon < self._model.action_horizon:
                    raise ValueError("RTC requires 0 <= inference_delay <= execution_horizon < action horizon")
                if rtc_delay > raw_actions.shape[0]:
                    raise ValueError("RTC delay exceeds the previous chunk's remaining actions")
                rtc_weight = float(rtc.get("max_guidance_weight", 10.0))
                if not np.isfinite(rtc_weight) or rtc_weight <= 0:
                    raise ValueError("RTC max_guidance_weight must be positive and finite")
                # SO101Inputs, DeltaActions, Normalize and PadStatesAndActions must
                # process the old absolute targets exactly like training actions.
                inputs["actions"] = np.array(raw_actions, dtype=np.float32, copy=True)
        inputs = self._input_transform(inputs)
        if isinstance(rtc, dict) and rtc.get("enabled", False):
            rtc_actions = inputs.pop("actions", None)
            if rtc_actions is None or np.asarray(rtc_actions).shape != (raw_actions.shape[0], self._model.action_dim):
                raise ValueError("RTC action transforms did not produce model-space actions")
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        if rtc_actions is not None:
            sample_kwargs.update(
                prev_chunk_left_over=jnp.asarray(rtc_actions, dtype=jnp.float32)[None, ...],
                inference_delay=rtc_delay,
                execution_horizon=rtc_horizon,
                max_guidance_weight=rtc_weight,
            )

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
