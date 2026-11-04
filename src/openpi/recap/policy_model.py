"""RECAP loss in a subclass: legacy Pi0 and its RTC sampler remain untouched."""
import jax
import jax.numpy as jnp
from openpi.models.pi0 import Pi0


class RecapPi0(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.recap_time_weighting = config.recap_time_weighting

    def compute_loss(self, rng, observation, actions, *, train=False):
        loss = super().compute_loss(rng, observation, actions, train=train)
        if not self.recap_time_weighting:
            return loss
        # Same flow time sampled by Pi0.compute_loss; preserve its RNG convention.
        _, _, time_rng = jax.random.split(rng, 3)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        return loss * (0.5 * jnp.exp(-0.5 * (1 - time)))[..., None]
