"""Stochastic gradient Langevin Dynamics (SGLD) algorithm."""

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from blackjax.sgmcmc.diffusions import overdamped_langevin

from src.kernels.base import Sampler
from src.types import DataSet, ParamTree, PRNGKey


class SGLD(Sampler):
    """Stochastic gradient Langevin Dynamics (SGLD) algorithm.

    The SGLD algorithm is a variant of the Langevin Monte Carlo algorithm where
    the gradient of the log-density is estimated using mini-batches of data.
    """

    def _sample_step(
        self,
        state: Sampler.State,
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float = 0.001,
        temperature: float = 1.0,
    ) -> Sampler.State:
        """Return sample.

        At the moment the temperature is kept constant during sampling. Overwrite
        `SGLD._compile_sample_step` to change that.
        """
        state.position = overdamped_langevin()(
            rng_key=rng_key,
            position=state.position,
            logdensity_grad=self._grad_estimator(
                state.position, x=minibatch[0], y=minibatch[1]
            )[1],
            step_size=step_size,
            temperature=temperature,
        )
        return state
