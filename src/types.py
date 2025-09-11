"""Type definitions for the src module."""

import typing
from pathlib import Path
from typing import Callable, Protocol

import jax
from blackjax.base import SamplingAlgorithm

ParamTree: typing.TypeAlias = dict[str, "jax.Array | ParamTree"]
FileTree: typing.TypeAlias = dict[str, "Path | FileTree"]
PRNGKey: typing.TypeAlias = jax.Array
DataSet: typing.TypeAlias = tuple[jax.Array, jax.Array]
Kernel = Callable[..., SamplingAlgorithm]


class PosteriorFunction(Protocol):
    """Protocol for Posterior Function used in full-batch sampling.

    Signature:
        `(position: ParamTree) -> jax.Array`
    """

    def __call__(self, position: ParamTree) -> jax.Array:
        """Posterior Function for full-batch sampling."""
        ...


class GradEstimator(Protocol):
    """Protocol for Gradient Estimator function used in mini-batch sampling.

    Signature:
        `(position: ParamTree, x: jax.Array, y: jax.Array) -> jax.Array`
    """

    def __call__(self, position: ParamTree, x: jax.Array, y: jax.Array) -> jax.Array:
        """Gradient Estimator function for mini-batch sampling."""
        ...


# class Runner(Protocol):
#     """Protocol to describe the Runner callables that are used in MCMC sampling.

#     Signature:
#         `(rng_key: jax.Array, state: S, batch: tuple[jax.Array, jax.Array], *args) -> S`
#     """

#     def __call__(
#         self,
#         rng_key: PRNGKey,
#         state: Sampler.State,
#         batch: tuple[jax.Array, jax.Array],
#         step_size: jax.Array,
#         *args,
#     ) -> Sampler.State:
#         """Runner callable for MCMC sampling."""
#         ...


# Warmup Functions must return warmup state and tuned parameters as a dictionary
WarmupResult = tuple[typing.Any, dict[str, typing.Any]]
