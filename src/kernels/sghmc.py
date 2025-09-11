"""Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) kernel."""

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from blackjax.util import generate_gaussian_noise

from src.kernels.base import Sampler
from src.types import DataSet, ParamTree, PRNGKey


class SGHMC(Sampler):
    """SGHMC algorithm as described in [1].

    [1] T. Chen, E. Fox, C. Guestrin
        In International conference on machine learning (2014).
        Stochastic gradient hamiltonian monte carlo.
    """

    @partial(
        jax.tree_util.register_dataclass,
        data_fields=["position", "momentum", "elementwise_sd", "logdensity_grad"],
        meta_fields=[],
    )
    @dataclass
    class State(Sampler.State):
        """The class for the state of the SGHMC sampler."""

        momentum: ParamTree
        elementwise_sd: ParamTree
        logdensity_grad: ParamTree

        def __init__(
            self,
            position: Optional[ParamTree] = None,
            momentum: Optional[ParamTree] = None,
            elementwise_sd: Optional[ParamTree] = None,
            logdensity_grad: Optional[ParamTree] = None,
            state: "Optional[SGHMC.State]" = None,
        ):
            """Initialize the state of the SGHMC sampler."""
            super().__init__(position=position, state=state)

            if momentum is not None:
                self.momentum = momentum
            else:
                self.momentum = self.zeros

            if elementwise_sd is not None:
                self.elementwise_sd = elementwise_sd
            elif state is not None:
                self.elementwise_sd = state.elementwise_sd
            else:
                self.elementwise_sd = self.ones

            if logdensity_grad is not None:
                self.logdensity_grad = logdensity_grad
            else:
                self.logdensity_grad = self.zeros

    def _sample_step(  # type: ignore
        self,
        state: "SGHMC.State",
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float = 0.001,
        mresampling: float = 0.0,
        num_integration_steps: int = 1,
        mdecay: float = 0.05,
    ) -> "SGHMC.State":
        """Generate a new sample.

        Args:
            state: SGHMC state.
            rng_key: RNG key.
            minibatch: Data set.
            step_size: Step size.
            mresampling: Probability for momentum resampling.
            num_integration_steps: Number of integration steps.
            mdecay: Momentum decay.
        """
        key_resampling, key_steps = jax.random.split(rng_key)
        state = self._resample_momentum(
            rng_key=key_resampling, state=state, mresampling=mresampling
        )
        state, _ = jax.lax.scan(
            f=partial(
                self._integration_step,
                minibatch=minibatch,
                step_size=step_size,
                mdecay=mdecay,
            ),
            init=state,
            xs=jax.random.split(key_steps, num_integration_steps),
        )
        return state

    def _integration_step(
        self,
        state: "SGHMC.State",
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float,
        mdecay: float,
    ) -> tuple["SGHMC.State", None]:
        """Sghmc's modified Euler's step."""
        state = self._update_logdensity_grad(state=state, minibatch=minibatch)
        state = self._update_momentum(
            state=state,
            rng_key=rng_key,
            mdecay=mdecay,
            step_size=step_size,
        )
        state = self._update_position(state=state, step_size=step_size)
        return (state, None)

    def _update_logdensity_grad(
        self,
        state: "SGHMC.State",
        minibatch: DataSet,
    ) -> "SGHMC.State":
        """Update gradient values."""
        max_grad = 1e6  # Gradient clipping threshold
        state.logdensity_grad = jax.tree_map(
            lambda g: jnp.where(jnp.isnan(g), 0.0, jnp.clip(g, -max_grad, max_grad)),
            self._grad_estimator(state.position, x=minibatch[0], y=minibatch[1])[1],
        )
        return state

    @staticmethod
    def _resample_momentum(
        state: "SGHMC.State",
        rng_key: PRNGKey,
        mresampling: float,
    ) -> "SGHMC.State":
        """Optionally resample the momentum."""
        state.momentum = jax.lax.cond(
            jax.random.bernoulli(rng_key, mresampling),
            lambda x: jax.tree.map(jnp.zeros_like, x),
            lambda x: x,
            state.momentum,
        )
        return state

    @staticmethod
    def _update_momentum(
        state: "SGHMC.State",
        rng_key: PRNGKey,
        mdecay: float,
        step_size: float,
    ) -> "SGHMC.State":
        """Update momentum."""
        state.momentum = jax.tree.map(
            lambda mom, grad, sd, n: (1 - mdecay) * mom
            + step_size * grad
            + n
            * jnp.sqrt(
                jnp.clip(2 * mdecay * sd - jnp.square(step_size * sd), a_min=1e-7)
            ),
            state.momentum,
            state.logdensity_grad,
            state.elementwise_sd,
            generate_gaussian_noise(rng_key, state.position),
        )
        return state

    @staticmethod
    def _update_position(state: "SGHMC.State", step_size: float) -> "SGHMC.State":
        """Update position."""
        state.position = jax.tree.map(
            lambda pos, mom, sd: pos + step_size * mom / sd,
            state.position,
            state.momentum,
            state.elementwise_sd,
        )
        return state


class AdaSGHMCWarmup(SGHMC):
    """Warmup kernel of the adaptive SGHMC algorithm as described in [1].

    [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
        In Advances in Neural Information Processing Systems 29 (2016).
        <https://papers.nips.cc/paper_files/paper/2016/file/
        a96d3afec184766bfeca7a9f989fc7e7-Paper.pdf>

    """

    @partial(
        jax.tree_util.register_dataclass,
        data_fields=[
            "position",
            "momentum",
            "elementwise_sd",
            "logdensity_grad",
            "smooth_windowsize",
            "smooth_grad",
        ],
        meta_fields=[],
    )
    @dataclass
    class State(SGHMC.State):
        """The class for the warmup state of the adaptive SGHMC sampler."""

        smooth_windowsize: ParamTree
        smooth_grad: ParamTree

        def __init__(
            self,
            position: Optional[ParamTree] = None,
            momentum: Optional[ParamTree] = None,
            elementwise_sd: Optional[ParamTree] = None,
            logdensity_grad: Optional[ParamTree] = None,
            smooth_windowsize: Optional[ParamTree] = None,
            smooth_grad: Optional[ParamTree] = None,
            state: "Optional[SGHMC.State]" = None,
        ):
            """Initialize the warmup state of the adaptive SGHMC sampler."""
            super().__init__(
                position=position,
                momentum=momentum,
                elementwise_sd=elementwise_sd,
                logdensity_grad=logdensity_grad,
                state=state,
            )

            if smooth_windowsize is not None:
                self.smooth_windowsize = smooth_windowsize
            else:
                self.smooth_windowsize = self.ones

            if smooth_grad is not None:
                self.smooth_grad = smooth_grad
            else:
                self.smooth_grad = self.ones

    def _integration_step(
        self,
        state: "AdaSGHMCWarmup.State",  # type: ignore
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float,
        mdecay: float,
    ) -> tuple["AdaSGHMCWarmup.State", None]:
        """Warmup step for adaSGHMC like [1].

        [1] https://github.com/automl/pybnn/blob/master/pybnn/sampler/adaptive_sghmc.py
        """
        state = self._update_windowsize(state=state)
        state = self._update_smooth_grad(state=state)
        state = self._update_elementwise_sd(state=state)
        return super()._integration_step(
            state=state,
            rng_key=rng_key,
            minibatch=minibatch,
            step_size=step_size,
            mdecay=mdecay,
        )  # type: ignore

    @staticmethod
    def _update_windowsize(state: "AdaSGHMCWarmup.State") -> "AdaSGHMCWarmup.State":
        """Update smooth window size."""
        state.smooth_windowsize = jax.tree.map(
            lambda t, sg, sd: t * (1 - jnp.square(sg) / (jnp.square(sd) + 1e-7)) + 1.0,
            state.smooth_windowsize,
            state.smooth_grad,
            state.elementwise_sd,
        )
        return state

    @staticmethod
    def _update_smooth_grad(state: "AdaSGHMCWarmup.State") -> "AdaSGHMCWarmup.State":
        """Update smooth gradient with clipping."""
        max_sg = 1e6  # Gradient clipping threshold
        state.smooth_grad = jax.tree_map(
            lambda sg, t, grad: jnp.clip(
                sg * (1 - 1 / (t + 1)) - grad / (t + 1),
                -max_sg,
                max_sg,
            ),
            state.smooth_grad,
            state.smooth_windowsize,
            state.logdensity_grad,
        )
        return state

    @staticmethod
    def _update_elementwise_sd(
        state: "AdaSGHMCWarmup.State",
    ) -> "AdaSGHMCWarmup.State":
        """Update elementwise sd."""
        state.elementwise_sd = jax.tree.map(
            lambda sd, grad, t: jnp.sqrt(
                jnp.clip(
                    jnp.square(sd) * (1 - 1 / (t + 1)) + jnp.square(grad) / (t + 1),
                    min=1e-7,
                )
            ),
            state.elementwise_sd,
            state.logdensity_grad,
            state.smooth_windowsize,
        )
        return state


class RMSPropWarmup(SGHMC):
    """Warmup kernel of the SGHMC algorithm using RMSProp preconditioning.

    The implementation is based on the one in the fortuna package. (as of April 2024)
    """

    def _sample_step(  # type: ignore
        self,
        state: SGHMC.State,
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float = 0.001,
        mresampling: float = 0.0,
        num_integration_steps: int = 1,
        mdecay: float = 0.05,
        running_avg_factor: float = 0.99,
    ) -> SGHMC.State:
        """Generate a new sample.

        Args:
            state: SGHMC state.
            rng_key: RNG key.
            minibatch: Data set.
            step_size: Step size.
            mresampling: Probability for momentum resampling.
            num_integration_steps: Number of integration steps.
            mdecay: Momentum decay.
            running_avg_factor: Running average factor.
        """
        key_resampling, key_steps = jax.random.split(rng_key)
        state = self._resample_momentum(
            rng_key=key_resampling, state=state, mresampling=mresampling
        )
        state, _ = jax.lax.scan(
            f=partial(
                self._integration_step,
                minibatch=minibatch,
                step_size=step_size,
                mdecay=mdecay,
                running_avg_factor=running_avg_factor,
            ),
            init=state,
            xs=jax.random.split(key_steps, num_integration_steps),
        )
        return state

    def _integration_step(
        self,
        state: SGHMC.State,
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float,
        mdecay: float,
        running_avg_factor: float = 0.99,
    ) -> tuple[SGHMC.State, None]:
        """Rmsprop preconditioning step."""
        state = self._update_elementwise_sd(
            state=state, running_avg_factor=running_avg_factor
        )
        return super()._integration_step(
            state=state,
            rng_key=rng_key,
            minibatch=minibatch,
            step_size=step_size,
            mdecay=mdecay,
        )

    @staticmethod
    def _update_elementwise_sd(
        state: SGHMC.State,
        running_avg_factor: float,
    ) -> SGHMC.State:
        """Update RMSProp running elementwise gradient sd."""
        state.elementwise_sd = jax.tree.map(
            lambda sd, grad: jnp.sqrt(
                jnp.clip(
                    jnp.square(sd) * running_avg_factor
                    + (1 - running_avg_factor) * jnp.square(grad),
                    min=1e-7,
                )
            ),
            state.elementwise_sd,
            state.logdensity_grad,
        )
        return state
