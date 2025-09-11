"""Stochastic Microcanonical Langevin MC kernel."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
from blackjax.mcmc.integrators import (
    IntegratorState,
    esh_dynamics_momentum_update_one_step,
    generalized_two_stage_integrator,
    mclachlan_coefficients,
)

from src.kernels.base import Sampler
from src.types import DataSet, GradEstimator, ParamTree, PRNGKey


class SMILE(Sampler):
    """SMILE algorithm."""

    @partial(
        jax.tree_util.register_dataclass,
        data_fields=[
            "position",
            "momentum",
            "elementwise_sd",
            "logdensity_grad",
            "step_size",
        ],
        meta_fields=[],
    )
    @dataclass
    class State(Sampler.State):
        """The class for the state of the SMILE sampler."""

        momentum: ParamTree
        elementwise_sd: ParamTree
        logdensity_grad: ParamTree
        step_size: jnp.ndarray

        def __init__(
            self,
            position: Optional[ParamTree] = None,
            momentum: Optional[ParamTree] = None,
            elementwise_sd: Optional[ParamTree] = None,
            logdensity_grad: Optional[ParamTree] = None,
            state: "Optional[SMILE.State]" = None,
            step_size: Optional[jnp.ndarray] = None,
        ):
            """Initialize the state of the SMILE sampler."""
            super().__init__(position=position, state=state)
            if state is not None:
                self.step_size = state.step_size
            else:
                if step_size is not None:
                    self.step_size = step_size
                else:
                    self.step_size = jnp.repeat(
                        jnp.array(0.0), jax.tree.leaves(position)[0].shape[0]
                    )

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
        state: "SMILE.State",
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float = 0.001,
        num_integration_steps: int = 1,
    ) -> "SMILE.State":
        """Generate a new sample.

        Args:
            state: SMILE state.
            rng_key: RNG key.
            minibatch: Data set.
            step_size: Step size.
            num_integration_steps: Number of integration steps.
        """
        _, key_steps = jax.random.split(rng_key)
        self.minibatch = minibatch
        int_state = IntegratorState(
            position=state.position,
            momentum=state.momentum,
            logdensity=None,
            logdensity_grad=state.logdensity_grad,
        )
        int_state, _ = jax.lax.scan(
            f=partial(
                self._integration_step,
                step_size=step_size * (state.step_size == 0.0)
                + (1 - (state.step_size == 0.0)) * state.step_size,
            ),
            init=int_state,
            xs=jax.random.split(key_steps, num_integration_steps),
        )
        return self.State(
            position=int_state.position,
            momentum=int_state.momentum,
            elementwise_sd=state.elementwise_sd,
            logdensity_grad=int_state.logdensity_grad,
            step_size=state.step_size,
        )

    def _integration_step(
        self,
        state: "SMILE.State",
        rng_key: PRNGKey,
        step_size: float,
    ) -> tuple["SMILE.State", None]:
        """Sghmc's modified Euler's step."""
        integrator = self._isokinetic_integrator()
        # one step of the deterministic dynamics
        state = integrator(state, step_size)
        return (state, None)

    def _isokinetic_integrator(
        self,
        sqrt_diag_cov: float = 1.0,
    ):
        position_update_fn = self._update_euclidean_position()
        one_step = generalized_two_stage_integrator(
            esh_dynamics_momentum_update_one_step(sqrt_diag_cov),
            position_update_fn,
            mclachlan_coefficients,
            format_output_fn=self._format_smile_state_output,
        )
        return one_step

    def _format_smile_state_output(
        self,
        position,
        momentum,
        logdensity,
        logdensity_grad,
        kinetic_grad,
        position_update_info,
        momentum_update_info,
    ):
        del (
            kinetic_grad,
            position_update_info,
            logdensity,
            momentum_update_info,
        )
        return IntegratorState(
            position=position,
            momentum=momentum,
            logdensity=None,
            logdensity_grad=logdensity_grad,
        )

    def _update_euclidean_position(
        self,
    ):
        def update(
            position: ParamTree,
            kinetic_grad: ParamTree,
            step_size: float,
            coef: float,
            auxiliary_info=None,
        ):
            del auxiliary_info
            new_position = jax.tree_util.tree_map(
                lambda x, grad: x + step_size * coef * grad,
                position,
                kinetic_grad,
            )
            max_grad = 1e6
            logdensity_grad = jax.tree_map(
                lambda g: jnp.where(
                    jnp.isnan(g), 0.0, jnp.clip(g, -max_grad, max_grad)
                ),
                self._grad_estimator(
                    new_position, x=self.minibatch[0], y=self.minibatch[1]
                )[1],
            )
            return new_position, None, logdensity_grad, None

        return update


class SMILET(Sampler):
    """SMILE Tuned algorithm."""

    def __init__(
        self,
        grad_estimator: GradEstimator,
        position: Optional[ParamTree] = None,
        state: "Optional[SMILET.State]" = None,
        **kwargs,
    ):
        """Initialize Sampler.

        Args:
            grad_estimator: Gradient function of the log-density at current position.
            position: Passed to Sampler.State.
            state: Passed to Sampler.State.
            **kwargs: Runtime static args passed to sampling step.
        """
        self._grad_estimator = grad_estimator
        step_size = kwargs.pop("step_size", 0.001)
        self._compile_sample_step(**kwargs)
        self.state = self.State(position=position, state=state, step_size=step_size)

    @partial(
        jax.tree_util.register_dataclass,
        data_fields=[
            "position",
            "momentum",
            "elementwise_sd",
            "logdensity_grad",
            "logdensity",
            "step_size",
            "energy_var_mean",
            "energy_var_std",
            "time",
        ],
        meta_fields=[],
    )
    @dataclass
    class State(Sampler.State):
        """The class for the state of the SMILET sampler."""

        momentum: ParamTree
        elementwise_sd: ParamTree
        logdensity_grad: ParamTree
        logdensity: jnp.ndarray
        step_size: jnp.ndarray
        energy_var_mean: jnp.ndarray
        energy_var_std: jnp.ndarray
        time: jnp.ndarray

        def __init__(
            self,
            position: Optional[ParamTree] = None,
            momentum: Optional[ParamTree] = None,
            elementwise_sd: Optional[ParamTree] = None,
            logdensity_grad: Optional[ParamTree] = None,
            state: "Optional[SMILET.State]" = None,
            step_size: float = 0.001,
            logdensity: float = 0.0,
            energy_var_mean: float = 0.0,
            energy_var_std: float = 0.0,
            time: float = 1.0,
        ):
            """Initialize the state of the SMILE sampler."""
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

            if state is not None:
                self.step_size = state.step_size
                self.energy_var_mean = state.energy_var_mean
                self.energy_var_std = state.energy_var_std
                self.time = state.time
            else:
                if isinstance(step_size, float):
                    self.step_size = jnp.repeat(
                        step_size, jax.tree.leaves(self.position)[0].shape[0]
                    )
                    self.energy_var_mean = jnp.repeat(
                        energy_var_mean, jax.tree.leaves(self.position)[0].shape[0]
                    )
                    self.energy_var_std = jnp.repeat(
                        energy_var_std, jax.tree.leaves(self.position)[0].shape[0]
                    )
                    self.time = jnp.repeat(
                        time, jax.tree.leaves(self.position)[0].shape[0]
                    )
                else:
                    self.step_size = step_size
                    self.energy_var_mean = energy_var_mean
                    self.energy_var_std = energy_var_std
                    self.time = time

            if isinstance(logdensity, float):
                self.logdensity = jnp.repeat(
                    logdensity, jax.tree.leaves(self.position)[0].shape[0]
                )
            else:
                self.logdensity = logdensity

    def _sample_step(  # type: ignore
        self,
        state: "SMILET.State",
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float = 0.001,
        num_integration_steps: int = 1,
        reset_prob: float = 0.99,
        adaption_prob: float = 0.05,
        adaption_strength: float = 0.02,
        smoothing_factor: float = 0.01,
    ) -> "SMILET.State":
        """Generate a new sample.

        Args:
            state: SMILE state.
            rng_key: RNG key.
            minibatch: Data set.
            step_size: Step size.
            num_integration_steps: Number of integration steps.
            reset_prob: Quantile of the energy error distribution to trigger a reset (e.g., 0.999).
            adaption_prob: Total probability mass in the tails for adaptation (algorithm's 'a').
            adaption_strength: Factor for step size update (algorithm's 'δ').
            smoothing_factor: Smoothing factor for the energy EMAs (algorithm's 'β').
        """
        _, key_steps = jax.random.split(rng_key)
        self.minibatch = minibatch  
        # ? Implement preconditioning
        # ? casting states back and forth is inefficient! Refactor.
        init_state = (
            IntegratorState(
                position=state.position,
                momentum=state.momentum,
                logdensity=state.logdensity,
                logdensity_grad=state.logdensity_grad,
            ),
            jnp.array(0.0),
        )
        (int_state, kinetic_change), _ = jax.lax.scan(
            f=partial(
                self._integration_step,
                step_size=state.step_size,
            ),
            init=init_state,
            xs=jax.random.split(key_steps, num_integration_steps),
        )
        energy_change = kinetic_change + (state.logdensity - int_state.logdensity) * (
            state.logdensity != 0.0
        )
        abs_energy_change = jnp.abs(energy_change)
        gamma_scale = (jnp.square(state.energy_var_std) + 1e-6) / state.energy_var_mean
        gamma_shape = jnp.square(state.energy_var_mean) / (
            jnp.square(state.energy_var_std) + 1e-6
        )

        def gamma_ppf(prob, a, scale):
            """Wilson-Hilferty approximation of the gamma quantile function."""
            z = jax.scipy.stats.norm.ppf(prob)
            tmp = 1 - 1 / (9 * a) + z / (3 * jnp.sqrt(a))
            return scale * (a * tmp**3)

        reset_threshold = gamma_ppf(reset_prob, a=gamma_shape, scale=gamma_scale)
        adaption_upper_threshold = gamma_ppf(
            1 - (2 * adaption_prob / 3), a=gamma_shape, scale=gamma_scale
        )
        adaption_lower_threshold = gamma_ppf(
            (1 * adaption_prob / 3), a=gamma_shape, scale=gamma_scale
        )

        is_divergent = abs_energy_change > reset_threshold
        is_divergent = jax.lax.select(state.time < 10.0, False, is_divergent)

        # ! log how often we exceed the energy_var_std x factor (ratio > 1.0), use with care (blows up the log)
        # jax.debug.print(
        #     "energy_change ratio {ecl}",
        #     ecl=(abs_energy_change / reset_threshold),
        # )
        new_position = jax.tree.map(
            lambda old, new: jax.lax.select(
                is_divergent,
                old,
                new,
            ),
            state.position,
            int_state.position,
        )
        new_momentum = jax.tree.map(
            lambda zer, new: jax.lax.select(
                is_divergent,
                zer,
                new,
            ),
            jax.tree.map(lambda m: jnp.zeros_like(m), state.momentum),
            int_state.momentum,
        )

        # update the exponential moving average of the energy variance
        new_energy_var_mean = jax.lax.select(
            state.time < 3.0,
            abs_energy_change,
            state.energy_var_mean * (1 - smoothing_factor)
            + smoothing_factor * abs_energy_change,
        )
        # clip mean
        new_energy_var_mean = jnp.clip(new_energy_var_mean, 1e-6, 1e6)
        new_energy_var_std = jnp.square(state.energy_var_std) * (state.time != 1.0) * (
            1 - smoothing_factor
        ) + smoothing_factor * jnp.square(abs_energy_change - new_energy_var_mean)
        bias_correction = jax.lax.select(
            state.time < 5.0, 1.0, 1.0 - ((0.5) ** (state.time - 4.0))
        )
        new_energy_var_std = jnp.sqrt(new_energy_var_std / bias_correction)
        new_energy_var_std = jnp.clip(new_energy_var_std, 1e-6, 1e7)

        step_size = state.step_size
        step_size = jnp.where(
            abs_energy_change > adaption_upper_threshold,
            step_size * (1 - adaption_strength),
            step_size,
        )
        step_size = jnp.where(
            abs_energy_change < adaption_lower_threshold,
            step_size * (1 + adaption_strength),
            step_size,
        )
        step_size = jnp.clip(step_size, 1e-8, 10.0)
        step_size = jax.lax.select(state.time < 10.0, state.step_size, step_size)

        return self.State(
            position=new_position,
            momentum=new_momentum,
            elementwise_sd=state.elementwise_sd,
            logdensity_grad=int_state.logdensity_grad,
            logdensity=int_state.logdensity,
            step_size=step_size,
            energy_var_mean=new_energy_var_mean,
            energy_var_std=new_energy_var_std,
            time=state.time + 1.0,
        )

    def _integration_step(
        self,
        state: tuple["SMILET.State", float],
        rng_key: PRNGKey,
        step_size: float,
    ) -> tuple[tuple[tuple[State, float], Any], None]:
        """Sghmc's modified Euler's step."""
        integrator = self._isokinetic_integrator()
        # one step of the deterministic dynamics
        state, kinetic_change = integrator(state[0], step_size)
        # no momentum for now just use the stochastic noise from the gradient.
        return ((state, kinetic_change), None)

    def _isokinetic_integrator(
        self,
        sqrt_diag_cov: float = 1.0,
    ):
        position_update_fn = self._update_euclidean_position()
        one_step = generalized_two_stage_integrator(
            esh_dynamics_momentum_update_one_step(sqrt_diag_cov),
            position_update_fn,
            mclachlan_coefficients,
            format_output_fn=self._format_smile_state_output,
        )
        return one_step

    def _format_smile_state_output(
        self,
        position,
        momentum,
        logdensity,
        logdensity_grad,
        kinetic_grad,
        position_update_info,
        momentum_update_info,
    ):
        del kinetic_grad, position_update_info
        return IntegratorState(
            position=position,
            momentum=momentum,
            logdensity=logdensity,
            logdensity_grad=logdensity_grad,
        ), momentum_update_info

    def _update_euclidean_position(
        self,
    ):
        def update(
            position: ParamTree,
            kinetic_grad: ParamTree,
            step_size: float,
            coef: float,
            auxiliary_info=None,
        ):
            del auxiliary_info
            new_position = jax.tree_util.tree_map(
                lambda x, grad: x + step_size * coef * grad,
                position,
                kinetic_grad,
            )
            max_grad = 1e6  # Gradient clipping threshold
            logdensity, logdensity_grad = self._grad_estimator(
                new_position, x=self.minibatch[0], y=self.minibatch[1]
            )
            logdensity_grad = jax.tree_map(
                lambda g: jnp.where(
                    jnp.isnan(g), 0.0, jnp.clip(g, -max_grad, max_grad)
                ),
                logdensity_grad,
            )
            return new_position, logdensity, logdensity_grad, None

        return update


class NOISYSMILE(Sampler):
    """SMILE Tuned algorithm."""

    def __init__(
        self,
        grad_estimator: GradEstimator,
        position: Optional[ParamTree] = None,
        state: "Optional[NOISYSMILE.State]" = None,
        **kwargs,
    ):
        """Initialize Sampler.

        Args:
            grad_estimator: Gradient function of the log-density at current position.
            position: Passed to Sampler.State.
            state: Passed to Sampler.State.
            **kwargs: Runtime static args passed to sampling step.
        """
        self._grad_estimator = grad_estimator
        step_size = kwargs.pop("step_size", 0.001)
        self._compile_sample_step(**kwargs)
        self.state = self.State(position=position, state=state, step_size=step_size)

    @partial(
        jax.tree_util.register_dataclass,
        data_fields=[
            "position",
            "momentum",
            "elementwise_sd",
            "smooth_windowsize",
            "smooth_grad",
            "logdensity_grad",
            "logdensity",
            "step_size",
            "energy_var_mean",
            "energy_var_std",
            "time",
        ],
        meta_fields=[],
    )
    @dataclass
    class State(Sampler.State):
        """The class for the state of the NOISYSMILE sampler."""

        momentum: ParamTree
        elementwise_sd: ParamTree
        smooth_windowsize: ParamTree
        smooth_grad: ParamTree
        logdensity_grad: ParamTree
        logdensity: jnp.ndarray
        step_size: jnp.ndarray
        energy_var_mean: jnp.ndarray
        energy_var_std: jnp.ndarray
        time: jnp.ndarray

        def __init__(
            self,
            position: Optional[ParamTree] = None,
            momentum: Optional[ParamTree] = None,
            elementwise_sd: Optional[ParamTree] = None,
            smooth_windowsize: Optional[ParamTree] = None,
            smooth_grad: Optional[ParamTree] = None,
            logdensity_grad: Optional[ParamTree] = None,
            state: "Optional[NOISYSMILE.State]" = None,
            step_size: float = 0.001,
            logdensity: float = 0.0,
            energy_var_mean: float = 0.0,
            energy_var_std: float = 0.0,
            time: float = 1.0,
        ):
            """Initialize the state of the NOISYSMILE sampler."""
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

            if smooth_windowsize is not None:
                self.smooth_windowsize = smooth_windowsize
            elif state is not None:
                self.smooth_windowsize = state.smooth_windowsize
            else:
                self.smooth_windowsize = self.ones

            if smooth_grad is not None:
                self.smooth_grad = smooth_grad
            elif state is not None:
                self.smooth_grad = state.smooth_grad
            else:
                self.smooth_grad = self.ones

            if logdensity_grad is not None:
                self.logdensity_grad = logdensity_grad
            else:
                self.logdensity_grad = self.zeros

            if state is not None:
                self.step_size = state.step_size
                self.energy_var_mean = state.energy_var_mean
                self.energy_var_std = state.energy_var_std
                self.time = state.time
            else:
                if isinstance(step_size, float):
                    self.step_size = jnp.repeat(
                        step_size, jax.tree.leaves(self.position)[0].shape[0]
                    )
                    self.energy_var_mean = jnp.repeat(
                        energy_var_mean, jax.tree.leaves(self.position)[0].shape[0]
                    )
                    self.energy_var_std = jnp.repeat(
                        energy_var_std, jax.tree.leaves(self.position)[0].shape[0]
                    )
                    self.time = jnp.repeat(
                        time, jax.tree.leaves(self.position)[0].shape[0]
                    )
                else:
                    self.step_size = step_size
                    self.energy_var_mean = energy_var_mean
                    self.energy_var_std = energy_var_std
                    self.time = time

            if isinstance(logdensity, float):
                self.logdensity = jnp.repeat(
                    logdensity, jax.tree.leaves(self.position)[0].shape[0]
                )
            else:
                self.logdensity = logdensity

    def _sample_step(  # type: ignore
        self,
        state: "NOISYSMILE.State",
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float = 0.001,
        num_integration_steps: int = 1,
        reset_prob: float = 0.99,
        adaption_prob: float = 0.05,
        adaption_strength: float = 0.02,
        smoothing_factor: float = 0.01,
    ) -> "NOISYSMILE.State":
        """Generate a new sample.

        Args:
            state: NOISYSMILE state.
            rng_key: RNG key.
            minibatch: Data set.
            step_size: Step size.
            num_integration_steps: Number of integration steps.
            reset_prob: Quantile of the energy error distribution to trigger a reset (e.g., 0.999).
            adaption_prob: Total probability mass in the tails for adaptation (algorithm's 'a').
            adaption_strength: Factor for step size update (algorithm's 'δ').
            smoothing_factor: Smoothing factor for the energy EMAs (algorithm's 'β').
        """
        _, key_steps = jax.random.split(rng_key)
        self.minibatch = minibatch 
        squared_sum = jax.tree.reduce(
            lambda acc, x: acc + jnp.sum(jnp.clip(x, 1e-2, 1e4) ** 2),
            state.elementwise_sd,
            initializer=0.0,
        )
        total_elements = jax.tree.reduce(
            lambda acc, x: acc + x.size,
            state.elementwise_sd,
            initializer=0,
        )
        normalization_factor = jnp.sqrt(squared_sum / total_elements)
        elementwise_sd_normalized = jax.tree.map(
            lambda x: jnp.clip(x, 1e-2, 1e4) / normalization_factor,
            state.elementwise_sd,
        )

        def scan_body(carry, rng_key):
            """Body function for the scan over integration steps."""
            # carry contains (current_state, kinetic_change, sd_norm)
            current_state, kinetic_change, sd_norm = carry
            integrator = self._isokinetic_integrator(elementwise_sd_normalized=sd_norm)
            new_state, new_kinetic_change = integrator(current_state, state.step_size)
            return (new_state, new_kinetic_change, sd_norm), None

        init_for_scan = (
            IntegratorState(
                position=state.position,
                momentum=state.momentum,
                logdensity=state.logdensity,
                logdensity_grad=state.logdensity_grad,
            ),
            jnp.array(0.0),
            elementwise_sd_normalized,
        )

        (int_state, kinetic_change, _), _ = jax.lax.scan(
            f=scan_body,
            init=init_for_scan,
            xs=jax.random.split(key_steps, num_integration_steps),
        )

        modified_position = jax.tree.map(
            lambda new: new,
            int_state.position,
        )
        int_state = IntegratorState(
            position=modified_position,
            momentum=int_state.momentum,
            logdensity=int_state.logdensity,
            logdensity_grad=int_state.logdensity_grad,
        )

        energy_change = kinetic_change + (state.logdensity - int_state.logdensity) * (
            state.logdensity != 0.0
        )
        abs_energy_change = jnp.abs(energy_change)
        gamma_scale = (jnp.square(state.energy_var_std) + 1e-6) / state.energy_var_mean
        gamma_shape = jnp.square(state.energy_var_mean) / (
            jnp.square(state.energy_var_std) + 1e-6
        )

        def gamma_ppf(prob, a, scale):
            """Wilson-Hilferty approximation of the gamma quantile function."""
            z = jax.scipy.stats.norm.ppf(prob)
            tmp = 1 - 1 / (9 * a) + z / (3 * jnp.sqrt(a))
            return scale * (a * tmp**3)

        reset_threshold = gamma_ppf(reset_prob, a=gamma_shape, scale=gamma_scale)
        adaption_upper_threshold = gamma_ppf(
            1 - (2 * adaption_prob / 3), a=gamma_shape, scale=gamma_scale
        )
        adaption_lower_threshold = gamma_ppf(
            (1 * adaption_prob / 3), a=gamma_shape, scale=gamma_scale
        )

        is_divergent = abs_energy_change > reset_threshold
        is_divergent = jax.lax.select(state.time < 10.0, False, is_divergent)
        # ! log how often we exceed the energy_var_std x factor (ratio > 1.0), use with care (blows up the log)
        # jax.debug.print(
        #     "energy_change ratio {ecl}",
        #     ecl=(abs_energy_change / reset_threshold),
        # )
        new_position = jax.tree.map(
            lambda old, new: jax.lax.select(
                is_divergent,
                old,
                new,
            ),
            state.position,
            int_state.position,
        )
        new_momentum = jax.tree.map(
            lambda zer, new: jax.lax.select(
                is_divergent,
                zer,
                new,
            ),
            jax.tree.map(lambda m: jnp.zeros_like(m), state.momentum),
            int_state.momentum,
        )

        new_energy_var_mean = jax.lax.select(
            state.time < 3.0,
            abs_energy_change,
            state.energy_var_mean * (1 - smoothing_factor)
            + smoothing_factor * abs_energy_change,
        )
        new_energy_var_mean = jnp.clip(new_energy_var_mean, 1e-6, 1e6)
        new_energy_var_std = jnp.square(state.energy_var_std) * (state.time != 1.0) * (
            1 - smoothing_factor
        ) + smoothing_factor * jnp.square(abs_energy_change - new_energy_var_mean)
        bias_correction = jax.lax.select(
            state.time < 5.0, 1.0, 1.0 - ((0.5) ** (state.time - 4.0))
        )
        new_energy_var_std = jnp.sqrt(new_energy_var_std / bias_correction)
        new_energy_var_std = jnp.clip(new_energy_var_std, 1e-6, 1e7)

        step_size = state.step_size
        step_size = jnp.where(
            abs_energy_change > adaption_upper_threshold,
            step_size * (1 - adaption_strength),
            step_size,
        )
        step_size = jnp.where(
            abs_energy_change < adaption_lower_threshold,
            step_size * (1 + adaption_strength),
            step_size,
        )
        step_size = jnp.clip(step_size, 1e-8, 10.0)
        step_size = jax.lax.select(state.time < 10.0, state.step_size, step_size)

        normalized_grad = int_state.logdensity_grad
        unnormalized_grad = jax.tree.map(
            lambda g, sd: g * sd,
            int_state.logdensity_grad,
            elementwise_sd_normalized,
        )

        newstate = self.State(
            position=new_position,
            momentum=new_momentum,
            elementwise_sd=state.elementwise_sd,
            logdensity_grad=unnormalized_grad,
            logdensity=int_state.logdensity,
            step_size=step_size,
            energy_var_mean=new_energy_var_mean,
            energy_var_std=new_energy_var_std,
            time=state.time + 1.0,
        )

        # update EMAs
        newstate = self._update_smooth_grad(state=newstate)
        newstate = self._update_elementwise_sd(state=newstate)

        newstate = self.State(
            position=new_position,
            momentum=new_momentum,
            elementwise_sd=newstate.elementwise_sd,
            logdensity_grad=normalized_grad,
            logdensity=int_state.logdensity,
            step_size=step_size,
            energy_var_mean=new_energy_var_mean,
            energy_var_std=new_energy_var_std,
            smooth_grad=newstate.smooth_grad,
            time=state.time + 1.0,
        )

        return newstate

    def _isokinetic_integrator(
        self,
        elementwise_sd_normalized: ParamTree,
        sqrt_diag_cov: float = 1.0,
    ):
        position_update_fn = self._update_euclidean_position()
        position_update_fn = partial(
            position_update_fn, elementwise_sd_normalized=elementwise_sd_normalized
        )

        one_step = generalized_two_stage_integrator(
            esh_dynamics_momentum_update_one_step(sqrt_diag_cov),
            position_update_fn,
            mclachlan_coefficients,
            format_output_fn=self._format_smile_state_output,
        )
        return one_step

    def _format_smile_state_output(
        self,
        position,
        momentum,
        logdensity,
        logdensity_grad,
        kinetic_grad,
        position_update_info,
        momentum_update_info,
    ):
        del kinetic_grad, position_update_info
        return IntegratorState(
            position=position,
            momentum=momentum,
            logdensity=logdensity,
            logdensity_grad=logdensity_grad,
        ), momentum_update_info

    @staticmethod
    def _update_smooth_grad(state: "NOISYSMILE.State") -> "NOISYSMILE.State":
        """Update smooth gradient with clipping."""
        max_sg = 1e6  # Gradient clipping threshold
        state.smooth_grad = jax.tree_map(
            lambda sg, grad: jnp.clip(
                sg * (1 - 0.01) + grad * 0.01,
                -max_sg,
                max_sg,
            ),
            state.smooth_grad,
            state.logdensity_grad,
        )
        return state

    @staticmethod
    def _update_elementwise_sd(
        state: "NOISYSMILE.State",
    ) -> "NOISYSMILE.State":
        """Update elementwise sd."""
        state.elementwise_sd = jax.tree.map(
            lambda sd, grad, sg: jnp.sqrt(
                jnp.clip(
                    jnp.square(sd) * (1 - 0.01) + jnp.square(grad - sg) * (0.01),
                    min=1e-7,
                )
            ),
            state.elementwise_sd,
            state.logdensity_grad,
            state.smooth_grad,
        )
        return state

    def _update_euclidean_position(
        self,
    ):
        def update(
            position: ParamTree,
            kinetic_grad: ParamTree,
            step_size: float,
            coef: float,
            auxiliary_info=None,
            elementwise_sd_normalized=None,
        ):
            del auxiliary_info
            new_position = jax.tree_util.tree_map(
                lambda x, grad, sd: x + step_size * coef * grad / sd,
                position,
                kinetic_grad,
                elementwise_sd_normalized,
            )
            max_grad = 1e6  # Gradient clipping threshold
            logdensity, logdensity_grad = self._grad_estimator(
                new_position, x=self.minibatch[0], y=self.minibatch[1]
            )
            logdensity_grad = jax.tree_map(
                lambda g: jnp.where(
                    jnp.isnan(g), 0.0, jnp.clip(g, -max_grad, max_grad)
                ),
                logdensity_grad,
            )
            logdensity_grad = jax.tree.map(
                lambda g, sd: g / sd,
                logdensity_grad,
                elementwise_sd_normalized,
            )
            return new_position, logdensity, logdensity_grad, None

        return update
