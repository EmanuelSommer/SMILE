"""Toolbox for Handling a (flax) BNN with Blackjax."""

import logging
import pickle
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.experimental
import jax.numpy as jnp
import optax
from blackjax import nuts
from dataserious.base import JsonSerializableDict
from optax._src.base import GradientTransformation
from tqdm import tqdm

from src.config.sampler import GetSampler, SamplerConfig
from src.dataset.base import BaseLoader
from src.kernels.base import Sampler
from src.kernels.warmup import custom_mclmc_warmup, custom_window_adaptation
from src.training.callbacks import (
    check_balance_condition,
    progress_bar_scan,
    save_position,
)
from src.types import (
    GradEstimator,
    Kernel,
    ParamTree,
    PosteriorFunction,
    PRNGKey,
    WarmupResult,
)

logger = logging.getLogger(__name__)


def inference_loop(
    unnorm_log_posterior: PosteriorFunction,
    config: SamplerConfig,
    rng_key: jax.Array,
    init_params: ParamTree,
    step_ids: jax.Array,
    saving_path: Path,
    saving_path_warmup: Optional[Path] = None,
):
    """Blackjax inference loop for full-batch sampling.

    Args:
        unnorm_log_posterior: PosteriorFunction
        config: Sampler configuration.
        rng_key: Random chainwise key.
        init_params: Initial parameters to start the sampling from.
        step_ids: Step ids of the chain to be sampled.
        saving_path: Path to save the sampling samples.
        saving_path_warmup: Path to save the warmup samples, by default None.

    Note:
        - Currently only supports nuts & mclmc kernel.
    """
    info: JsonSerializableDict = {}  # Put any information you might need later for analysis
    n_devices = len(step_ids)
    assert config.warmup_steps > 0, "Number of warmup steps must be greater than 0."

    keys = jax.vmap(jax.random.split)(rng_key)

    # Warmup
    logger.info(f"> Starting {config.name.value} Warmup Sampling...")
    match config.name:
        case GetSampler.NUTS:
            warmup_state, parameters = warmup_nuts(
                kernel=nuts,  # config.kernel,
                config=config,
                rng_key=keys[..., 0],
                init_params=init_params,
                step_ids=step_ids,
                unnorm_log_posterior=unnorm_log_posterior,
                n_devices=n_devices,
                saving_path=saving_path_warmup,
            )
        case GetSampler.MCLMC:
            warmup_state, parameters = warmup_mclmc(
                config=config,
                rng_key=keys[..., 0],
                init_params=init_params,
                unnorm_log_posterior=unnorm_log_posterior,
            )
            # Save the warmup parameters in a .txt file
            if not saving_path.exists():
                saving_path.mkdir(parents=True)
            with open((saving_path.parent / "warmup_params.txt"), "w") as f:
                if parameters["step_size"].shape == ():
                    parameters["step_size"] = jnp.array([parameters["step_size"]])
                    parameters["L"] = jnp.array([parameters["L"]])
                f.write(",".join([str(sts) for sts in parameters["step_size"]]) + "\n")
                f.write(",".join([str(sts) for sts in parameters["L"]]) + "\n")
        case _:
            raise NotImplementedError(
                f"{config.name} does not have warmup implemented."
            )

    warmup_state = jax.block_until_ready(warmup_state)

    logger.info(f"> {config.name.value} Warmup sampling completed successfully.")

    # Sampling with tuned parameters
    if config.n_thinning == 1:

        def _inference_loop(
            rng_key: PRNGKey,
            state: Sampler.State,
            parameters: ParamTree,
            step_id: jax.Array,
        ):
            def one_step(state: Sampler.State, xs: tuple[jax.Array, jax.Array]):
                idx, rng_key = xs
                state, info = sampler.step(rng_key, state)
                # dump y to disk
                jax.experimental.io_callback(
                    partial(save_position, base=saving_path),
                    result_shape_dtypes=state.position,
                    position=state.position,
                    idx=step_id,
                    n=idx,
                )
                return state, info

            one_step_ = jax.jit(
                progress_bar_scan(
                    n_steps=config.n_samples * n_devices,
                    name=f"{config.name.value} Sampling",
                )(one_step)
            )

            sampler = config.kernel(logdensity_fn=unnorm_log_posterior, **parameters)
            keys = jax.random.split(rng_key, config.n_samples)
            _, infos = jax.lax.scan(
                f=one_step_, init=state, xs=(jnp.arange(config.n_samples), keys)
            )
            return infos

    elif config.n_thinning > 1:
        # only save every n_thinning samples and thus do not scan but loop
        def _inference_loop(
            rng_key: PRNGKey,
            state: Sampler.State,
            parameters: ParamTree,
            step_id: jax.Array,
        ):
            def one_step(state: Sampler.State, xs: tuple[jax.Array, jax.Array]):
                idx, rng_key = xs
                state, info = sampler.step(rng_key, state)

                def save_if_thinned():
                    jax.experimental.io_callback(
                        partial(save_position, base=saving_path),
                        result_shape_dtypes=state.position,
                        position=state.position,
                        idx=step_id,
                        n=idx,
                    )
                    return None

                jax.lax.cond(
                    idx % config.n_thinning == 0, save_if_thinned, lambda: None
                )
                return state, info

            one_step_ = jax.jit(
                progress_bar_scan(
                    n_steps=config.n_samples * n_devices,
                    name=f"{config.name.value} Sampling",
                )(one_step)
            )

            sampler = config.kernel(logdensity_fn=unnorm_log_posterior, **parameters)
            keys = jax.random.split(rng_key, config.n_samples)
            _, infos = jax.lax.scan(
                f=one_step_, init=state, xs=(jnp.arange(config.n_samples), keys)
            )
            return infos

    # Run the sampling loop
    logger.info(f"> Starting {config.name.value} Sampling...")
    runner_info = jax.pmap(_inference_loop)(
        keys[..., 1], warmup_state, parameters, step_ids
    )

    # Explicitly wait for the computation to finish, doesnt matter if
    # we need runner_info or not.
    runner_info = jax.block_until_ready(runner_info)

    logger.info(f"> {config.name.value} Sampling completed successfully.")

    match config.name:
        case GetSampler.NUTS:
            info.update(
                {
                    "num_integration_steps": runner_info.num_integration_steps,
                    "acceptance_rate": runner_info.acceptance_rate,
                    "num_trajectory_expansions": runner_info.num_trajectory_expansions,
                    "is_divergent": runner_info.is_divergent,
                    "energy": runner_info.energy,
                    "is_turning": runner_info.is_turning,
                }
            )

    # Save information to disk
    if not saving_path.exists():
        saving_path.mkdir(parents=True)

    # Dump Information
    with open(saving_path / "info.pkl", "wb") as f:  # type: ignore[assignment]
        pickle.dump(info, f)  # type: ignore[arg-type]


def inference_loop_batch(
    grad_estimator: GradEstimator,
    config: SamplerConfig,
    rng_key: jax.Array,
    init_params: ParamTree,
    loader: BaseLoader,
    step_ids: jnp.ndarray,
    saving_path: Path,
    saving_path_warmup: Path | None = None,
):
    """Blackjax inference loop for mini-batch sampling.

    Args:
        grad_estimator: Grad estimator for mini-batch sampling.
        config: Sampler configuration.
        rng_key: Random chainwise key.
        init_params: Initial parameters to start the sampling from.
        loader: Data loader for the dataset.
        step_ids: Step ids of the chain to be sampled.
        saving_path: Path to save the samples.
        saving_path_warmup: Path to save the warmup samples, by default None (no saving).

    Note:
        - Currently only supports `sghmc` and `adasghmc` samplers.
    """
    info: JsonSerializableDict = {}  # Put any information you might need later for analysis

    n_devices = len(step_ids)
    n_warmup = config.warmup_steps
    n_samples = config.n_samples
    n_thinning = config.n_thinning

    if config.epoch_wise_sampling:
        n_train = len(loader.data_train[0])
        batch_size = config.batch_size or n_train
        n_batches = n_train // batch_size
        n_warmup = n_warmup * n_batches
        n_samples = n_samples * n_batches
        n_thinning = n_thinning * n_batches

    # Get step size schedulers
    scheduler_warmup = config.scheduler(n_steps=n_warmup, warmup=True)
    scheduler_sampling = config.scheduler(n_steps=n_samples, warmup=False)

    logger.info(
        f"> Running sampling with the following configuration:"
        f"\n\t- Warmup steps: {n_warmup}"
        f"\n\t- Sampling steps: {n_samples}"
        f"\n\t- Thinning: {n_thinning}"
        f"\n\t- Warmup Scheduler: {scheduler_warmup}"
        f"\n\t- Sampling Scheduler: {scheduler_sampling}"
    )

    keys = jax.vmap(jax.random.split)(rng_key)

    sampler_warmup = config.warmup_kernel(
        grad_estimator=grad_estimator,
        position=init_params,
    )
    if sampler_warmup is not None:
        logger.info(f"> Starting {config.name.value} Warmup sampling...")
        _inference_loop_batch(
            rng_key=keys[..., 0],
            sampler=sampler_warmup,
            loader=loader,
            n_samples=n_warmup,
            batch_size=config.batch_size,
            n_thinning=n_thinning,
            step_size=jnp.array(config.step_size),
            n_devices=n_devices,
            step_ids=step_ids,
            saving_path=saving_path_warmup,
            scheduler=scheduler_warmup,
            optimizer=None,
            grad_estimator=None,
            desc="Warmup",
        )
        state_warmup = jax.block_until_ready(sampler_warmup.state)
        logger.info(f"> {config.name.value} Warmup sampling completed successfully.")

        if config.name == GetSampler.ADASGHMC:
            logger.info(f"Checking balance condition for chains {step_ids}...")
            balance_check = jax.vmap(check_balance_condition, in_axes=(0, None, None))(
                sampler_warmup.state, config.step_size, config.mdecay
            )
            if jnp.all(balance_check):
                logger.info("Warmup converged to a balanced state.")
            else:
                logger.warning(
                    "Warmup didn't converge to a balanced state for chains "
                    f"{step_ids[~balance_check]}."
                )
        if config.name == GetSampler.SMILE:
            logger.info(f"Step sizes from warmup: {state_warmup.step_size} ")
        elif config.name in [GetSampler.SMILET, GetSampler.NOISYSMILE]:
            logger.info(f"Step sizes from warmup: {state_warmup.step_size} ")
            logger.info(f"energy_var_mean: {state_warmup.energy_var_mean} ")
            logger.info(f"energy_var_std: {state_warmup.energy_var_std} ")
            # if config.name == GetSampler.NOISYSMILE:
            #     logger.info(f"elementwise_sd: {state_warmup.elementwise_sd} ")

        sampler = config.kernel(grad_estimator=grad_estimator, state=state_warmup)
    else:
        sampler = config.kernel(grad_estimator=grad_estimator, position=init_params)

    logger.info(f"> Starting {config.name.value} Sampling...")
    _inference_loop_batch(
        rng_key=keys[..., 1],
        sampler=sampler,
        loader=loader,
        n_samples=n_samples,
        batch_size=config.batch_size,
        n_thinning=n_thinning,
        step_size=jnp.array(config.step_size),
        n_devices=n_devices,
        step_ids=step_ids,
        saving_path=saving_path,
        scheduler=scheduler_sampling,
        optimizer=None,
        grad_estimator=None,
        desc="Sampling",
    )
    jax.block_until_ready(sampler.state)
    if config.name in [GetSampler.SMILET, GetSampler.NOISYSMILE]:
        logger.info(f"Final step sizes sampling: {sampler.state.step_size} ")
        logger.info(f"Final energy_var_mean: {sampler.state.energy_var_mean} ")
        logger.info(f"Final energy_var_std: {sampler.state.energy_var_std} ")
    logger.info(f"> {config.name.value} Sampling completed successfully.")

    # Dump Information
    with open(saving_path / "info.pkl", "wb") as f:
        pickle.dump(info, f)


# Implementation of warmup process for Full-Batch Sampling #


def warmup_nuts(
    kernel: Kernel,
    config: SamplerConfig,
    rng_key: jax.Array,  # chainwise key!
    init_params: ParamTree,
    step_ids: jax.Array,
    unnorm_log_posterior: PosteriorFunction,
    n_devices: int,
    saving_path: Optional[Path] = None,
) -> WarmupResult:
    """Perform warmup for NUTS."""
    warmup_algo = custom_window_adaptation(
        algorithm=kernel,
        logdensity_fn=unnorm_log_posterior,
        progress_bar=True,
        saving_path=saving_path,
    )
    warmup_state, parameters = jax.pmap(
        fun=warmup_algo.run,
        in_axes=(0, 0, 0, None, None),
        static_broadcasted_argnums=(3, 4),
    )(
        rng_key,
        init_params,
        step_ids,
        config.warmup_steps,
        n_devices,
    )
    return warmup_state, parameters


def warmup_mclmc(
    config: SamplerConfig,
    rng_key: jax.Array,  # chainwise key!
    init_params: ParamTree,
    unnorm_log_posterior: PosteriorFunction,
) -> WarmupResult:
    """Perform warmup for MCLMC."""
    warmup_algo = custom_mclmc_warmup(
        logdensity_fn=unnorm_log_posterior,
        diagonal_preconditioning=config.diagonal_preconditioning,
        desired_energy_var_start=config.desired_energy_var_start,
        desired_energy_var_end=config.desired_energy_var_end,
        trust_in_estimate=config.trust_in_estimate,
        num_effective_samples=config.num_effective_samples,
        step_size_init=config.step_size_init,
    )
    warmup_state, parameters = jax.pmap(
        fun=warmup_algo.run,
        in_axes=(0, 0, None),
        static_broadcasted_argnums=2,
    )(
        rng_key,
        init_params,
        config.warmup_steps,
    )
    parameters = {"step_size": parameters.step_size, "L": parameters.L}
    return warmup_state, parameters


def _inference_loop_batch(
    rng_key: jax.Array,
    sampler: Sampler,
    loader: BaseLoader,
    n_samples: int,
    batch_size: int | None,
    n_thinning: int,
    step_size: float,
    n_devices: int,
    step_ids: jax.Array,
    saving_path: Optional[Path] = None,
    scheduler: Optional[Callable] = None,
    optimizer: Optional[GradientTransformation] = None,
    grad_estimator: Optional[GradEstimator] = None,
    desc: Optional[str] = "Sampling",
):
    """Blackjax inference loop for mini-batch sampling.

    Args:
        rng_key: Random chainwise keys.
        sampler: The sampler to use.
        loader: Data loader for the dataset.
        saving_path: Path to save the samples.
        n_samples: The number of samples.
        batch_size: The batch size.
        n_thinning: How to thin the samples.
        step_size: The step size.
        n_devices: The number of devices.
        step_ids: Chain IDs.
        scheduler: tbd.
        optimizer: tbd.
        grad_estimator: Grad estimator for mini-batch sampling.
        desc: Name of the sampling loop.
    """
    # if optimizer is not None:
    #     opt_state = optimizer.init(sampler.state.position)

    keys = jax.vmap(jax.random.split, in_axes=(0, None))(rng_key, n_samples)
    with tqdm(total=n_samples, desc=desc) as progress_bar:
        _step_count = 0
        while _step_count < n_samples:
            for batch in loader.iter(
                split="train",
                batch_size=batch_size,
                chains=step_ids,
            ):
                if scheduler is not None:
                    scheduler_state = scheduler(_step_count)
                    step_size = jnp.array(scheduler_state.lr)

                    if scheduler_state.explore:
                        logger.debug(f"Exploring with step size: {step_size}")
                        sampler.update_state(keys[:, _step_count], batch, step_size)
                        # ! currently no sgd exploration is enabled
                        # ! (simply explore with the kernel)
                        # sampler.state, opt_state = one_sgd_step(
                        #     sampler.state,
                        #     batch,
                        #     step_size,
                        #     grad_estimator,
                        #     opt_state,
                        #     optimizer,
                        # )
                        _step_count += 1
                        progress_bar.update(1)
                        if _step_count == n_samples:
                            break
                        continue  # Skip the rest of the loop

                logger.debug(f"Sampling with step size: {step_size}")

                sampler.update_state(keys[:, _step_count], batch, step_size)
                # ! log some details about the current step. blows up the log so use with care
                # logger.info(
                #     f"Step {_step_count + 1}/{n_samples}, step_size: {sampler.state.step_size if hasattr(sampler.state, 'step_size') else 'N/A'}"
                #     f", logdensity: {sampler.state.logdensity if hasattr(sampler.state, 'logdensity') else 'N/A'}"
                #     f", energy_var_mean: {sampler.state.energy_var_mean if hasattr(sampler.state, 'energy_var_mean') else 'N/A'}"
                #     f", energy_var_std: {sampler.state.energy_var_std if hasattr(sampler.state, 'energy_var_std') else 'N/A'}"
                # )

                if saving_path and (_step_count % n_thinning == 0):
                    for i, chain_id in enumerate(step_ids):
                        save_position(
                            position=jax.tree.map(
                                lambda x: x[i], sampler.state.position
                            ),
                            base=saving_path,
                            idx=chain_id,
                            n=_step_count,
                        )
                progress_bar.update(1)
                _step_count += 1
                if _step_count == n_samples:
                    break


@partial(jax.pmap, in_axes=(0, 0, None, None, 0, None))
def one_sgd_step(
    state: Sampler.State,
    batch: tuple[jnp.ndarray, jnp.ndarray],
    step_size: float,
    func: GradEstimator,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> tuple[Sampler.State, optax.OptState]:
    """Single SG step."""
    grads = func(state.position, batch[0], batch[1])
    grads_scaled = jax.tree.map(lambda x: -1.0 * step_size * x, grads)
    updates, new_opt_state = optimizer.update(grads_scaled, opt_state)
    state.position = optax.apply_updates(state.position, updates)
    return state, new_opt_state
