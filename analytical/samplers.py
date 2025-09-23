from functools import partial
from typing import Tuple, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np


def make_sampling_state(blackjax: Any, logdensity_fn, initial_position, key, algorithm: str = "mclmc"):
    if algorithm == "mclmc":
        return blackjax.mcmc.mclmc.init(position=initial_position, logdensity_fn=logdensity_fn, rng_key=key)
    if algorithm == "sgld":
        return blackjax.sgld.init(position=initial_position)
    if algorithm == "sghmc":
        return blackjax.sghmc.init(position=initial_position)
    raise ValueError(f"Unknown algorithm {algorithm}")



def get_likelihood_factories(
    likelihood_model: str,
    *,
    Condition_numbers: float,
    N_total: int,
    dim_val: int,
    constant_val: float,
    batch_size: Optional[int],
    seed: int = 0,
    noise_type: Optional[str] = None,
    noise_scale: Optional[float] = None,
    random_rotation: Optional[bool] = None,
    center: Optional[bool] = None,
):
    """Resolve dataset/logp factory from inference_model(s) based on a string name.

    Attempts to import `inference_model` (singular) first, then `inference_models` (plural).
    Looks up a callable matching the provided likelihood_model or an alias.
    Falls back to ill-conditioned gaussian defaults if not found.
    """
    # Map friendly names to canonical factory callables inside the module
    alias_map = {
        # default from previous script
        "ill_condition_Gaussian_precond": "Ill_condition_Gaussian_log_density",
        "ill_condition_Gaussian": "Ill_condition_Gaussian_log_density",
        "ill_condition_Gaussian_with_data": "Ill_condition_Gaussian_log_density_with_data",
        "funnel": "Funnel_log_density",
        "rosenbrock": "Rosenbrock_log_density",
    }

    target_name = alias_map.get(likelihood_model, likelihood_model)

    module = None
    try:
        import inference_model as module  # type: ignore
    except Exception:
        try:
            import inference_models as module  # type: ignore
        except Exception:
            module = None

    if module is None:
        raise ImportError(
            f"Could not import 'inference_model' or 'inference_models'. Cannot resolve likelihood_model='{likelihood_model}'."
        )

    if not hasattr(module, target_name):
        raise AttributeError(
            f"Likelihood model '{likelihood_model}' (resolved to '{target_name}') not found in module {module.__name__}."
        )

    factory = getattr(module, target_name)
    try:
        kwargs = dict(
            Condition_numbers=Condition_numbers,
            N_total=N_total,
            dim_val=dim_val,
            constant_val=constant_val,
            batch_size=batch_size,
            seed=seed,
        )
        if noise_type is not None:
            kwargs["noise_type"] = noise_type
        if noise_scale is not None:
            kwargs["noise_scale"] = noise_scale
        if random_rotation is not None:
            kwargs["random_rotation"] = random_rotation
        if center is not None:
            kwargs["center"] = center
        out = factory(**kwargs)
    except TypeError:
        # Fallback to a minimal signature for broader compatibility
        out = factory(N_total=N_total, dim_val=dim_val, constant_val=constant_val, seed=seed)

    if not isinstance(out, tuple) or len(out) != 4:
        raise ValueError(
            f"Factory '{target_name}' must return a 4-item tuple (create_dataset, make_batched_log_p, E_f, Var_f); got {type(out)} with length {len(out) if isinstance(out, tuple) else 'N/A'}."
        )
    return out


@partial(jax.jit, static_argnums=(3, 4, 5,6,7))
def scan_sghmc_body(carry, _, y_full, N_total, current_batch_size, current_L, current_step_size, make_batched_log_p_fn ):
    current_state, current_key = carry
    next_key, mc_key, batch_key = jax.random.split(current_key, 3)
    indices = jax.random.choice(batch_key, N_total, shape=(current_batch_size,), replace=False)
    #indices = jax.random.permutation(batch_key, N_total)[:current_batch_size]
    y_batch = y_full[indices, :]

    logp_batch = make_batched_log_p_fn(y_batch, N_total, current_batch_size)
    grad_logp_batch = lambda x, _: jax.grad(logp_batch)(x)

    def kernel_factory(bjx):
        return bjx.sghmc(grad_logp_batch, num_integration_steps=int(current_L), alpha=0.01, beta=0)

    # Carry blackjax via closure to keep function JIT-friendly
    sghmc_kernel = kernel_factory(scan_sghmc_body.blackjax)  # type: ignore[attr-defined]
    next_state = sghmc_kernel.step(mc_key, current_state, y_batch, current_step_size)
    new_carry = (next_state, next_key)
    return new_carry, next_state


@partial(jax.jit, static_argnums=(3, 4, 5, 6,7))
def scan_sgld_body(carry, _, y_full, N_total, current_batch_size, current_L, current_step_size, make_batched_log_p_fn):
    current_state, current_key = carry
    next_key, mc_key, batch_key = jax.random.split(current_key, 3)
    indices = jax.random.choice(batch_key, N_total, shape=(current_batch_size,), replace=False)
    #indices = jax.random.permutation(batch_key, N_total)[:current_batch_size]
    y_batch = y_full[indices, :]

    logp_batch = make_batched_log_p_fn(y_batch, N_total, current_batch_size)
    grad_logp_batch = lambda x, _: jax.grad(logp_batch)(x)

    def kernel_factory(bjx):
        return bjx.sgld(grad_logp_batch)

    sgld_kernel = kernel_factory(scan_sgld_body.blackjax)  # type: ignore[attr-defined]
    next_state = sgld_kernel.step(mc_key, current_state, y_batch, current_step_size, current_L)
    new_carry = (next_state, next_key)
    return new_carry, next_state


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def scan_mclmc_body(carry, _, y_full, N_total, current_batch_size, current_L, current_step_size, make_batched_log_p_fn, use_preconditioning=True):
    current_state, current_key, moving_mean, moving_std = carry
    next_key, mc_key, batch_key = jax.random.split(current_key, 3)
    #total = y_full.shape[0]
    indices = jax.random.choice(batch_key, N_total, shape=(current_batch_size,), replace=False)
    #indices = jax.random.permutation(batch_key, N_total)[:current_batch_size]
    y_batch = y_full[indices, :]

    logp_batch = make_batched_log_p_fn(y_batch, N_total, current_batch_size)

    current_gradient = jax.grad(logp_batch)
    grad_logp_batch = current_gradient(current_state.position)

    moving_mean = 0.99 * moving_mean + 0.01 * grad_logp_batch
    moving_std = jnp.sqrt(0.99 * moving_std**2 + 0.01 * (grad_logp_batch - moving_mean) ** 2)

    noise_condition = jax.lax.cond(
        use_preconditioning,
        lambda: 1/moving_std * jnp.sqrt(10)/jnp.linalg.norm(1/moving_std)  ,
        lambda: jnp.ones_like(current_state.position),
    )

    def kernel_factory(bjx):
        return bjx.mclmc(
            logdensity_fn=logp_batch,
            L=current_L,
            step_size=current_step_size,
            integrator=bjx.mcmc.integrators.isokinetic_mclachlan,
            sqrt_diag_cov=noise_condition,
        )

    mclmc_kernel = kernel_factory(scan_mclmc_body.blackjax)  # type: ignore[attr-defined]
    next_state, info = mclmc_kernel.step(mc_key, current_state)
    new_carry = (next_state, next_key, moving_mean, moving_std)
    collected_output = next_state.position
    return new_carry, collected_output


def bind_blackjax_for_scan_bodies(blackjax):
    """Attach blackjax module onto JITed scan bodies for kernel creation.

    This avoids capturing large modules in closures at jit-time.
    """
    scan_sghmc_body.blackjax = blackjax  # type: ignore[attr-defined]
    scan_sgld_body.blackjax = blackjax  # type: ignore[attr-defined]
    scan_mclmc_body.blackjax = blackjax  # type: ignore[attr-defined]
