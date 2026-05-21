from functools import partial
from typing import Tuple, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np


def _add_gradient_noise(logp_fn, key, dim, noise_type, noise_scale, noise_df,
                        noise_structure, grad_conds, grad_Q_mat,
                        spatial_dim=2, spatial_scale=6.5):
    """Wrap logp_fn to inject structured non-Gaussian noise into its gradient.

    Adds a linear term to logp so that
        grad_x[noisy_logp] = grad_x[logp] + noise_vec
    (or with an additional position-dependent factor for spatially_varied).

    The base per-component samples are normalised to unit variance before
    being multiplied by noise_scale, so the same noise_scale gives identical
    gradient-noise magnitude across distributions.

    noise_type: "laplacian", "student_t", "lognormal", or None / "none".
    noise_structure:
        "isotropic"        - all dims have std = noise_scale
        "anisotropic"      - per-dim std = noise_scale * grad_conds[d]
        "correlated"       - anisotropic noise rotated by grad_Q_mat
        "spatially_varied" - correlated noise scaled by exp(-x[spatial_dim] / spatial_scale)
    """
    if noise_type is None or noise_type == "none":
        return logp_fn

    key1, _ = jax.random.split(key)

    # Generate unit-variance base noise from chosen distribution
    if noise_type == "laplacian":
        u = jax.random.uniform(key1, shape=(dim,), minval=1e-7, maxval=1 - 1e-7)
        base_noise = jnp.sign(u - 0.5) * jnp.log1p(-2.0 * jnp.abs(u - 0.5))
        # Laplace(0, 1) has variance 2; rescale to unit variance
        base_noise = base_noise / jnp.sqrt(2.0)
    elif noise_type == "student_t":
        raw = jax.random.t(key1, df=noise_df, shape=(dim,))
        # Student-t(df) has variance df/(df-2) for df>2
        base_noise = raw / jnp.sqrt(noise_df / (noise_df - 2.0))
    elif noise_type == "lognormal":
        # noise_df is reused as the lognormal sigma parameter.
        sigma_ln = noise_df
        z = jax.random.normal(key1, shape=(dim,))
        raw = jnp.exp(sigma_ln * z)
        mean_ln = jnp.exp(sigma_ln ** 2 / 2.0)
        std_ln = jnp.sqrt(jnp.expm1(sigma_ln ** 2) * jnp.exp(sigma_ln ** 2))
        base_noise = (raw - mean_ln) / std_ln
    else:
        return logp_fn

    if noise_structure == "anisotropic":
        noise_vec = noise_scale * grad_conds * base_noise
    elif noise_structure in ("correlated", "spatially_varied"):
        noise_vec = noise_scale * grad_Q_mat @ (grad_conds * base_noise)
    else:  # isotropic
        noise_vec = noise_scale * base_noise

    if noise_structure == "spatially_varied":
        def noisy_logp(x):
            return logp_fn(x) + jnp.dot(noise_vec, x) * jnp.exp(-x[spatial_dim] / spatial_scale)
        return noisy_logp

    def noisy_logp(x):
        return logp_fn(x) + jnp.dot(noise_vec, x)
    return noisy_logp


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


# const_logp_fn (optional): when not None, skip mini-batch sampling each step and
# use this pre-baked full-dataset log-density. Useful when gradient noise comes
# entirely from explicit injection (laplacian / student_t / lognormal) rather
# than subsampling, removing the CLT-suppressed mini-batch noise.
#
# grad_conds, grad_Q_mat: traced JAX arrays for anisotropic / correlated /
# spatially_varied gradient-noise structure. Ignored when noise_structure is
# "isotropic" or grad_noise_type is None.


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 11, 12, 15, 16))
def scan_sghmc_body(carry, _, y_full, N_total, current_batch_size, current_L,
                    current_step_size, make_batched_log_p_fn,
                    grad_noise_type=None, grad_noise_scale=1.0, grad_noise_df=3.0,
                    const_logp_fn=None, grad_noise_structure="isotropic",
                    grad_conds=None, grad_Q_mat=None,
                    grad_noise_spatial_dim=2, grad_noise_spatial_scale=6.5):
    current_state, current_key = carry
    dim = y_full.shape[1]

    if const_logp_fn is not None:
        next_key, mc_key, noise_key = jax.random.split(current_key, 3)
        logp_batch = const_logp_fn
        y_batch = y_full[:current_batch_size]  # dummy — ignored by grad
    else:
        next_key, mc_key, batch_key, noise_key = jax.random.split(current_key, 4)
        indices = jax.random.choice(batch_key, N_total, shape=(current_batch_size,), replace=False)
        y_batch = y_full[indices, :]
        logp_batch = make_batched_log_p_fn(y_batch, N_total, current_batch_size)

    logp_batch = _add_gradient_noise(logp_batch, noise_key, dim,
                                     grad_noise_type, grad_noise_scale, grad_noise_df,
                                     grad_noise_structure, grad_conds, grad_Q_mat,
                                     grad_noise_spatial_dim, grad_noise_spatial_scale)
    grad_logp_batch = lambda x, _: jax.grad(logp_batch)(x)

    def kernel_factory(bjx):
        return bjx.sghmc(grad_logp_batch, num_integration_steps=int(current_L), alpha=0.01, beta=0)

    # Carry blackjax via closure to keep function JIT-friendly
    sghmc_kernel = kernel_factory(scan_sghmc_body.blackjax)  # type: ignore[attr-defined]
    next_state = sghmc_kernel.step(mc_key, current_state, y_batch, current_step_size)
    new_carry = (next_state, next_key)
    return new_carry, next_state


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 11, 12, 15, 16))
def scan_sgld_body(carry, _, y_full, N_total, current_batch_size, current_L,
                   current_step_size, make_batched_log_p_fn,
                   grad_noise_type=None, grad_noise_scale=1.0, grad_noise_df=3.0,
                   const_logp_fn=None, grad_noise_structure="isotropic",
                   grad_conds=None, grad_Q_mat=None,
                   grad_noise_spatial_dim=2, grad_noise_spatial_scale=6.5):
    current_state, current_key = carry
    dim = y_full.shape[1]

    if const_logp_fn is not None:
        next_key, mc_key, noise_key = jax.random.split(current_key, 3)
        logp_batch = const_logp_fn
        y_batch = y_full[:current_batch_size]  # dummy — ignored by grad
    else:
        next_key, mc_key, batch_key, noise_key = jax.random.split(current_key, 4)
        indices = jax.random.choice(batch_key, N_total, shape=(current_batch_size,), replace=False)
        y_batch = y_full[indices, :]
        logp_batch = make_batched_log_p_fn(y_batch, N_total, current_batch_size)

    logp_batch = _add_gradient_noise(logp_batch, noise_key, dim,
                                     grad_noise_type, grad_noise_scale, grad_noise_df,
                                     grad_noise_structure, grad_conds, grad_Q_mat,
                                     grad_noise_spatial_dim, grad_noise_spatial_scale)
    grad_logp_batch = lambda x, _: jax.grad(logp_batch)(x)

    def kernel_factory(bjx):
        return bjx.sgld(grad_logp_batch)

    sgld_kernel = kernel_factory(scan_sgld_body.blackjax)  # type: ignore[attr-defined]
    next_state = sgld_kernel.step(mc_key, current_state, y_batch, current_step_size, current_L)
    new_carry = (next_state, next_key)
    return new_carry, next_state


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17))
def scan_mclmc_body(carry, _, y_full, N_total, current_batch_size, current_L,
                    current_step_size, make_batched_log_p_fn, use_preconditioning=True,
                    grad_noise_type=None, grad_noise_scale=1.0, grad_noise_df=3.0,
                    const_logp_fn=None, grad_noise_structure="isotropic",
                    grad_conds=None, grad_Q_mat=None,
                    grad_noise_spatial_dim=2, grad_noise_spatial_scale=6.5):
    current_state, current_key, moving_mean, moving_std = carry
    dim = y_full.shape[1]

    if const_logp_fn is not None:
        next_key, mc_key, noise_key = jax.random.split(current_key, 3)
        logp_batch = const_logp_fn
    else:
        next_key, mc_key, batch_key, noise_key = jax.random.split(current_key, 4)
        indices = jax.random.choice(batch_key, N_total, shape=(current_batch_size,), replace=False)
        y_batch = y_full[indices, :]
        logp_batch = make_batched_log_p_fn(y_batch, N_total, current_batch_size)

    logp_batch = _add_gradient_noise(logp_batch, noise_key, dim,
                                     grad_noise_type, grad_noise_scale, grad_noise_df,
                                     grad_noise_structure, grad_conds, grad_Q_mat,
                                     grad_noise_spatial_dim, grad_noise_spatial_scale)

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
