"""Implementation of the Improved Variational Online Newton (IVON) optimizer.

The script is inspired by the implementation in: https://github.com/ysngshn/ivon-optax
"""

import dataclasses
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax


def randn_like(rng: jax.random.PRNGKey, params):
    """Generate random normal noise with same structure as params."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(rng, len(leaves))
    noise_leaves = [
        jax.random.normal(key, leaf.shape, leaf.dtype)
        for key, leaf in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, noise_leaves)


@dataclasses.dataclass
class IVONState:
    """State for the IVON optimizer using dataclass for easier updates."""

    count: jnp.ndarray
    momentum: optax.Updates
    hessian: optax.Updates
    sigma: optax.Updates
    grad_acc: Optional[optax.Updates]
    noise_acc: Optional[optax.Updates]
    acc_count: jnp.ndarray
    init_seed: jnp.ndarray

    def replace(self, **kwargs):
        """Create a new state with updated fields."""
        return dataclasses.replace(self, **kwargs)

    def tree_flatten(self):
        """Flatten the state for PyTree."""
        arrays = (
            self.count,
            self.momentum,
            self.hessian,
            self.sigma,
            self.grad_acc,
            self.noise_acc,
            self.acc_count,
            self.init_seed,
        )
        return arrays, None

    @classmethod
    def tree_unflatten(cls, aux_data, arrays):
        """Unflatten the state for PyTree."""
        return cls(*arrays)


# Register the custom dataclass as a PyTree
jax.tree_util.register_pytree_node(
    IVONState,
    IVONState.tree_flatten,
    IVONState.tree_unflatten,
)


def init_fn(
    params,
    h0: float,
    weight_decay: float,
    ess: float,
    seed: int,
) -> IVONState:
    """Initialize IVON state."""
    zeros_like = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t), params)
    h0_like = jax.tree_util.tree_map(lambda t: jnp.full_like(t, h0), params)
    sigma = jax.tree_util.tree_map(
        lambda h: 1.0 / jnp.sqrt(ess * (h + weight_decay)), h0_like
    )

    return IVONState(
        count=jnp.ones([], jnp.int32),  # start at 1 to avoid division by zero
        momentum=zeros_like,
        hessian=h0_like,
        sigma=sigma,
        grad_acc=zeros_like,
        noise_acc=zeros_like,
        acc_count=jnp.zeros([], jnp.int32),
        init_seed=jnp.array(seed, jnp.int32),
    )


def update_fn(
    updates: optax.Updates,
    state: IVONState,
    samples: Optional[optax.Params],
    beta1: float,
    beta2: float,
    weight_decay: float,
    ess: float,
    noise: Optional[optax.Updates],
) -> Tuple[optax.Updates, IVONState]:
    """Update IVON state and compute parameter updates."""
    if noise is not None:
        grad = updates
        _noise = noise
    else:  # Final update step after accumulating gradients
        acc_count = state.acc_count
        grad = jax.tree_util.tree_map(lambda g: g / acc_count, state.grad_acc)
        _noise = jax.tree_util.tree_map(lambda n: n / acc_count, state.noise_acc)

    momentum_new = jax.tree_util.tree_map(
        lambda grad, momentum: beta1 * momentum + (1 - beta1) * grad,
        grad,
        state.momentum,
    )

    hessian_hat = jax.tree_util.tree_map(
        lambda g, n, s: g * n / (s**2),
        grad,
        _noise,
        state.sigma,
    )

    hessian_new = jax.tree_util.tree_map(
        lambda h, h_hat: (
            beta2 * h
            + (1 - beta2) * h_hat
            + 0.5 * (1 - beta2) ** 2 * (h - h_hat) ** 2 / (h + weight_decay)
        ),
        state.hessian,
        hessian_hat,
    )

    bias_correction = 1 - beta1**state.count
    momentum_corr = jax.tree_util.tree_map(lambda m: m / bias_correction, momentum_new)

    updates = jax.tree_util.tree_map(
        lambda momentum_corr, hessian_new, p: (momentum_corr + weight_decay * p)
        / (hessian_new + weight_decay),
        momentum_corr,
        hessian_new,
        samples,
    )

    sigma_new = jax.tree_util.tree_map(
        lambda h: 1.0 / jnp.sqrt(ess * (h + weight_decay)), hessian_new
    )

    if noise is not None:
        return updates, state.replace(
            grad_acc=jax.tree_util.tree_map(lambda a, g: a + g, state.grad_acc, grad),
            noise_acc=jax.tree_util.tree_map(
                lambda a, n: a + n, state.noise_acc, noise
            ),
            acc_count=state.acc_count + 1,
        )
    else:
        return updates, state.replace(
            count=state.count + 1,
            momentum=momentum_corr,
            hessian=hessian_new,
            sigma=sigma_new,
            grad_acc=jax.tree_util.tree_map(
                lambda t: jnp.zeros_like(t), state.grad_acc
            ),
            noise_acc=jax.tree_util.tree_map(
                lambda t: jnp.zeros_like(t), state.noise_acc
            ),
            acc_count=jnp.zeros([], jnp.int32),
            init_seed=state.init_seed,
        )


def get_param_sample(
    rng: jax.random.PRNGKey,
    params: optax.Updates,
    state: IVONState,
    ess: float,
    weight_decay: float,
) -> Tuple[optax.Updates, optax.Updates]:
    """Sample parameters following paper's method."""
    noise = jax.tree_util.tree_map(
        lambda n, h: n / jnp.sqrt(ess * (h + weight_decay)),
        randn_like(rng, params),
        state.hessian,
    )
    sample = jax.tree_util.tree_map(lambda p, n: p + n, params, noise)
    return sample, noise


def scale_by_ivon(
    beta1: float,
    beta2: float,
    h0: float,
    weight_decay: float,
    ess: float,
    num_mc_samples: int,
    seed: int,
) -> optax.GradientTransformation:
    """Create IVON gradient transformation."""

    def init(params):
        return init_fn(params, h0, weight_decay, ess, seed)

    def update(updates, state, params=None):
        if num_mc_samples > 0:
            base_rng = jax.random.PRNGKey(state.init_seed)
            rng = jax.random.fold_in(base_rng, state.count)
            keys = jax.random.split(rng, num_mc_samples)

            for key in keys:
                sample, noise = get_param_sample(key, params, state, ess, weight_decay)
                _updates, state = update_fn(
                    updates, state, sample, beta1, beta2, weight_decay, ess, noise
                )
        else:
            noise = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t), params)
            _updates, state = update_fn(
                updates, state, params, beta1, beta2, weight_decay, ess, noise
            )

        return update_fn(
            updates,
            state,
            params,
            beta1,
            beta2,
            weight_decay,
            ess,
            None,
        )

    return optax.GradientTransformation(init, update)


def ivon(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    h0: float = 1.0,
    weight_decay: float = 1e-4,
    ess: float = 100.0,
    rescale_lr: bool = True,
    clip_radius: float = float("inf"),
    num_mc_samples: int = 0,
    seed: int = 0,
) -> optax.GradientTransformation:
    """Create IVON optimizer.

    Args:
        learning_rate: Learning rate scaling factor
        beta1: Exponential decay rate for momentum estimation
        beta2: Exponential decay rate for Hessian estimation
        h0: Initial Hessian value
        weight_decay: Weight decay coefficient (δ in paper)
        ess: Effective sample size (λ in paper)
        rescale_lr: Whether to rescale learning rate by (h0 + weight_decay)
        clip_radius: Maximum allowed gradient norm. Default: infinity (no clipping)
        num_mc_samples: Number of Monte Carlo samples to use for sampling parameters
        seed: Random seed for parameter sampling
    Returns:
        An optax.GradientTransformation implementing IVON with the specified options
    """
    ivon_transform = scale_by_ivon(
        beta1=beta1,
        beta2=beta2,
        h0=h0,
        weight_decay=weight_decay,
        ess=ess,
        num_mc_samples=num_mc_samples,
        seed=seed,
    )

    if rescale_lr:
        lr_scale = [
            optax.scale_by_learning_rate(learning_rate),
            optax.scale(h0 + weight_decay),
        ]
    else:
        lr_scale = [optax.scale_by_learning_rate(learning_rate)]

    transforms = [ivon_transform]

    if clip_radius < float("inf"):
        transforms.append(optax.clip(clip_radius))

    transforms.extend(lr_scale)

    return optax.chain(*transforms)
