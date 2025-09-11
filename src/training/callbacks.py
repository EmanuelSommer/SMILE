"""Callbacks used in training."""

import logging
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from tqdm import tqdm

from src.kernels.sghmc import AdaSGHMCWarmup
from src.types import ParamTree
from src.utils import get_flattened_keys

logger = logging.getLogger(__name__)


def check_balance_condition(
    state: AdaSGHMCWarmup.State,
    step_size: float,
    mdecay: float,
):
    """Check wheather mdecay is big enough after adaSGHMC warmup."""
    max_sd = jnp.max(ravel_pytree(state.elementwise_sd)[0])
    return max_sd < 2 * mdecay / jnp.square(step_size)


def save_position(position: ParamTree, base: Path, idx: jnp.ndarray, n: jnp.ndarray):
    """Save the position of the model. TODO: Needs REFACTORING.

    Args:
        position: Position of the model to save.
        base: Base path to save the samples.
        idx: Index of the current chain.
        n: Index of the current sample.
    """
    leafs, _ = jax.tree.flatten(position)
    param_names = get_flattened_keys(position)
    path = base / f"{idx.item()}/sample_{n}.npz"
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    np.savez_compressed(
        path, **{name: np.array(leaf) for name, leaf in zip(param_names, leafs)}
    )
    return position


def progress_bar_scan(n_steps: int, name: str):
    """Progress bar designed for lax.scan.

    Args:
        n_steps: Number of steps in the scan to show progress for.
        name: Name of the progress bar to display.
    """
    progress_bar = tqdm(total=n_steps, desc=name)

    def _update_progress_bar(iter_num: int):
        # Update the progress bar
        _ = jax.lax.cond(
            iter_num >= 0,
            lambda _: jax.debug.callback(partial(progress_bar.update, 1)),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            iter_num == n_steps - 1,
            lambda _: jax.debug.callback(partial(progress_bar.close)),
            lambda _: None,
            operand=None,
        )

    def _progress_bar_scan(f: Callable):
        def inner(carry, xs):
            if isinstance(xs, tuple):
                n, *_ = xs
            else:
                n = xs
            result = f(carry, xs)
            _update_progress_bar(n)
            return result

        return inner

    return _progress_bar_scan
