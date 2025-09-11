"""Learning rate schedules for optimizers."""

from typing import Callable


def cosine_decay_schedule(
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0,
) -> Callable:
    """Create a cosine decay learning rate schedule.

    Args:
        learning_rate: Initial learning rate
        total_steps: Total number of steps
        alpha: Minimum learning rate value as a fraction of the initial value

    Returns:
        Learning rate schedule function
    """
    try:
        import optax  # type: ignore
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install 'optax' module for learning rate scheduling"
        )

    return optax.cosine_decay_schedule(
        init_value=init_value,
        decay_steps=decay_steps,
        alpha=alpha,
    )


def warmup_cosine_decay_schedule(
    init_value: float,
    peak_value: float,
    decay_steps: int,
    warmup_steps: int,
    exponent: float = 1.0,
    end_value: float = 0.0,
) -> Callable:
    """Create a cosine decay with linear warmup learning rate schedule.

    Args:
        init_value: Initial learning rate
        peak_value: Peak learning rate after warmup
        decay_steps: Total number of steps
        warmup_steps: Number of warmup steps
        exponent: Exponent for the cosine decay
        end_value: minimallerning rate value at the end of the schedule

    Returns:
        Learning rate schedule function
    """
    try:
        import optax  # type: ignore
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install 'optax' module for learning rate scheduling"
        )

    return optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        exponent=exponent,
        end_value=end_value,
    )


def cosine_warmup_schedule(
    init_value: float,
    decay_steps: int,
    warmup_steps: int,
    alpha: float = 0.0,
    warmup_init_value: float | None = None,
) -> Callable:
    """Create a cosine decay with linear warmup learning rate schedule.

    Args:
        learning_rate: Peak learning rate after warmup
        total_steps: Total number of steps
        warmup_steps: Number of warmup steps
        alpha: Minimum learning rate value as a fraction of the initial value
        warmup_init_value: Initial learning rate for warmup (defaults to 10% of peak)

    Returns:
        Learning rate schedule function
    """
    try:
        import optax  # type: ignore
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install 'optax' module for learning rate scheduling"
        )

    if warmup_init_value is None:
        warmup_init_value = init_value * 0.1

    warmup_fn = optax.linear_schedule(
        init_value=warmup_init_value,
        end_value=init_value,
        transition_steps=warmup_steps,
    )

    cosine_fn = optax.cosine_decay_schedule(
        init_value=init_value,
        decay_steps=decay_steps,
        alpha=alpha,
    )

    return optax.join_schedules([warmup_fn, cosine_fn], [warmup_steps])


def linear_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
) -> Callable:
    """Create a linear learning rate schedule.

    Args:
        init_value: Initial learning rate
        end_value: Final learning rate
        transition_steps: Number of steps to transition from init to end value

    Returns:
        Learning rate schedule function
    """
    try:
        import optax  # type: ignore
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install 'optax' module for learning rate scheduling"
        )

    return optax.linear_schedule(
        init_value=init_value,
        end_value=end_value,
        transition_steps=transition_steps,
    )
