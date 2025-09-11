"""Optimizers used for warmstarting in the module_sandbox."""

from typing import Callable

from optax import (
    adam,
    adamw,
    sgd,
)
from optax._src.base import GradientTransformation

from src.training.optimizers.ivon import ivon
from src.training.optimizers.lr_schedules import (
    cosine_decay_schedule,
    cosine_warmup_schedule,
    linear_schedule,
    warmup_cosine_decay_schedule,
)

__all__ = ["IVON"]

OPTIMIZERS: dict[str, Callable[..., GradientTransformation]] = {
    "adam": adam,
    "adamw": adamw,
    "sgd": sgd,
    "ivon": ivon,
}

LR_SCHEDULES: dict[str, Callable] = {
    "cosine": cosine_decay_schedule,
    "cosine_warmup": cosine_warmup_schedule,
    "cosine_decay": warmup_cosine_decay_schedule,
    "linear": linear_schedule,
}
