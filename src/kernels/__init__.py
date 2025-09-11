"""What to load from training part."""

from typing import Callable, Optional, TypeAlias

from blackjax import (  # maybe wrap them in kernels.py for cleaner sampling
    hmc,
    mclmc,
    nuts,
)

from src.kernels.sghmc import (
    SGHMC,
    AdaSGHMCWarmup,
    RMSPropWarmup,
)
from src.kernels.sgld import SGLD
from src.kernels.smile import NOISYSMILE, SMILE, SMILET

__all__ = [
    "nuts",
    "hmc",
    "mclmc",
    "adasghmc",
    "sgld",
    "smile",
    "smiley",
    "sgdmom",
    "smilet",
    "noisysmile",
]

KernelRegistry: TypeAlias = dict[str, Optional[Callable]]

KERNELS: KernelRegistry = {
    "nuts": nuts,
    "hmc": hmc,
    "mclmc": mclmc,
    "adasghmc": SGHMC,
    "rmsprop": SGHMC,
    "sghmc": SGHMC,
    "sgld": SGLD,
    "smiley": SMILE, # SMILE-naive
    "smilet": SMILET, # SMILE
    "noisysmile": NOISYSMILE, # pSMILE(-naive)
}

WARMUP_KERNELS: KernelRegistry = {
    "adasghmc": AdaSGHMCWarmup,
    "rmsprop": RMSPropWarmup,
    "sghmc": None,
    "sgld": None,
    "smiley": SMILE,
    "smilet": SMILET,
    "noisysmile": NOISYSMILE,
}
