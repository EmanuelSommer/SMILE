"""Sampler Configuration."""

import warnings
from dataclasses import field
from enum import Enum
from typing import Any, Optional

from dataserious import BaseConfig

from src.bnns.priors import PriorDist
from src.training.scheduler import (
    cosine_annealing_scheduler,
    linear_decay_scheduler,
)


class GetSampler(str, Enum):
    """Sampler Names.

    Notes:
        This enum class defines the samplers that can be used for Bayesian Inference.
        To extend the possible samplers, add a new value to the `GetSampler` enum.
        The `get_kernel` and `get_warmup_kernel` methods are used to get the kernel
        and warmup kernel for the sampler respectively.
        The `get_kernel` and `get_warmup_kernel` methods return the kernel
        not the sampler itself. The sampler is later initialized in the
        `src.training.trainer` module.
    """

    NUTS = "nuts"
    MCLMC = "mclmc"
    HMC = "hmc"
    SGLD = "sgld"
    ADASGHMC = "adasghmc"
    SGHMC = "sghmc"
    RMSPROP = "rmsprop"
    SMILE = "smile"
    SMILEY = "smiley"
    SMILET = "smilet"
    NOISYSMILE = "noisysmile"
    SGDMOM = "sgdmom"

    def get_kernel(self):
        """Get sampling kernel."""
        from src.kernels import KERNELS

        if self.value not in KERNELS:
            raise NotImplementedError(
                f"Sampler for {self.value} is not yet implemented."
            )
        return KERNELS[self.value]

    def get_warmup_kernel(self):
        """Get warmup kernel."""
        from src.kernels import WARMUP_KERNELS

        if self.value not in WARMUP_KERNELS:
            raise NotImplementedError(
                f"Warmup Kernel for {self.value} is not yet implemented."
            )
        return WARMUP_KERNELS[self.value]

    def is_minibatch(self):
        """Check if the sampler is minibatch."""
        return self.value in [
            GetSampler.ADASGHMC,
            GetSampler.SGHMC,
            GetSampler.RMSPROP,
            GetSampler.SMILE,
            GetSampler.SMILEY,
            GetSampler.SMILET,
            GetSampler.NOISYSMILE,
            GetSampler.SGLD,
            GetSampler.SGDMOM,
        ]


class Scheduler(str, Enum):
    """Learning Rate Scheduler Names."""

    COSINE = "Cosine"
    LINEAR = "Linear"

    def get_scheduler(self):
        """Get the learning rate scheduler."""
        if self == Scheduler.COSINE:
            return cosine_annealing_scheduler
        if self == Scheduler.LINEAR:
            return linear_decay_scheduler
        raise NotImplementedError(
            f"Learning Rate Scheduler for {self.value} is not yet implemented."
        )


class PriorConfig(BaseConfig):
    """Configuration for the prior distribution on the model parameters.

    Note:
        The `name` should be a `PriorDist` enum value which defines the complete
        prior distribution, it can be a general distribution or a pre-defined one.
        To extend the possible priors, add a new value to the `PriorDist` enum.
        and extend the `get_prior` method accordingly. Through `parameters` field
        the user can pass as many keyword arguments from the configuration file
        as needed for the initialization of the prior distribution.
    """

    name: PriorDist = field(
        default=PriorDist.StandardNormal,
        metadata={"description": "Prior to Use", "searchable": True},
    )
    parameters: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Parameters for the prior distribution.",
            "searchable": True,
        },
    )

    def get_prior(self):
        """Get the prior distribution.

        Note:
            Get the prior by passing the parameters from the config to `get_prior`
            method of the `PriorDist` enum. See the `PriorDist` enum for more details.
        """
        return self.name.get_prior(**self.parameters)


class SchedulerConfig(BaseConfig):
    """Scheduler Configuration."""

    name: Optional[Scheduler] = field(
        default=None,
        metadata={"description": "Scheduler to Use.", "searchable": True},
    )
    exploration: float = field(
        default=0.25,
        metadata={"description": "Exploration Ratio.", "searchable": True},
    )
    init_step_size: float = field(
        default=1e-3,
        metadata={"description": "Initial Learning Rate.", "searchable": True},
    )
    target_step_size: float = field(
        default=0.0,
        metadata={"description": "Target Learning Rate.", "searchable": True},
    )
    n_cycles: int = field(
        default=4,
        metadata={
            "description": "Number of Cycles [Cosine Scheduler].",
            "searchable": True,
        },
    )

    def __post_init__(self):
        """Post Initialization for the Scheduler Configuration."""
        super().__post_init__()
        n_cycles_default = self.__class__.__dataclass_fields__["n_cycles"].default
        if self.name == Scheduler.LINEAR and self.n_cycles != n_cycles_default:
            self._modify_field(**{"n_cycles": n_cycles_default})
            warnings.warn("Ignoring n_cycles in Linear Scheduler.", UserWarning)

    def get_scheduler(self, n_steps: int):
        """Get the learning rate scheduler."""
        if self.name == Scheduler.COSINE:
            return self.name.get_scheduler()(
                n_steps=n_steps,
                n_cycles=self.n_cycles,
                init_lr=self.init_step_size,
                target_lr=self.target_step_size,
                exploration_ratio=self.exploration,
            )
        if self.name == Scheduler.LINEAR:
            return self.name.get_scheduler()(
                n_steps=n_steps,
                init_lr=self.init_step_size,
                target_lr=self.target_step_size,
                exploration_ratio=self.exploration,
            )
        return None


class SamplerConfig(BaseConfig):
    """Sampler Configuration."""

    name: GetSampler = field(
        default=GetSampler.NUTS, metadata={"description": "Sampler to Use."}
    )
    epoch_wise_sampling: bool = field(
        default=False,
        metadata={
            "description": "Perform epoch-wise or batch-wise in minibatch sampling."
        },
    )
    params_frozen: list[str] = field(
        default_factory=list,
        metadata={
            "description": (
                "Point delimited parameter names in pytree to freeze."
                "(not yet fully implemented)"
            )
        },
    )
    batch_size: int | None = field(
        default=None,
        metadata={"description": "Batch Size in SBI Training.", "searchable": True},
    )
    burn_in: int = field(
        default=0,
        metadata={
            "description": "Number of samples to discard from the main sampling phase.",
            "searchable": True,
        },
    )
    warmup_steps: int = field(
        default=50,
        metadata={"description": "Number of warmup steps.", "searchable": True},
    )
    n_chains: int = field(
        default=2,
        metadata={"description": "Number of chains to run.", "searchable": True},
    )
    n_samples: int = field(
        default=1000,
        metadata={"description": "Number of samples to draw.", "searchable": True},
    )
    use_warmup_as_init: bool = field(
        default=True,
        metadata={
            "description": "Use params resulting from warmup as initial for sampling."
        },
    )
    n_thinning: int = field(
        default=1, metadata={"description": "Thinning.", "searchable": True}
    )
    diagonal_preconditioning: bool = field(
        default=False,
        metadata={
            "description": "Use Diagonal Preconditioning (MCLMC).",
            "searchable": True,
        },
    )
    desired_energy_var: float = field(
        default=1.0,
        metadata={
            "description": "Desired Energy Variance (SMILE).",
            "searchable": True,
        },
    )
    desired_energy_var_start: float = field(
        default=5e-4,
        metadata={
            "description": "Desired Energy Variance (MCLMC) at start of lin. decay.",
            "searchable": True,
        },
    )
    desired_energy_var_end: float = field(
        default=1e-4,
        metadata={
            "description": "Desired Energy Variance (MCLMC) at end of lin. decay.",
            "searchable": True,
        },
    )
    trust_in_estimate: float = field(
        default=1.5,
        metadata={"description": "Trust in Estimate (MCLMC).", "searchable": True},
    )
    num_effective_samples: int = field(
        default=100,
        metadata={
            "description": "Number of Effective Samples (MCLMC).",
            "searchable": True,
        },
    )
    step_size_init: float = field(
        default=0.005,
        metadata={"description": "Initial Step Size (MCLMC).", "searchable": True},
    )
    step_size: float = field(
        default=0.0001, metadata={"description": "Step Size.", "searchable": True}
    )
    mdecay: float = field(
        default=0.05, metadata={"description": "Momentum Decay.", "searchable": True}
    )
    n_integration_steps: int = field(
        default=1,
        metadata={"description": "Number of Integration Steps.", "searchable": True},
    )
    momentum_resampling: float = field(
        default=0.0,
        metadata={"description": "Momentum Resampling (adaSGHMC)", "searchable": True},
    )
    temperature: float = field(
        default=1.0, metadata={"description": "Temperature (SGLD)", "searchable": True}
    )
    smile_tuning: dict = field(
        default_factory=lambda: {
            "reset_prob": 0.99,
            "adaption_prob": 0.05,
            "adaption_strength": 0.02,
            "smoothing_factor": 0.01,
            # "abort_sd_factor": 4.0,
            # "step_size_decrease_factor": 0.98,
            # "step_size_increase_factor": 1.02,
            # "step_size_decrease_thres": 1.6449,
            # "step_size_increase_thres": 0.1257,
            # "alpha": 0.01,
        },
        metadata={
            "description": "SMILE Tuning Parameters.",
            "searchable": True,
        },
    )
    running_avg_factor: float = field(
        default=0.0,
        metadata={
            "description": "Running average factor (rmsprop)",
            "searchable": True,
        },
    )

    keep_warmup: bool = field(
        default=False, metadata={"description": "Keep warmup samples."}
    )
    prior_config: PriorConfig = field(
        default_factory=PriorConfig,
        metadata={"description": "Prior configuration for the model."},
    )
    scheduler_config_warm: SchedulerConfig = field(
        default_factory=SchedulerConfig,
        metadata={"description": "Step size Warmup Scheduler"},
    )
    scheduler_config_samp: SchedulerConfig = field(
        default_factory=SchedulerConfig,
        metadata={"description": "Step size Sampling Scheduler"},
    )

    def scheduler(self, n_steps: int, warmup: bool = True):
        """Get the learning rate scheduler."""
        if warmup:
            return self.scheduler_config_warm.get_scheduler(n_steps=n_steps)
        else:
            return self.scheduler_config_samp.get_scheduler(n_steps=n_steps)

    def __post_init__(self):
        """Post Initialization for the Sampler Configuration."""
        super().__post_init__()
        mini_batch_only = ["batch_size", "n_integration_steps", "mdecay", "step_size"]
        if self.name == GetSampler.NUTS:
            for fn in mini_batch_only:
                if getattr(self, fn) is not None:
                    default = self.__class__.__dataclass_fields__[fn].default
                    warnings.warn(f"Ignoring {fn} in NUTS Sampling.", UserWarning)
                    self._modify_field(**{fn: default})

    @property
    def prior(self):
        """Get the prior."""
        return self.prior_config.get_prior()

    def kernel(self, **kwargs):
        """Returns the kernel: see src.training.kernels for more details."""
        return self.name.get_kernel()(**self._sampler_kwargs, **kwargs)

    def warmup_kernel(self, **kwargs):
        """Returns the warmup kernel: see src.training.kernels."""
        kernel = self.name.get_warmup_kernel()
        if kernel is None:
            return None
        else:
            return kernel(**self._warmup_kwargs, **kwargs)

    @property
    def _warmup_dir_name(self):
        """Return the directory name for saving warmup samples."""
        return "sampling_warmup"

    @property
    def _dir_name(self):
        """Return the directory name for saving samples."""
        return "samples"

    @property
    def _sampler_kwargs(self):
        """Sampler configs."""
        if self.name.value in [
            GetSampler.ADASGHMC,
            GetSampler.SGHMC,
            GetSampler.RMSPROP,
        ]:
            return {
                "num_integration_steps": self.n_integration_steps,
                "mdecay": self.mdecay,
                "mresampling": self.momentum_resampling,
            }
        elif self.name.value == GetSampler.SMILE:
            return {
                "desired_energy_var": self.desired_energy_var,
                "num_integration_steps": self.n_integration_steps,
            }
        elif self.name.value in [GetSampler.SMILEY]:
            return {"num_integration_steps": self.n_integration_steps}
        elif self.name.value in [GetSampler.SMILET, GetSampler.NOISYSMILE]:
            return {
                "num_integration_steps": self.n_integration_steps,
                # "alpha": self.smile_tuning["alpha"],
                # "abort_sd_factor": self.smile_tuning["abort_sd_factor"],
                # "step_size_decrease_factor": self.smile_tuning[
                #     "step_size_decrease_factor"
                # ],
                # "step_size_increase_factor": self.smile_tuning[
                #     "step_size_increase_factor"
                # ],
                # "step_size_decrease_thres": self.smile_tuning[
                #     "step_size_decrease_thres"
                # ],
                # "step_size_increase_thres": self.smile_tuning[
                #     "step_size_increase_thres"
                # ],
                "adaption_prob": self.smile_tuning["adaption_prob"],
                "adaption_strength": self.smile_tuning["adaption_strength"],
                "smoothing_factor": self.smile_tuning["smoothing_factor"],
                "reset_prob": self.smile_tuning["reset_prob"],
                "step_size": self.step_size,
            }
        elif self.name.value == GetSampler.SGDMOM:
            return {"mdecay": self.mdecay}
        elif self.name.value == GetSampler.SGLD:
            return {"temperature": self.temperature}
        else:
            return {}

    @property
    def _warmup_kwargs(self):
        """Sampler configs."""
        if self.name.value in [GetSampler.ADASGHMC, GetSampler.SGHMC, GetSampler.SGLD]:
            return {
                "num_integration_steps": self.n_integration_steps,
                "mdecay": self.mdecay,
                "mresampling": self.momentum_resampling,
            }
        elif self.name.value == GetSampler.RMSPROP:
            return {
                "num_integration_steps": self.n_integration_steps,
                "mdecay": self.mdecay,
                "mresampling": self.momentum_resampling,
                "running_avg_factor": self.running_avg_factor,
            }
        elif self.name.value == GetSampler.SMILE:
            return {
                "desired_energy_var": self.desired_energy_var,
                "num_integration_steps": self.n_integration_steps,
            }
        elif self.name.value in [GetSampler.SMILEY]:
            return {"num_integration_steps": self.n_integration_steps}
        elif self.name.value in [GetSampler.SMILET, GetSampler.NOISYSMILE]:
            return {
                "num_integration_steps": self.n_integration_steps,
                "adaption_prob": self.smile_tuning["adaption_prob"],
                "adaption_strength": self.smile_tuning["adaption_strength"],
                "smoothing_factor": self.smile_tuning["smoothing_factor"],
                "reset_prob": self.smile_tuning["reset_prob"],
                # "alpha": self.smile_tuning["alpha"],
                # "abort_sd_factor": self.smile_tuning["abort_sd_factor"],
                # "step_size_decrease_factor": self.smile_tuning[
                #     "step_size_decrease_factor"
                # ],
                # "step_size_increase_factor": self.smile_tuning[
                #     "step_size_increase_factor"
                # ],
                # "step_size_decrease_thres": self.smile_tuning[
                #     "step_size_decrease_thres"
                # ],
                # "step_size_increase_thres": self.smile_tuning[
                #     "step_size_increase_thres"
                # ],
                "step_size": self.step_size,
            }
        elif self.name.value == GetSampler.SGDMOM:
            return {
                "mdecay": self.mdecay,
            }
        else:
            return {}

    @property
    def is_minibatch(self):
        """Check if the sampler is minibatch."""
        return self.name.is_minibatch()
