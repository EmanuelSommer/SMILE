import argparse
import json
from dataclasses import asdict
from typing import Any, Dict

from run_experiments import Config, run
from bjx_utils import DEFAULT_BLACKJAX_PARENT_DIR


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Constant Regression with Stochastic Gradient MCMC.")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    # CLI overrides (optional); when provided they override JSON values
    parser.add_argument("--algorithm", type=str, choices=["sgld", "sghmc", "mclmc"], default=None)
    parser.add_argument("--likelihood_model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--min_n_samples", type=int, default=None)
    parser.add_argument("--max_n_samples", type=int, default=None)
    parser.add_argument("--min_step_size", type=float, default=None)
    parser.add_argument("--max_step_size", type=float, default=None)
    parser.add_argument("--L", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--blackjax_path", type=str, default=None)
    # Explicit non-Gaussian gradient-noise injection (optional)
    parser.add_argument("--grad_noise_type", type=str,
                        choices=["laplacian", "student_t", "lognormal", "none"], default=None)
    parser.add_argument("--grad_noise_scale", type=float, default=None)
    parser.add_argument("--grad_noise_df", type=float, default=None)
    parser.add_argument("--grad_noise_structure", type=str,
                        choices=["isotropic", "anisotropic", "correlated", "spatially_varied"], default=None)
    ns = parser.parse_args()

    # Load JSON if provided
    cfg_dict: Dict[str, Any] = {}
    if ns.config:
        with open(ns.config, "r") as f:
            try:
                cfg_dict = json.load(f)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Failed to parse JSON config {ns.config}: {e}")

    # Overlay CLI overrides where provided (not None)
    for key in [
        "algorithm",
        "likelihood_model",
        "batch_size",
        "min_n_samples",
        "max_n_samples",
        "min_step_size",
        "max_step_size",
        "L",
        "seed",
        "blackjax_path",
        "grad_noise_type",
        "grad_noise_scale",
        "grad_noise_df",
        "grad_noise_structure",
    ]:
        val = getattr(ns, key)
        if val is not None:
            cfg_dict[key] = val

    # Validate required key 'algorithm'
    if "algorithm" not in cfg_dict or cfg_dict["algorithm"] is None:
        raise SystemExit("Missing required 'algorithm'. Provide it in --config JSON or via --algorithm.")

    # Instantiate Config; relies on dataclass defaults for any missing optional keys
    if "blackjax_path" not in cfg_dict or cfg_dict["blackjax_path"] is None:
        cfg_dict["blackjax_path"] = DEFAULT_BLACKJAX_PARENT_DIR

    cfg = Config(**cfg_dict)
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    print(f"Running with config: {asdict(cfg)}")
    run(cfg)