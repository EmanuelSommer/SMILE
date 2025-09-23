import os
import re
import time
from dataclasses import dataclass
from datetime import date
from functools import partial
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np

from bjx_utils import setup_blackjax_path, DEFAULT_BLACKJAX_PARENT_DIR
from samplers import (
    get_likelihood_factories,
    make_sampling_state,
    scan_mclmc_body,
    scan_sgld_body,
    scan_sghmc_body,
    bind_blackjax_for_scan_bodies,
)
from analysis import post_analysis


@dataclass
class Config:
    algorithm: Literal["sgld", "sghmc", "mclmc"]
    likelihood_model: str = "ill_condition_Gaussian_precond"
    batch_size: int = 256
    min_n_samples: int = 100
    max_n_samples: int = 1000
    min_step_size: float = 1e-4
    max_step_size: float = 1e-1
    L: float = 10.0
    seed: int = int(date.today().strftime("%Y%m%d"))
    blackjax_path: str = DEFAULT_BLACKJAX_PARENT_DIR
    condition_numbers: float = 100.0
    noise_type: Optional[str] = None
    noise_scale: Optional[float] = None
    random_rotation: Optional[bool] = None
    center: Optional[bool] = None
    use_preconditioning: bool = False


def run(cfg: Config):
    blackjax = setup_blackjax_path(cfg.blackjax_path)
    bind_blackjax_for_scan_bodies(blackjax)

    # Problem constants
    DIM = 10
    N_TOTAL = 409600
    CONSTANT_VAL = 0.0

    rng_key_master = jax.random.key(cfg.seed)
    rng_key_data_gen_init, rng_key_runs_master = jax.random.split(rng_key_master)

    # Resolve likelihood factories by name
    create_dataset, make_batched_log_p, E_f, Var_f = get_likelihood_factories(
        cfg.likelihood_model,
        Condition_numbers=cfg.condition_numbers,
        N_total=N_TOTAL,
        dim_val=DIM,
        constant_val=CONSTANT_VAL,
        batch_size=None,
        seed=0,
        noise_type=cfg.noise_type,
        noise_scale=cfg.noise_scale,
        random_rotation=cfg.random_rotation,
        center=cfg.center,
    )
    y_data = create_dataset(N_TOTAL, DIM, CONSTANT_VAL, seed=0)
    initial_position = np.mean(y_data, axis=0)
    initial_position = jnp.zeros_like(initial_position)
    if cfg.likelihood_model == "rosenbrock":
        initial_position = jnp.array([1.0,]*(DIM//2)+[2.0,]*(DIM//2))
    y_data_jnp = jnp.array(y_data)

    step_sizes_to_run = np.logspace(np.log10(cfg.min_step_size), np.log10(cfg.max_step_size), 15)
    n_samples_to_run = np.unique(
        np.logspace(np.log10(cfg.min_n_samples), np.log10(cfg.max_n_samples), 2).astype(int)
    )

    total_runs = len(step_sizes_to_run) * len(n_samples_to_run)
    current_run_count = 0

    base_results_dir = "/pscratch/sd/d/dkn16/constant_reg_results/"
    algorithm_results_dir = os.path.join(base_results_dir, cfg.algorithm)
    # sanitize likelihood model for folder name
    llh_folder = re.sub(r"[^A-Za-z0-9._-]", "_", str(cfg.likelihood_model))
    model_results_dir = os.path.join(algorithm_results_dir, llh_folder)
    os.makedirs(model_results_dir, exist_ok=True)

    for current_n_samples_val in n_samples_to_run:
        for current_step_size_val in step_sizes_to_run:
            current_run_count += 1
            print(
                f"\n--- Running experiment {current_run_count}/{total_runs} ---\n"
                f"Algorithm: {cfg.algorithm}, N_samples: {current_n_samples_val}, "
                f"Step_size: {current_step_size_val:.3e}, L: {cfg.L}, Batch_size: {cfg.batch_size}"
            )

            # Prepare scan body once per (n_samples, step_size)
            if cfg.algorithm == "mclmc":
                scan_body_partial = partial(
                    scan_mclmc_body,
                    y_full=y_data_jnp,
                    N_total=N_TOTAL,
                    current_batch_size=cfg.batch_size,
                    current_L=cfg.L,
                    current_step_size=current_step_size_val,
                    make_batched_log_p_fn=make_batched_log_p,
                    use_preconditioning=cfg.use_preconditioning,
                )
            elif cfg.algorithm == "sgld":
                scan_body_partial = partial(
                    scan_sgld_body,
                    y_full=y_data_jnp,
                    N_total=N_TOTAL,
                    current_batch_size=cfg.batch_size,
                    current_L=cfg.L,
                    current_step_size=current_step_size_val,
                    make_batched_log_p_fn=make_batched_log_p,
                )
            else:
                scan_body_partial = partial(
                    scan_sghmc_body,
                    y_full=y_data_jnp,
                    N_total=N_TOTAL,
                    current_batch_size=cfg.batch_size,
                    current_L=cfg.L,
                    current_step_size=current_step_size_val,
                    make_batched_log_p_fn=make_batched_log_p,
                )

            logdensity_fn_for_mclmc_init = make_batched_log_p(y_data_jnp[: cfg.batch_size], N_TOTAL, cfg.batch_size)

            # Replications over seeds: seed, seed+1, seed+2
            rep_metrics = []
            for rep_idx, rep_seed in enumerate([cfg.seed + i + 3 for i in range(7)]):
                print(f"  Replication {rep_idx + 1}/7 with seed {rep_seed}")
                rng_key_master_rep = jax.random.key(rep_seed)
                rng_key_init_run, rng_key_sampling_run = jax.random.split(rng_key_master_rep)

                state = make_sampling_state(
                    blackjax,
                    logdensity_fn_for_mclmc_init,
                    initial_position,
                    rng_key_sampling_run,
                    algorithm=cfg.algorithm,
                )

                initial_carry = (state, rng_key_init_run)
                if cfg.algorithm == "mclmc":
                    initial_carry = (
                        state,
                        rng_key_init_run,
                        jnp.zeros_like(initial_position),
                        jnp.ones_like(initial_position),
                    )

                start_time = time.time()
                final_carry, samples_tuple = jax.lax.scan(
                    scan_body_partial, initial_carry, None, length=current_n_samples_val
                )
                elapsed = time.time() - start_time

                samples_to_save = np.array(samples_tuple)

                filename_parts = [
                    "constant_reg_samples",
                    f"bs{cfg.batch_size}",
                    f"ns{current_n_samples_val}",
                    f"step{current_step_size_val:.3e}",
                    f"seed{rep_seed}",
                ]
                if cfg.algorithm in ["sghmc", "mclmc"]:
                    filename_parts.append(f"L{cfg.L}")

                out_path = os.path.join(model_results_dir, "_".join(filename_parts) + ".npy")
                np.save(out_path, samples_to_save)
                print(f"    Sampling took {elapsed:.2f}s. Samples saved to {out_path}")

                # Run placeholder analysis on the samples
                metric = post_analysis(samples_to_save, E_f=E_f, var_f=Var_f)
                rep_metrics.append(metric)
                print(f"    Post-analysis metric: {metric:.6f}")

            avg_metric = float(np.mean(rep_metrics)) if len(rep_metrics) > 0 else float("nan")
            std_metric = float(np.std(rep_metrics)) if len(rep_metrics) > 0 else float("nan")
            print(f"  Average post-analysis metric over 7 replications: {avg_metric:.6f}")
            print(f"  Standard deviation of post-analysis metric over 7 replications: {std_metric:.6f}")
