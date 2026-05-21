# Stochastic Microcanonical Langevin Ensembles (SMILE) 😃

Code for the ICML 2026 paper: [**Can Microcanonical Langevin Dynamics Leverage Mini-Batch Gradient Noise?**](https://arxiv.org/abs/2602.06500)

Below you can find a qualitative contextualization of the newly proposed and explored methods (in blue) relative to prior work (in grey).

<p align="center">
    <img src="overviewsmile.png" alt="Flowchart" style="width: 66%;">
</p>


## Setup

We use python `3.12` (newer stable versions should work as well) and [Poetry](https://python-poetry.org/) (make sure you have it installed) to manage dependencies. To install the dependencies within a virtual environment, run the following commands:


```bash
python -m venv venv
source venv/bin/activate
poetry install --all-extras
```

## File Structure

```
.
├── data/                    Data folder
├── scripts/
│   ├── experiment_configs/  Folder with folders of detailed experiment configuration files
│   ├── experiment_utils/    Folder with a CLI tool making it easy to aggregate results across multiple experiments
│   └── analytical/          Folder with folders of detailed analytical benchmark configuration files
├── src/                     Source code of the project
├── analytical/              Analytical benchmark source code
├── README.md                This file
├── pyproject.toml           Poetry configuration file
└── poetry.lock              Poetry generated file for managing dependencies
```

> **Note:** This codebase is partly an adaptation/extension of the following codebases: [MILE](https://github.com/EmanuelSommer/MILE), [SAI](https://github.com/EmanuelSommer/sampled-approx-posteriors), and [dataserious](https://github.com/Noza23/dataserious).

## Usage

The **Bayesian Neural Network experiments** can be easily exectuted using the `src` module. To see all available options, run:

```bash
python -m src -h
```

To run a single experiment on 10 available cores, use the following command:

```bash
python -m src -c scripts/experiment_configs/uci_benchmarks/tabular_regr_psmile_naive.yaml -d 10
```

To run hyperparameter sweeps or replicate experiments across multiple seeds and data splits, use the following command:

```bash
python -m src \
    -c scripts/experiment_configs/uci_benchmarks/tabular_regr_psmile_naive.yaml \
    -d 10 \
    -s scripts/experiment_configs/uci_benchmarks/uci_search_config.yaml
```

### Analytical benchmark — required blackjax fork

The analytical benchmark relies on a customized MCLMC kernel (`sqrt_diag_cov=` preconditioning) that is **not part of upstream `blackjax-devs/blackjax`**. Clone the pinned fork as a sibling of this repo and check out the pinned branch before running:

```bash
git clone https://github.com/reubenharry/blackjax.git ../blackjax
cd ../blackjax && git checkout uhmc      # pinned commit: ff71d6c
cd -
```

The path to the parent directory of the `blackjax/` package is configurable per run via the `blackjax_path` field in the JSON config (or `--blackjax_path` on the CLI); the default is `../blackjax/`.

> **SGHMC integrator note.** Upstream `reubenharry/blackjax@uhmc` ships SGHMC with the Euler-Maruyama integrator from Chen et al. (2014), which is what we report in the paper. A small additional patch — wiring an exact-OU ("exponential integrator") variant in as the SGHMC default and adding an extra `noise_condition=` knob to `blackjax.mclmc` — is available for reference at [`dkn16/blackjax_for_SMILE`](https://github.com/dkn16/blackjax_for_SMILE) (commit `ca8d40d`). It is **not** required to reproduce the numbers in this repository.

To run the **analytical benchmark**:
```bash
python analytical/run_analytical.py --config scripts/analytical/pmclmc/icg_correlated.json
```

For the non-Gaussian-noise experiments (Appendix D.2), set `grad_noise_type` (`laplacian`, `student_t`, or `lognormal`), `grad_noise_scale`, optionally `grad_noise_df` (Student-t degrees of freedom or LogNormal sigma), and `grad_noise_structure` (`isotropic` | `anisotropic` | `correlated` | `spatially_varied`) in the config JSON. When `grad_noise_type` is set, mini-batch subsampling is bypassed in favour of a fresh per-step injected noise vector, so the injected non-Gaussian noise is not diluted by the CLT.

## Results Storage

After executing experiments, all results will be automatically stored in a dedicated subfolder within the (automatically generated) `results/` directory. Each experiment's output includes:

- A copy of the `config.yaml` used for configuration
- Trained deep ensemble models in the `warmstart/` subdirectory
- Model posterior samples saved in the `samples/` subdirectory
- Evaluation metrics and outputs in the `eval/` subdirectory
- Detailed training logs

The bias for analytical examples are reported in log file, within the `log/` folder.

## Citation

```bibtex
@inproceedings{sommer2026can,
  title  = {{Can Microcanonical Langevin Dynamics Leverage Mini-Batch Gradient Noise?}},
  author = {Emanuel Sommer and Kangning Diao and Jakob Robnik and Uroš Seljak and David R{\"u}gamer},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=D2evvc90tF}
}
```