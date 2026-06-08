# Losing Dimensions: Geometric Memorization in Generative Diffusion

Official implementation accompanying:

**Losing Dimensions: Geometric Memorization in Generative Diffusion**  
Beatrice Achilli, Enrico Ventura, Gianluigi Silvestri, Bao Pham, Gabriel Raya,
Dmitry Krotov, Carlo Lucibello, Luca Ambrogioni

Paper: https://arxiv.org/abs/2410.08727

## Overview

This repository contains code used to study geometric memorization in diffusion
models. The experiments characterize dimensional collapse through spectra of
score-function Jacobians and compare neural score models with theoretical and
empirical predictions on synthetic data.

The repository contains:

* image diffusion experiments,
* synthetic low-dimensional spectrum experiments,
* score-Jacobian/SVD analysis code,
* notebooks for reproducing the figures.

## Repository Structure

```text
.
├── image_experiments/
│   ├── README.md
│   ├── run_svd.py
│   ├── stats_utils.py
│   ├── train_utils.py
│   ├── plot_svds_data.ipynb
│   └── src/
│
├── synthetic-experiments/
│   ├── README.md
│   ├── requirements.txt
│   ├── synthetic_score_common.py
│   ├── train_spectrum.py
│   ├── analyze_spectrum.py
│   ├── reproduce_synthetic_training_analysis.ipynb
│   └── plots_synthetic_theory.ipynb
│
├── environment.yml
└── README.md
```

## Image Experiments

Training and analysis of diffusion models on image datasets live in
`image_experiments/`. See `image_experiments/README.md`.

## Synthetic Experiments

The minimal local reproduction path for the synthetic linear-spectrum
experiments lives in `synthetic-experiments/`.

From that folder, install:

```bash
pip install -r requirements.txt
```

Train the default synthetic spectrum run locally:

```bash
python train_spectrum.py
```

Add `accelerator=mps` on Apple Silicon or `accelerator=cuda` on CUDA machines if you want to run off CPU.

Analyze the checkpoint:

```bash
python analyze_spectrum.py checkpoint=outputs/local_runs/linear_model_spectrum_ds_20000_N_30_s_1_d1_5_d2_10_v1_normalized/checkpoints/final.pt
```

The analysis writes legacy-compatible files under `synthetic-experiments/pickles/`
so that `plots_synthetic_theory.ipynb` can load them without modification.
For a step-by-step walkthrough, open
`synthetic-experiments/reproduce_synthetic_training_analysis.ipynb`.

## Installation

The root `environment.yml` is primarily for the image experiments:

```bash
conda env create -f environment.yml
conda activate latent_dim_diff
```

For synthetic-only reproduction, the smaller
`synthetic-experiments/requirements.txt` is sufficient.

## Citation

```bibtex
@article{achilli2024losing,
  title={Losing Dimensions: Geometric Memorization in Generative Diffusion},
  author={Achilli, Beatrice and Ventura, Enrico and Silvestri, Gianluigi and Pham, Bao and Raya, Gabriel and Krotov, Dmitry and Lucibello, Luca Ambrogioni},
  journal={arXiv preprint arXiv:2410.08727},
  year={2024}
}
```

## License

See the LICENSE file for licensing information.
