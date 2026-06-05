# Losing Dimensions: Geometric Memorization in Generative Diffusion

Official implementation accompanying:

**Losing Dimensions: Geometric Memorization in Generative Diffusion**
Beatrice Achilli, Enrico Ventura, Gianluigi Silvestri, Bao Pham, Gabriel Raya, Dmitry Krotov, Carlo Lucibello, Luca Ambrogioni

Paper: https://arxiv.org/abs/2410.08727

## Overview

This repository contains the code used to study geometric memorization in diffusion models.

The paper introduces a geometric perspective on memorization under the manifold hypothesis. Rather than appearing abruptly as exact replication of training examples, memorization emerges through a progressive reduction of effective dimensionality in the learned score field. This loss of geometric degrees of freedom can be characterized through the spectrum of score-function Jacobians and compared against theoretical predictions derived from synthetic low-dimensional datasets.

The repository contains implementations for:

* training diffusion models,
* computing score-function Jacobians,
* estimating singular value spectra,
* analyzing effective dimensionality,
* comparing empirical observations with theoretical predictions,
* reproducing the main experiments of the paper.

## Repository Structure

```text
.
├── image_experiments/
│   ├── README.md
│   ├── diffusion.py
│   ├── run_svd.py
│   ├── stats_utils.py
│   ├── train_utils.py
│   ├── lpips.py
│   ├── plot_svds_data.ipynb
│   └── src/
│       └── models/
│
├── synthetic_experiments/
│   ├── README.md
│   ├── data_generation/
│   ├── training/
│   ├── analysis/
│   └── plots_synthetic_theory.ipynb
│
├── environment.yml
├── LICENSE
└── README.md
```

### Image Experiments

Training and analysis of diffusion models on image datasets.
See [image_experiments/README.md](image_experiments/README.md).

### Synthetic Experiments

Experiments on low-dimensional synthetic manifolds and comparison with theory.
See [synthetic_experiments/README.md](synthetic_experiments/README.md).

## Installation

Create the environment with:

```bash
conda env create -f environment.yml
conda activate diffusion_env
```

## Reproducing Experiments

After installing the environment, users can:

1. Train diffusion models.
2. Compute score Jacobian spectra.
3. Analyze effective dimensionality.
4. Generate the plots and comparisons reported in the paper.

Detailed instructions for each experiment are provided in the corresponding subdirectories.

## Citation

```bibtex
@article{achilli2024losing,
  title={Losing Dimensions: Geometric Memorization in Generative Diffusion},
  author={Achilli, Beatrice and Ventura, Enrico and Silvestri, Gianluigi and Pham, Bao and Raya, Gabriel and Krotov, Dmitry and Lucibello, Carlo and Ambrogioni, Luca},
  journal={arXiv preprint arXiv:2410.08727},
  year={2024}
}
```

## License

See the LICENSE file for licensing information.
