# Synthetic Experiments

This folder contains the minimal local code needed to reproduce the synthetic
linear-spectrum score-model experiments used by `plots_synthetic_theory.ipynb`.

The training and analysis scripts are standalone PyTorch + Hydra scripts. They
do not use Lightning or W&B.

## Setup

From this folder:

```bash
pip install -r requirements.txt
```

On Apple Silicon or CUDA machines, install the PyTorch build matching your hardware if needed. The examples below run with the default CPU setting; add `accelerator=mps` or `accelerator=cuda` to use those devices.

## Train

By default, `train_spectrum.py` matches the old spectrum training override:
`data_size=20000`, `input_size=30`, `d1=5`, `d2=10`, `scale=1`,
`is_normalized=True`, and `total_training_steps=2000000`.

Run the default configuration locally:

```bash
python train_spectrum.py
```

For a short trial run:

```bash
python train_spectrum.py total_training_steps=100000
```

The run name includes the legacy `v1` tag by default:

```text
linear_model_spectrum_ds_20000_N_30_s_1_d1_5_d2_10_v1_normalized
```

The final checkpoint is saved under:

```text
outputs/local_runs/<run_name>/checkpoints/final.pt
```

To reproduce the dataset-size comparison cells in the existing notebook, use
the notebook's legacy parameters:

```bash
python train_spectrum.py data_size=500 d1=2 d2=5
```

Repeat with `data_size=250`, `500`, and `20000` as needed.

## Analyze

After training, compute the orthogonal-group singular-value curves:

```bash
python analyze_spectrum.py checkpoint=outputs/local_runs/linear_model_spectrum_ds_20000_N_30_s_1_d1_5_d2_10_v1_normalized/checkpoints/final.pt
```

The analysis script writes:

```text
pickles/<run_name>.pickle
pickles/<run_name>.npz
```

The pickle intentionally keeps the legacy structure expected by
`plots_synthetic_theory.ipynb`:

```python
{
    "time_steps": np.array(...),
    "singular_values_orthg": [torch.Tensor, ...],
}
```

By default, the analysis uses 100 time steps from `0.8` to `1e-5`, matching the
indexing used in the existing notebook. Do not change
`plots_synthetic_theory.ipynb`; once the expected pickles exist in `pickles/`,
the notebook can load them directly.
