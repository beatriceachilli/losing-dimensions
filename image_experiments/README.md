# Diffusion Model SVD Computation

This folder contains tools for computing the Singular Value Decomposition (SVD) of score function Jacobians for trained diffusion models (UNet).

**Note:** This pipeline is designed to run in a multi-GPU distributed environment managed by SLURM and will not run locally without modification.

---

## 1. Setup & Installation

We use Conda to manage the Python environment and gracefully handle CUDA/PyTorch binary dependencies. 

1. Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed on your cluster.
2. Save the provided `environment.yml` file in the root of this project.
3. Build and activate the environment:

```bash
# Create the environment (this may take a few minutes)
conda env create -f environment.yml

# Activate the environment
conda activate diffusion_env
```

## 2. Data Preparation

Before running the SVD computation, you must tell the script where to pull the evaluation images from. The script accepts one of two data inputs:

- Raw Dataset (`--data-path`): Points to a standard dataset directory (e.g., ImageNet, CIFAR10). The script will automatically load and crop the images based on the model's training configuration.

- Pre-computed Samples (`--sample-path`): Points to a .npz file containing a pre-generated array of images (e.g., from an evaluation run).

If you want to limit how many images are processed (useful for testing), use the `--sample-size` flag (e.g., `--sample-size 100`).


## 3. Running the Code (SLURM)

Because `run_svd.py` relies on MPI/SLURM environment variables to distribute the workload across multiple GPUs, you must run it via a batch submission script. But you can convert that file into one that can easily run on a non-slurm system. See the `example_slurm.sh` script.

```bash
sbatch submit.slurm
```