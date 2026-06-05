"""
=============================================================================
run_svd.py: Distributed SVD Computation for Diffusion Model Score Functions
=============================================================================

This script computes the Singular Value Decomposition (SVD) of the score 
function Jacobians for a trained diffusion model (UNet). It is designed to 
run in a multi-GPU distributed environment (e.g., SLURM) and processes images 
in parallel.

The script supports evaluating the intrinsic dimensionality of the model's 
score function under standard, symmetrized, or non-symmetrized orthogonal noise.

Usage Example (via SLURM/torchrun):
-----------------------------------------------------------------------------
srun -n <num_gpus> python run_svd.py \
    --ckpt-path ./checkpoints/model.pt \
    --result-path ./results/svd_outputs/ \
    --data-path ./data/cifar10/ \
    --time 15 \
    --batch-size 512 \
    --score-fn-type symmetrized \
    --use-lobpcg

Required Arguments:
-----------------------------------------------------------------------------
--ckpt-path       : (str) Path to the trained model checkpoint (.pt or .pth).
--result-path     : (str) Directory where the resulting .npz files (singular 
                    values) will be saved.

Data Sourcing (Must provide ONE of the following):
-----------------------------------------------------------------------------
--data-path       : (str) Path to the raw dataset to sample from.
--sample-path     : (str) Path to a pre-generated .npz file containing 
                    evaluation samples.

Key Optional Arguments:
-----------------------------------------------------------------------------
--time            : (int) The diffusion timestep (t) to compute the SVD at. 
                    Default is 15.
--score-fn-type   : (str) The mathematical method for computing the score 
                    matrix. Options: "symmetrized", "non-symmetrized", or 
                    "original". Default is "non-symmetrized".
--batch-size      : (int) Number of matrix chunks to process at once. Reduce 
                    this if you run into CUDA Out of Memory (OOM) errors.
--k-mult          : (float) Multiplier for the matrix dimension (K = k_mult * d).
--use-lobpcg      : (flag) If set, uses the LOBPCG algorithm to approximate 
                    the lowest singular values, which is highly memory-efficient 
                    for very large matrices.
--overwrite       : (flag) If set, forces recalculation of samples that already 
                    have a saved .npz result.

Output:
-----------------------------------------------------------------------------
The script generates individual compressed NumPy files (`.npz`) for each 
processed sample index inside the `--result-path` directory. These files 
contain the calculated singular values array (`singular`) and the original 
checkpoint arguments (`ckpt_args`).
"""

import os
from time import time
from functools import partial

import torch
import torch.distributed as dist
import numpy as np
from torch import lobpcg
from diffusers import DDPMScheduler
from simple_parsing import ArgumentParser

from train_utils import get_dist_info, get_dataset
from stats_utils import create_dirs, get_unet, get_train_loader


@torch.no_grad()
def score_fn(x_t, t, model, sigma_t) -> torch.Tensor:
    """Computes the estimated score: s_t = -eps_t / sigma_t."""
    eps = model(x_t, t)
    return -eps / sigma_t


def chunk_dot_prod(A, x, chunk_size: int = 2048):
    """
    Computes a dot product in chunks to maintain memory efficiency
    when dealing with large matrices.
    """
    prods = []
    for i in range(0, len(A), chunk_size):
        j = min(i + chunk_size, len(A))
        A_chunk = A[i:j]
        p_chunk = A_chunk @ x
        prods.append(p_chunk)

    if len(A) % chunk_size > 0:
        assert len(prods) == len(A) // chunk_size + 1
    else:
        assert len(prods) == len(A) // chunk_size

    return torch.cat(prods, dim=0)


@torch.no_grad()
def get_ortho_noise(I, atol=1e-6):
    """
    Generates a random orthogonal noise matrix of size (b, d, d) using QR decomposition.

    Args:
        I (torch.Tensor): Identity matrix of shape (b, d, d).
        atol (float): Absolute tolerance for the orthogonality check.

    Returns:
        torch.Tensor: Random orthogonal matrix of size (b, d, d).
    """

    def initialize(b, d, device):
        # Generate a random matrix and perform QR decomposition
        random_matrix = torch.randn((b, d, d), device=device)
        q, r = torch.linalg.qr(random_matrix)

        # Ensure the determinant is +1 (proper orthogonal matrix)
        r_diag = torch.diagonal(r, offset=0, dim1=-2, dim2=-1).sign()
        q = q * r_diag[..., None]
        return q

    b, d, d = I.shape

    while True:
        q = initialize(b, d, I.device)
        dot_prod = torch.bmm(q.permute(0, 2, 1), q)

        # Loop until orthogonality strictly satisfies the tolerance
        if torch.allclose(dot_prod, I, atol=atol):
            break

    return q


def sym_score(x0, t, model, I, alphas_cumprod_t, sigma_t):
    """Computes the symmetrized score using orthogonal noise (x_plus and x_minus)."""
    noise = get_ortho_noise(I)[: len(x0) * x0.shape[1]]
    noise = noise.view(x0.shape).contiguous()

    # Forward process scaling
    sqrt_1m_alphas_cumprod = sigma_t
    sqrt_alphas_cumprod = alphas_cumprod_t**0.5

    scaled_x0 = sqrt_alphas_cumprod * x0
    scaled_noise = sqrt_1m_alphas_cumprod * noise

    x_plus = scaled_x0 + scaled_noise
    x_minus = scaled_x0 - scaled_noise

    # Compute scores for both directions
    s_plus, s_minus = map(
        lambda x_t: score_fn(x_t, t, model, sigma_t), (x_plus, x_minus)
    )

    # Return symmetrized score
    return (s_plus - s_minus) / 2


def non_sym_score(x0, t, model, I, alphas_cumprod_t, sigma_t):
    """Computes the standard score using only positive orthogonal noise."""
    noise_plus = get_ortho_noise(I)[: len(x0) * x0.shape[1]]
    noise_plus = noise_plus.view(x0.shape).contiguous()

    sqrt_1m_alphas_cumprod = sigma_t
    sqrt_alphas_cumprod = alphas_cumprod_t**0.5
    x_plus = sqrt_alphas_cumprod * x0 + sqrt_1m_alphas_cumprod * noise_plus

    return score_fn(x_plus, t, model, sigma_t)


def original_score(x0, t, model, I, alphas_cumprod_t, sigma_t):
    """Computes the baseline score using standard Gaussian noise."""
    noise = torch.randn_like(x0)

    sqrt_1m_alphas_cumprod = sigma_t
    sqrt_alphas_cumprod = alphas_cumprod_t**0.5
    z = sqrt_alphas_cumprod * x0 + sqrt_1m_alphas_cumprod * noise

    return score_fn(z, t, model, sigma_t)


def compute_svd_ortho(
    img: torch.Tensor,
    K: int,
    model,
    scheduler,
    time: int,
    batch_size: int = 512,
    use_lobpcg: bool = False,
    svs_frac: float = 1.0,
    fn_type: str = "symmetrized",
) -> torch.Tensor:
    """
    Computes intrinsic dimensionality with orthogonalization of the noise under DDPM setting.
    """
    assert batch_size > 1
    model.eval()

    if use_lobpcg and svs_frac > 0.35:
        svs_frac = 0.35

    S = []
    alphas_cumprod = scheduler.alphas_cumprod[time]
    sigma_t = (1.0 - alphas_cumprod) ** 0.5

    x = img.repeat(batch_size, 1, 1, 1)
    t = torch.full([len(x)], time).long().to(x.device)

    _, c, h, w = x.shape
    identity = torch.eye(h, device=x.device)[None].repeat(batch_size * c, 1, 1)

    # Dispatch to the appropriate score function implementation
    fn_type = fn_type.lower()
    if fn_type == "symmetrized":
        fn = partial(
            sym_score,
            model=model,
            I=identity,
            alphas_cumprod_t=alphas_cumprod,
            sigma_t=sigma_t,
        )
    elif fn_type == "non-symmetrized":
        fn = partial(
            non_sym_score,
            model=model,
            I=identity,
            alphas_cumprod_t=alphas_cumprod,
            sigma_t=sigma_t,
        )
    else:
        fn = partial(
            original_score,
            model=model,
            I=identity,
            alphas_cumprod_t=alphas_cumprod,
            sigma_t=sigma_t,
        )

    # Batch process S computation
    for i in range(0, K, batch_size):
        if i + batch_size > K:
            batch_size = K - i
        score = fn(x[:batch_size], t[:batch_size])
        S.append(torch.flatten(score, 1))

    # Concatenate all the scores (kD x D)
    S = torch.cat(S, dim=0)
    num_svs = max(int(S.shape[1] * svs_frac), 384)

    # Grab least singular values using either standard SVD or LOBPCG
    if not use_lobpcg:
        # compute singular values with repsect to the kD columns
        S = torch.linalg.svdvals(S.T)[-num_svs:]
    else:
        # compute S^T S and singular values = sqrt(eigenvalues of S)
        S_dot_S = chunk_dot_prod(S, S, chunk_size=2048)
        S = lobpcg(S_dot_S, k=num_svs, largest=False).abs() ** 0.5

    return S.detach().cpu()


def filter_samples(save_path, samples, overwrite=False):
    """Filters out sample indices that have already been processed and saved."""
    if overwrite:
        return np.arange(0, len(samples))

    unfinished_indices = []
    for idx, _ in enumerate(samples):
        result_path = os.path.join(save_path, str(idx))
        if not os.path.exists(result_path + ".npz"):
            unfinished_indices.append(idx)
    return unfinished_indices


def main(rank, local_rank, args):
    """Main distributed entry point for computing and saving SVDs."""
    result_path = args.result_path
    create_dirs([result_path])

    # Basic Path Validation
    if not (args.ckpt_path.endswith(".pt") or args.ckpt_path.endswith(".pth")):
        raise ValueError("Checkpoint path must end with .pt or .pth")
    if (
        args.data_path is None
        and args.sample_path is not None
        and not args.sample_path.endswith(".npz")
    ):
        raise ValueError("Sample path must end with .npz")
    if args.result_path is None:
        raise ValueError("Result path must be specified")

    ckpt_file = torch.load(args.ckpt_path, map_location="cpu")
    ckpt_args = ckpt_file["args"]

    # Load dataset or existing evaluation samples
    if args.data_path is not None and args.data_path != "":
        dataset = get_dataset(
            args.data_path,
            ckpt_args.data.data_name,
            (None if not (ckpt_args.train.centercrop) else ckpt_args.model.image_size),
        )
        train_loader = get_train_loader(dataset, 1, ckpt_args)
        samples = []
        for x, _ in train_loader.dataset:
            if len(samples) >= (
                args.sample_size
                if args.sample_size is not None and args.sample_size > 0
                else len(train_loader.dataset)
            ):
                break
            samples.append(x)
    else:
        try:
            eval_file = np.load(args.sample_path, allow_pickle=True)
            samples = eval_file["samples"]
            top_size, _ = eval_file["sizes"]
            samples = samples[:top_size]

            if args.sample_size is not None and args.sample_size > 0:
                samples = samples[: args.sample_size]
        except Exception as e:
            print(f"Warning: Could not load sample path cleanly. Exception: {e}")

    # Initialize diffusion components
    diffusion = DDPMScheduler(
        beta_schedule=ckpt_args.train.beta_schedule,
        prediction_type=ckpt_args.train.prediction_type,
    )
    diffusion.set_timesteps(ckpt_args.train.timesteps)

    # Initialize and load model
    ema = get_unet(ckpt_args, False)
    ema.load_state_dict(ckpt_file["ema"])
    ema = ema.to(local_rank)
    ema.eval()

    # Distribute workload across MPI ranks
    start_end_idx = filter_samples(args.result_path, samples, args.overwrite)
    num_samples_per_rank = len(start_end_idx) // dist.get_world_size()

    start_idx = local_rank * num_samples_per_rank
    end_idx = (local_rank + 1) * num_samples_per_rank
    if local_rank == dist.get_world_size() - 1:
        end_idx = len(start_end_idx)

    print_time = False
    if local_rank == 0:
        print(f"Computing SVD at t={args.time} with orthogonalization.")
        total_params = sum(p.numel() for p in ema.parameters() if p.requires_grad)
        print(f"Total parameters in the model: {total_params / 1e6:.2f}M")
        print_time = True

    print(f"Rank {rank} processing samples from index {start_idx} to {end_idx}.")

    # Processing Loop
    for idx in range(start_idx, end_idx):
        i = start_end_idx[idx]
        save_path = os.path.join(result_path, str(i))
        if os.path.exists(save_path + ".npz") and not args.overwrite:
            continue

        if isinstance(samples[i], np.ndarray):
            img = torch.from_numpy(samples[i]).to(local_rank)
        else:
            img = samples[i].to(local_rank)

        K = int(args.k_mult * img.numel())

        if print_time and local_rank == 0:
            st_time = time()

        sv = compute_svd_ortho(
            img[None],
            K,
            ema,
            diffusion,
            args.time,
            args.batch_size,
            args.use_lobpcg,
            args.svs_frac,
            args.score_fn_type,
        )

        if print_time and local_rank == 0:
            print(
                f"Rank {rank} processed sample {i} in {time() - st_time:.5f} seconds. Estimated total time: {(end_idx - start_idx) * (time() - st_time) / 60:.5f} minutes."
            )
            print_time = False

        np.savez_compressed(save_path, singular=sv, ckpt_args=ckpt_args)

    dist.barrier(device_ids=[rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_argument(
        "--result-path",
        type=str,
        help="Path to stored the results. If specified None, then results are stored in the same path as ckpt path.",
    )
    parser.add_argument("--ckpt-path", type=str, help="Path to .pth file")
    parser.add_argument("--sample-path", type=str, help="Path to evaluation .npz file.")
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument(
        "--score-fn-type",
        type=str,
        default="non-symmetrized",
        help="Function type for computing score functions matrix.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to use from the eval file.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--time",
        type=int,
        default=15,
        help="The timestep at which you want to compute the SVD for the image.",
    )
    parser.add_argument(
        "--k-mult",
        type=float,
        default=4.0,
        help="Multiplier to d-dimension for computing S.",
    )
    parser.add_argument(
        "--svs-frac",
        type=float,
        default=1.0,
        help="Fraction of number of singular values to compute.",
    )
    parser.add_argument(
        "--use-lobpcg",
        action="store_true",
        help="Use LOBPCG to compute singular values.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing results."
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    rank, world_size, gpus_per_node = get_dist_info()
    assert gpus_per_node == torch.cuda.device_count()

    # Calculate Local GPU rank
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)

    main(rank, local_rank, args)
