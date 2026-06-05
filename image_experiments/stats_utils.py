"""
stats_utils.py
Contains utility functions for model loading, evaluation metric computation
(LPIPS, L2 Norm), and relative potential score measurements for memorization
and generalization analysis in diffusion models.

Note: Assume all image samples are scaled between (-1, 1).
"""

import os
import glob
from pathlib import Path
from typing import List, Union, Callable, Tuple

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from lpips import ModifiedLPIPS

from src import Unet
from train_utils import split_data


def get_unet(args, data_parallel: bool = True) -> nn.Module:
    """Creates a UNet model with configurations passed via args."""
    model = Unet(
        image_size=args.model.image_size,
        in_channels=args.model.in_channels,
        dim=args.model.dim,
        dim_mults=args.model.dim_mults,
        attn_resolutions=args.model.attn_resolutions,
        num_res_blocks=args.model.num_res_blocks,
        dropout=args.model.dropout,
        conditional=args.model.conditional,
        resamp_with_conv=args.model.resamp_with_conv,
        nonlinearity=args.model.nonlinearity,
        scale_by_sigma=args.model.scale_by_sigma,
    )
    if data_parallel:
        model = DataParallel(model)
    return model


def create_dirs(paths: Union[str, List[str]]):
    """Creates directory structures for a string or list of path strings."""
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    elif isinstance(paths, list):
        for path in paths:
            os.makedirs(path, exist_ok=True)
    else:
        raise TypeError("Must be a str or a sequence of strs.")


def remove_pth(path: str) -> str:
    """Removes the .pth or .pt extension from a filepath string."""
    if path.endswith(".pth"):
        return path[:-4]
    elif path.endswith(".pt"):
        return path[:-3]
    else:
        raise ValueError("Path does not end with .pth or .pt")


def sort_files(files: List[str]) -> List[str]:
    """Sorts filepaths assuming their name acts as an integer suffix (e.g. unet_5.pt)."""
    files.sort(key=lambda file: int(Path(file).name.split(".")[0]))
    return files


def to_zero_one(x: torch.Tensor) -> torch.Tensor:
    """Converts a (-1, 1) tensor to a (0, 1) scale inline."""
    if x is None:
        return x
    x.mul_(0.5).add_(0.5)
    return x


def convert_dataset(train_loader: DataLoader) -> torch.Tensor:
    """Concatenates all samples from a data iterator (assumes shuffle is off)."""
    xs = []
    for x, _ in train_loader:
        if len(x.shape) == 3:
            x = x[None]
        xs.append(x)
    return torch.cat(xs, dim=0)


def get_train_loader(
    dataset, batch_size: int, args, shuffle: bool = False
) -> DataLoader:
    """
    Returns a PyTorch DataLoader mapped to a specific dataset split.
    Notes: num_workers=0, pin_memory=False, drop_last=False.
    """
    train_data, _ = split_data(
        dataset,
        args.train.train_size,
        args.train.valid_size,
        args.train.global_seed,
    )

    return DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def filter_and_sort(
    sample_path: str, ckpt_files: List[str]
) -> Tuple[List[str], List[str]]:
    """Filters and sorts checkpoint files based on generated evaluation file numbers."""
    if sample_path is None:
        eval_files = [None] * len(ckpt_files)
        ckpt_files = sort_files(ckpt_files)
        return eval_files, ckpt_files

    if isinstance(sample_path, str) and sample_path.endswith(".npz"):
        eval_files = [sample_path]
    elif isinstance(sample_path, str):
        eval_files = glob.glob(os.path.join(sample_path, "*.npz"))

    ckpt_files, eval_files = map(sort_files, (ckpt_files, eval_files))

    eval_sizes = set([int(Path(f).name.split(".")[0]) for f in eval_files])
    new_files = [f for f in ckpt_files if int(Path(f).name.split(".")[0]) in eval_sizes]

    assert len(eval_files) == len(new_files)
    return eval_files, new_files


@torch.no_grad()
def compute_norm(
    x1: torch.Tensor, x2: torch.Tensor, unsqueeze: bool = True
) -> torch.Tensor:
    """Computes the inverse L2 norm / distance between two tensor batches."""
    x1, x2 = map(lambda z: torch.flatten(z, start_dim=1), (x1, x2))

    if unsqueeze:
        x1 = x1[:, None]
        x2 = x2[None]

    diff = x1 - x2
    # Add epsilon to prevent division by zero
    return 1.0 / (torch.norm(diff, dim=-1) + 1e-8)


def get_metric_fn(
    use_lpips: bool,
    network: str = "alex",
    unsqueeze: bool = True,
    rescale: bool = True,
    device: str = "cuda",
) -> Callable:
    """Returns a function for computing metrics (either LPIPS or inverse L2 Norm)."""
    rescale_fn = (lambda z: z * 0.5 + 0.5) if rescale else (lambda z: z)

    if use_lpips:
        metric = ModifiedLPIPS(network=network, reduction="none").to(device)
        return lambda x, y: metric(x, y, fn=rescale_fn, unsqueeze=unsqueeze)
    else:
        return lambda x, y: compute_norm(x, y, unsqueeze=unsqueeze)


@torch.no_grad()
def compute_metrics(
    m_top1_dists: torch.Tensor,
    s_top1_dists: torch.Tensor,
    deltas: Tuple[float, float],
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes boolean tensor bins representing memorized, spurious, and generalized scores."""
    delta_m, delta_s = deltas

    if device is not None:
        m_top1_dists, s_top1_dists = map(
            lambda z: z.to(device), (m_top1_dists, s_top1_dists)
        )

    # Threshold checks
    m_bins = torch.where(m_top1_dists <= delta_m, 1, 0)
    s_bins = torch.where(s_top1_dists <= delta_s, 1, 0)

    # Boolean logic masks
    neg_m_bins = m_bins ^ 1
    s_bins = torch.logical_and(neg_m_bins, s_bins)
    g_bins = torch.logical_and(neg_m_bins, s_bins ^ 1)

    return map(lambda z: z.float().cpu(), (m_bins, s_bins, g_bins))


def get_nonzero_entries(targets: torch.Tensor, binaries: torch.Tensor) -> torch.Tensor:
    """Extracts entries from `targets` where the corresponding binary mask is 1."""
    indices = torch.where(binaries == 1)
    return targets[indices]


def find_nns(target, indices):
    """Finds corresponding nearest neighbors in the target tensor given specific indices."""
    if indices is None or len(indices) == 0:
        return None

    def not_tuple(item):
        return item[0] if isinstance(item, tuple) else item

    nns = []
    for idx in indices:
        try:
            nns.append(not_tuple(target[idx]))
        except Exception:
            # Handle case where idx is an iterable sequence of indices
            nn_i = [not_tuple(target[i]) for i in idx]
            nns.append(np.stack(nn_i, axis=0))

    return np.stack(nns, axis=0)


@torch.no_grad()
def compute_rel_potential(
    model: nn.Module,
    target: torch.Tensor,
    reference: torch.Tensor,
    alpha_misc: Tuple[float, torch.Tensor, torch.Tensor],
    diff_misc: Tuple[torch.Tensor, torch.Tensor],
    mult_fn: nn.Module,
    t: int = 0,
):
    """
    Computes the relative potential between a target tensor and a reference tensor
    based on score function gradients over a structural transition path.
    """
    dA, cos_alpha, sin_alpha = alpha_misc
    betas, std = diff_misc

    b, p = len(target), len(cos_alpha)

    # Define interpolation (x) and orthogonal (v) paths
    x = cos_alpha * target[None] + sin_alpha * reference[None]
    v = -sin_alpha * target[None] + cos_alpha * reference[None]

    # Batch collapse for parallel model inference
    x = rearrange(x, "p b ... -> (p b) ...")
    t_tensor = torch.zeros(len(x), device=x.device).long() + t

    std_t = std[t]
    beta_t = betas[t][:, None, None, None]

    # Score projection
    score = -model(x, t_tensor) / std_t[:, None, None, None]
    grad_u = -beta_t * score - 0.5 * beta_t * x
    grad_u = rearrange(grad_u, "(p b) ... -> p b ...", p=p, b=b)

    cumprod_u = 0
    v.mul_(dA)

    # Aggregate potential across the discrete path
    for i in range(p):
        val = mult_fn(grad_u[i].view(b, -1), v[i].view(b, -1))
        cumprod_u += val.sum(-1)

    return cumprod_u.cpu().numpy()


def batch_potential(
    model: nn.Module,
    x1: torch.Tensor,  # Set of target images of size B
    x2: torch.Tensor,  # Reference image of size 1
    betas: torch.Tensor,  # Diffusion variances of size T
    batch_size: int = 128,
    device: str = "cuda",
):
    """
    Calculates the relative potential between batches of images and a reference image,
    following "Spontaneous Symmetry..." (https://arxiv.org/abs/2305.19693).
    """

    class Model(nn.Module):
        """Helper module for DataParallel to perform batched matrix multiplication."""

        def __init__(self):
            super().__init__()

        def __call__(self, x1: torch.Tensor, x2: torch.Tensor):
            return x1 * x2

    assert batch_size > 0 and len(x2) == 1

    # Setup path integration variables
    alpha = torch.linspace(0, 0.5 * torch.pi, 20)
    dA = (alpha[1] - alpha[0]).item()
    cos_alpha = torch.cos(alpha)[:, None, None, None, None].to(device)
    sin_alpha = torch.sin(alpha)[:, None, None, None, None].to(device)
    alpha_misc = (dA, cos_alpha, sin_alpha)

    betas = betas.to(device)
    alpha_cps = torch.cumprod(betas, dim=0)
    diff_misc = (betas, (1 - alpha_cps) ** 0.5)

    mult_fn = nn.DataParallel(Model())
    model = model.eval()

    reference = x2.to(device)
    batch_size = min(len(x1), batch_size)
    rel_potentials = []

    # Process in chunks to prevent memory blowouts
    for i in range(0, len(x1), batch_size):
        j = min(i + batch_size, len(x1))
        target = x1[i:j].to(device)
        rel_potential = compute_rel_potential(
            model, target, reference, alpha_misc, diff_misc, mult_fn
        )
        rel_potentials.append(rel_potential)

    if len(rel_potentials) == 1:
        return rel_potentials[0]
    return np.concatenate(rel_potentials)
