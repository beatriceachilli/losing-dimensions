"""Shared utilities for synthetic spectrum score-model experiments."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(accelerator: str) -> torch.device:
    if accelerator == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if accelerator == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if accelerator == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def generate_linear_model_spectrum(
    *,
    n_samples: int,
    input_size: int,
    d1: int,
    d2: int,
    scale: float,
    is_normalized: bool,
) -> np.ndarray:
    """Generate the synthetic spectrum dataset used in the paper notebook."""

    d3 = input_size - d1 - d2
    if d3 < 0:
        raise ValueError(f"d1 + d2 must be <= input_size; got {d1} + {d2} > {input_size}")

    stds = np.concatenate(
        (
            scale * np.ones((d1,)),
            scale * 0.3 * np.ones((d2,)),
            scale * 0.01 * np.ones((d3,)),
        ),
        axis=0,
    )
    samples = stds[None, :] * np.random.normal(0.0, 1.0, (n_samples, input_size))
    if is_normalized:
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        samples = samples / np.clip(norms, a_min=1e-12, a_max=None)
    return samples.astype(np.float32)


def make_run_name(
    *,
    data_size: int,
    input_size: int,
    scale: float,
    d1: int,
    d2: int,
    is_normalized: bool,
    tag: str = "v1",
) -> str:
    """Return the legacy filename stem expected by plots_synthetic_theory.ipynb."""

    scale_text = f"{scale:g}"
    name = f"linear_model_spectrum_ds_{data_size}_N_{input_size}_s_{scale_text}_d1_{d1}_d2_{d2}"
    if tag:
        name += f"_{tag}"
    if is_normalized:
        name += "_normalized"
    return name


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class TimestepBlock(nn.Module):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, temb)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    def __init__(self, input_size: int, output_size: int, tdim: int, droprate: float = 0.0) -> None:
        super().__init__()
        self.block_1 = nn.Sequential(nn.LayerNorm(input_size), nn.SiLU(), nn.Linear(input_size, output_size))
        self.temb_proj = nn.Sequential(nn.SiLU(), nn.Linear(tdim, output_size))
        self.block_2 = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.SiLU(),
            nn.Dropout(p=droprate),
            nn.Linear(output_size, output_size),
        )
        self.residual = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        latent = self.block_1(x)
        latent = latent + self.temb_proj(temb)
        latent = self.block_2(latent)
        return latent + self.residual(x)


class DDPMTabular(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_res_blocks: int = 2,
        hidden_size: int = 128,
        droprate: float = 0.0,
    ) -> None:
        super().__init__()
        time_embed_dim = 512
        self.time_embedding = SinusoidalEmbedding(128)
        self.temb_layer = nn.Sequential(
            nn.Linear(128, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.layers = nn.ModuleList([TimestepEmbedSequential(nn.Linear(input_size, hidden_size))])
        for _ in range(num_res_blocks):
            self.layers.append(TimestepEmbedSequential(ResBlock(hidden_size, hidden_size, time_embed_dim, droprate)))
        self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.SiLU(), nn.Linear(hidden_size, input_size))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.temb_layer(self.time_embedding(t))
        h = x
        for block in self.layers:
            h = block(h, temb)
        return self.out(h)


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


class VESDE:
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0, n_steps: int = 1000) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = n_steps
        self.T = 1.0

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return x, std


class ScoreModel(nn.Module):
    def __init__(self, score_model: nn.Module, eps: float = 1e-5) -> None:
        super().__init__()
        self.sde = VESDE()
        self.eps = eps
        self.score_model = score_model

    def _labels(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.sde.marginal_prob(torch.zeros_like(x), t)[1]

    def get_score(self, x: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.ones(x.shape[0], device=x.device, dtype=x.dtype) * float(t)
        labels = self._labels(x, t)
        sigma = append_dims(self.sde.marginal_prob(x, t)[1], x.ndim)
        return -self.score_model(x, labels) / sigma

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.rand(x.shape[0], device=x.device, dtype=x.dtype) * (self.sde.T - self.eps) + self.eps
        z = torch.randn_like(x)
        mean, std = self.sde.marginal_prob(x, t)
        perturbed_data = mean + append_dims(std, x.ndim) * z
        labels = self._labels(x, t)
        score = self.score_model(perturbed_data, labels)
        loss = torch.square(score - z)
        return torch.mean(loss.view(loss.shape[0], -1), dim=-1)


class EMA:
    def __init__(self, model: nn.Module, beta: float = 0.9999) -> None:
        self.beta = beta
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if torch.is_floating_point(param)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            value.mul_(self.beta).add_(state[name].detach(), alpha=1.0 - self.beta)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.shadow.items()}


def build_model(input_size: int, device: torch.device) -> ScoreModel:
    return ScoreModel(DDPMTabular(input_size=input_size)).to(device)


def load_checkpoint_model(checkpoint_path: Path, device: torch.device, use_ema: bool = True) -> tuple[ScoreModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model = build_model(int(cfg["input_size"]), device)
    if use_ema and "ema_state_dict" in checkpoint:
        state = model.state_dict()
        for name, value in checkpoint["ema_state_dict"].items():
            if name in state:
                state[name] = value.to(device)
        model.load_state_dict(state)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
