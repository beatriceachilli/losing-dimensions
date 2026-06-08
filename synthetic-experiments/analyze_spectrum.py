"""Compute notebook-compatible orthogonal-group SVD pickles for synthetic runs."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import scipy.stats
import torch
from hydra.core.config_store import ConfigStore

from synthetic_score_common import (
    append_dims,
    choose_device,
    generate_linear_model_spectrum,
    load_checkpoint_model,
)


@dataclass
class AnalysisConfig:
    checkpoint: str = ""
    accelerator: str = "auto"
    use_ema: bool = True
    point_index: int = 0
    time_steps: list[float] = field(default_factory=lambda: np.linspace(0.8, 1e-5, 100).tolist())
    normalize: bool = True
    drop_first: bool = True
    output_dir: str = "pickles"


ConfigStore.instance().store(name="analyze_spectrum_config", node=AnalysisConfig)


def compute_singular_values_orthg(score_fn, sde, *, x0: torch.Tensor, t0: float) -> torch.Tensor:
    """Match the legacy orthogonal-group spectrum calculation for tabular data."""

    if x0.shape[0] != 1:
        raise ValueError("x0 must contain exactly one point")
    dim = x0[0].numel()
    t = torch.ones(x0.shape[0], device=x0.device, dtype=x0.dtype) * t0
    mean, std = sde.marginal_prob(x0, t)
    std = append_dims(std, x0.ndim)

    z = torch.tensor(scipy.stats.ortho_group.rvs(dim), dtype=x0.dtype, device=x0.device)
    perturbed = z * std + mean
    score_t = torch.ones(perturbed.shape[0], device=x0.device, dtype=x0.dtype) * t0
    scores = score_fn(perturbed, score_t).detach() / torch.norm(std)
    return torch.linalg.svdvals(torch.flatten(scores, 1))


@hydra.main(version_base=None, config_name="analyze_spectrum_config")
def main(cfg: AnalysisConfig) -> None:
    if not cfg.checkpoint:
        raise ValueError("Pass checkpoint=/path/to/final.pt")

    original_cwd = Path(hydra.utils.get_original_cwd())
    checkpoint_path = Path(cfg.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = original_cwd / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = choose_device(cfg.accelerator)
    model, checkpoint = load_checkpoint_model(checkpoint_path, device, use_ema=cfg.use_ema)
    train_cfg = checkpoint["config"]
    data = generate_linear_model_spectrum(
        n_samples=int(train_cfg["data_size"]),
        input_size=int(train_cfg["input_size"]),
        d1=int(train_cfg["d1"]),
        d2=int(train_cfg["d2"]),
        scale=float(train_cfg["scale"]),
        is_normalized=bool(train_cfg["is_normalized"]),
    )
    x0 = torch.tensor(data[int(cfg.point_index)], dtype=torch.float32, device=device).unsqueeze(0)

    singular_values_orthg = []
    with torch.no_grad():
        for t0 in cfg.time_steps:
            sv = compute_singular_values_orthg(model.get_score, model.sde, x0=x0, t0=float(t0)).cpu()
            if cfg.drop_first and sv.numel() > 1:
                sv = sv[1:]
            if cfg.normalize:
                sv = sv / torch.clamp(torch.max(sv), min=1e-12)
            singular_values_orthg.append(sv)

    result = {
        "time_steps": np.array(cfg.time_steps),
        "singular_values_orthg": singular_values_orthg,
    }
    output_dir = original_cwd / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = output_dir / f"{checkpoint['run_name']}.pickle"
    with open(pickle_path, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    npz_path = output_dir / f"{checkpoint['run_name']}.npz"
    np.savez_compressed(
        npz_path,
        time_steps=np.array(cfg.time_steps),
        singular_values_orthg=np.stack([sv.numpy() for sv in singular_values_orthg], axis=0),
        singular=np.stack([sv.numpy() for sv in singular_values_orthg], axis=0),
    )
    print(f"saved_pickle={pickle_path}")
    print(f"saved_npz={npz_path}")


if __name__ == "__main__":
    main()
