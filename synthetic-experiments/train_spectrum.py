"""Train a local score model on the synthetic linear-model spectrum dataset."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from synthetic_score_common import (
    EMA,
    build_model,
    choose_device,
    generate_linear_model_spectrum,
    make_run_name,
    seed_everything,
)


@dataclass
class TrainConfig:
    data_size: int = 20000
    input_size: int = 30
    d1: int = 5
    d2: int = 10
    scale: float = 1.0
    is_normalized: bool = True
    tag: str = "v1"
    total_training_steps: int = 2000000
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    log_frequency: int = 10000
    accelerator: str = "cpu"
    seed: int = 42
    output_dir: str = "outputs/local_runs"


ConfigStore.instance().store(name="train_spectrum_config", node=TrainConfig)


@hydra.main(version_base=None, config_name="train_spectrum_config")
def main(cfg: TrainConfig) -> None:
    seed_everything(int(cfg.seed))
    device = choose_device(cfg.accelerator)
    original_cwd = Path(hydra.utils.get_original_cwd())
    run_name = make_run_name(
        data_size=cfg.data_size,
        input_size=cfg.input_size,
        scale=cfg.scale,
        d1=cfg.d1,
        d2=cfg.d2,
        is_normalized=cfg.is_normalized,
        tag=cfg.tag,
    )
    run_dir = original_cwd / cfg.output_dir / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data = generate_linear_model_spectrum(
        n_samples=cfg.data_size,
        input_size=cfg.input_size,
        d1=cfg.d1,
        d2=cfg.d2,
        scale=cfg.scale,
        is_normalized=cfg.is_normalized,
    )
    loader = DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=cfg.batch_size, shuffle=True)

    model = build_model(cfg.input_size, device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ema = EMA(model)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    final_loss = None

    for step, (batch,) in zip(range(1, cfg.total_training_steps + 1), cycle(loader)):
        batch = batch.to(device)
        loss = torch.mean(model.loss(batch))
        loss.backward()
        optimizer.step()
        ema.update(model)
        optimizer.zero_grad(set_to_none=True)

        final_loss = float(loss.detach().cpu())
        if step == 1 or step % cfg.log_frequency == 0 or step == cfg.total_training_steps:
            elapsed_min = (time.time() - start_time) / 60
            print(f"step={step}/{cfg.total_training_steps} loss={final_loss:.6f} elapsed_min={elapsed_min:.2f}", flush=True)

    checkpoint = {
        "format_version": 1,
        "run_name": run_name,
        "global_step": cfg.total_training_steps,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_loss": final_loss,
    }
    checkpoint_path = checkpoint_dir / "final.pt"
    torch.save(checkpoint, checkpoint_path)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "checkpoint": str(checkpoint_path),
                "global_step": cfg.total_training_steps,
                "final_loss": final_loss,
                "device": str(device),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
