"""
train.py
=============================================================================
Training entry point for the image diffusion experiments in
"Losing Dimensions: Geometric Memorization in Generative Diffusion".

This script was missing from the released repository: run_svd.py assumes a
checkpoint already exists, but nothing actually produces one. This script
glues together the existing building blocks:

    - parse_utils.py      (DataOptions / TrainOptions / ModelOptions schema,
                            matching Table 1 of the paper)
    - train_utils.py       (dataset loading/splitting, the DDPM training loss)
    - stats_utils.py       (get_unet, get_train_loader)
    - ema.py                (torch's AveragedModel, used for the EMA weights)
    - diffusers.DDPMScheduler (the same scheduler run_svd.py uses at eval time,
                            so training and analysis stay consistent)

IMPORTANT — checkpoint format compatibility:
run_svd.py loads checkpoints as
    ckpt = torch.load(ckpt_path)
    ckpt_args = ckpt["args"]              # nested Namespace: .data .train .model
    ema = get_unet(ckpt_args, False)      # data_parallel=False -> no "module." prefix
    ema.load_state_dict(ckpt["ema"])
This script saves checkpoints in exactly that shape, so run_svd.py works
unmodified on top of it.

Usage (single dataset size, single GPU or CPU):
-----------------------------------------------------------------------------
python train.py \
    --data-path ./data \
    --data-name cifar10 \
    --train-size 2000 \
    --valid-size 1000 \
    --iterations 500000 \
    --image-size 32 --in-channels 3 --dim 128 --dim-mults 1,2,2,2 \
    --results-path ./results

Resuming:
-----------------------------------------------------------------------------
python train.py --resume ./results/cifar10_n2000/checkpoints/latest.pt ...

Multi-GPU (single node):
-----------------------------------------------------------------------------
Handled automatically via nn.DataParallel when more than one CUDA device is
visible (mirrors the convention already used by stats_utils.get_unet). For
multi-node scaling, swap the DataParallel block below for DDP + torchrun.
=============================================================================
"""

import os
import time

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from simple_parsing import ArgumentParser

from parse_utils import DataOptions, TrainOptions, ModelOptions
from train_utils import (
    get_dataset,
    create_logger,
    train_loss,
    sample_data,
    str2tuple,
)
from stats_utils import get_unet, get_train_loader, create_dirs
from ema import AveragedModel, get_ema_multi_avg_fn
from run_options import RunOptions


def build_scheduler(train_opts: TrainOptions, run_opts: RunOptions) -> DDPMScheduler:
    """Builds the same DDPMScheduler class run_svd.py uses at eval time, so the
    (alpha_cumprod, sigma_t) convention is guaranteed identical between
    training and the score-Jacobian analysis."""
    scheduler = DDPMScheduler(
        num_train_timesteps=train_opts.timesteps,
        beta_start=run_opts.beta_start,
        beta_end=run_opts.beta_end,
        beta_schedule=train_opts.beta_schedule,
        prediction_type=train_opts.prediction_type,
    )
    return scheduler


def make_run_dir(data_opts: DataOptions, train_opts: TrainOptions) -> str:
    run_name = f"{data_opts.data_name}_n{train_opts.train_size}"
    run_dir = os.path.join(data_opts.results_path, run_name)
    create_dirs([run_dir, os.path.join(run_dir, "checkpoints")])
    return run_dir


def save_checkpoint(
    path: str,
    step: int,
    raw_model: nn.Module,
    ema_model: AveragedModel,
    optimizer: torch.optim.Optimizer,
    args,
) -> None:
    """Saves in the exact shape run_svd.py expects (`args`, `ema`), plus
    `model`/`opt`/`step` for resuming training later."""
    torch.save(
        {
            "step": step,
            "model": raw_model.state_dict(),
            # ema_model.module is the *unwrapped* averaged copy -> no
            # "module." prefix, matching get_unet(ckpt_args, False) at load time.
            "ema": ema_model.module.state_dict(),
            "opt": optimizer.state_dict(),
            "args": args,
        },
        path,
    )


def main(args) -> None:
    torch.manual_seed(args.run.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir(args.data, args.train)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    logger = create_logger(run_dir)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config: data={args.data} train={args.train} model={args.model}")

    # ---------------------------------------------------------------- data --
    img_size_for_crop = None if not args.train.centercrop else args.model.image_size
    dataset = get_dataset(args.data.data_path, args.data.data_name, img_size_for_crop)
    train_loader = get_train_loader(
        dataset, args.train.global_batch_size, args, shuffle=True
    )
    logger.info(
        f"Dataset '{args.data.data_name}': using {len(train_loader.dataset)} "
        f"training samples (requested train_size={args.train.train_size})."
    )
    data_gen = sample_data(train_loader)

    # --------------------------------------------------------------- model --
    use_data_parallel = torch.cuda.device_count() > 1
    model = get_unet(args, data_parallel=use_data_parallel).to(device)
    raw_model = model.module if use_data_parallel else model

    ema_model = AveragedModel(
        raw_model,
        device=device,
        multi_avg_fn=get_ema_multi_avg_fn(args.train.ema_decay),
    )
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.lr)
    scheduler = build_scheduler(args.train, args.run)

    start_step = 0
    if args.run.resume:
        logger.info(f"Resuming from {args.run.resume}")
        ckpt = torch.load(args.run.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        ema_model.module.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]

    total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params / 1e6:.2f}M trainable parameters.")

    # -------------------------------------------------------------- train --
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for step in range(start_step, args.train.iterations):
        x, _ = next(data_gen)
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(scheduler, model, x, prediction_type=args.train.prediction_type)
        loss.backward()

        if args.train.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if (step + 1) % args.run.ema_update_every == 0:
            ema_model.update_parameters(raw_model)

        running_loss += loss.item()

        if (step + 1) % args.train.log_every == 0:
            avg_loss = running_loss / args.train.log_every
            steps_per_sec = args.train.log_every / (time.time() - t0)
            logger.info(
                f"step {step + 1:>8d}/{args.train.iterations} | "
                f"loss {avg_loss:.4f} | {steps_per_sec:.2f} it/s"
            )
            running_loss = 0.0
            t0 = time.time()

        if (step + 1) % args.train.ckpt_every == 0 or (step + 1) == args.train.iterations:
            ckpt_path = os.path.join(ckpt_dir, f"{step + 1}.pt")
            save_checkpoint(ckpt_path, step + 1, raw_model, ema_model, optimizer, args)
            latest_path = os.path.join(ckpt_dir, "latest.pt")
            save_checkpoint(latest_path, step + 1, raw_model, ema_model, optimizer, args)
            logger.info(f"Saved checkpoint at step {step + 1} -> {ckpt_path}")

        if (step + 1) % args.train.ckpt_every == 0 or (step + 1) == args.train.iterations:
            latest_path = os.path.join(ckpt_dir, "latest.pt")
            save_checkpoint(latest_path, step + 1, raw_model, ema_model, optimizer, args)
            logger.info(f"Saved checkpoint at step {step + 1} -> {latest_path}")

            if args.run.keep_milestone_every > 0:
                n_ckpts_so_far = (step + 1) // args.train.ckpt_every
                if n_ckpts_so_far % args.run.keep_milestone_every == 0:
                    milestone_path = os.path.join(ckpt_dir, f"{step + 1}.pt")
                    save_checkpoint(
                        milestone_path, step + 1, raw_model, ema_model, optimizer, args
                    )
                    logger.info(f"Kept milestone checkpoint -> {milestone_path}")

    final_path = os.path.join(ckpt_dir, "final.pt")
    save_checkpoint(final_path, args.train.iterations, raw_model, ema_model, optimizer, args)
    logger.info(f"Training complete. Final checkpoint -> {final_path}")


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(DataOptions, dest="data")
    parser.add_arguments(TrainOptions, dest="train")
    parser.add_arguments(ModelOptions, dest="model")
    parser.add_arguments(RunOptions, dest="run")
    parsed_args = parser.parse_args()

    # ModelOptions parses dim_mults/attn_resolutions as comma-separated
    # strings (e.g. "1,2,2,2"), but Unet's constructor (src/models/ddpm.py)
    # expects tuples of ints. stats_utils.get_unet passes these straight
    # through without converting them, so we convert once here — both for
    # this run and for whatever gets pickled into the checkpoint's "args",
    # since run_svd.py later calls get_unet(ckpt_args, ...) again on load.
    parsed_args.model.dim_mults = str2tuple(parsed_args.model.dim_mults)
    parsed_args.model.attn_resolutions = str2tuple(parsed_args.model.attn_resolutions)

    # `get_unet` (in stats_utils.py) expects a flat namespace with .model.*,
    # and `get_train_loader` expects .train.* — both already satisfied by
    # the dest= groups above, so `parsed_args` can be passed straight through.
    main(parsed_args)
