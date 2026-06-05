"""
train_utils.py
Utilities handling dataset ingestion, preprocessing logic (like ADM crops),
distributed logging, and the core forward/reverse diffusion steps.
"""

import os
import logging
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder, MNIST, LSUN, FashionMNIST

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def str2tuple(v):
    """Safely converts string config values to tuples."""
    try:
        return tuple([int(v)])
    except ValueError:
        return tuple([int(c) for c in v.split(",")])


def str2bool(v):
    """Converts a standard set of string flags to python booleans."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    return False


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation pulled directly from ADM.
    Ensures images are structurally scaled before executing the final center crop.
    Ref: https://github.com/openai/guided-diffusion
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2

    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def create_logger(logging_dir, index: int = 1):
    """Creates a basic formatted logger outputting to file and stdout stream."""
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log_{index}.txt"),
        ],
    )
    return logging.getLogger(__name__)


def get_transform(image_size=None, single=False):
    """Returns a torchvision transform pipeline depending on grayscale/RGB needs."""
    mean = MEAN if not single else MEAN[0]
    std = STD if not single else STD[0]

    if image_size is None:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std, inplace=True),
            ]
        )

    return transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std, inplace=True),
        ]
    )


def get_cifar10(data_path):
    return CIFAR10(data_path, train=True, download=True, transform=get_transform(None))


def get_celebahq(data_path, img_size: int = None):
    """Custom data loader specifically targeting CelebA-HQ numpy structures."""

    def is_valid_file(path):
        if path.endswith(".npy") or path.endswith(".npz"):
            return path
        raise ValueError("Invalid file. CelebA-HQ loader expects .npy or .npz")

    def loader_fn(path):
        try:
            # Assuming data is stored as C x H x W
            arr = np.load(path, mmap_mode="r").squeeze(0).transpose(1, 2, 0)
        except Exception:
            arr = np.load(path, mmap_mode="r").transpose(1, 2, 0)
        return Image.fromarray(arr)

    return ImageFolder(
        data_path,
        transform=get_transform(img_size),
        loader=loader_fn,
        is_valid_file=is_valid_file,
    )


def get_dataset(data_path: str, name: str = "cifar10", img_size: int = None):
    """Dataset routing and initialization logic."""
    name = name.lower()

    if name == "cifar10":
        return get_cifar10(data_path)
    elif name in ("celeba", "imagenet"):
        return ImageFolder(data_path, transform=get_transform(img_size))
    elif name == "celebahq":
        return get_celebahq(data_path, img_size)
    elif name == "mnist":
        return MNIST(data_path, download=True, transform=get_transform(None, True))
    elif name == "fashionmnist":
        return FashionMNIST(
            data_path, download=True, transform=get_transform(None, True)
        )
    elif name == "lsun-church":
        return LSUN(
            data_path,
            classes=["church_outdoor_train"],
            transform=get_transform(img_size),
        )
    else:
        raise ValueError("Invalid Dataset requested.")


def split_data(data, memorize_size, validate_size, seed):
    """Splits a PyTorch dataset into fixed, seeded subsets based on config sizes."""
    max_size = len(data)
    generator = torch.Generator().manual_seed(seed)

    if memorize_size >= max_size:
        _, valid_data = torch.utils.data.random_split(
            data, [max_size - validate_size, validate_size], generator=generator
        )
        return data, valid_data

    elif validate_size >= max_size:
        train_data, _ = torch.utils.data.random_split(
            data, [memorize_size, max_size - memorize_size], generator=generator
        )
        return train_data, data

    train_data, valid_data, _ = torch.utils.data.random_split(
        data,
        [memorize_size, validate_size, max_size - (memorize_size + validate_size)],
        generator=generator,
    )
    return train_data, valid_data


def get_dist_info():
    """Fetches variables bound by SLURM for multi-node distributed processing."""
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    return rank, world_size, gpus_per_node


@torch.no_grad()
def to_identity(x):
    """Passthrough structural wrapper."""
    return x


@torch.no_grad()
def to_real(x, vae):
    """Decodes latent features mapping back to pixel space using a VAE."""
    x = vae.decode(x / 0.18215).sample
    return x


def sample(scheduler, model, x):
    """Runs the reverse diffusion sample steps evaluating the UNet per timestep."""
    model.eval()
    for t in scheduler.timesteps:
        x = scheduler.scale_model_input(x, t)
        with torch.no_grad():
            score = model(x, t.repeat(x.shape[0]).to(x.device))
        x = scheduler.step(score, t, x).prev_sample
    return x


def train_loss(scheduler, model, x, prediction_type: str = "epsilon"):
    """
    Executes a standard forward diffusion forward pass and loss evaluation step.
    Supports either standard Epsilon noise matching or direct sample recovery.
    """
    noise = torch.randn_like(x)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, [x.shape[0]], device=x.device
    )

    # Forward diffusion
    x_t = scheduler.add_noise(x, noise=noise, timesteps=timesteps)
    score = model(x_t, timesteps)

    # Loss computation
    if prediction_type == "epsilon":
        return torch.square(noise - score).mean()
    elif prediction_type == "sample":
        return torch.square(x - score).mean()
    else:
        raise ValueError("Invalid Prediction Type. Must be 'epsilon' or 'sample'.")


def sample_data(loader):
    """Infinite generator wrapper around standard PyTorch dataloaders."""
    loader_iter = iter(loader)
    while True:
        try:
            yield next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            yield next(loader_iter)
