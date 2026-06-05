from dataclasses import dataclass
from simple_parsing import choice, field, ArgumentParser


@dataclass
class DataOptions:
    data_path: str = field("../data", alias="--data-path")
    results_path: str = field("results", alias="--results-path")
    data_name: str = choice(
        "celeba",
        "celebahq",
        "mnist",
        "cifar10",
        "lsun-church",
        "fashionmnist",
        default="cifar10",
        alias="--data-name",
    )


@dataclass
class TrainOptions:
    global_batch_size: int = field(128, alias="--global-batch-size")
    iterations: int = field(400_000, alias="--iterations")
    num_workers: int = field(4, alias="--num-workers")
    log_every: int = field(500, alias="--log-every")
    ckpt_every: int = field(500, alias="--ckpt-every")
    train_size: int = field(1_000, alias="--train-size")
    valid_size: int = field(1_000, alias="--valid-size")
    global_seed: int = field(3407, alias="--global-seed")
    lr: float = field(1e-4, alias="--lr")
    ema_decay: float = field(0.9999, alias="--ema-decay")
    clip_grad: bool = field(True, alias="--clip-grad")
    prediction_type: str = choice(
        "epsilon", "sample", default="epsilon", alias="--prediction-type"
    )
    beta_schedule: str = choice(
        "linear",
        default="linear",
        alias="--beta-schedule",
    )
    timesteps: int = field(1_000, alias="--timesteps")
    centercrop: bool = field(False, alias="--centercrop")


@dataclass
class ModelOptions:
    image_size: int = field(32, alias="--image-size")
    in_channels: int = field(3, alias="--in-channels")
    dim: int = field(128, alias="--dim")
    dim_mults: str = field("1,2,2,2", alias="--dim-mults")
    attn_resolutions: str = field("16", alias="--attn-resolutions")
    num_res_blocks: int = field(2, alias="--num-res-blocks")
    dropout: float = field(0.0, alias="--dropout")
    conditional: bool = field(True, alias="--conditional")
    resamp_with_conv: bool = field(True, alias="--resamp-with-conv")
    nonlinearity: str = field("swish", alias="--nonlinearity")
    scale_by_sigma: bool = field(False, alias="--scale-by-sigma")
