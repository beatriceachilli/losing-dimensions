import math
import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from typing import Sequence, Optional, Union, Tuple

Tensor = torch.Tensor


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]
    

class VPScheduler(object):
    def __init__(self, b_min: float = 1e-4, b_max: float = 0.02, T_max: int = 1_000):
        super().__init__()

        self.T_max = T_max
        self.betas = torch.linspace(b_min, b_max, T_max)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = 0)


    def set_variables(self, device):
        if self.betas.device != device:
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.betas = self.betas.to(device)
            self.alphas = self.alphas.to(device)
        

    @torch.no_grad()
    def apply_q_x0(self, x0: Tensor, t: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward process q(x_t | x_0)
        """

        device = x0.device

        training = False
        if t is None:
            t = torch.randint(0, self.T_max, [len(x0)]).to(device)
            training = True
            
        noise = torch.randn_like(x0)
        self.set_variables(x0.device)
        
        var_t = self.alphas_cumprod[t]
        var_t = append_dims(var_t, len(x0.shape))

        xt = torch.sqrt(var_t) * x0 + torch.sqrt(1 - var_t) * noise

        if training:
            return xt, noise, t
        return xt

    
    @torch.no_grad()
    def apply_reverse(self, xt: Tensor, st: Tensor, t: Tensor) -> Tensor:
        """Reverse process p(x_{t - 1} | x_t, x_0)
        """
        assert len(xt) == len(t) and len(xt) == len(st)

        self.set_variables(xt.device)

        alpha_t = self.alphas[t]
        alpha_cp_t = self.alphas_cumprod[t] 
        alpha_t, alpha_cp_t = map(lambda z: append_dims(z, 4), (alpha_t, alpha_cp_t))

        beta_t = 1 - alpha_t # assume beta_t = sigma^2 
        return (xt - beta_t / torch.sqrt(1 - alpha_cp_t) * st) / torch.sqrt(alpha_t) + torch.sqrt(beta_t) * torch.randn_like(xt)


    @torch.no_grad()
    def ddpm(self, model: nn.Module, z: Tensor, clamp: bool = False):
        # basically run DDPM
        model.eval()
        self.set_variables(z.device)

        for t in range(self.T_max - 1, -1, -1): # T - 1 ... 0
            t = torch.full([len(z)], t).to(z.device)
            z = self.apply_reverse(z, model(z, t), t)
        return z.clamp_(-1., 1.)


    @torch.no_grad()
    def ddpm_all(self, model: nn.Module, z: Tensor, clamp: bool = True, store_score: bool = False):
        # basically run DDPM
        model.eval()
        self.set_variables(z.device)

        zs = []
        for t in range(self.T_max - 1, -1, -1): # T - 1 ... 0
            t = torch.full([len(z)], t).to(z.device)
            s_t = model(z, t)
            z = self.apply_reverse(z, s_t, t)
            if store_score:
                zs.append(s_t.cpu())
            else:
                zs.append(z.cpu())

        if not store_score and clamp:
            zs[-1] = zs[-1].clamp(-1, 1)
        return zs[::-1]
            
        