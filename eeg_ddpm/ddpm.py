from __future__ import annotations

import math

import torch
from torch import nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int, cosine_s: float = 0.008):
        super().__init__()
        betas = cosine_beta_schedule(timesteps=timesteps, s=cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @staticmethod
    def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, timesteps)
        return out.view(timesteps.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self.extract(self.sqrt_alphas_cumprod, timesteps, x0.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        return (sqrt_alpha_bar * x0) + (sqrt_one_minus_alpha_bar * noise)

    def predict_eps(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
        guidance_w: float = 0.0,
    ) -> torch.Tensor:
        if guidance_w == 0.0:
            return model(xt, timesteps, cond)
        eps_cond = model(xt, timesteps, cond)
        eps_uncond = model(xt, timesteps, cond, force_uncond=True)
        return eps_uncond + guidance_w * (eps_cond - eps_uncond)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        cond: torch.Tensor,
        output_shape: tuple[int, int, int, int],
        guidance_w: float = 0.0,
    ) -> torch.Tensor:
        xt = torch.randn(output_shape, device=cond.device)
        for timestep in reversed(range(self.timesteps)):
            t = torch.full((output_shape[0],), timestep, device=cond.device, dtype=torch.long)
            beta_t = self.extract(self.betas, t, xt.shape)
            sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
            sqrt_recip_alpha_t = self.extract(self.sqrt_recip_alphas, t, xt.shape)
            posterior_variance_t = self.extract(self.posterior_variance, t, xt.shape)

            eps = self.predict_eps(model, xt, t, cond, guidance_w=guidance_w)
            model_mean = sqrt_recip_alpha_t * (xt - (beta_t / sqrt_one_minus_alpha_bar_t) * eps)
            if timestep > 0:
                noise = torch.randn_like(xt)
                xt = model_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                xt = model_mean
        return xt
