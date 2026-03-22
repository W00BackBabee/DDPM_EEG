from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig


def group_norm_groups(channels: int) -> int:
    for group in (32, 16, 8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        angles = timesteps.float()[:, None] * frequencies[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class EEGConditionEncoder(nn.Module):
    def __init__(self, cond_in_ch: int, cond_ch: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cond_in_ch, cond_ch, kernel_size=3, padding=1),
            nn.GroupNorm(group_norm_groups(cond_ch), cond_ch),
            nn.SiLU(),
            nn.Conv2d(cond_ch, cond_ch, kernel_size=3, padding=1),
            nn.GroupNorm(group_norm_groups(cond_ch), cond_ch),
            nn.SiLU(),
        )
        self.vec_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cond_ch, emb_dim),
        )

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond_map = self.net(eeg)
        cond_vec = self.vec_proj(cond_map)
        return cond_map, cond_vec


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, cond_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_norm_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(group_norm_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        self.cond_proj = nn.Conv2d(cond_ch, out_ch, kernel_size=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, cond_map: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        cond_resized = F.interpolate(cond_map, size=h.shape[-2:], mode="bilinear", align_corners=False)
        h = h + self.cond_proj(cond_resized)

        scale_shift = self.emb_proj(F.silu(emb))[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + residual


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(self, config: ExperimentConfig, cond_in_ch: int, target_in_ch: int = 1):
        super().__init__()
        if target_in_ch != 1:
            raise ValueError(f"Only single-channel audio targets are supported, got target_in_ch={target_in_ch}")

        widths = [config.base_ch * mult for mult in config.ch_mults]
        if not widths:
            raise ValueError("ch_mults must not be empty.")

        self.config = config
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.emb_dim),
            nn.Linear(config.emb_dim, config.emb_dim),
            nn.SiLU(),
            nn.Linear(config.emb_dim, config.emb_dim),
        )
        self.cond_encoder = EEGConditionEncoder(cond_in_ch, config.cond_ch, config.emb_dim)
        self.inj_proj = nn.Conv2d(config.cond_ch, config.inj_ch, kernel_size=1)
        self.input_proj = nn.Conv2d(1 + config.inj_ch, widths[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        current_ch = widths[0]
        for level, width in enumerate(widths):
            self.down_blocks.append(ResBlock(current_ch, width, config.emb_dim, config.cond_ch))
            current_ch = width
            if level < len(widths) - 1:
                self.downsamplers.append(Downsample(current_ch))

        self.mid_block1 = ResBlock(current_ch, current_ch, config.emb_dim, config.cond_ch)
        self.mid_block2 = ResBlock(current_ch, current_ch, config.emb_dim, config.cond_ch)

        self.upsamplers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.upsamplers.append(Upsample(current_ch))
            self.up_blocks.append(ResBlock(current_ch + width, width, config.emb_dim, config.cond_ch))
            current_ch = width

        self.out_norm = nn.GroupNorm(group_norm_groups(current_ch), current_ch)
        self.out_conv = nn.Conv2d(current_ch, target_in_ch, kernel_size=3, padding=1)

    def encode_condition(
        self,
        eeg: torch.Tensor,
        uncond_mask: torch.Tensor | None = None,
        force_uncond: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_map, cond_vec = self.cond_encoder(eeg)
        if force_uncond:
            cond_map = torch.zeros_like(cond_map)
            cond_vec = torch.zeros_like(cond_vec)
            return cond_map, cond_vec
        if uncond_mask is not None:
            mask_4d = uncond_mask[:, None, None, None].float()
            mask_2d = uncond_mask[:, None].float()
            cond_map = cond_map * (1.0 - mask_4d)
            cond_vec = cond_vec * (1.0 - mask_2d)
        return cond_map, cond_vec

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        eeg: torch.Tensor,
        uncond_mask: torch.Tensor | None = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        cond_map, cond_vec = self.encode_condition(eeg, uncond_mask=uncond_mask, force_uncond=force_uncond)
        time_emb = self.time_embed(timesteps)
        emb = time_emb + (self.config.cond_scale * cond_vec)

        inj_map = self.inj_proj(cond_map)
        inj_map = F.interpolate(inj_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        h = self.input_proj(torch.cat([x, inj_map], dim=1))

        skips: list[torch.Tensor] = []
        for level, block in enumerate(self.down_blocks):
            h = block(h, emb, cond_map)
            if level < len(self.downsamplers):
                skips.append(h)
                h = self.downsamplers[level](h)

        h = self.mid_block1(h, emb, cond_map)
        h = self.mid_block2(h, emb, cond_map)

        for upsample, block, skip in zip(self.upsamplers, self.up_blocks, reversed(skips)):
            h = upsample(h, size=skip.shape[-2:])
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, emb, cond_map)

        return self.out_conv(F.silu(self.out_norm(h)))
