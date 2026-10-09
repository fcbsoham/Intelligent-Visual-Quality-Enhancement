"""
models/unet.py
==============
Module 2 - U-Net for joint Colorization + Super-Resolution.

Input  : low-res grayscale ``[B, 1, h, w]`` in [0, 1]
Output : RGB ``[B, 3, scale*h, scale*w]`` in [0, 1]   (scale in {1, 2, 4})

The input is first resized with bicubic interpolation to the target size, then
the U-Net predicts colour + detail. ``scale=1`` gives pure colorization.

Shapes for a 32x32 target with base channels c:

    inc    : 1   -> c    @ 32x32   (kept as skip x1)
    down1  : c   -> 2c   @ 16x16   (kept as skip x2)
    down2  : 2c  -> 4c   @ 8x8     (kept as skip x3)
    down3  : 4c  -> 8c   @ 4x4     bottleneck
    up1    : 8c  -> 4c   @ 8x8   cat([x3, up]) -> 8c -> DoubleConv -> 4c
    up2    : 4c  -> 2c   @ 16x16 cat([x2, up]) -> 4c -> DoubleConv -> 2c
    up3    : 2c  -> c    @ 32x32 cat([x1, up]) -> 2c -> DoubleConv -> c
    outc   : c   -> 3 (1x1 conv) + Sigmoid
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import make_dropout  # noqa: E402


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2 followed by optional spatial dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            make_dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Contracting step: MaxPool(2) then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Expanding step: ConvTranspose2d halves the channels and doubles H, W; the
    result is padded to the skip tensor's size (handles odd inputs) and
    concatenated with ``torch.cat([skip, x], dim=1)``.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # after the cat: (in_ch // 2) skip channels + (in_ch // 2) upsampled = in_ch
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_h, diff_w = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        scale: int = 2,
        base_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if scale not in (1, 2, 4):
            raise ValueError("scale must be 1 (colorize only), 2 or 4")
        self.init_args = dict(
            in_channels=in_channels, out_channels=out_channels, scale=scale,
            base_channels=base_channels, dropout=dropout,
        )
        self.scale = scale
        c = base_channels

        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, 2 * c)
        self.down2 = Down(2 * c, 4 * c)
        self.down3 = Down(4 * c, 8 * c, dropout)   # bottleneck (regularised)
        self.up1 = Up(8 * c, 4 * c)
        self.up2 = Up(4 * c, 2 * c)
        self.up3 = Up(2 * c, c)
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
            x = x.clamp(0.0, 1.0)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y = self.up1(x4, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)
        return torch.sigmoid(self.outc(y))
