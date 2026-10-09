"""
models/autoencoder.py
=====================
Module 1 - Convolutional Denoising Autoencoder.

Encoder : [Conv-BN-ReLU] x2 -> MaxPool -> [Conv-BN-ReLU] x2 -> MaxPool
          -> Conv-BN-ReLU -> Dropout2d            (H, W)  ->  (H/4, W/4)
Decoder : ConvTranspose2d (x2 up) -> Conv -> ConvTranspose2d (x2 up) -> Conv
          -> Conv(3x3) -> Sigmoid                 (H/4, W/4) -> (H, W)

Channel flow (grayscale): 1 -> c -> 2c -> 4c (latent) -> 2c -> c -> 1
Input / output are ``[B, 1, H, W]`` in [0, 1]; the network is fully
convolutional, so any H, W works (odd sizes are re-aligned at the end).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import make_dropout  # noqa: E402


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """3x3 Conv -> BatchNorm -> ReLU (size preserving)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """ConvTranspose2d (2x upsampling) -> BatchNorm -> ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.init_args = dict(in_channels=in_channels, base_channels=base_channels, dropout=dropout)
        c = base_channels

        self.encoder = nn.Sequential(
            conv_block(in_channels, c),
            conv_block(c, c),
            nn.MaxPool2d(2),                 # H/2
            conv_block(c, 2 * c),
            conv_block(2 * c, 2 * c),
            nn.MaxPool2d(2),                 # H/4
            conv_block(2 * c, 4 * c),        # latent code
            make_dropout(dropout),           # Module 4: regularisation
        )
        self.decoder = nn.Sequential(
            up_block(4 * c, 2 * c),          # H/2
            conv_block(2 * c, 2 * c),
            up_block(2 * c, c),              # H
            conv_block(c, c),
            nn.Conv2d(c, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),                    # pixels in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(self.encoder(x))
        if out.shape[-2:] != x.shape[-2:]:  # odd sizes lose a pixel in MaxPool
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out
