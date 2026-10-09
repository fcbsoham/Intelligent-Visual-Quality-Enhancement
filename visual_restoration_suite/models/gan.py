"""
models/gan.py
=============
Module 3 - High-fidelity texture enhancement.

Generator     : residual refinement network (SRResNet-style, no up-sampling).
                head conv -> N x ResidualBlock -> conv+BN (+ long skip) ->
                conv -> Tanh residual added to the input and clamped to [0, 1].
                The final conv is initialised near zero, so training starts from
                the identity mapping (stable) and learns only the missing texture.
Discriminator : PatchGAN. Outputs a grid of real/fake *logits*, one per
                receptive-field patch. Uses LeakyReLU(0.2); the sigmoid is folded
                into ``nn.BCEWithLogitsLoss`` for numerical stability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import make_dropout  # noqa: E402


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU-(Dropout)-Conv-BN with an identity skip connection."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            make_dropout(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_res_blocks: int = 6,
        dropout: float = 0.0,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.init_args = dict(
            in_channels=in_channels, base_channels=base_channels,
            num_res_blocks=num_res_blocks, dropout=dropout, residual_scale=residual_scale,
        )
        self.residual_scale = residual_scale
        self.head = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels, dropout) for _ in range(num_res_blocks)])
        self.mid = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
        )
        self.tail = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        nn.init.normal_(self.tail.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = self.mid(self.blocks(feat)) + feat           # long skip connection
        delta = torch.tanh(self.tail(feat))                 # texture residual in [-1, 1]
        return (x + self.residual_scale * delta).clamp(0.0, 1.0)


class PatchDiscriminator(nn.Module):
    """
    70x70-style PatchGAN (shallower, sized for 32x32 inputs).
    32x32 -> 16x16 -> 8x8 -> 7x7 -> 6x6 grid of logits.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.init_args = dict(in_channels=in_channels, base_channels=base_channels, dropout=dropout)
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(2 * c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * c, 4 * c, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(4 * c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            make_dropout(dropout),
            nn.Conv2d(4 * c, 1, 4, stride=1, padding=1),    # patch logits
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
