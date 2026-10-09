"""
dataset.py
==========
Data loading, degradation (noise injection / down-sampling) and a synthetic
offline fallback.

Data conventions (used by every module)
---------------------------------------
* Source images : CIFAR-10, RGB, 32x32, float in [0, 1].
* ``hr_rgb``    : ground-truth colour image            [3, H,   W  ]
* ``hr_gray``   : luminance of ``hr_rgb``              [1, H,   W  ]
* ``lr_gray``   : ``hr_gray`` box-downsampled by scale [1, H/s, W/s]

Noise is applied to ``lr_gray`` *on the fly* (see ``add_noise``) so every epoch
sees different corruptions, while validation uses a seeded generator so that
validation numbers are comparable between epochs.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

NOISE_TYPES = ("gaussian", "salt_pepper", "mixed")


# --------------------------------------------------------------------------- #
# Colour / resolution helpers
# --------------------------------------------------------------------------- #
def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luminance. Works for ``[3,H,W]`` and ``[B,3,H,W]`` tensors."""
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x * weights).sum(dim=-3, keepdim=True)


def downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Box (area) downsampling by an integer factor. ``[C,H,W]`` or ``[B,C,H,W]``."""
    if scale == 1:
        return x
    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(0)
    out = F.avg_pool2d(x, kernel_size=scale, stride=scale)
    return out.squeeze(0) if squeeze else out


# --------------------------------------------------------------------------- #
# Noise injection (operates on CPU or GPU tensors in [0, 1], shape [B,C,H,W])
# --------------------------------------------------------------------------- #
def _rand(shape, generator: Optional[torch.Generator]) -> torch.Tensor:
    return torch.rand(shape, generator=generator)


def add_gaussian_noise(
    x: torch.Tensor,
    std_range: Tuple[float, float] = (0.05, 0.30),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Additive white Gaussian noise with a per-sample random std."""
    lo, hi = std_range
    std = _rand((x.size(0), 1, 1, 1), generator) * (hi - lo) + lo
    noise = torch.randn(x.shape, generator=generator) * std
    return (x + noise.to(x.device)).clamp(0.0, 1.0)


def add_salt_pepper_noise(
    x: torch.Tensor,
    amount_range: Tuple[float, float] = (0.02, 0.15),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Impulse noise: a random fraction of pixels is set to 0 (pepper) or 1 (salt)."""
    lo, hi = amount_range
    b, _, h, w = x.shape
    amount = (_rand((b, 1, 1, 1), generator) * (hi - lo) + lo).to(x.device)
    mask = _rand((b, 1, h, w), generator).to(x.device)
    out = torch.where(mask < amount / 2, torch.zeros_like(x), x)
    out = torch.where(mask > 1 - amount / 2, torch.ones_like(x), out)
    return out


def add_noise(
    x: torch.Tensor,
    noise_type: str = "mixed",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Corrupt a batch. ``noise_type``: ``gaussian`` | ``salt_pepper`` | ``mixed``
    (``mixed`` picks Gaussian or salt-and-pepper independently per sample).
    ``generator`` must be a CPU generator (used for reproducible validation).
    """
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got '{noise_type}'")
    if noise_type == "gaussian":
        return add_gaussian_noise(x, generator=generator)
    if noise_type == "salt_pepper":
        return add_salt_pepper_noise(x, generator=generator)
    gauss = add_gaussian_noise(x, generator=generator)
    impulse = add_salt_pepper_noise(x, generator=generator)
    pick_gauss = (_rand((x.size(0), 1, 1, 1), generator) < 0.5).to(x.device)
    return torch.where(pick_gauss, gauss, impulse)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class SyntheticImageDataset(Dataset):
    """
    Procedural colour images (gradients + discs + faint texture). Used only when
    CIFAR-10 cannot be downloaded, so the whole suite still runs offline.
    Images are deterministic functions of their index.
    """

    def __init__(self, length: int, size: int = 32, seed: int = 0) -> None:
        self.length, self.size, self.seed = length, size, seed
        axis = torch.linspace(0, 1, size)
        self._yy, self._xx = torch.meshgrid(axis, axis, indexing="ij")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        c1, c2 = torch.rand(3, generator=g), torch.rand(3, generator=g)
        angle = float(torch.rand(1, generator=g)) * 2 * math.pi
        t = math.cos(angle) * self._xx + math.sin(angle) * self._yy
        t = (t - t.min()) / (t.max() - t.min() + 1e-6)
        img = c1[:, None, None] * (1 - t) + c2[:, None, None] * t
        for _ in range(2 + int(torch.randint(0, 3, (1,), generator=g))):
            cx, cy = torch.rand(2, generator=g).tolist()
            radius = 0.1 + 0.2 * float(torch.rand(1, generator=g))
            color = torch.rand(3, generator=g)
            disc = (self._xx - cx) ** 2 + (self._yy - cy) ** 2 < radius**2
            img = torch.where(disc, color[:, None, None], img)
        img = img + 0.03 * torch.randn(img.shape, generator=g)
        return img.clamp(0, 1), 0


class RestorationDataset(Dataset):
    """Wraps an (image, label) dataset and yields the dict described at the top."""

    def __init__(self, base: Dataset, scale: int = 2) -> None:
        if scale not in (1, 2, 4):
            raise ValueError("scale must be 1, 2 or 4")
        self.base, self.scale = base, scale

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, _ = self.base[idx]
        hr_gray = rgb_to_gray(image)
        return {
            "hr_rgb": image,
            "hr_gray": hr_gray,
            "lr_gray": downsample(hr_gray, self.scale),
        }


def _build_base(train: bool, root: str, use_synthetic: bool, seed: int):
    """Return ``(dataset, source_name)``; falls back to synthetic data on failure."""
    if not use_synthetic:
        try:
            tf = transforms.ToTensor()
            if train:
                tf = transforms.Compose([transforms.RandomHorizontalFlip(), tf])
            ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tf)
            return ds, "cifar10"
        except Exception as exc:
            warnings.warn(
                f"CIFAR-10 unavailable ({type(exc).__name__}: {exc}). "
                "Falling back to the synthetic dataset."
            )
    length = 50_000 if train else 10_000
    return SyntheticImageDataset(length, 32, seed=seed + (0 if train else 777)), "synthetic"


def _subset(ds: Dataset, n: int, seed: int) -> Dataset:
    if n <= 0 or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    return Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())


def get_dataloaders(
    root: str = "data",
    batch_size: int = 64,
    scale: int = 2,
    train_subset: int = 10_000,
    val_subset: int = 2_000,
    num_workers: int = 0,
    use_synthetic: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, str]:
    """
    Build train / validation loaders (``batch_size`` defaults to 64).
    ``*_subset <= 0`` means "use the full split". Returns the data source name.
    """
    train_base, source = _build_base(True, root, use_synthetic, seed)
    val_base, _ = _build_base(False, root, source == "synthetic", seed)
    train_ds = RestorationDataset(_subset(train_base, train_subset, seed), scale)
    val_ds = RestorationDataset(_subset(val_base, val_subset, seed + 1), scale)

    pin = torch.cuda.is_available()
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, source


def get_test_dataset(
    root: str = "data", scale: int = 2, use_synthetic: bool = False, seed: int = 42
) -> Tuple[RestorationDataset, str]:
    """Held-out split (no augmentation) used by the demonstration pipeline."""
    base, source = _build_base(False, root, use_synthetic, seed)
    return RestorationDataset(base, scale), source


def load_image_file(path: str, size: int = 32, scale: int = 2) -> torch.Tensor:
    """
    Load an arbitrary image as a ``[1, 3, S, S]`` tensor in [0, 1]. The short
    side is resized to ``size`` then centre-cropped; ``S`` is rounded down to a
    multiple of ``scale`` so that it can be downsampled exactly.
    """
    from PIL import Image

    size = max(scale, (size // scale) * scale)
    tf = transforms.Compose(
        [transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()]
    )
    with Image.open(Path(path)) as img:
        return tf(img.convert("RGB")).unsqueeze(0)
