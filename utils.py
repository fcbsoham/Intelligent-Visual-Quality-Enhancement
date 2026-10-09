"""
utils.py
========
Shared infrastructure for the Intelligent Visual Quality Enhancement &
Restoration Suite.

Contents
--------
* Reproducibility / device routing / explicit accelerator memory cleanup
* Module 4 regularisation helpers (modular Dropout, L2 via ``weight_decay``)
* torch.autograd utilities (gradient norms, input-saliency via autograd.grad)
* Metrics: PSNR, SSIM (pure PyTorch), running metric tracker, training history
* Loss helpers: pixel losses and a VGG19 perceptual loss (with offline fallback)
* Plotting: training curves and side-by-side comparison grids
* Checkpoint I/O and the end-to-end ``RestorationPipeline``

This file intentionally does NOT import from ``models`` so that the model
package can import from it without creating a circular dependency.
"""

from __future__ import annotations

import gc
import json
import math
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # headless backend: safe on servers / CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


# --------------------------------------------------------------------------- #
# Reproducibility, device routing, memory management
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: Optional[str] = "auto") -> torch.device:
    """Route to ``cuda`` -> ``mps`` -> ``cpu`` (or honour an explicit choice)."""
    if preference and preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_memory(device: torch.device) -> None:
    """Run the garbage collector and release cached accelerator memory."""
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:  # pragma: no cover - depends on torch build
            pass


def memory_report(device: torch.device) -> str:
    """Human-readable accelerator memory usage (empty string on CPU/MPS)."""
    if device.type == "cuda" and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device) / 2**20
        reserved = torch.cuda.memory_reserved(device) / 2**20
        return f"GPU mem {alloc:.0f}MB alloc / {reserved:.0f}MB reserved"
    return ""


# --------------------------------------------------------------------------- #
# Module 4: regularisation (Dropout + L2)
# --------------------------------------------------------------------------- #
def make_dropout(p: float, spatial: bool = True) -> nn.Module:
    """
    Modular dropout factory.

    ``spatial=True`` -> ``nn.Dropout2d`` (drops whole feature maps, the right
    choice for convolutional activations). ``p == 0`` returns ``nn.Identity``
    so the layer can stay in the architecture at zero cost.
    """
    if not 0.0 <= p < 1.0:
        raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
    if p == 0.0:
        return nn.Identity()
    return nn.Dropout2d(p) if spatial else nn.Dropout(p)


def build_optimizer(
    target: Union[nn.Module, Iterable[torch.nn.Parameter]],
    name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    momentum: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """
    Build an optimizer with L2 regularisation through ``weight_decay``.

    If a module is passed, parameters are split into two groups: conv/linear
    *weights* receive weight decay, while biases and normalisation parameters
    (1-D tensors) do not, which is standard practice.
    """
    if isinstance(target, nn.Module):
        decay, no_decay = [], []
        for param in target.parameters():
            if not param.requires_grad:
                continue
            (decay if param.ndim > 1 else no_decay).append(param)
        params = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    else:
        params = list(target)

    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
        )
    raise ValueError(f"Unknown optimizer '{name}' (use adam | adamw | sgd)")


def l2_penalty(model: nn.Module, coeff: float) -> torch.Tensor:
    """Explicit L2 penalty ``coeff * sum(w^2)`` over conv/linear weights."""
    total = sum((p.pow(2).sum() for p in model.parameters() if p.ndim > 1), start=torch.zeros(()))
    return coeff * total


# --------------------------------------------------------------------------- #
# torch.autograd utilities
# --------------------------------------------------------------------------- #
def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Enable / disable gradient tracking for every parameter of ``module``."""
    for param in module.parameters():
        param.requires_grad_(flag)


def safe_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, max_norm: float = 1.0
) -> Tuple[float, bool]:
    """
    Clip gradients, measure the global gradient norm (taken from the autograd
    graph populated by ``loss.backward()``) and step only if it is finite.

    Returns ``(grad_norm_before_clipping, stepped)``.
    """
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0, False
    limit = max_norm if max_norm and max_norm > 0 else float("inf")
    norm = float(torch.nn.utils.clip_grad_norm_(params, limit))
    if math.isfinite(norm):
        optimizer.step()
        return norm, True
    optimizer.zero_grad(set_to_none=True)
    return norm, False


def input_saliency(model: nn.Module, x: torch.Tensor, target: torch.Tensor, loss_fn) -> float:
    """
    Mean absolute gradient of the loss w.r.t. the *input* tensor, obtained with
    ``torch.autograd.grad``. A healthy value (non-zero, finite) shows the whole
    graph from input to loss is differentiable; it is logged every epoch.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            x_in = x.detach().clone().requires_grad_(True)
            loss = loss_fn(model(x_in), target)
            (grad,) = torch.autograd.grad(loss, x_in)
        return float(grad.abs().mean().item())
    finally:
        model.train(was_training)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, reduce: bool = True):
    """Peak signal-to-noise ratio in dB for ``[B, C, H, W]`` tensors."""
    mse = (x - y).pow(2).flatten(1).mean(dim=1).clamp_min(1e-10)
    value = 10.0 * torch.log10(data_range**2 / mse)
    return value.mean() if reduce else value


def _gaussian_kernel(win: int, sigma: float, channels: int, like: torch.Tensor) -> torch.Tensor:
    coords = torch.arange(win, dtype=like.dtype, device=like.device) - (win - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel.expand(channels, 1, win, win).contiguous()


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    sigma: float = 1.5,
    reduce: bool = True,
):
    """
    Structural similarity (Gaussian window, valid convolution) in pure PyTorch.
    The window shrinks automatically for very small images (e.g. 8x8 inputs).
    """
    _, channels, height, width = x.shape
    win = min(win_size, height, width)
    if win % 2 == 0:
        win -= 1
    kernel = _gaussian_kernel(win, sigma, channels, x)

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, groups=channels)

    mu_x, mu_y = blur(x), blur(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = blur(x * x) - mu_x2
    sigma_y = blur(y * y) - mu_y2
    sigma_xy = blur(x * y) - mu_xy

    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    per_image = ssim_map.flatten(1).mean(dim=1)
    return per_image.mean() if reduce else per_image


class MetricTracker:
    """Sample-weighted running averages for a dict of scalar metrics."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        for key, value in metrics.items():
            self._sums[key] += float(value) * n
            self._counts[key] += n

    def averages(self) -> Dict[str, float]:
        return {k: self._sums[k] / max(self._counts[k], 1) for k in self._sums}


class History:
    """Per-epoch metric log (aligned lists) with JSON export."""

    def __init__(self) -> None:
        self.data: Dict[str, List[float]] = defaultdict(list)
        self.meta: Dict[str, object] = {}

    def log(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self.data[key].append(float(value))

    def to_dict(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self.data.items()}

    def save_json(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump({"history": self.to_dict(), "meta": self.meta}, handle, indent=2)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def get_pixel_loss(name: str = "smooth_l1") -> nn.Module:
    """Pixel-wise reconstruction loss: ``l1`` | ``smooth_l1`` | ``mse``."""
    name = name.lower()
    if name == "l1":
        return nn.L1Loss()
    if name in ("smooth_l1", "huber"):
        return nn.SmoothL1Loss(beta=0.1)
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown pixel loss '{name}'")


class PerceptualLoss(nn.Module):
    """
    Content / perceptual loss: MSE between frozen deep features (VGG19 up to
    ``relu3_4``). If the pretrained weights cannot be downloaded (offline
    machines) a frozen, deterministically initialised conv extractor is used so
    that the pipeline still runs; ``self.backend`` reports which was used.
    Inputs are RGB tensors in ``[0, 1]``.
    """

    def __init__(self, use_vgg: bool = True, layer_index: int = 18) -> None:
        super().__init__()
        extractor: Optional[nn.Module] = None
        self.backend = "random-conv"
        if use_vgg:
            try:
                from torchvision.models import VGG19_Weights, vgg19

                vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
                extractor = nn.Sequential(*list(vgg.features.children())[:layer_index])
                self.backend = "vgg19"
            except Exception as exc:  # network / cache problems
                warnings.warn(
                    f"Could not load pretrained VGG19 ({type(exc).__name__}); "
                    "falling back to a frozen random conv feature extractor."
                )
        if extractor is None:
            with torch.random.fork_rng():
                torch.manual_seed(1234)
                extractor = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                )
                for module in extractor.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                        nn.init.zeros_(module.bias)
        self.extractor = extractor.eval()
        set_requires_grad(self.extractor, False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor((x - self.mean) / self.std)

    def forward(self, restored: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        feat_restored = self._features(restored)
        with torch.no_grad():
            feat_target = self._features(target)
        return F.mse_loss(feat_restored, feat_target)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_history(history: History, path: Union[str, Path], title: str = "Training history") -> None:
    """
    One subplot per metric. Keys named ``train_<m>`` / ``val_<m>`` are drawn
    together; ``val_baseline_<m>`` is drawn as a dashed reference line.
    """
    data = history.to_dict()
    groups: List[str] = []
    for key in data:
        if key.startswith("val_baseline_"):
            continue
        for prefix in ("train_", "val_"):
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                if suffix not in groups:
                    groups.append(suffix)
    if not groups:
        return

    cols = min(3, len(groups))
    rows = math.ceil(len(groups) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
    for ax, group in zip(axes.flat, groups):
        for prefix, style in (("train_", "-"), ("val_", "--")):
            key = prefix + group
            if key in data:
                ax.plot(range(1, len(data[key]) + 1), data[key], style, label=prefix[:-1])
        base_key = f"val_baseline_{group}"
        if base_key in data:
            ax.plot(range(1, len(data[base_key]) + 1), data[base_key], ":", color="gray", label="baseline")
        ax.set_title(group)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _to_display(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu().float().clamp(0, 1)
    if img.size(0) == 1:
        return img[0].numpy()
    return img.permute(1, 2, 0).numpy()


def save_comparison_grid(
    columns: Sequence[Tuple[str, torch.Tensor]],
    path: Union[str, Path],
    captions: Optional[Sequence[Optional[Sequence[str]]]] = None,
    max_rows: int = 8,
    cell: float = 2.1,
) -> None:
    """
    Save a grid where every column is ``(title, tensor[N, C, H, W])`` and each
    row is one sample. Grayscale and RGB columns can be mixed; ``captions[j][i]``
    is an optional small caption under cell (row i, column j).
    """
    rows = min(max_rows, min(t.size(0) for _, t in columns))
    cols = len(columns)
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols, (cell + 0.35) * rows), squeeze=False)
    for j, (title, tensor) in enumerate(columns):
        for i in range(rows):
            ax = axes[i][j]
            img = _to_display(tensor[i])
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=9)
            if captions is not None and captions[j] is not None:
                ax.set_xlabel(captions[j][i], fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    history: Optional[History] = None,
    extra: Optional[dict] = None,
    state_dict: Optional[dict] = None,
) -> None:
    """Persist weights + constructor kwargs (``model.init_args``) + history."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict if state_dict is not None else model.state_dict(),
            "init_args": dict(getattr(model, "init_args", {})),
            "history": history.to_dict() if history is not None else {},
            "extra": extra or {},
        },
        path,
    )


def load_model(model_cls, path: Union[str, Path], device: torch.device) -> nn.Module:
    """Rebuild ``model_cls`` from a checkpoint and return it in eval mode."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = model_cls(**ckpt["init_args"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# End-to-end pipeline: Denoise -> Colorize / Super-Resolve -> GAN refine
# --------------------------------------------------------------------------- #
class RestorationPipeline(nn.Module):
    """
    Chains the three trained modules.

    Input : noisy low-resolution grayscale ``[B, 1, h, w]`` in ``[0, 1]``
    Output: dict with ``denoised`` ``[B,1,h,w]``, ``colorized_sr`` ``[B,3,s*h,s*w]``
            and (if a generator is attached) ``refined`` ``[B,3,s*h,s*w]``.
    """

    def __init__(self, autoencoder: nn.Module, unet: nn.Module, generator: Optional[nn.Module] = None):
        super().__init__()
        self.autoencoder = autoencoder
        self.unet = unet
        self.generator = generator

    @torch.no_grad()
    def stage12(self, noisy_lr_gray: torch.Tensor) -> torch.Tensor:
        """Denoise then colorize + super-resolve (used to train the GAN)."""
        return self.unet(self.autoencoder(noisy_lr_gray))

    @torch.no_grad()
    def restore(self, noisy_lr_gray: torch.Tensor) -> Dict[str, torch.Tensor]:
        denoised = self.autoencoder(noisy_lr_gray)
        colorized_sr = self.unet(denoised)
        outputs = {"denoised": denoised, "colorized_sr": colorized_sr}
        outputs["refined"] = self.generator(colorized_sr) if self.generator is not None else colorized_sr
        return outputs
