"""
train.py
========
Training loops for all three learned modules plus a ``ensure_models`` helper
that loads existing checkpoints or trains whatever is missing.

    python train.py --module autoencoder      # Module 1
    python train.py --module unet             # Module 2
    python train.py --module gan              # Module 3 (needs AE + U-Net checkpoints)
    python train.py --module all              # everything, in order

Every loop tracks Training / Validation loss, PSNR and SSIM per epoch, the
global gradient norm and an autograd-derived input saliency, applies Dropout +
L2 (``weight_decay``) regularisation, and releases cached accelerator memory at
the end of every epoch.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from dataset import add_noise, get_dataloaders
from models import DenoisingAutoencoder, Generator, PatchDiscriminator, UNet
from utils import (
    History,
    MetricTracker,
    PerceptualLoss,
    RestorationPipeline,
    build_optimizer,
    clear_memory,
    get_device,
    get_pixel_loss,
    input_saliency,
    l2_penalty,
    load_model,
    memory_report,
    plot_history,
    psnr,
    safe_step,
    save_checkpoint,
    set_requires_grad,
    set_seed,
    ssim,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    # generic
    epochs: int = 20
    batch_size: int = 64
    seed: int = 42
    device: str = "auto"
    scale: int = 2                      # super-resolution factor (1, 2 or 4)
    data_root: str = "data"
    ckpt_dir: str = "checkpoints"
    out_dir: str = "outputs"
    train_subset: int = 10_000          # <= 0 means the full 50k CIFAR-10 train split
    val_subset: int = 2_000             # <= 0 means the full 10k test split
    num_workers: int = 0
    synthetic: bool = False             # force the offline synthetic dataset
    noise_type: str = "mixed"           # gaussian | salt_pepper | mixed
    # optimisation / regularisation (Module 4)
    optimizer: str = "adam"
    lr: float = 1e-3
    gan_lr: float = 2e-4
    weight_decay: float = 1e-5          # L2 regularisation
    explicit_l2: float = 0.0            # optional extra explicit L2 penalty term
    dropout: float = 0.1
    grad_clip: float = 1.0
    detect_anomaly: bool = False        # torch.autograd.set_detect_anomaly (slow)
    # architecture sizes
    ae_channels: int = 32
    unet_channels: int = 32
    g_channels: int = 64
    g_blocks: int = 6
    d_channels: int = 64
    # U-Net
    pixel_loss: str = "smooth_l1"       # l1 | smooth_l1 | mse
    unet_input_jitter: float = 0.02     # tiny input noise so the U-Net tolerates AE artefacts
    # GAN
    w_pixel: float = 1.0
    w_perceptual: float = 0.05
    w_adv: float = 0.005
    gan_warmup_epochs: int = 2          # content-only generator pre-training
    use_vgg: bool = True                # pretrained VGG19 perceptual loss (needs download)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Expose every ``TrainConfig`` field as a ``--kebab-case`` CLI flag."""
    for field in dataclasses.fields(TrainConfig):
        flag = "--" + field.name.replace("_", "-")
        default = field.default
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig)})


def ckpt_path(cfg: TrainConfig, name: str) -> Path:
    return Path(cfg.ckpt_dir) / f"{name}_x{cfg.scale}.pt"


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def _setup(cfg: TrainConfig, tag: str) -> torch.device:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n=== [{tag}] device={device} | epochs={cfg.epochs} | batch={cfg.batch_size} | scale=x{cfg.scale}")
    return device


def _get_loaders(cfg: TrainConfig, loaders=None):
    if loaders is not None:
        return loaders
    loaders = get_dataloaders(
        root=cfg.data_root, batch_size=cfg.batch_size, scale=cfg.scale,
        train_subset=cfg.train_subset, val_subset=cfg.val_subset,
        num_workers=cfg.num_workers, use_synthetic=cfg.synthetic, seed=cfg.seed,
    )
    print(f"    data source: {loaders[2]} | train batches: {len(loaders[0])} | val batches: {len(loaders[1])}")
    return loaders


def _with_l2(loss: torch.Tensor, model: nn.Module, coeff: float) -> torch.Tensor:
    return loss + l2_penalty(model, coeff).to(loss.device) if coeff > 0 else loss


def _fmt(metrics: Dict[str, float], keys=("loss", "psnr", "ssim")) -> str:
    return " ".join(f"{k} {metrics[k]:.4f}" if k != "psnr" else f"psnr {metrics[k]:.2f}dB"
                    for k in keys if k in metrics)


def _val_generator(cfg: TrainConfig) -> torch.Generator:
    """Fresh seeded generator -> identical validation noise every epoch."""
    return torch.Generator().manual_seed(cfg.seed + 1)


def _snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _finalize(tag, cfg, model, history, best_state, device, ckpt, extra=None):
    """Restore best weights, write the training-curve plot + JSON, free memory."""
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    plot_history(history, Path(cfg.out_dir) / f"{tag}_history_x{cfg.scale}.png", f"{tag} training")
    history.save_json(Path(cfg.out_dir) / f"{tag}_history_x{cfg.scale}.json")
    torch.autograd.set_detect_anomaly(False)
    clear_memory(device)
    print(f"    [{tag}] best checkpoint -> {ckpt}")
    return model, history


def _bicubic_gray_to_rgb(lr_gray: torch.Tensor, scale: int) -> torch.Tensor:
    up = torch.nn.functional.interpolate(lr_gray, scale_factor=scale, mode="bicubic", align_corners=False)
    return up.clamp(0, 1).repeat(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# Module 1: Denoising autoencoder
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _validate_ae(model, loader, device, cfg, criterion) -> Dict[str, float]:
    model.eval()
    tracker, gen = MetricTracker(), _val_generator(cfg)
    for batch in loader:
        clean = batch["lr_gray"]
        noisy = add_noise(clean, cfg.noise_type, generator=gen)
        clean, noisy = clean.to(device), noisy.to(device)
        recon = model(noisy)
        tracker.update(
            {
                "loss": criterion(recon, clean).item(),
                "psnr": psnr(recon, clean).item(),
                "ssim": ssim(recon, clean).item(),
                "baseline_psnr": psnr(noisy, clean).item(),
                "baseline_ssim": ssim(noisy, clean).item(),
            },
            n=clean.size(0),
        )
    return tracker.averages()


def train_autoencoder(cfg: TrainConfig, loaders=None) -> Tuple[DenoisingAutoencoder, History]:
    device = _setup(cfg, "autoencoder")
    train_loader, val_loader, _ = _get_loaders(cfg, loaders)

    model = DenoisingAutoencoder(1, cfg.ae_channels, cfg.dropout).to(device)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr, cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.MSELoss()

    probe = next(iter(val_loader))["lr_gray"]
    probe_noisy = add_noise(probe, cfg.noise_type, generator=_val_generator(cfg))
    probe, probe_noisy = probe.to(device), probe_noisy.to(device)

    history, best_loss, best_state = History(), float("inf"), None
    ckpt = ckpt_path(cfg, "autoencoder")
    for epoch in range(1, cfg.epochs + 1):
        start, tracker = time.time(), MetricTracker()
        model.train()
        for batch in train_loader:
            clean = batch["lr_gray"]
            noisy = add_noise(clean, cfg.noise_type)
            clean, noisy = clean.to(device, non_blocking=True), noisy.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            recon = model(noisy)
            loss = criterion(recon, clean)
            _with_l2(loss, model, cfg.explicit_l2).backward()       # autograd backward pass
            grad_norm, _ = safe_step(model, optimizer, cfg.grad_clip)

            with torch.no_grad():
                tracker.update(
                    {"loss": loss.item(), "psnr": psnr(recon, clean).item(),
                     "ssim": ssim(recon, clean).item(), "grad_norm": grad_norm},
                    n=clean.size(0),
                )
        scheduler.step()

        train_m = tracker.averages()
        val_m = _validate_ae(model, val_loader, device, cfg, criterion)
        val_m["saliency"] = input_saliency(model, probe_noisy, probe, criterion)

        history.log(
            lr=optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        )
        if val_m["loss"] < best_loss:
            best_loss, best_state = val_m["loss"], _snapshot(model)
            save_checkpoint(ckpt, model, history, {"epoch": epoch, "val_loss": best_loss}, best_state)
        print(f"[AE   {epoch:02d}/{cfg.epochs}] train {_fmt(train_m)} | val {_fmt(val_m)} "
              f"(noisy {val_m['baseline_psnr']:.2f}dB) | grad {train_m['grad_norm']:.3f} "
              f"| sal {val_m['saliency']:.2e} | {time.time() - start:.1f}s {memory_report(device)}")
        clear_memory(device)

    return _finalize("autoencoder", cfg, model, history, best_state, device, ckpt)


# --------------------------------------------------------------------------- #
# Module 2: U-Net (colorization + super-resolution)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _validate_unet(model, loader, device, cfg, criterion) -> Dict[str, float]:
    model.eval()
    tracker = MetricTracker()
    for batch in loader:
        lr_gray, hr = batch["lr_gray"].to(device), batch["hr_rgb"].to(device)
        pred = model(lr_gray)
        base = _bicubic_gray_to_rgb(lr_gray, cfg.scale)
        tracker.update(
            {
                "loss": criterion(pred, hr).item(),
                "psnr": psnr(pred, hr).item(),
                "ssim": ssim(pred, hr).item(),
                "baseline_psnr": psnr(base, hr).item(),
                "baseline_ssim": ssim(base, hr).item(),
            },
            n=hr.size(0),
        )
    return tracker.averages()


def train_unet(cfg: TrainConfig, loaders=None) -> Tuple[UNet, History]:
    device = _setup(cfg, "unet")
    train_loader, val_loader, _ = _get_loaders(cfg, loaders)

    model = UNet(1, 3, cfg.scale, cfg.unet_channels, cfg.dropout).to(device)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr, cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = get_pixel_loss(cfg.pixel_loss)

    probe_batch = next(iter(val_loader))
    probe_x, probe_y = probe_batch["lr_gray"].to(device), probe_batch["hr_rgb"].to(device)

    history, best_loss, best_state = History(), float("inf"), None
    ckpt = ckpt_path(cfg, "unet")
    for epoch in range(1, cfg.epochs + 1):
        start, tracker = time.time(), MetricTracker()
        model.train()
        for batch in train_loader:
            lr_gray, hr = batch["lr_gray"].to(device), batch["hr_rgb"].to(device)
            if cfg.unet_input_jitter > 0:
                lr_gray = (lr_gray + cfg.unet_input_jitter * torch.randn_like(lr_gray)).clamp(0, 1)

            optimizer.zero_grad(set_to_none=True)
            pred = model(lr_gray)
            loss = criterion(pred, hr)
            _with_l2(loss, model, cfg.explicit_l2).backward()
            grad_norm, _ = safe_step(model, optimizer, cfg.grad_clip)

            with torch.no_grad():
                tracker.update(
                    {"loss": loss.item(), "psnr": psnr(pred, hr).item(),
                     "ssim": ssim(pred, hr).item(), "grad_norm": grad_norm},
                    n=hr.size(0),
                )
        scheduler.step()

        train_m = tracker.averages()
        val_m = _validate_unet(model, val_loader, device, cfg, criterion)
        val_m["saliency"] = input_saliency(model, probe_x, probe_y, criterion)

        history.log(
            lr=optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        )
        if val_m["loss"] < best_loss:
            best_loss, best_state = val_m["loss"], _snapshot(model)
            save_checkpoint(ckpt, model, history, {"epoch": epoch, "val_loss": best_loss}, best_state)
        print(f"[UNET {epoch:02d}/{cfg.epochs}] train {_fmt(train_m)} | val {_fmt(val_m)} "
              f"(bicubic {val_m['baseline_psnr']:.2f}dB) | grad {train_m['grad_norm']:.3f} "
              f"| sal {val_m['saliency']:.2e} | {time.time() - start:.1f}s {memory_report(device)}")
        clear_memory(device)

    return _finalize("unet", cfg, model, history, best_state, device, ckpt)


# --------------------------------------------------------------------------- #
# Module 3: GAN texture refinement
# --------------------------------------------------------------------------- #
def train_gan(
    cfg: TrainConfig,
    autoencoder: nn.Module,
    unet: nn.Module,
    loaders=None,
) -> Tuple[Generator, History]:
    """
    Train the residual generator + PatchGAN discriminator on the *real* outputs
    of the frozen upstream stages (noisy -> AE -> U-Net), so the GAN learns to
    repair exactly the artefacts it will meet at inference time.

    G loss = w_pixel * pixel + w_perceptual * VGG-feature MSE + w_adv * BCE(D(G(x)), 1)
    D loss = 0.5 * [BCE(D(real), 0.9) + BCE(D(G(x).detach()), 0)]   (one-sided label smoothing)
    """
    device = _setup(cfg, "gan")
    train_loader, val_loader, _ = _get_loaders(cfg, loaders)

    autoencoder, unet = autoencoder.to(device).eval(), unet.to(device).eval()
    set_requires_grad(autoencoder, False)
    set_requires_grad(unet, False)
    pipeline = RestorationPipeline(autoencoder, unet).to(device)

    gen = Generator(3, cfg.g_channels, cfg.g_blocks, dropout=0.0).to(device)
    disc = PatchDiscriminator(3, cfg.d_channels, cfg.dropout).to(device)
    opt_g = build_optimizer(gen, cfg.optimizer, cfg.gan_lr, cfg.weight_decay, betas=(0.5, 0.999))
    opt_d = build_optimizer(disc, cfg.optimizer, cfg.gan_lr, cfg.weight_decay, betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=cfg.epochs)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=cfg.epochs)

    bce = nn.BCEWithLogitsLoss()
    pixel_loss = get_pixel_loss(cfg.pixel_loss)
    perceptual = PerceptualLoss(use_vgg=cfg.use_vgg).to(device)
    print(f"    perceptual backend: {perceptual.backend}")

    def content_loss(fake, real):
        pix, perc = pixel_loss(fake, real), perceptual(fake, real)
        return cfg.w_pixel * pix + cfg.w_perceptual * perc, pix, perc

    warmup = min(cfg.gan_warmup_epochs, max(cfg.epochs - 1, 0))

    @torch.no_grad()
    def validate() -> Dict[str, float]:
        gen.eval()
        tracker, noise_gen = MetricTracker(), _val_generator(cfg)
        for batch in val_loader:
            clean, hr = batch["lr_gray"], batch["hr_rgb"].to(device)
            noisy = add_noise(clean, cfg.noise_type, generator=noise_gen).to(device)
            restored = pipeline.stage12(noisy)
            fake = gen(restored)
            total, _, _ = content_loss(fake, hr)
            tracker.update(
                {"loss": total.item(), "psnr": psnr(fake, hr).item(), "ssim": ssim(fake, hr).item(),
                 "baseline_psnr": psnr(restored, hr).item(), "baseline_ssim": ssim(restored, hr).item()},
                n=hr.size(0),
            )
        return tracker.averages()

    probe = next(iter(val_loader))
    probe_noisy = add_noise(probe["lr_gray"], cfg.noise_type, generator=_val_generator(cfg)).to(device)
    probe_in = pipeline.stage12(probe_noisy)
    probe_hr = probe["hr_rgb"].to(device)
    probe_loss = lambda out, tgt: pixel_loss(out, tgt)  # noqa: E731

    history, best_loss, best_state = History(), float("inf"), None
    ckpt = ckpt_path(cfg, "generator")
    for epoch in range(1, cfg.epochs + 1):
        start, tracker = time.time(), MetricTracker()
        adversarial = epoch > warmup
        gen.train()
        disc.train()
        for batch in train_loader:
            hr = batch["hr_rgb"].to(device, non_blocking=True)
            noisy = add_noise(batch["lr_gray"], cfg.noise_type).to(device, non_blocking=True)
            restored = pipeline.stage12(noisy)           # frozen upstream stages
            fake = gen(restored)

            # ---- discriminator step -------------------------------------- #
            d_loss_value = None
            if adversarial:
                set_requires_grad(disc, True)
                opt_d.zero_grad(set_to_none=True)
                real_logits, fake_logits = disc(hr), disc(fake.detach())
                d_loss = 0.5 * (
                    bce(real_logits, torch.full_like(real_logits, 0.9))
                    + bce(fake_logits, torch.zeros_like(fake_logits))
                )
                d_loss.backward()
                safe_step(disc, opt_d, cfg.grad_clip)
                d_loss_value = d_loss.item()

            # ---- generator step ------------------------------------------ #
            opt_g.zero_grad(set_to_none=True)
            if adversarial:
                set_requires_grad(disc, False)           # no wasted grads w.r.t. D
                g_logits = disc(fake)
                adv = bce(g_logits, torch.ones_like(g_logits))
            else:
                adv = torch.zeros((), device=device)
            content, pix, perc = content_loss(fake, hr)
            g_total = content + cfg.w_adv * adv
            _with_l2(g_total, gen, cfg.explicit_l2).backward()
            grad_norm, _ = safe_step(gen, opt_g, cfg.grad_clip)
            set_requires_grad(disc, True)

            with torch.no_grad():
                metrics = {
                    "loss": content.item(), "g_total": g_total.item(), "pixel": pix.item(),
                    "perceptual": perc.item(), "adv": adv.item(),
                    "psnr": psnr(fake, hr).item(), "ssim": ssim(fake, hr).item(),
                    "grad_norm": grad_norm,
                }
                if d_loss_value is not None:
                    metrics["d_loss"] = d_loss_value
                tracker.update(metrics, n=hr.size(0))
        sched_g.step()
        if adversarial:                                   # D's optimizer only steps after warm-up
            sched_d.step()

        train_m = tracker.averages()
        train_m.setdefault("d_loss", float("nan"))        # keeps history lists aligned in warm-up
        val_m = validate()
        val_m["saliency"] = input_saliency(gen, probe_in, probe_hr, probe_loss)

        history.log(
            lr=opt_g.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        )
        if val_m["loss"] < best_loss:
            best_loss, best_state = val_m["loss"], _snapshot(gen)
            save_checkpoint(ckpt, gen, history, {"epoch": epoch, "val_loss": best_loss}, best_state)
        phase = "adv" if adversarial else "warm"
        print(f"[GAN  {epoch:02d}/{cfg.epochs}|{phase}] train {_fmt(train_m)} d {train_m['d_loss']:.3f} "
              f"adv {train_m['adv']:.3f} | val {_fmt(val_m)} (input {val_m['baseline_psnr']:.2f}dB) "
              f"| grad {train_m['grad_norm']:.3f} | {time.time() - start:.1f}s {memory_report(device)}")
        clear_memory(device)

    return _finalize("gan", cfg, gen, history, best_state, device, ckpt)


# --------------------------------------------------------------------------- #
# Load-or-train orchestration (used by main.py)
# --------------------------------------------------------------------------- #
def ensure_models(cfg: TrainConfig, retrain: bool = False):
    """
    Load each module's checkpoint if present, otherwise train it. The GAN is
    retrained whenever the AE or U-Net it depends on was (re)trained.
    Returns ``(autoencoder, unet, generator, device, data_source)``.
    """
    device = get_device(cfg.device)
    ae_path, unet_path, gen_path = (ckpt_path(cfg, n) for n in ("autoencoder", "unet", "generator"))
    train_ae = retrain or not ae_path.exists()
    train_unet_flag = retrain or not unet_path.exists()
    train_gan_flag = retrain or train_ae or train_unet_flag or not gen_path.exists()

    loaders = _get_loaders(cfg) if (train_ae or train_unet_flag or train_gan_flag) else None
    source = loaders[2] if loaders else "n/a (all checkpoints loaded)"

    if train_ae:
        autoencoder, _ = train_autoencoder(cfg, loaders)
    else:
        print(f"Loading autoencoder  <- {ae_path}")
        autoencoder = load_model(DenoisingAutoencoder, ae_path, device)
    clear_memory(device)

    if train_unet_flag:
        unet, _ = train_unet(cfg, loaders)
    else:
        print(f"Loading U-Net        <- {unet_path}")
        unet = load_model(UNet, unet_path, device)
    clear_memory(device)

    if train_gan_flag:
        generator, _ = train_gan(cfg, autoencoder, unet, loaders)
    else:
        print(f"Loading generator    <- {gen_path}")
        generator = load_model(Generator, gen_path, device)
    clear_memory(device)

    return autoencoder.to(device).eval(), unet.to(device).eval(), generator.to(device).eval(), device, source


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train modules of the Visual Restoration Suite")
    parser.add_argument("--module", choices=["autoencoder", "unet", "gan", "all"], default="all")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)

    loaders = _get_loaders(cfg)
    device = get_device(cfg.device)
    if args.module in ("autoencoder", "all"):
        autoencoder, _ = train_autoencoder(cfg, loaders)
    if args.module in ("unet", "all"):
        unet, _ = train_unet(cfg, loaders)
    if args.module in ("gan", "all"):
        if args.module == "gan":
            autoencoder = load_model(DenoisingAutoencoder, ckpt_path(cfg, "autoencoder"), device)
            unet = load_model(UNet, ckpt_path(cfg, "unet"), device)
        train_gan(cfg, autoencoder, unet, loaders)


if __name__ == "__main__":
    main()
