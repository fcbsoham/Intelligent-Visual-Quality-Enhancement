"""
main.py
=======
End-to-end demonstration of the Intelligent Visual Quality Enhancement &
Restoration Suite.

    python main.py                        # train missing models (20 epochs each), then demo
    python main.py --quick                # smaller data subsets -> much faster on CPU
    python main.py --scale 4              # 4x super-resolution
    python main.py --image my_photo.jpg   # run the pipeline on your own picture
    python main.py --retrain              # ignore existing checkpoints

Pipeline:   original RGB -> grayscale -> low-res -> + noise
            -> [1] Denoising AE -> [2] U-Net colorize + super-resolve -> [3] GAN refine

Outputs (``--out-dir``, default ``outputs/``):
    comparison_original_noisy_restored_x{s}.png   original | noisy low-res | restored
    pipeline_stages_x{s}.png                      every intermediate stage + metrics
    metrics_x{s}.json                             per-stage PSNR / SSIM
    *_history_x{s}.png / .json                    training curves (when training ran)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import add_noise, downsample, get_test_dataset, load_image_file, rgb_to_gray
from train import add_config_args, config_from_args, ensure_models
from utils import (
    RestorationPipeline,
    clear_memory,
    psnr,
    save_comparison_grid,
    set_seed,
    ssim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intelligent Visual Quality Enhancement & Restoration Suite")
    add_config_args(parser)
    parser.add_argument("--quick", action="store_true",
                        help="use small data subsets (2000 train / 500 val) for fast CPU runs")
    parser.add_argument("--retrain", action="store_true", help="retrain even if checkpoints exist")
    parser.add_argument("--image", type=str, default=None, help="optional path to your own image")
    parser.add_argument("--size", type=int, default=32, help="side length used for --image (HR size)")
    parser.add_argument("--num-samples", type=int, default=8, help="test images shown in the figures")
    return parser.parse_args()


def _caption(p: torch.Tensor, s: torch.Tensor):
    return [f"PSNR {a:.1f} | SSIM {b:.2f}" for a, b in zip(p.tolist(), s.tolist())]


def build_inputs(args, cfg):
    """Return ``(hr_rgb, lr_clean_gray, noisy_lr_gray, data_source)`` as CPU tensors."""
    if args.image:
        hr = load_image_file(args.image, args.size, cfg.scale)
        source = f"user image: {args.image}"
    else:
        dataset, source = get_test_dataset(cfg.data_root, cfg.scale, cfg.synthetic, cfg.seed)
        n = min(args.num_samples, len(dataset))
        hr = torch.stack([dataset[i]["hr_rgb"] for i in range(n)])
    lr_clean = downsample(rgb_to_gray(hr), cfg.scale)
    generator = torch.Generator().manual_seed(cfg.seed)          # reproducible corruption
    noisy = add_noise(lr_clean, cfg.noise_type, generator=generator)
    return hr, lr_clean, noisy, source


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    if args.quick:
        cfg = dataclasses.replace(cfg, train_subset=2000, val_subset=500)
    set_seed(cfg.seed)

    # 1) get trained models (load checkpoints or train what is missing) ------- #
    autoencoder, unet, generator, device, train_source = ensure_models(cfg, retrain=args.retrain)
    pipeline = RestorationPipeline(autoencoder, unet, generator).to(device).eval()

    # 2) build the degraded demo inputs --------------------------------------- #
    hr, lr_clean, noisy, source = build_inputs(args, cfg)
    print(f"\nDemo data: {source} | {hr.size(0)} image(s) | HR {tuple(hr.shape[-2:])} -> "
          f"LR {tuple(noisy.shape[-2:])} (x{cfg.scale}) | noise: {cfg.noise_type}")

    # 3) run the full restoration pipeline ------------------------------------ #
    outputs = {k: v.cpu() for k, v in pipeline.restore(noisy.to(device)).items()}
    denoised, colorized, refined = outputs["denoised"], outputs["colorized_sr"], outputs["refined"]
    bicubic = F.interpolate(noisy, scale_factor=cfg.scale, mode="bicubic", align_corners=False)
    bicubic = bicubic.clamp(0, 1).repeat(1, 3, 1, 1) if cfg.scale > 1 else noisy.repeat(1, 3, 1, 1)

    # 4) metrics --------------------------------------------------------------- #
    def stats(pred, ref):
        return psnr(pred, ref, reduce=False), ssim(pred, ref, reduce=False)

    m = {
        "noisy_lr (vs clean LR)": stats(noisy, lr_clean),
        "denoised (vs clean LR)": stats(denoised, lr_clean),
        "bicubic baseline (vs original)": stats(bicubic, hr),
        "U-Net colorized+SR (vs original)": stats(colorized, hr),
        "GAN refined (vs original)": stats(refined, hr),
    }
    print(f"\n{'stage':<36}{'PSNR (dB)':>10}{'SSIM':>8}")
    summary = {}
    for name, (p, s) in m.items():
        summary[name] = {"psnr": p.mean().item(), "ssim": s.mean().item()}
        print(f"{name:<36}{p.mean().item():>10.2f}{s.mean().item():>8.3f}")

    # 5) save figures ---------------------------------------------------------- #
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"x{cfg.scale}"

    save_comparison_grid(
        [("Original", hr), (f"Noisy low-res (1/{cfg.scale})", noisy), ("Restored (full pipeline)", refined)],
        out_dir / f"comparison_original_noisy_restored_{tag}.png",
        captions=[None, _caption(*m["noisy_lr (vs clean LR)"]), _caption(*m["GAN refined (vs original)"])],
        max_rows=args.num_samples,
    )
    save_comparison_grid(
        [
            ("Original", hr),
            ("Noisy low-res", noisy),
            ("1) Denoised (AE)", denoised),
            ("Bicubic baseline", bicubic),
            ("2) Colorized + SR (U-Net)", colorized),
            ("3) GAN refined", refined),
        ],
        out_dir / f"pipeline_stages_{tag}.png",
        captions=[
            None,
            _caption(*m["noisy_lr (vs clean LR)"]),
            _caption(*m["denoised (vs clean LR)"]),
            _caption(*m["bicubic baseline (vs original)"]),
            _caption(*m["U-Net colorized+SR (vs original)"]),
            _caption(*m["GAN refined (vs original)"]),
        ],
        max_rows=args.num_samples,
    )
    with open(out_dir / f"metrics_{tag}.json", "w") as handle:
        json.dump({"data_source": source, "training_data": train_source,
                   "scale": cfg.scale, "noise": cfg.noise_type, "stages": summary}, handle, indent=2)

    print(f"\nSaved figures + metrics to: {out_dir.resolve()}")
    for name in (f"comparison_original_noisy_restored_{tag}.png", f"pipeline_stages_{tag}.png", f"metrics_{tag}.json"):
        print(f"  - {out_dir / name}")
    clear_memory(device)


if __name__ == "__main__":
    main()
