"""
infer_resunet.py
================
Inference script for Residual U-Net (ResUNet) on random images from any split.

Usage:
  python infer_resunet.py --split train
  python infer_resunet.py --split val
  python infer_resunet.py --split test
  python infer_resunet.py --split test --file 1008.png
"""

import os
import sys
import random
import math
import argparse
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.transforms import ValTestTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import TRAIN_DIR, VAL_DIR, TEST_DIR
from models.resunet import ResUNet
from models.losses import ssim


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = float(torch.mean((pred - target) ** 2))
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def run_inference(split: str = "test", filename: str = None):
    split = split.lower().strip()
    if split == "train":
        img_dir = TRAIN_DIR
    elif split == "val":
        img_dir = VAL_DIR
    elif split == "test":
        img_dir = TEST_DIR
    else:
        raise ValueError(f"Invalid split '{split}'. Must be one of ['train', 'val', 'test']")

    # Select image file
    all_files = sorted([p for p in img_dir.glob("*.png")])
    if not all_files:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    if filename:
        target_path = img_dir / filename
        if not target_path.exists():
            raise FileNotFoundError(f"Specified file '{filename}' not found in {img_dir}")
        selected_file = target_path
    else:
        selected_file = random.choice(all_files)

    print("=" * 65, flush=True)
    print("RESUNET RANDOM IMAGE INFERENCE", flush=True)
    print("=" * 65, flush=True)
    print(f"Dataset Split   : {split.upper()}")
    print(f"Selected Image  : {selected_file.name}")

    # Setup device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet(in_channels=3, out_channels=3).to(device)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "resunet_best.pth"
    if not ckpt_path.exists():
        # Fallback to latest or un-trained if running before 100-epoch completion
        ckpt_path_alt = PROJECT_ROOT / "checkpoints" / "resunet_latest.pth"
        if ckpt_path_alt.exists():
            ckpt_path = ckpt_path_alt
        else:
            print("  [Note] No trained checkpoint found in checkpoints/. Running in demonstration mode.")
            ckpt_path = None

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        trained_epoch = checkpoint.get("epoch", "N/A")
        print(f"Loaded Checkpoint: {ckpt_path.name} (Trained Epoch: {trained_epoch})")

    model.eval()

    # Preprocess clean image
    transform = ValTestTransform(crop_size=256)
    noise_fn = SpeckleNoise(num_looks=4)

    with Image.open(selected_file) as raw_img:
        if raw_img.mode != "RGB":
            raw_img = raw_img.convert("RGB")
        clean_tensor = transform(raw_img)

    # Generate synthetic speckle noise (L=4)
    noisy_tensor = noise_fn(clean_tensor)

    # Run ResUNet Inference
    with torch.no_grad():
        noisy_batch = noisy_tensor.unsqueeze(0).to(device)
        denoised_batch = model(noisy_batch)
        denoised_tensor = denoised_batch.squeeze(0).cpu()

    # Calculate evaluation metrics against clean target
    psnr_val = calculate_psnr(denoised_tensor, clean_tensor)
    ssim_val = float(ssim(denoised_tensor.unsqueeze(0), clean_tensor.unsqueeze(0)))

    print(f"\nInference Performance Metrics:")
    print(f"  PSNR : {psnr_val:.2f} dB")
    print(f"  SSIM : {ssim_val:.4f}")

    # Render 3-panel RGB visualization
    results_dir = PROJECT_ROOT / "results" / "residual_unet_inference"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"inference_{split}_{selected_file.stem}.png"

    clean_np    = clean_tensor.permute(1, 2, 0).numpy()
    noisy_np    = noisy_tensor.permute(1, 2, 0).numpy()
    denoised_np = denoised_tensor.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(clean_np)
    axes[0].set_title(f"Original Clean Target\nSplit: {split.upper()} | {selected_file.name}", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(noisy_np)
    axes[1].set_title(f"Speckle Noisy Input\n(Gamma Speckle L=4)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(denoised_np)
    axes[2].set_title(f"ResUNet Denoised Output\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}", fontsize=11)
    axes[2].axis("off")

    plt.suptitle("Residual U-Net Image Denoising Inference Result", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(str(out_file), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Inference 3-panel visualization saved to: {out_file.name}")
    print("=" * 65, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResUNet Random Image Inference")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--file", type=str, default=None, help="Optional specific filename (e.g. 1008.png)")
    args = parser.parse_args()

    run_inference(split=args.split, filename=args.file)
