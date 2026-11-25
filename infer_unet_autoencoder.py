"""
infer_unet_autoencoder.py
=========================
Inference script for U-Net Denoising Autoencoder (UNet-Autoencoder) on random images.

Usage:
  python infer_unet_autoencoder.py --split test
  python infer_unet_autoencoder.py --split val
  python infer_unet_autoencoder.py --split train
  python infer_unet_autoencoder.py --split test --file 1008.png
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
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Strict D: drive redirect
os.environ["TORCH_HOME"]         = str(PROJECT_ROOT / ".cache" / "torch")
os.environ["TEMP"]               = str(PROJECT_ROOT / ".temp")
os.environ["TMP"]                = str(PROJECT_ROOT / ".temp")
os.environ["MPLCONFIGDIR"]       = str(PROJECT_ROOT / ".cache" / "matplotlib")
os.environ["CUDA_CACHE_PATH"]    = str(PROJECT_ROOT / ".cache" / "nv")
os.environ["PYTHONPYCACHEPREFIX"] = str(PROJECT_ROOT / ".cache" / "pycache")

from preprocessing.transforms import ValTestTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import TRAIN_DIR, VAL_DIR, TEST_DIR
from models.unet_autoencoder import UNetAutoencoder
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

    all_files = sorted([p for p in img_dir.glob("*.png")])
    if not all_files:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    if filename:
        target_path = img_dir / filename
        if not target_path.exists():
            raise FileNotFoundError(f"Image {filename} not found in {img_dir}")
    else:
        target_path = random.choice(all_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading UNet-Autoencoder on {device}...")
    model = UNetAutoencoder(in_channels=3, out_channels=3, base_channels=64).to(device)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "unet_autoencoder_best.pth"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "unet_autoencoder_latest.pth"

    if ckpt_path.exists():
        st = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(st["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path.name}")
    else:
        print(f"[!] Warning: No checkpoint found at {ckpt_path}. Running with initialized weights.")

    model.eval()

    # Load and process image
    raw_img = Image.open(target_path).convert("RGB")
    transform = ValTestTransform(crop_size=256)
    clean_tensor = transform(raw_img)

    noise_fn = SpeckleNoise(num_looks=4)
    noisy_tensor = noise_fn(clean_tensor)

    inp = noisy_tensor.unsqueeze(0).to(device)
    tgt = clean_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)

    p_val = calculate_psnr(out[0], tgt[0])
    s_val = float(ssim(out, tgt))

    print(f"Inference complete for {target_path.name}:")
    print(f"  PSNR: {p_val:.2f} dB")
    print(f"  SSIM: {s_val:.4f}")

    # Visualization
    out_dir = PROJECT_ROOT / "results" / "unet_autoencoder_inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"inference_{split}_{target_path.stem}.png"

    c_np = clean_tensor.permute(1, 2, 0).numpy()
    n_np = noisy_tensor.permute(1, 2, 0).numpy()
    d_np = out[0].cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(c_np, 0, 1))
    axes[0].set_title(f"Target Clean ({target_path.name})", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(np.clip(n_np, 0, 1))
    axes[1].set_title("Speckle Noisy (L=4)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(np.clip(d_np, 0, 1))
    axes[2].set_title(f"UNet-Autoencoder Denoised\nPSNR: {p_val:.2f} dB | SSIM: {s_val:.4f}", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    run_inference(split=args.split, filename=args.file)
