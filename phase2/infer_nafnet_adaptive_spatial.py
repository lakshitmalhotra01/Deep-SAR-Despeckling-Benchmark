"""
infer_nafnet_adaptive_spatial.py
================================
Inference script for NAFNet with Adaptive Gate + Spatial Attention (M3 — Phase 2)
on random images from any dataset split or a specified file.

Usage:
  python phase2/infer_nafnet_adaptive_spatial.py --split test
  python phase2/infer_nafnet_adaptive_spatial.py --split test --file 1008.png
"""

import sys
import random
import math
import argparse
from pathlib import Path
from PIL import Image

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.transforms import ValTestTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import TRAIN_DIR, VAL_DIR, TEST_DIR
from models.nafnet_adaptive_spatial import NAFNetAdaptiveSpatial
from models.losses import ssim


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = float(torch.mean((pred.float() - target.float()) ** 2))
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def run_inference(split: str = "test", filename: str = None):
    split = split.lower().strip()
    split_map = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}
    if split not in split_map:
        raise ValueError(f"Invalid split '{split}'. Choose from: train, val, test")

    img_dir   = split_map[split]
    all_files = sorted(img_dir.glob("*.png"))
    if not all_files:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    if filename:
        selected = img_dir / filename
        if not selected.exists():
            raise FileNotFoundError(f"File '{filename}' not found in {img_dir}")
    else:
        selected = random.choice(all_files)

    print("=" * 65, flush=True)
    print("NAFNET ADAPTIVE + SPATIAL ATTN (M3) -- RANDOM INFERENCE", flush=True)
    print("=" * 65, flush=True)
    print(f"Dataset Split   : {split.upper()}")
    print(f"Selected Image  : {selected.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = PROJECT_ROOT / "checkpoints" / "nafnet_adaptive_spatial_best.pth"
    if not ckpt_path.exists():
        alt = PROJECT_ROOT / "checkpoints" / "nafnet_adaptive_spatial_latest.pth"
        ckpt_path = alt if alt.exists() else None

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        config = ckpt.get("config", {})
        base_ch = config.get("base_ch", 32)
        num_blocks = config.get("num_blocks", [1, 1, 1, 4])
        model = NAFNetAdaptiveSpatial(in_channels=3, out_channels=3, base_ch=base_ch, num_blocks=num_blocks).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Checkpoint : {ckpt_path.name} (Trained Epoch: {ckpt.get('epoch', '?')})")
    else:
        model = NAFNetAdaptiveSpatial(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1, 1, 1, 4]).to(device)
        print("[Warning] No checkpoint found in checkpoints/ -- running in demo mode (untrained)")

    model.eval()

    transform = ValTestTransform(crop_size=256)
    noise_fn  = SpeckleNoise(num_looks=4)

    with Image.open(selected) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        clean_tensor = transform(img)

    noisy_tensor = noise_fn(clean_tensor)

    with torch.no_grad():
        denoised_tensor = model(noisy_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()

    psnr_val = calculate_psnr(denoised_tensor, clean_tensor)
    ssim_val = float(ssim(denoised_tensor.unsqueeze(0), clean_tensor.unsqueeze(0)))

    print(f"\nInference Performance Metrics:")
    print(f"  PSNR : {psnr_val:.2f} dB")
    print(f"  SSIM : {ssim_val:.4f}")

    results_dir = PROJECT_ROOT / "results" / "nafnet_adaptive_spatial_inference"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"inference_{split}_{selected.stem}.png"

    c_np = clean_tensor.permute(1, 2, 0).numpy()
    n_np = noisy_tensor.permute(1, 2, 0).numpy()
    d_np = denoised_tensor.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(c_np)
    axes[0].set_title(f"Original Clean Target\nSplit: {split.upper()} | {selected.name}", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(n_np)
    axes[1].set_title("Speckle Noisy Input\n(Gamma Speckle L=4)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(d_np)
    axes[2].set_title(f"NAFNet-AdaptiveSpatial Output\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}", fontsize=11)
    axes[2].axis("off")

    plt.suptitle("NAFNet with Adaptive Gate + Spatial Attn (M3) Despeckling", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(str(out_file), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Visualization saved to: {out_file}")
    print("=" * 65, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAFNet Adaptive Spatial Inference")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--file",  type=str, default=None,
                        help="Optional specific filename (e.g. 1008.png)")
    args = parser.parse_args()
    run_inference(split=args.split, filename=args.file)
