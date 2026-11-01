"""
infer_nafnet_adaptive_channel.py
================================
Inference script for NAFNet with Adaptive Gate + Channel Attention (M4 — Phase 2)
on random images from any dataset split or a specified file.

Usage:
  python phase2/infer_nafnet_adaptive_channel.py --split test
  python phase2/infer_nafnet_adaptive_channel.py --split test --file 1008.png
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
from models.nafnet_adaptive_channel import NAFNetAdaptiveChannel, nafnet_adaptive_channel_model
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
    print("NAFNET ADAPTIVE + CHANNEL ATTN (M4) -- RANDOM INFERENCE", flush=True)
    print("=" * 65, flush=True)
    print(f"Dataset Split   : {split.upper()}", flush=True)
    print(f"Selected Image  : {selected.name}", flush=True)

    # 1. Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = nafnet_adaptive_channel_model().to(device)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "nafnet_adaptive_channel_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded Checkpoint : {ckpt_path.name} (Trained Epoch: {ckpt.get('epoch', 'N/A')})", flush=True)

    # 2. Image Loading and Speckle Synthesis
    raw_img = Image.open(selected).convert("RGB")
    transform = ValTestTransform()
    clean_tensor = transform(raw_img)

    speckle_gen  = SpeckleNoise(num_looks=4)
    noisy_tensor = speckle_gen(clean_tensor)

    # 3. Model Inference
    use_cuda = device.type == "cuda"
    inp = noisy_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_cuda):
            pred_tensor = model(inp)
        pred_tensor = pred_tensor.squeeze(0).clamp(0.0, 1.0)

    # 4. Metrics
    clean_dev = clean_tensor.to(device)
    psnr_val  = calculate_psnr(pred_tensor, clean_dev)
    ssim_val  = float(ssim(pred_tensor.unsqueeze(0), clean_dev.unsqueeze(0)).item())

    print("\nInference Performance Metrics:", flush=True)
    print(f"  PSNR : {psnr_val:.2f} dB", flush=True)
    print(f"  SSIM : {ssim_val:.4f}", flush=True)

    # 5. Save Visualization
    out_dir = PROJECT_ROOT / "results" / "nafnet_adaptive_channel_inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"inference_{split}_{selected.stem}.png"

    clean_np = clean_tensor.permute(1, 2, 0).cpu().numpy()
    noisy_np = noisy_tensor.permute(1, 2, 0).cpu().numpy()
    pred_np  = pred_tensor.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(clean_np); axes[0].set_title(f"Target: {selected.name}", fontsize=11); axes[0].axis("off")
    axes[1].imshow(noisy_np); axes[1].set_title("Speckle Input (L=4)", fontsize=11); axes[1].axis("off")
    axes[2].imshow(pred_np);  axes[2].set_title(f"NAFNet-AdaptiveChannel Output\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}", fontsize=11); axes[2].axis("off")

    plt.suptitle(f"NAFNet Adaptive + Channel Attn (M4) Inference on {split.upper()} Image ({selected.name})", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Visualization saved to: {out_path}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer with NAFNet Adaptive + Channel Attention")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split (train, val, test)")
    parser.add_argument("--file", type=str, default=None,
                        help="Optional specific image filename (e.g. 1008.png)")
    args = parser.parse_args()

    run_inference(split=args.split, filename=args.file)
