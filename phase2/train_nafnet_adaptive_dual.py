"""
train_nafnet_adaptive_dual.py  (Phase 2 -- M5)
==============================================
Full-dataset training script for NAFNet with Adaptive Gate + Dual Attention (SAM + CAM)
on the S2Looking satellite optical image despeckling dataset.

Ablation Study for IEEE GRSL Paper:
  - Combines all three novel contributions: AGM + SAM + CAM (Dual Attention).
  - 3,500 train / 500 val / 1,000 test (S2Looking RGB).
  - Loss: Combined L1 + SSIM (0.84*L1 + 0.16*(1-SSIM)).
  - AdamW (lr=5e-4, wd=1e-4) + CosineAnnealingLR.
  - Strict D: drive cache isolation (zero C: drive usage).
  - Clean per-epoch progress logging.

Usage:
  python phase2/train_nafnet_adaptive_dual.py --epochs 100 --batch_size 4 --grad_accum 2 --val_every 10 --patience 100
"""

import os
import sys
import math
import time
import json
import csv
from pathlib import Path
from typing import Tuple, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Strict Cache & Temp Redirection to D: Drive (No C: Drive usage)
CACHE_DIR = PROJECT_ROOT / ".cache"
TEMP_DIR  = PROJECT_ROOT / ".temp"

os.environ["TORCH_HOME"]          = str(CACHE_DIR / "torch")
os.environ["HF_HOME"]             = str(CACHE_DIR / "huggingface")
os.environ["TRANSFORMERS_CACHE"]  = str(CACHE_DIR / "huggingface")
os.environ["XDG_CACHE_HOME"]      = str(CACHE_DIR)
os.environ["TEMP"]                = str(TEMP_DIR)
os.environ["TMP"]                 = str(TEMP_DIR)
os.environ["TMPDIR"]              = str(TEMP_DIR)
os.environ["MPLCONFIGDIR"]        = str(CACHE_DIR / "matplotlib")
os.environ["CUDA_CACHE_PATH"]     = str(CACHE_DIR / "nv")
os.environ["PYTHONPYCACHEPREFIX"] = str(CACHE_DIR / "pycache")

for sub in ["torch", "huggingface", "matplotlib", "nv", "pycache"]:
    (CACHE_DIR / sub).mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.transforms import TrainTransform, ValTestTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import SatelliteSpeckleDataset, TRAIN_DIR, VAL_DIR, TEST_DIR
from models.nafnet_adaptive_dual import NAFNetAdaptiveDual, nafnet_adaptive_dual_model
from models.losses import CombinedL1SSIMLoss, ssim


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = float(torch.mean((pred.float() - target.float()) ** 2))
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def calculate_pixel_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.05) -> Tuple[float, float, float]:
    abs_diff = torch.abs(pred.float() - target.float())
    correct  = (abs_diff <= threshold).float()
    acc = float(correct.mean())
    return acc, acc, acc


# ---------------------------------------------------------------------------
# Training & Validation Loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    grad_accum_steps: int = 1) -> float:
    model.train()
    running_loss, total_samples = 0.0, 0
    use_cuda = device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)

    for step_idx, (noisy, clean) in enumerate(loader):
        noisy = noisy.to(device, non_blocking=use_cuda)
        clean = clean.to(device, non_blocking=use_cuda)

        with torch.amp.autocast('cuda', enabled=use_cuda):
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == num_batches:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_sz = noisy.size(0)
        running_loss += float(loss.item()) * batch_sz
        total_samples += batch_sz

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_val(model, loader, criterion, device) -> Tuple[float, float, float]:
    model.eval()
    running_loss, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0
    use_cuda = device.type == "cuda"

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=use_cuda)
        clean = clean.to(device, non_blocking=use_cuda)

        with torch.amp.autocast('cuda', enabled=use_cuda):
            pred = model(noisy)
            loss = criterion(pred, clean)

        batch_sz = noisy.size(0)
        running_loss += float(loss.item()) * batch_sz
        for i in range(batch_sz):
            p = pred[i:i+1].clamp(0.0, 1.0)
            c = clean[i:i+1]
            psnr_sum += calculate_psnr(p, c)
            ssim_sum += float(ssim(p, c).item())
        count += batch_sz

    return running_loss / count, psnr_sum / count, ssim_sum / count


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train NAFNet Adaptive Dual Attention (M5)")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--grad_accum",   type=int,   default=2)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_every",    type=int,   default=10)
    parser.add_argument("--patience",     type=int,   default=100)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--resume",       type=str,   default=None)
    args = parser.parse_args()

    # D: drive output directories
    results_dir     = PROJECT_ROOT / "results" / "nafnet_adaptive_dual"
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ckpt_best   = checkpoints_dir / "nafnet_adaptive_dual_best.pth"
    ckpt_latest = checkpoints_dir / "nafnet_adaptive_dual_latest.pth"
    csv_path    = results_dir / "training_log.csv"

    # Hardware detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("NAFNET + AGM + DUAL ATTENTION (M5 -- PROPOSED MODEL) TRAINING")
    print("=" * 70)
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Batch Size (eff.)    : {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Target Epochs        : {args.epochs}")
    print(f"Validation Frequency : Every {args.val_every} epochs")
    print(f"Cache Directory      : {CACHE_DIR}")
    print(f"Output Checkpoints   : {checkpoints_dir}")
    print(f"Output Results       : {results_dir}")
    print("=" * 70)

    # Datasets
    noise_fn      = SpeckleNoise(num_looks=4)
    train_dataset = SatelliteSpeckleDataset(TRAIN_DIR, transform=TrainTransform(crop_size=256), noise_fn=noise_fn)
    val_dataset   = SatelliteSpeckleDataset(VAL_DIR,   transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)

    print(f"Train Dataset Size   : {len(train_dataset)} images")
    print(f"Val Dataset Size     : {len(val_dataset)} images")

    # DataLoader (Windows multiprocessing safe)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # Model instantiation
    model = nafnet_adaptive_dual_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters     : {total_params:,} (Balanced Dual-Attention Model)")

    criterion = CombinedL1SSIMLoss(alpha=0.84).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Resume handling
    start_epoch = 1
    best_psnr   = -float("inf")
    best_epoch  = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt.get("scaler_state_dict", scaler.state_dict()))
        start_epoch = ckpt["epoch"] + 1
        best_psnr   = ckpt.get("best_psnr", -float("inf"))
        best_epoch  = ckpt.get("best_epoch", 0)
        print(f"Resumed from epoch {start_epoch-1}. Best Val PSNR: {best_psnr:.2f} dB (epoch {best_epoch})")

    # Initialize CSV log
    if start_epoch == 1 or not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_PSNR_dB", "Val_SSIM", "LR", "Time_sec"])

    print("\nStarting Training...")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_accum_steps=args.grad_accum
        )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_time = time.time() - t0

        val_loss, val_psnr, val_ssim = None, None, None

        # Validation check
        if epoch % args.val_every == 0 or epoch == args.epochs:
            t_val = time.time()
            val_loss, val_psnr, val_ssim = evaluate_val(model, val_loader, criterion, device)
            val_time = time.time() - t_val

            print(f"[Epoch {epoch:03d}/{args.epochs:03d}] "
                  f"Train Loss: {train_loss:.5f} | "
                  f"Val Loss: {val_loss:.5f} | "
                  f"Val PSNR: {val_psnr:.2f} dB | "
                  f"Val SSIM: {val_ssim:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s (+{val_time:.1f}s val)")

            # Check for best model
            if val_psnr > best_psnr:
                best_psnr  = val_psnr
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_psnr": best_psnr,
                    "best_epoch": best_epoch,
                    "total_params": total_params,
                }, str(ckpt_best))
                print(f"  >>> Best checkpoint updated! (Val PSNR: {best_psnr:.2f} dB at epoch {best_epoch})")
        else:
            print(f"[Epoch {epoch:03d}/{args.epochs:03d}] "
                  f"Train Loss: {train_loss:.5f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_psnr": best_psnr,
            "best_epoch": best_epoch,
            "total_params": total_params,
        }, str(ckpt_latest))

        # Append to CSV log
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}" if val_loss is not None else "",
                f"{val_psnr:.4f}" if val_psnr is not None else "",
                f"{val_ssim:.4f}" if val_ssim is not None else "",
                f"{current_lr:.6e}",
                f"{epoch_time:.1f}"
            ])

    print("-" * 70)
    print("TRAINING COMPLETE!")
    print(f"Best Val PSNR: {best_psnr:.2f} dB (achieved at Epoch {best_epoch})")
    print(f"Best checkpoint: {ckpt_best}")
    print(f"Latest checkpoint: {ckpt_latest}")
    print(f"Training log: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
