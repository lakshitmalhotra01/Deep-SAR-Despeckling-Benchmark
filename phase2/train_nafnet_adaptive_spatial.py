"""
train_nafnet_adaptive_spatial.py  (Phase 2 — M3)
================================================
Full-dataset training script for NAFNet with Adaptive Gate + Spatial Attention (SAM)
on the S2Looking satellite optical image despeckling dataset.

Ablation Study for IEEE GRSL Paper:
  - Evaluates explicit Spatial Attention combined with Adaptive Gating.
  - 3,500 train / 500 val / 1,000 test (S2Looking RGB).
  - Loss: Combined L1 + SSIM (0.84*L1 + 0.16*(1-SSIM)).
  - AdamW (lr=5e-4, wd=1e-4) + CosineAnnealingLR.
  - Strict D: drive cache isolation (zero C: drive usage).
  - Clean per-epoch progress logging.

Usage:
  python phase2/train_nafnet_adaptive_spatial.py --epochs 100 --batch_size 4 --grad_accum 2 --val_every 10 --patience 100
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

# ─────────────────────────────────────────────────────────────────────────────
# Strict Cache & Temp Redirection to D: Drive (No C: Drive usage)
# ─────────────────────────────────────────────────────────────────────────────
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
from models.nafnet_adaptive_spatial import NAFNetAdaptiveSpatial
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

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            outputs = model(noisy)
            loss    = criterion(outputs, clean)
            loss_scaled = loss / grad_accum_steps

        if use_cuda:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == num_batches:
            if use_cuda:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_sz = noisy.size(0)
        running_loss  += loss.item() * batch_sz
        total_samples += batch_sz

    return running_loss / total_samples


def validate(model, loader, criterion, device) -> Tuple[float, float, float]:
    model.eval()
    running_loss = total_psnr = total_ssim = total = 0.0
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device, non_blocking=use_cuda)
            clean = clean.to(device, non_blocking=use_cuda)
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(noisy)
                loss    = criterion(outputs, clean)
            bs = noisy.size(0)
            running_loss += loss.item() * bs
            total_psnr   += calculate_psnr(outputs, clean) * bs
            total_ssim   += float(ssim(outputs, clean)) * bs
            total        += bs

    return running_loss / total, total_psnr / total, total_ssim / total


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def run_full_training(epochs: int = 100, batch_size: int = 4, lr: float = 5e-4,
                      patience: int = 100, val_every: int = 10, grad_accum: int = 2,
                      base_ch: int = 32, num_blocks: list = None,
                      num_workers: int = 2):

    if num_blocks is None:
        num_blocks = [1, 1, 1, 4]

    MODEL_TAG = "NAFNet-AdaptiveSpatial"

    print("=" * 70, flush=True)
    print(f"{MODEL_TAG} (Adaptive Gate + Spatial Attn) TRAINING -- Phase 2 M3 ({epochs} EPOCHS)", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_prop = torch.cuda.get_device_properties(0)
        gpu_mem  = (gpu_prop.total_mem if hasattr(gpu_prop, 'total_mem') else gpu_prop.total_memory) / (1024 ** 3)
        print(f"Target Device            : CUDA ({gpu_name}, {gpu_mem:.1f} GB VRAM)", flush=True)
    else:
        print("Target Device            : CPU", flush=True)

    effective_batch = batch_size * grad_accum
    print(f"Training Set Size        : 3,500 images")
    print(f"Validation Set Size      : 500 images")
    print(f"Test Set Size            : 1,000 images")
    print(f"Epochs                   : {epochs}")
    print(f"Micro-Batch Size         : {batch_size}")
    print(f"Gradient Accumulation    : {grad_accum} steps (Effective Batch Size: {effective_batch})")
    print(f"Base Channels            : {base_ch}")
    print(f"Block Architecture       : {num_blocks}")
    print(f"Attention Architecture   : Adaptive Gating (AGM) + Spatial Attention (SAM 7x7)")
    print(f"Initial Learning Rate    : {lr}")
    print(f"Validation Frequency     : every {val_every} epochs")
    print(f"Early Stopping Patience  : {patience} val checks (monitored: Val PSNR)")
    print(f"Optimizer                : AdamW (weight_decay=1e-4)")
    print(f"Scheduler                : CosineAnnealingLR (T_max={epochs})")
    print(f"Loss Function            : Combined L1 + SSIM Loss (0.84 L1 + 0.16 SSIM-loss)")
    print(f"Noise Model              : Gamma Multiplicative Speckle (L=4)")

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    results_dir    = PROJECT_ROOT / "results" / "nafnet_adaptive_spatial"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    use_pin      = device.type == "cuda"
    noise_fn     = SpeckleNoise(num_looks=4)
    train_ds     = SatelliteSpeckleDataset(TRAIN_DIR, transform=TrainTransform(crop_size=256), noise_fn=noise_fn)
    val_ds       = SatelliteSpeckleDataset(VAL_DIR,   transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=use_pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)

    # Model
    model = NAFNetAdaptiveSpatial(in_channels=3, out_channels=3, base_ch=base_ch, num_blocks=num_blocks).to(device)
    criterion = CombinedL1SSIMLoss(alpha=0.84)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters         : {total_params:,} total  |  {trainable_params:,} trainable")

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_psnr": [], "val_ssim": [], "lr": [], "epoch_time": []}

    best_val_psnr      = float('-inf')
    best_val_ssim      = 0.0
    best_epoch         = 0
    early_stop_counter = 0

    csv_file = results_dir / "training_log.csv"
    with open(csv_file, "w", newline="") as f:
        csv.writer(f).writerow(["Epoch", "Train_Loss", "Val_Loss",
                                 "Val_PSNR_dB", "Val_SSIM", "LR", "Time_sec"])

    print("\nStarting Training Loop...\n", flush=True)
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum_steps=grad_accum
        )

        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        do_val = (epoch % val_every == 0) or (epoch == epochs)

        if do_val:
            val_loss, val_psnr, val_ssim_val = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_psnr, val_ssim_val = None, None, None

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss if do_val else float('nan'))
        history["val_psnr"].append(val_psnr if do_val else float('nan'))
        history["val_ssim"].append(val_ssim_val if do_val else float('nan'))
        history["lr"].append(curr_lr)
        history["epoch_time"].append(elapsed)

        with open(csv_file, "a", newline="") as f:
            if do_val:
                csv.writer(f).writerow([
                    epoch, f"{train_loss:.5f}", f"{val_loss:.5f}",
                    f"{val_psnr:.2f}", f"{val_ssim_val:.4f}",
                    f"{curr_lr:.6f}", f"{elapsed:.1f}"
                ])
            else:
                csv.writer(f).writerow([
                    epoch, f"{train_loss:.5f}", "", "", "", f"{curr_lr:.6f}", f"{elapsed:.1f}"
                ])

        if do_val:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Val PSNR: {val_psnr:.2f} dB | "
                f"Val SSIM: {val_ssim_val:.4f} | "
                f"LR: {curr_lr:.6f}",
                flush=True
            )

            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_val_ssim = val_ssim_val
                best_epoch    = epoch
                early_stop_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim_val,
                    "config": {
                        "architecture": "NAFNet-AdaptiveSpatial (AGM+SAM)",
                        "base_ch": base_ch,
                        "num_blocks": num_blocks,
                        "dw_expand": 2,
                        "ffn_expand": 2,
                    },
                }, str(checkpoint_dir / "nafnet_adaptive_spatial_best.pth"))
            else:
                early_stop_counter += 1
        else:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.5f} | "
                f"LR: {curr_lr:.6f}",
                flush=True
            )

        # Always save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_psnr": best_val_psnr,
            "val_ssim": best_val_ssim,
            "config": {
                "architecture": "NAFNet-AdaptiveSpatial (AGM+SAM)",
                "base_ch": base_ch,
                "num_blocks": num_blocks,
            },
        }, str(checkpoint_dir / "nafnet_adaptive_spatial_latest.pth"))

        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} val checks without improvement.", flush=True)
            break

    total_time = time.time() - total_start
    print(f"\nTraining complete in {total_time/3600:.2f} hours.")
    print(f"Best Epoch: {best_epoch} | Best Val PSNR: {best_val_psnr:.2f} dB | Best Val SSIM: {best_val_ssim:.4f}")

    # Generate training curves
    ep = history["epoch"]
    val_ep     = [e for e, v in zip(ep, history["val_psnr"]) if not math.isnan(v)]
    val_psnrs  = [v for v in history["val_psnr"] if not math.isnan(v)]
    val_ssims  = [v for v in history["val_ssim"] if not math.isnan(v)]
    val_losses = [v for v in history["val_loss"] if not math.isnan(v)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(ep, history["train_loss"], label="Train Loss", color="blue", linewidth=1.5)
    if val_ep:
        axes[0, 0].plot(val_ep, val_losses, label="Val Loss", color="red", linestyle="--", marker="o", linewidth=1.5)
    axes[0, 0].set_title("Training & Validation Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    if val_ep:
        axes[0, 1].plot(val_ep, val_psnrs, label="Val PSNR (dB)", color="green", marker="o", linewidth=1.5)
    axes[0, 1].set_title("Validation PSNR (dB)"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    if val_ep:
        axes[1, 0].plot(val_ep, val_ssims, label="Val SSIM", color="purple", marker="o", linewidth=1.5)
    axes[1, 0].set_title("Validation SSIM"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ep, history["lr"], label="Learning Rate", color="orange", linewidth=1.5)
    axes[1, 1].set_title("Learning Rate (Cosine Annealing)"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("LR")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"{MODEL_TAG} Training History ({len(ep)} Epochs)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(results_dir / "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] training_curves.png saved.")

    print("\n" + "=" * 70)
    print(f"{MODEL_TAG} TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train NAFNet with Adaptive Gate + Spatial Attn (M3)")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--patience",     type=int,   default=100)
    parser.add_argument("--val_every",    type=int,   default=10)
    parser.add_argument("--grad_accum",   type=int,   default=2)
    parser.add_argument("--base_ch",      type=int,   default=32, help="Base channel width (default: 32)")
    parser.add_argument("--num_blocks",   type=str,   default="1,1,1,4", help="Comma-separated blocks per level")
    parser.add_argument("--num_workers",  type=int,   default=2, help="DataLoader workers (default: 2)")
    args = parser.parse_args()

    blocks = [int(x.strip()) for x in args.num_blocks.split(",")]

    run_full_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        val_every=args.val_every,
        grad_accum=args.grad_accum,
        base_ch=args.base_ch,
        num_blocks=blocks,
        num_workers=args.num_workers
    )
