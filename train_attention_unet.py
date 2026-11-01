"""
train_attention_unet.py
=======================
Full-dataset training script for Attention U-Net on the S2Looking satellite
optical dataset.

Features:
  - 3,500 train images, 500 validation images, 1,000 test images.
  - Loss: 0.84 * L1 + 0.16 * (1 - SSIM).
  - AdamW Optimizer (lr=5e-4, weight_decay=1e-4) + CosineAnnealingLR.
  - Automatic Mixed Precision (AMP FP16) for accelerated GPU training.
  - Early stopping on Val PSNR (configurable --patience, default=15).
  - Per-epoch tracking: Train Loss, Val Loss, Val PSNR, Val SSIM, LR, Time.
  - Saves best + latest checkpoints (selected on Val PSNR — NOT test set).
  - Generates training curves (only epochs actually run) and CSV log.
  - Final test evaluation on 1,000 untouched test images after training.
  - 3-panel representative visualizations: Clean | Noisy | Denoised.
  - JSON experiment report for comparison with ResUNet baseline.
"""

import os
import sys
import math
import time
import json
import csv
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Redirect all caching and temporary allocations to D: drive
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".cache" / "torch")
os.environ["TEMP"]       = str(PROJECT_ROOT / ".temp")
os.environ["TMP"]        = str(PROJECT_ROOT / ".temp")
(PROJECT_ROOT / ".cache" / "torch").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".temp").mkdir(parents=True, exist_ok=True)

from preprocessing.transforms import TrainTransform, ValTestTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import SatelliteSpeckleDataset, TRAIN_DIR, VAL_DIR, TEST_DIR
from models.attention_unet import AttentionUNet
from models.losses import CombinedL1SSIMLoss, ssim


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = float(torch.mean((pred.float() - target.float()) ** 2))
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def calculate_pixel_metrics(pred: torch.Tensor, target: torch.Tensor,
                             threshold: float = 0.05) -> Tuple[float, float, float]:
    """
    Pixel-level restoration metrics at tolerance threshold 0.05.
    A pixel is 'Restored' if |pred - target| <= threshold.
    Acc = Prec = Recall (in this spatial domain formulation — documented).
    """
    abs_diff = torch.abs(pred.float() - target.float())
    correct  = (abs_diff <= threshold).float()
    acc = float(correct.mean())
    return acc, acc, acc


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def save_triplet_plot(clean: torch.Tensor, noisy: torch.Tensor,
                      denoised: torch.Tensor, title: str, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    c_np = clean.permute(1, 2, 0).detach().cpu().numpy()
    n_np = noisy.permute(1, 2, 0).detach().cpu().numpy()
    d_np = denoised.permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(c_np);   axes[0].set_title("Original Clean Target",    fontsize=11); axes[0].axis("off")
    axes[1].imshow(n_np);   axes[1].set_title("Speckle Noisy Input (L=4)", fontsize=11); axes[1].axis("off")
    axes[2].imshow(d_np);   axes[2].set_title("Attention U-Net Denoised",  fontsize=11); axes[2].axis("off")
    plt.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    model.train()
    running_loss, total = 0.0, 0
    use_cuda = device.type == "cuda"

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=use_cuda)
        clean = clean.to(device, non_blocking=use_cuda)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            outputs = model(noisy)
            loss    = criterion(outputs, clean)

        if use_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * noisy.size(0)
        total        += noisy.size(0)

    return running_loss / total


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


def evaluate_test_set(model, loader, device) -> Dict:
    model.eval()
    psnr_list, ssim_list, acc_list = [], [], []
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device, non_blocking=use_cuda)
            clean = clean.to(device, non_blocking=use_cuda)
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(noisy)

            for i in range(noisy.size(0)):
                p, c = outputs[i], clean[i]
                psnr_list.append(calculate_psnr(p, c))
                ssim_list.append(float(ssim(p.unsqueeze(0), c.unsqueeze(0))))
                acc, _, _ = calculate_pixel_metrics(p, c, threshold=0.05)
                acc_list.append(acc)

    return {
        "test_count":           len(psnr_list),
        "mean_psnr":            float(np.mean(psnr_list)),
        "median_psnr":          float(np.median(psnr_list)),
        "mean_ssim":            float(np.mean(ssim_list)),
        "median_ssim":          float(np.median(ssim_list)),
        "mean_pixel_accuracy":  float(np.mean(acc_list)),
        "mean_pixel_precision": float(np.mean(acc_list)),   # same formulation
        "mean_pixel_recall":    float(np.mean(acc_list)),
        "threshold":            0.05,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def save_training_curves(history: Dict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ep = history["epoch"]

    val_epochs = [e for e, v in zip(ep, history["val_loss"]) if not math.isnan(v)]
    val_loss   = [v for v in history["val_loss"] if not math.isnan(v)]
    val_psnr   = [v for v in history["val_psnr"] if not math.isnan(v)]
    val_ssim   = [v for v in history["val_ssim"] if not math.isnan(v)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(ep, history["train_loss"], label="Train Loss", color="blue", linewidth=1.5)
    if val_epochs:
        axes[0, 0].plot(val_epochs, val_loss, label="Val Loss", color="red", linestyle="--", marker="o", linewidth=1.5)
    axes[0, 0].set_title("Training & Validation Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    if val_epochs:
        axes[0, 1].plot(val_epochs, val_psnr, label="Val PSNR (dB)", color="green", marker="o", linewidth=1.5)
    axes[0, 1].set_title("Validation PSNR (dB)"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    if val_epochs:
        axes[1, 0].plot(val_epochs, val_ssim, label="Val SSIM", color="purple", marker="o", linewidth=1.5)
    axes[1, 0].set_title("Validation SSIM"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ep, history["lr"], label="Learning Rate", color="orange", linewidth=1.5)
    axes[1, 1].set_title("Learning Rate (Cosine Annealing)"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("LR")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Attention U-Net Training & Validation History", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def run_full_training(epochs: int = 100, batch_size: int = 8,
                      lr: float = 5e-4, patience: int = 100, val_every: int = 10):

    print("=" * 70, flush=True)
    print(f"ATTENTION U-NET FULL-DATASET EXPERIMENT ({epochs} EPOCHS)", flush=True)
    print("=" * 70, flush=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Target Device            : CUDA ({gpu_name}, {gpu_mem:.1f} GB VRAM)", flush=True)
    else:
        print("Target Device            : CPU", flush=True)

    print(f"Training Set Size        : 3,500 images")
    print(f"Validation Set Size      : 500 images")
    print(f"Test Set Size            : 1,000 images")
    print(f"Epochs                   : {epochs}")
    print(f"Batch Size               : {batch_size}")
    print(f"Initial Learning Rate    : {lr}")
    print(f"Early Stopping Patience  : {patience} epochs (monitored: Val PSNR)")
    print(f"Validation Frequency     : every {val_every} epochs")
    print(f"Optimizer                : AdamW (weight_decay=1e-4)")
    print(f"Scheduler                : CosineAnnealingLR (T_max={epochs})")
    print(f"Loss Function            : Combined L1 + SSIM Loss (0.84 L1 + 0.16 SSIM-loss)")
    print(f"Noise Model              : Gamma Multiplicative Speckle (L=4)")

    # Directories
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    results_dir    = PROJECT_ROOT / "results" / "attention_unet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    noise_fn     = SpeckleNoise(num_looks=4)
    train_ds     = SatelliteSpeckleDataset(TRAIN_DIR, transform=TrainTransform(crop_size=256),    noise_fn=noise_fn)
    val_ds       = SatelliteSpeckleDataset(VAL_DIR,   transform=ValTestTransform(crop_size=256),  noise_fn=noise_fn)
    test_ds      = SatelliteSpeckleDataset(TEST_DIR,  transform=ValTestTransform(crop_size=256),  noise_fn=noise_fn)

    use_pin      = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=use_pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin)

    # Model, loss, optimiser, scheduler
    model     = AttentionUNet(in_channels=3, out_channels=3).to(device)
    criterion = CombinedL1SSIMLoss(alpha=0.84)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters         : {total_params:,} total  |  {trainable_params:,} trainable")

    # History
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_psnr": [], "val_ssim": [], "lr": [], "epoch_time": []}

    best_val_psnr     = float('-inf')
    best_val_ssim     = 0.0
    best_epoch        = 0
    early_stop_counter = 0

    total_start = time.time()

    # CSV log
    csv_file = results_dir / "training_log.csv"
    with open(csv_file, "w", newline="") as f:
        csv.writer(f).writerow(["Epoch", "Train_Loss", "Val_Loss",
                                 "Val_PSNR_dB", "Val_SSIM", "LR", "Time_sec"])

    print("\nStarting Training Loop...\n", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        curr_lr    = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed    = time.time() - t0

        # ── Decide whether to run validation this epoch ───────────────────────
        do_val = (epoch % val_every == 0) or (epoch == epochs)

        if do_val:
            val_loss, val_psnr, val_ssim_val = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_psnr, val_ssim_val = None, None, None

        # ── Record history (only val epochs get val metrics) ──────────────────
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss   if do_val else float('nan'))
        history["val_psnr"].append(val_psnr   if do_val else float('nan'))
        history["val_ssim"].append(val_ssim_val if do_val else float('nan'))
        history["lr"].append(curr_lr)
        history["epoch_time"].append(elapsed)

        # ── CSV row ───────────────────────────────────────────────────────────
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

        # ── Console output ────────────────────────────────────────────────────
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
        else:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.5f} | "
                f"LR: {curr_lr:.6f}",
                flush=True
            )

        # ── Checkpoint + early stopping (only on val epochs) ──────────────────
        if do_val:
            # Latest checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim_val,
                "config": {"in_channels": 3, "out_channels": 3, "crop_size": 256,
                           "noise_L": 4, "lr": lr, "batch_size": batch_size},
            }, checkpoint_dir / "attention_unet_latest.pth")

            # Best checkpoint
            if val_psnr > best_val_psnr:
                best_val_psnr      = val_psnr
                best_val_ssim      = val_ssim_val
                best_epoch         = epoch
                early_stop_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim_val,
                    "config": {"in_channels": 3, "out_channels": 3, "crop_size": 256,
                               "noise_L": 4, "lr": lr, "batch_size": batch_size},
                }, checkpoint_dir / "attention_unet_best.pth")
                print(f"  [Checkpoint] New Best Model Saved "
                      f"(Val PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim_val:.4f}) "
                      f"-> attention_unet_best.pth", flush=True)
            else:
                early_stop_counter += 1
                print(f"  [EarlyStopping] No improvement for {early_stop_counter}/{patience} "
                      f"val checks (best: {best_val_psnr:.2f} dB @ epoch {best_epoch})", flush=True)
                if early_stop_counter >= patience:
                    print("\n" + "=" * 70, flush=True)
                    print("Early stopping triggered", flush=True)
                    print(f"  Best Epoch    : {best_epoch}", flush=True)
                    print(f"  Best Val PSNR : {best_val_psnr:.2f} dB", flush=True)
                    print(f"  Best Val SSIM : {best_val_ssim:.4f}", flush=True)
                    print("=" * 70, flush=True)
                    break

    # ── Post-loop stats ───────────────────────────────────────────────────────
    total_training_time = time.time() - total_start
    epochs_run  = len(history["epoch"])
    avg_epoch_time = total_training_time / max(epochs_run, 1)

    # Training curves (only epochs that actually ran)
    save_training_curves(history, results_dir / "training_curves.png")

    # ── Final test evaluation on best checkpoint ──────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("FINAL TEST SET EVALUATION (ON 1,000 UNTOUCHED TEST IMAGES)", flush=True)
    print("=" * 70, flush=True)

    best_ckpt_path = checkpoint_dir / "attention_unet_best.pth"
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded Best Checkpoint from Epoch {ckpt['epoch']} "
          f"(Val PSNR: {ckpt['val_psnr']:.2f} dB)", flush=True)

    test_metrics = evaluate_test_set(model, test_loader, device)

    print("\n--- FINAL TEST EVALUATION RESULTS ---", flush=True)
    print(f"Test Images Evaluated    : {test_metrics['test_count']} / 1,000")
    print(f"Mean Test PSNR           : {test_metrics['mean_psnr']:.2f} dB")
    print(f"Median Test PSNR         : {test_metrics['median_psnr']:.2f} dB")
    print(f"Mean Test SSIM           : {test_metrics['mean_ssim']:.4f}")
    print(f"Median Test SSIM         : {test_metrics['median_ssim']:.4f}")
    print(f"Pixel Restoration Acc    : {test_metrics['mean_pixel_accuracy']*100:.2f}% (threshold=0.05)")

    # ── Representative visualisations ────────────────────────────────────────
    print("\n--- Generating Representative 3-Panel Visualizations ---", flush=True)
    model.eval()
    with torch.no_grad():
        for split_name, ds_obj, label in [
            ("train", train_ds, "Train Split"),
            ("val",   val_ds,   "Validation Split"),
            ("test",  test_ds,  "Test Split"),
        ]:
            noisy_s, clean_s = ds_obj[0]
            denoised_s = model(noisy_s.unsqueeze(0).to(device)).squeeze(0).cpu()
            pv = calculate_psnr(denoised_s, clean_s)
            sv = float(ssim(denoised_s.unsqueeze(0), clean_s.unsqueeze(0)))
            out_vis = results_dir / f"representative_{split_name}_triplet.png"
            save_triplet_plot(
                clean=clean_s, noisy=noisy_s, denoised=denoised_s,
                title=f"Attention U-Net: {label} | PSNR: {pv:.2f} dB | SSIM: {sv:.4f}",
                save_path=out_vis,
            )
            print(f"  [OK] {split_name} triplet saved: {out_vis.name}")

    # ── JSON experiment report ────────────────────────────────────────────────
    summary = {
        "model": "Attention U-Net",
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "dataset_split":    {"train": 3500, "val": 500, "test": 1000},
        "preprocessing":    {"size": 256, "channels": 3, "range": "[0, 1]"},
        "speckle_noise":    {"model": "Gamma Multiplicative", "L": 4},
        "hyperparameters": {
            "epochs_configured": epochs, "epochs_run": epochs_run,
            "early_stopping_patience": patience,
            "val_every": val_every,
            "batch_size": batch_size, "learning_rate": lr,
            "optimizer": "AdamW", "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "loss": "0.84 * L1 + 0.16 * (1 - SSIM)",
        },
        "training_stats": {
            "total_time_sec":    round(total_training_time, 2),
            "avg_epoch_time_sec": round(avg_epoch_time, 2),
            "final_train_loss":  round(history["train_loss"][-1], 5),
            "final_val_loss":    round([v for v in history["val_loss"] if not math.isnan(v)][-1], 5) if any(not math.isnan(v) for v in history["val_loss"]) else None,
            "best_epoch":        best_epoch,
            "best_val_psnr":     round(best_val_psnr, 2),
            "best_val_ssim":     round(best_val_ssim, 4),
        },
        "test_results": test_metrics,
        "device": str(device),
    }

    report_path = results_dir / "attention_unet_experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[OK] Experiment Report  : results/attention_unet/attention_unet_experiment_report.json")
    print(f"[OK] Training Curves    : results/attention_unet/training_curves.png")
    print(f"[OK] Training CSV Log   : results/attention_unet/training_log.csv")
    print(f"[OK] Best Checkpoint    : checkpoints/attention_unet_best.pth  (epoch {best_epoch})")
    print("=" * 70, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Attention U-Net on S2Looking dataset")
    parser.add_argument("--epochs",     type=int,   default=100,  help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int,   default=8,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--patience",   type=int,   default=100,
                        help="Early stopping patience in val-check rounds (default: 100 = no early stop)")
    parser.add_argument("--val_every",  type=int,   default=10,
                        help="Run validation every N epochs (default: 10)")
    args = parser.parse_args()

    run_full_training(epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, patience=args.patience, val_every=args.val_every)
