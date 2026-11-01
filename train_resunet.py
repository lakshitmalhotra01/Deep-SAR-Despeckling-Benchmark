"""
train_resunet.py
================
Full-dataset training script for Residual U-Net (ResUNet) on the S2Looking satellite optical dataset.

Features:
  - 3,500 train images, 500 validation images, 1,000 test images.
  - Loss: 0.84 * L1 + 0.16 * (1 - SSIM).
  - AdamW Optimizer (lr=5e-4, weight_decay=1e-4) + CosineAnnealingLR.
  - Automatic Mixed Precision (AMP FP16) for accelerated training on NVIDIA GPU.
  - Per-epoch tracking: Train Loss, Val Loss, Val PSNR, Val SSIM, LR, Time.
  - Checkpoint management: saves best model checkpoint based on Validation PSNR.
  - Early stopping based on Validation PSNR (configurable --patience, default=15).
  - Generates training curves plot (only epochs actually run) and CSV log.
  - Evaluates best checkpoint on untouched 1,000 TEST images (Mean & Median PSNR, SSIM,
    and Pixel-Level Restoration Accuracy/Precision/Recall at threshold epsilon = 0.05).
  - Generates 3-panel representative visualizations: Original Clean | Speckle Noisy | ResUNet Denoised.
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
from models.resunet import ResUNet
from models.losses import CombinedL1SSIMLoss, ssim


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = float(torch.mean((pred.float() - target.float()) ** 2))
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def calculate_pixel_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.05) -> Tuple[float, float, float]:
    """
    Pixel-level restoration metrics:
      A pixel is defined as 'Restored' (True Positive) if |pred - target| <= threshold (5% intensity tolerance).
      - Pixel Accuracy  : Fraction of pixels within tolerance.
      - Pixel Precision : Fraction of restored pixels relative to total image pixels.
      - Pixel Recall    : Fraction of restored pixels relative to total target pixels.
    """
    abs_diff = torch.abs(pred.float() - target.float())
    correct_pixels = (abs_diff <= threshold).float()

    acc = float(correct_pixels.mean())
    prec = acc
    rec  = acc
    return acc, prec, rec


def save_triplet_plot(clean: torch.Tensor, noisy: torch.Tensor, denoised: torch.Tensor, title: str, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    c_np = clean.permute(1, 2, 0).detach().cpu().numpy()
    n_np = noisy.permute(1, 2, 0).detach().cpu().numpy()
    d_np = denoised.permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(c_np)
    axes[0].set_title("Original Clean Target", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(n_np)
    axes[1].set_title("Speckle Noisy Input (L=4)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(d_np)
    axes[2].set_title("ResUNet Denoised Output", fontsize=11)
    axes[2].axis("off")

    plt.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    use_cuda = device.type == "cuda"

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=use_cuda)
        clean = clean.to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            outputs = model(noisy)
            loss = criterion(outputs, clean)

        if use_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = noisy.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device, non_blocking=use_cuda)
            clean = clean.to(device, non_blocking=use_cuda)

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(noisy)
                loss = criterion(outputs, clean)

            batch_size = noisy.size(0)
            running_loss += loss.item() * batch_size
            total_psnr += calculate_psnr(outputs, clean) * batch_size
            total_ssim += float(ssim(outputs, clean)) * batch_size
            total_samples += batch_size

    return (running_loss / total_samples), (total_psnr / total_samples), (total_ssim / total_samples)


def evaluate_test_set(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    psnr_list = []
    ssim_list = []
    acc_list  = []
    prec_list = []
    rec_list  = []
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device, non_blocking=use_cuda)
            clean = clean.to(device, non_blocking=use_cuda)

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(noisy)

            for i in range(noisy.size(0)):
                p = outputs[i]
                c = clean[i]
                psnr_val = calculate_psnr(p, c)
                ssim_val = float(ssim(p.unsqueeze(0), c.unsqueeze(0)))
                acc, prec, rec = calculate_pixel_metrics(p, c, threshold=0.05)

                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                acc_list.append(acc)
                prec_list.append(prec)
                rec_list.append(rec)

    return {
        "test_count": len(psnr_list),
        "mean_psnr": float(np.mean(psnr_list)),
        "median_psnr": float(np.median(psnr_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "median_ssim": float(np.median(ssim_list)),
        "mean_pixel_accuracy": float(np.mean(acc_list)),
        "mean_pixel_precision": float(np.mean(prec_list)),
        "mean_pixel_recall": float(np.mean(rec_list)),
        "threshold": 0.05,
    }


def save_training_curves(history: Dict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curve
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="blue", linewidth=1.5)
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", color="red", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("Training & Validation Loss", fontsize=12)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss (0.84 L1 + 0.16 SSIM-loss)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # PSNR curve
    axes[0, 1].plot(epochs, history["val_psnr"], label="Val PSNR (dB)", color="green", linewidth=1.5)
    axes[0, 1].set_title("Validation PSNR (dB)", fontsize=12)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # SSIM curve
    axes[1, 0].plot(epochs, history["val_ssim"], label="Val SSIM", color="purple", linewidth=1.5)
    axes[1, 0].set_title("Validation SSIM", fontsize=12)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # LR curve
    axes[1, 1].plot(epochs, history["lr"], label="Learning Rate", color="orange", linewidth=1.5)
    axes[1, 1].set_title("Learning Rate Schedule (Cosine Annealing)", fontsize=12)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Residual U-Net (ResUNet) Training & Validation History", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_full_training(epochs: int = 100, batch_size: int = 8, lr: float = 5e-4, patience: int = 15):
    print("=" * 70, flush=True)
    print(f"RESIDUAL U-NET (ResUNet) FULL-DATASET EXPERIMENT ({epochs} EPOCHS)", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Target Device            : CUDA ({gpu_name}, {gpu_mem:.1f} GB VRAM)", flush=True)
    else:
        print(f"Target Device            : CPU", flush=True)

    print(f"Training Set Size        : 3,500 images")
    print(f"Validation Set Size      : 500 images")
    print(f"Test Set Size            : 1,000 images")
    print(f"Epochs                   : {epochs}")
    print(f"Batch Size               : {batch_size}")
    print(f"Initial Learning Rate    : {lr}")
    print(f"Early Stopping Patience  : {patience} epochs (monitored: Val PSNR)")
    print(f"Optimizer                : AdamW (weight_decay=1e-4)")
    print(f"Scheduler                : CosineAnnealingLR (T_max={epochs})")
    print(f"Loss Function            : Combined L1 + SSIM Loss (0.84 L1 + 0.16 SSIM-loss)")
    print(f"Noise Model              : Gamma Multiplicative Speckle (L=4)")

    # Output directories
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    results_dir    = PROJECT_ROOT / "results" / "residual_unet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    noise_fn = SpeckleNoise(num_looks=4)
    train_ds = SatelliteSpeckleDataset(TRAIN_DIR, transform=TrainTransform(crop_size=256), noise_fn=noise_fn)
    val_ds   = SatelliteSpeckleDataset(VAL_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)
    test_ds  = SatelliteSpeckleDataset(TEST_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)

    use_pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_pin)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin)

    # Model & Optimization
    model = ResUNet(in_channels=3, out_channels=3).to(device)
    criterion = CombinedL1SSIMLoss(alpha=0.84)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "lr": [],
        "epoch_time": []
    }

    best_val_psnr = float('-inf')
    best_val_ssim = 0.0
    best_epoch = 0
    early_stop_counter = 0   # counts consecutive epochs without improvement
    total_start_time = time.time()

    # CSV Log header
    csv_file = results_dir / "training_log.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_PSNR_dB", "Val_SSIM", "LR", "Time_sec"])

    print("\nStarting Training Loop...\n", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_psnr, val_ssim_val = validate(model, val_loader, criterion, device)
        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - t0

        # Update History
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim_val)
        history["lr"].append(curr_lr)
        history["epoch_time"].append(elapsed)

        # Append to CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}", f"{val_psnr:.2f}", f"{val_ssim_val:.4f}", f"{curr_lr:.6f}", f"{elapsed:.1f}"])

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val PSNR: {val_psnr:.2f} dB | "
            f"Val SSIM: {val_ssim_val:.4f} | "
            f"LR: {curr_lr:.6f}",
            flush=True
        )

        # Save Latest Checkpoint
        latest_ckpt = checkpoint_dir / "resunet_latest.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_psnr": val_psnr,
            "val_ssim": val_ssim_val,
            "config": {
                "in_channels": 3, "out_channels": 3, "crop_size": 256,
                "noise_L": 4, "lr": lr, "batch_size": batch_size
            }
        }, latest_ckpt)

        # Save Best Checkpoint (Model Selection on Validation PSNR)
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_val_ssim = val_ssim_val
            best_epoch = epoch
            early_stop_counter = 0   # reset: we improved
            best_ckpt = checkpoint_dir / "resunet_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim_val,
                "config": {
                    "in_channels": 3, "out_channels": 3, "crop_size": 256,
                    "noise_L": 4, "lr": lr, "batch_size": batch_size
                }
            }, best_ckpt)
            print(f"  [Checkpoint] New Best Model Saved (Val PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim_val:.4f}) -> {best_ckpt.name}", flush=True)
        else:
            early_stop_counter += 1
            print(f"  [EarlyStopping] No improvement for {early_stop_counter}/{patience} epochs "
                  f"(best Val PSNR so far: {best_val_psnr:.2f} dB @ epoch {best_epoch})", flush=True)
            if early_stop_counter >= patience:
                print("\n" + "=" * 70, flush=True)
                print("Early stopping triggered", flush=True)
                print(f"  Best Epoch       : {best_epoch}", flush=True)
                print(f"  Best Val PSNR    : {best_val_psnr:.2f} dB", flush=True)
                print(f"  Best Val SSIM    : {best_val_ssim:.4f}", flush=True)
                print("=" * 70, flush=True)
                break


    # ---- Post-loop: compute wall-clock stats based on epochs actually run ----
    total_training_time = time.time() - total_start_time
    epochs_run = len(history["epoch"])
    avg_epoch_time = total_training_time / max(epochs_run, 1)

    # Save training curves (only epochs that actually ran)
    save_training_curves(history, results_dir / "training_curves.png")


    print("\n" + "=" * 70, flush=True)
    print("FINAL TEST SET EVALUATION (ON 1,000 UNTOUCHED TEST IMAGES)", flush=True)
    print("=" * 70, flush=True)

    # Load Best Checkpoint for Final Test Evaluation
    best_ckpt = checkpoint_dir / "resunet_best.pth"
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded Best Validation Checkpoint from Epoch {checkpoint['epoch']} (Val PSNR: {checkpoint['val_psnr']:.2f} dB)", flush=True)

    test_metrics = evaluate_test_set(model, test_loader, device)

    print("\n--- FINAL TEST EVALUATION RESULTS ---", flush=True)
    print(f"Test Images Evaluated    : {test_metrics['test_count']} / 1,000")
    print(f"Mean Test PSNR           : {test_metrics['mean_psnr']:.2f} dB")
    print(f"Median Test PSNR         : {test_metrics['median_psnr']:.2f} dB")
    print(f"Mean Test SSIM           : {test_metrics['mean_ssim']:.4f}")
    print(f"Median Test SSIM         : {test_metrics['median_ssim']:.4f}")
    print(f"Pixel Restoration Acc    : {test_metrics['mean_pixel_accuracy']*100:.2f}% (tolerance = {test_metrics['threshold']})")
    print(f"Pixel Restoration Prec   : {test_metrics['mean_pixel_precision']*100:.2f}%")
    print(f"Pixel Restoration Rec    : {test_metrics['mean_pixel_recall']*100:.2f}%")

    # Generate Representative Visualizations for Train, Val, and Test
    print("\n--- Generating Representative 3-Panel Visualizations ---", flush=True)
    model.eval()
    splits_to_vis = [
        ("train", train_ds, "Train Split (Sample 1)"),
        ("val", val_ds, "Validation Split (Sample 1)"),
        ("test", test_ds, "Test Split (Sample 1)")
    ]

    with torch.no_grad():
        for split_name, ds_obj, label in splits_to_vis:
            noisy_sample, clean_sample = ds_obj[0]
            noisy_in = noisy_sample.unsqueeze(0).to(device)
            denoised_out = model(noisy_in).squeeze(0).cpu()

            psnr_v = calculate_psnr(denoised_out, clean_sample)
            ssim_v = float(ssim(denoised_out.unsqueeze(0), clean_sample.unsqueeze(0)))

            out_vis_path = results_dir / f"representative_{split_name}_triplet.png"
            save_triplet_plot(
                clean=clean_sample,
                noisy=noisy_sample,
                denoised=denoised_out,
                title=f"Residual U-Net: {label} | PSNR: {psnr_v:.2f} dB | SSIM: {ssim_v:.4f}",
                save_path=out_vis_path
            )
            print(f"  [OK] Saved {split_name} visual triplet to: {out_vis_path.name}")

    # Save summary report JSON
    summary_report = {
        "model": "Residual U-Net (ResUNet)",
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "dataset_split": {"train": 3500, "val": 500, "test": 1000},
        "preprocessing": {"size": 256, "channels": 3, "range": "[0, 1]"},
        "speckle_noise": {"model": "Gamma Multiplicative", "L": 4},
        "hyperparameters": {
            "epochs_configured": epochs, "epochs_run": epochs_run,
            "early_stopping_patience": patience,
            "batch_size": batch_size, "learning_rate": lr,
            "optimizer": "AdamW", "weight_decay": 1e-4, "scheduler": "CosineAnnealingLR",
            "loss": "0.84 * L1 + 0.16 * (1 - SSIM)"
        },
        "training_stats": {
            "total_time_sec": round(total_training_time, 2),
            "avg_epoch_time_sec": round(avg_epoch_time, 2),
            "final_train_loss": round(history["train_loss"][-1], 5),
            "final_val_loss": round(history["val_loss"][-1], 5),
            "best_epoch": best_epoch,
            "best_val_psnr": round(best_val_psnr, 2),
            "best_val_ssim": round(best_val_ssim, 4),
        },
        "test_results": test_metrics,
        "device": str(device)
    }

    report_json_path = results_dir / "resunet_experiment_report.json"
    with open(report_json_path, "w") as f:
        json.dump(summary_report, f, indent=4)

    print(f"\n[OK] Experiment Summary JSON saved to: {report_json_path.name}")
    print(f"[OK] Training Curves Plot saved to: results/residual_unet/training_curves.png")
    print(f"[OK] Training CSV Log saved to: results/residual_unet/training_log.csv")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ResUNet on S2Looking dataset")
    parser.add_argument("--epochs",     type=int,   default=100,  help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=8,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--patience",   type=int,   default=15,
                        help="Early stopping patience: stop if Val PSNR does not improve for this many consecutive epochs (default: 15)")
    args = parser.parse_args()

    run_full_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)
