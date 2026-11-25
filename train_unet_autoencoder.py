"""
train_unet_autoencoder.py
=========================
Full-dataset training script for U-Net Denoising Autoencoder (UNet-Autoencoder)
on the satellite optical/SAR speckle denoising benchmark.

Model:
  - 11th Model Benchmark: U-Net Autoencoder (31.04M parameters)
  - Contracting Encoder (64 -> 128 -> 256 -> 512)
  - Latent Bottleneck (1024)
  - Expanding Decoder with Skip Connections (512 -> 256 -> 128 -> 64)
  - Output: 1x1 Conv with Sigmoid activation strictly bounding output to [0, 1].

Training Settings:
  - Dataset: 3,500 train images, 500 validation images, 1,000 test images (S2Looking).
  - Noise Model: Multiplicative Gamma speckle noise (L = 4 looks).
  - Loss: Combined 0.84 * L1 + 0.16 * (1 - SSIM).
  - Optimizer: AdamW (lr=5e-4, weight_decay=1e-4) + CosineAnnealingLR.
  - Automatic Mixed Precision (AMP FP16) for accelerated training on NVIDIA GPU.
  - Checkpoint: checkpoints/unet_autoencoder_best.pth
  - Output Directory: results/unet_autoencoder/
  - Strict D: drive caching isolation (TEMP, TORCH_HOME).
"""

import os
import sys
import math
import time
import json
import csv
import argparse
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

# Force all caches and temp files to D: drive
os.environ["TORCH_HOME"]         = str(PROJECT_ROOT / ".cache" / "torch")
os.environ["TEMP"]               = str(PROJECT_ROOT / ".temp")
os.environ["TMP"]                = str(PROJECT_ROOT / ".temp")
os.environ["MPLCONFIGDIR"]       = str(PROJECT_ROOT / ".cache" / "matplotlib")
os.environ["CUDA_CACHE_PATH"]    = str(PROJECT_ROOT / ".cache" / "nv")
os.environ["PYTHONPYCACHEPREFIX"] = str(PROJECT_ROOT / ".cache" / "pycache")

for d in [".cache/torch", ".cache/matplotlib", ".cache/nv", ".cache/pycache", ".temp"]:
    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

from preprocessing.transforms import TrainTransform, ValTestTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import SatelliteSpeckleDataset, TRAIN_DIR, VAL_DIR, TEST_DIR
from models.unet_autoencoder import UNetAutoencoder
from models.losses import CombinedL1SSIMLoss, ssim


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = float(torch.mean((pred.float() - target.float()) ** 2))
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def calculate_pixel_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.05) -> Tuple[float, float, float]:
    abs_diff = torch.abs(pred.float() - target.float())
    correct_pixels = (abs_diff <= threshold).float()
    acc = float(correct_pixels.mean())
    return acc, acc, acc


def save_triplet_plot(clean: torch.Tensor, noisy: torch.Tensor, denoised: torch.Tensor, title: str, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    c_np = clean.permute(1, 2, 0).detach().cpu().numpy()
    n_np = noisy.permute(1, 2, 0).detach().cpu().numpy()
    d_np = denoised.permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(c_np, 0, 1))
    axes[0].set_title("Original Clean Target", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(np.clip(n_np, 0, 1))
    axes[1].set_title("Speckle Noisy Input (L=4)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(np.clip(d_np, 0, 1))
    axes[2].set_title("UNet-Autoencoder Denoised", fontsize=11)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train UNet-Autoencoder for Satellite/SAR Despeckling")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--eval-freq", type=int, default=1, help="Validation frequency (epochs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("  TRAINING MODEL 11: U-NET DENOISING AUTOENCODER (UNET-AE)")
    print("=" * 75)
    print(f"Device           : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Target Epochs    : {args.epochs}")
    print(f"Batch Size       : {args.batch_size}")
    print(f"Base LR          : {args.lr}")
    print(f"Early Patience   : {args.patience}")
    print("-" * 75)

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    results_dir = PROJECT_ROOT / "results" / "unet_autoencoder"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & DataLoaders
    noise_fn = SpeckleNoise(num_looks=4)
    train_ds = SatelliteSpeckleDataset(TRAIN_DIR, transform=TrainTransform(crop_size=256), noise_fn=noise_fn)
    val_ds   = SatelliteSpeckleDataset(VAL_DIR,   transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)
    test_ds  = SatelliteSpeckleDataset(TEST_DIR,  transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    print(f"Train samples    : {len(train_ds):,} (batches: {len(train_loader)})")
    print(f"Val samples      : {len(val_ds):,} (batches: {len(val_loader)})")
    print(f"Test samples     : {len(test_ds):,} (batches: {len(test_loader)})")

    # 2. Model & Optimizer
    model = UNetAutoencoder(in_channels=3, out_channels=3, base_channels=64).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Params : {total_params:,} (~{total_params/1e6:.2f}M)")

    criterion = CombinedL1SSIMLoss(alpha=0.84).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 3. CSV Logger
    csv_path = results_dir / "training_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "lr", "epoch_time_s"])

    best_val_psnr = -1.0
    best_epoch    = 0
    patience_cnt  = 0
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [], "epochs": []}

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_acc = 0.0

        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_acc += loss.item() * noisy_imgs.size(0)

        epoch_train_loss = train_loss_acc / len(train_ds)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_time = time.time() - t0

        # Validation
        if epoch % args.eval_freq == 0:
            model.eval()
            val_loss_acc = 0.0
            val_psnr_acc = 0.0
            val_ssim_acc = 0.0

            with torch.no_grad():
                for noisy_v, clean_v in val_loader:
                    noisy_v = noisy_v.to(device)
                    clean_v = clean_v.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        out_v = model(noisy_v)
                        v_loss = criterion(out_v, clean_v)

                    val_loss_acc += v_loss.item() * noisy_v.size(0)
                    for b in range(noisy_v.size(0)):
                        val_psnr_acc += calculate_psnr(out_v[b], clean_v[b])
                        val_ssim_acc += float(ssim(out_v[b:b+1], clean_v[b:b+1]))

            ep_val_loss = val_loss_acc / len(val_ds)
            ep_val_psnr = val_psnr_acc / len(val_ds)
            ep_val_ssim = val_ssim_acc / len(val_ds)

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(ep_val_loss)
            history["val_psnr"].append(ep_val_psnr)
            history["val_ssim"].append(ep_val_ssim)
            history["epochs"].append(epoch)

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{epoch_train_loss:.6f}", f"{ep_val_loss:.6f}",
                                 f"{ep_val_psnr:.4f}", f"{ep_val_ssim:.4f}", f"{current_lr:.6e}", f"{epoch_time:.1f}"])

            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                  f"Train Loss: {epoch_train_loss:.5f} | Val Loss: {ep_val_loss:.5f} | "
                  f"Val PSNR: {ep_val_psnr:.2f} dB | Val SSIM: {ep_val_ssim:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

            # Checkpoint saving
            latest_ckpt = ckpt_dir / "unet_autoencoder_latest.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": ep_val_psnr,
                "val_ssim": ep_val_ssim,
                "total_params": total_params
            }, str(latest_ckpt))

            if ep_val_psnr > best_val_psnr:
                best_val_psnr = ep_val_psnr
                best_epoch = epoch
                patience_cnt = 0
                best_ckpt = ckpt_dir / "unet_autoencoder_best.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_psnr": ep_val_psnr,
                    "val_ssim": ep_val_ssim,
                    "total_params": total_params
                }, str(best_ckpt))
                print(f"  [+] Saved new best model to {best_ckpt.name} (Val PSNR: {best_val_psnr:.2f} dB)")
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"\n[!] Early stopping triggered at epoch {epoch}. Best Val PSNR: {best_val_psnr:.2f} dB (epoch {best_epoch}).")
                    break
        else:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] Train Loss: {epoch_train_loss:.5f} | Time: {epoch_time:.1f}s")

    print("\n" + "=" * 75)
    print("  TRAINING COMPLETE! PROCEEDING TO FINAL 1,000 TEST EVALUATION")
    print("=" * 75)

    # 4. Final Evaluation on 1,000 Test Set
    best_ckpt = ckpt_dir / "unet_autoencoder_best.pth"
    if best_ckpt.exists():
        st = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(st["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {st.get('epoch', 'N/A')}")

    model.eval()
    test_psnr_list = []
    test_ssim_list = []
    test_acc_list  = []

    sample_clean, sample_noisy, sample_denoised = None, None, None

    with torch.no_grad():
        for i, (noisy_t, clean_t) in enumerate(test_loader):
            noisy_t = noisy_t.to(device)
            clean_t = clean_t.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                denoised_t = model(noisy_t)

            for b in range(noisy_t.size(0)):
                p = calculate_psnr(denoised_t[b], clean_t[b])
                s = float(ssim(denoised_t[b:b+1], clean_t[b:b+1]))
                acc, _, _ = calculate_pixel_metrics(denoised_t[b], clean_t[b], threshold=0.05)
                test_psnr_list.append(p)
                test_ssim_list.append(s)
                test_acc_list.append(acc)

            if i == 0:
                sample_clean = clean_t[0].cpu()
                sample_noisy = noisy_t[0].cpu()
                sample_denoised = denoised_t[0].cpu()

    mean_psnr = float(np.mean(test_psnr_list))
    median_psnr = float(np.median(test_psnr_list))
    mean_ssim = float(np.mean(test_ssim_list))
    mean_acc  = float(np.mean(test_acc_list)) * 100.0

    print("\n" + "-" * 75)
    print("  MODEL 11: UNET-AUTOENCODER FINAL TEST SET RESULTS (1,000 Images)")
    print("-" * 75)
    print(f"  Test PSNR (Mean)   : {mean_psnr:.2f} dB")
    print(f"  Test PSNR (Median) : {median_psnr:.2f} dB")
    print(f"  Test SSIM          : {mean_ssim:.4f}")
    print(f"  Pixel Acc (eps=0.05): {mean_acc:.2f}%")
    print("-" * 75)

    # Save summary report JSON
    report = {
        "model": "UNetAutoencoder",
        "total_parameters": total_params,
        "best_epoch": best_epoch,
        "best_val_psnr": best_val_psnr,
        "test_psnr_mean": mean_psnr,
        "test_psnr_median": median_psnr,
        "test_ssim": mean_ssim,
        "test_pixel_accuracy_005": mean_acc
    }
    with open(results_dir / "unet_autoencoder_experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save representative triplet visualization
    if sample_clean is not None:
        save_triplet_plot(sample_clean, sample_noisy, sample_denoised,
                          f"UNet-Autoencoder Representative Test Triplet\nPSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f}",
                          results_dir / "representative_test_triplet.png")

    # Plot training curves
    if len(history["epochs"]) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history["epochs"], history["train_loss"], label="Train Loss", color="blue")
        axes[0].plot(history["epochs"], history["val_loss"], label="Val Loss", color="orange")
        axes[0].set_title("Training & Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.6)

        axes[1].plot(history["epochs"], history["val_psnr"], label="Val PSNR (dB)", color="green")
        axes[1].set_title(f"Validation PSNR (Best: {best_val_psnr:.2f} dB)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("PSNR (dB)")
        axes[1].legend()
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(str(results_dir / "training_curves.png"), dpi=150)
        plt.close()

    print(f"Artifacts saved in: {results_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
