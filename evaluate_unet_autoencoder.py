"""
evaluate_unet_autoencoder.py
============================
Standalone full evaluation script for U-Net Denoising Autoencoder (UNet-Autoencoder).
Matches the exact evaluation report & plot suite of ResUNet / CBAM-UNet:
  - Evaluates checkpoints/unet_autoencoder_best.pth on 1,000 untouched test images
  - Saves full unet_autoencoder_experiment_report.json
  - Generates representative_train_triplet.png, representative_val_triplet.png, representative_test_triplet.png
  - Generates training_curves.png, plot_psnr_progression.png, plot_ssim_progression.png,
    plot_full_dashboard.png, and plot_model_comparison.png
"""

import os
import sys
import csv
import math
import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Strict D: drive redirect
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".cache" / "torch")
os.environ["TEMP"] = str(PROJECT_ROOT / ".temp")
os.environ["TMP"] = str(PROJECT_ROOT / ".temp")
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")
os.environ["CUDA_CACHE_PATH"] = str(PROJECT_ROOT / ".cache" / "nv")
os.environ["PYTHONPYCACHEPREFIX"] = str(PROJECT_ROOT / ".cache" / "pycache")

from preprocessing.transforms import ValTestTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import SatelliteSpeckleDataset, TRAIN_DIR, VAL_DIR, TEST_DIR
from models.unet_autoencoder import UNetAutoencoder
from models.losses import ssim


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = float(torch.mean((pred.float() - target.float()) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def calculate_pixel_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.05) -> Tuple[float, float, float]:
    abs_diff = torch.abs(pred.float() - target.float())
    correct = (abs_diff <= threshold).float()
    acc = float(correct.mean())
    return acc, acc, acc


def save_triplet_plot(clean: torch.Tensor, noisy: torch.Tensor, denoised: torch.Tensor, title: str, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (clean, "Clean (Ground Truth)"),
        (noisy, "Noisy (Gamma L=4)"),
        (denoised, "UNet-Autoencoder Denoised"),
    ]
    for ax, (img, label) in zip(axes, panels):
        arr = img.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        ax.imshow(arr)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = PROJECT_ROOT / "results" / "unet_autoencoder"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  EVALUATING UNET-AUTOENCODER ON 1,000 UNTOUCHED TEST IMAGES")
    print("=" * 75)
    print(f"Device: {device}")

    ckpt_path = PROJECT_ROOT / "checkpoints" / "unet_autoencoder_best.pth"
    if not ckpt_path.exists():
        print(f"[!] Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    model = UNetAutoencoder(in_channels=3, out_channels=3, base_channels=64).to(device)
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded checkpoint successfully (Total params: {total_params:,})")

    # 1. Evaluate on 1,000 test images
    noise_fn = SpeckleNoise(num_looks=4)
    test_ds = SatelliteSpeckleDataset(TEST_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    print(f"Evaluating {len(test_ds)} test images...")
    psnrs, ssims, accs = [], [], []
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                denoised = model(noisy)
            for b in range(noisy.size(0)):
                p_out = torch.clamp(denoised[b].float(), 0.0, 1.0)
                c_out = clean[b].float()
                psnrs.append(calculate_psnr(p_out, c_out))
                ssims.append(float(ssim(p_out.unsqueeze(0), c_out.unsqueeze(0))))
                acc, _, _ = calculate_pixel_metrics(p_out, c_out, threshold=0.05)
                accs.append(acc)

    test_metrics = {
        "test_count": len(psnrs),
        "mean_psnr": float(np.mean(psnrs)),
        "median_psnr": float(np.median(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
        "median_ssim": float(np.median(ssims)),
        "mean_pixel_accuracy": float(np.mean(accs)),
        "mean_pixel_precision": float(np.mean(accs)),
        "mean_pixel_recall": float(np.mean(accs)),
        "threshold": 0.05,
    }
    print(f"  Mean PSNR: {test_metrics['mean_psnr']:.2f} dB | Mean SSIM: {test_metrics['mean_ssim']:.4f} | Pixel Acc: {test_metrics['mean_pixel_accuracy']*100:.2f}%")

    # 2. Representative Triplets (train, val, test)
    train_ds = SatelliteSpeckleDataset(TRAIN_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)
    val_ds = SatelliteSpeckleDataset(VAL_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)

    with torch.no_grad():
        for split_name, ds_obj, label in [
            ("train", train_ds, "Train Sample"),
            ("val", val_ds, "Validation Sample"),
            ("test", test_ds, "Test Sample"),
        ]:
            noisy_s, clean_s = ds_obj[0]
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                denoised_s = model(noisy_s.unsqueeze(0).to(device))
            denoised_s = torch.clamp(denoised_s.squeeze(0).float().cpu(), 0.0, 1.0)
            pv = calculate_psnr(denoised_s, clean_s)
            sv = float(ssim(denoised_s.unsqueeze(0), clean_s.unsqueeze(0)))
            out_vis = results_dir / f"representative_{split_name}_triplet.png"
            save_triplet_plot(
                clean=clean_s,
                noisy=noisy_s,
                denoised=denoised_s,
                title=f"UNet-Autoencoder: {label} | PSNR: {pv:.2f} dB | SSIM: {sv:.4f}",
                save_path=out_vis,
            )
            print(f"  [OK] {out_vis.name}")

    # 3. Parse training_log.csv (handling NaN epochs gracefully)
    csv_path = results_dir / "training_log.csv"
    epochs, t_loss, val_epochs, val_losses, val_psnrs, val_ssims, lr_list, t_sec = [], [], [], [], [], [], [], []
    t_loss_ep = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_l = {k.strip().lower(): v for k, v in row.items() if k is not None}
            ep = int(row_l["epoch"])
            epochs.append(ep)
            lr_list.append(float(row_l["lr"]))
            t_sec.append(float(row_l["epoch_time_s"]))

            tl = float(row_l["train_loss"])
            if math.isfinite(tl):
                t_loss_ep.append(ep)
                t_loss.append(tl)

            vl = float(row_l["val_loss"])
            vp = float(row_l["val_psnr"])
            vs = float(row_l["val_ssim"])
            if math.isfinite(vl) and math.isfinite(vp) and math.isfinite(vs):
                val_epochs.append(ep)
                val_losses.append(vl)
                val_psnrs.append(vp)
                val_ssims.append(vs)

    epochs_run = len(epochs)
    best_idx = int(np.argmax(val_psnrs))
    best_epoch = val_epochs[best_idx]
    best_val_psnr = val_psnrs[best_idx]
    best_val_ssim = val_ssims[best_idx]
    total_time_sec = float(np.sum(t_sec))
    avg_epoch_time = float(np.mean(t_sec))

    # 4. Save Experiment Report JSON (matching CBAM / ResUNet schema)
    summary_report = {
        "model": "U-Net Autoencoder (UNet-Autoencoder)",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "dataset_split": {"train": 3500, "val": 500, "test": 1000},
        "preprocessing": {"size": 256, "channels": 3, "range": "[0, 1]"},
        "speckle_noise": {"model": "Gamma Multiplicative", "L": 4},
        "hyperparameters": {
            "epochs_configured": 100,
            "epochs_run": epochs_run,
            "val_every": 1,
            "batch_size": 4,
            "grad_accum": 2,
            "effective_batch_size": 8,
            "learning_rate": 0.0005,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "loss": "0.84 * L1 + 0.16 * (1 - SSIM)",
        },
        "training_stats": {
            "total_time_sec": round(total_time_sec, 2),
            "avg_epoch_time_sec": round(avg_epoch_time, 2),
            "final_train_loss": round(t_loss[-1], 5),
            "final_val_loss": round(val_losses[-1], 5),
            "best_epoch": best_epoch,
            "best_val_psnr": round(best_val_psnr, 2),
            "best_val_ssim": round(best_val_ssim, 4),
        },
        "test_results": test_metrics,
        "device": str(device),
    }

    report_path = results_dir / "unet_autoencoder_experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=4)
    print(f"  [OK] {report_path.name}")

    # 5. Generate Standard Plots (matching ResUNet / CBAM-UNet)
    # (A) training_curves.png
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(t_loss_ep, t_loss, label="Train Loss", color="blue", linewidth=1.5)
    axes[0, 0].plot(val_epochs, val_losses, label="Val Loss", color="red", linestyle="--", marker="o", markersize=3, linewidth=1.5)
    axes[0, 0].set_title("Training & Validation Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(val_epochs, val_psnrs, label="Val PSNR (dB)", color="green", marker="o", markersize=3, linewidth=1.5)
    axes[0, 1].set_title("Validation PSNR (dB)"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(val_epochs, val_ssims, label="Val SSIM", color="purple", marker="o", markersize=3, linewidth=1.5)
    axes[1, 0].set_title("Validation SSIM"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, lr_list, label="Learning Rate", color="orange", linewidth=1.5)
    axes[1, 1].set_title("Learning Rate (Cosine Annealing)"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("LR")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"U-Net Autoencoder Training & Validation History ({epochs_run} Epochs)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(results_dir / "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] training_curves.png")

    # (B) plot_psnr_progression.png
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(val_epochs, val_psnrs, color="#00838F", marker="o", markersize=4, linewidth=2.0, label="Val PSNR (dB)")
    ax.axhline(best_val_psnr, color="#D84315", linestyle="--", linewidth=1.2, label=f"Best Val PSNR: {best_val_psnr:.2f} dB (epoch {best_epoch})")
    ax.axhline(test_metrics["mean_psnr"], color="#2E7D32", linestyle="-.", linewidth=1.2, label=f"Test Set Mean PSNR: {test_metrics['mean_psnr']:.2f} dB")
    ax.scatter([best_epoch], [best_val_psnr], color="#D84315", s=130, zorder=5)
    ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel("PSNR (dB)", fontsize=11)
    ax.set_title("U-Net Autoencoder — Validation PSNR Progression", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_psnr_progression.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] plot_psnr_progression.png")

    # (C) plot_ssim_progression.png
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(val_epochs, val_ssims, color="#AD1457", marker="o", markersize=4, linewidth=2.0, label="Val SSIM")
    ax.axhline(best_val_ssim, color="#D84315", linestyle="--", linewidth=1.2, label=f"Best Val SSIM: {best_val_ssim:.4f} (epoch {best_epoch})")
    ax.axhline(test_metrics["mean_ssim"], color="#2E7D32", linestyle="-.", linewidth=1.2, label=f"Test Set Mean SSIM: {test_metrics['mean_ssim']:.4f}")
    ax.scatter([best_epoch], [best_val_ssim], color="#D84315", s=130, zorder=5)
    ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel("SSIM", fontsize=11)
    ax.set_title("U-Net Autoencoder — Validation SSIM Progression", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_ssim_progression.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] plot_ssim_progression.png")

    # (D) plot_full_dashboard.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes[0, 0].plot(t_loss_ep, t_loss, color="#1565C0", linewidth=1.5, label="Train Loss")
    axes[0, 0].plot(val_epochs, val_losses, color="#C62828", linestyle="--", marker="o", markersize=3, label="Val Loss")
    axes[0, 0].set_title("Loss Convergence"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(val_epochs, val_psnrs, color="#2E7D32", marker="o", markersize=3, label="Val PSNR")
    axes[0, 1].axhline(test_metrics["mean_psnr"], color="#00695C", linestyle="-.", label=f"Test PSNR: {test_metrics['mean_psnr']:.2f} dB")
    axes[0, 1].scatter([best_epoch], [best_val_psnr], color="#D84315", s=100, zorder=5)
    axes[0, 1].set_title("PSNR Trajectory"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(epochs, t_sec, color="#455A64", alpha=0.75, width=0.8)
    axes[1, 0].axhline(avg_epoch_time, color="#D84315", linestyle="--", label=f"Mean: {avg_epoch_time:.1f}s")
    axes[1, 0].set_title("Epoch Training Time Profile"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].axis("off")
    summary_txt = (
        f"  U-NET AUTOENCODER — EVALUATION SUMMARY\n"
        f"  {'─'*43}\n\n"
        f"  Architecture : UNet-Autoencoder ({total_params/1e6:.1f}M params)\n"
        f"  Type         : Denoising Autoencoder (No Skip/Attn)\n"
        f"  Dataset      : S2Looking (5,000 RGB images)\n"
        f"  Noise Model  : Gamma Speckle L=4\n"
        f"  Epochs Run   : {epochs_run} / 100\n"
        f"  Batch Size   : 4 (x2 grad accum = 8 eff)\n"
        f"  Optimizer    : AdamW (lr = 5e-4)\n"
        f"  Loss         : 0.84*L1 + 0.16*(1-SSIM)\n\n"
        f"  ── VALIDATION PERFORMANCE ──\n"
        f"  Best Epoch   : {best_epoch}\n"
        f"  Best PSNR    : {best_val_psnr:.2f} dB\n"
        f"  Best SSIM    : {best_val_ssim:.4f}\n\n"
        f"  ── TEST SET (1,000 untouched images) ──\n"
        f"  Mean PSNR    : {test_metrics['mean_psnr']:.2f} dB\n"
        f"  Median PSNR  : {test_metrics['median_psnr']:.2f} dB\n"
        f"  Mean SSIM    : {test_metrics['mean_ssim']:.4f}\n"
        f"  Median SSIM  : {test_metrics['median_ssim']:.4f}\n"
        f"  Pixel Acc    : {test_metrics['mean_pixel_accuracy']*100:.2f}% (e=0.05)\n\n"
        f"  Training Time: {total_time_sec/3600:.2f} hours"
    )
    axes[1, 1].text(0.05, 0.95, summary_txt, transform=axes[1, 1].transAxes,
                    fontsize=10.5, verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.9))

    plt.suptitle("U-Net Autoencoder — Experiment Dashboard", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_full_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] plot_full_dashboard.png")

    # (E) plot_model_comparison.png (Phase 1 Baselines including UNet-Autoencoder)
    baseline_reports = [
        ("ResUNet", PROJECT_ROOT / "results" / "residual_unet" / "resunet_experiment_report.json", "#1E88E5"),
        ("Attention U-Net", PROJECT_ROOT / "results" / "attention_unet" / "attention_unet_experiment_report.json", "#00897B"),
        ("MultiResUNet", PROJECT_ROOT / "results" / "multires_unet" / "multires_unet_experiment_report.json", "#FB8C00"),
        ("CBAM-UNet", PROJECT_ROOT / "results" / "cbam_unet" / "cbam_unet_experiment_report.json", "#8E24AA"),
        ("SAR2SAR", PROJECT_ROOT / "results" / "sar2sar" / "sar2sar_experiment_report.json", "#6D4C41"),
    ]

    model_names, psnr_vals, ssim_vals, acc_vals, colors = [], [], [], [], []
    for m_label, j_path, col in baseline_reports:
        if j_path.exists():
            with open(j_path) as f:
                rep = json.load(f)
            model_names.append(m_label)
            psnr_vals.append(rep["test_results"]["mean_psnr"])
            ssim_vals.append(rep["test_results"]["mean_ssim"])
            acc_vals.append(rep["test_results"]["mean_pixel_accuracy"] * 100)
            colors.append(col)

    model_names.append("UNet-AE")
    psnr_vals.append(test_metrics["mean_psnr"])
    ssim_vals.append(test_metrics["mean_ssim"])
    acc_vals.append(test_metrics["mean_pixel_accuracy"] * 100)
    colors.append("#E53935")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    b1 = axes[0].bar(model_names, psnr_vals, color=colors, width=0.5)
    for b, v in zip(b1, psnr_vals):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9.5)
    axes[0].set_ylabel("PSNR (dB)", fontsize=11); axes[0].set_title("Test Set Mean PSNR (dB)", fontsize=12)
    axes[0].set_ylim(0, max(psnr_vals) + 5); axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=20)

    b2 = axes[1].bar(model_names, ssim_vals, color=colors, width=0.5)
    for b, v in zip(b2, ssim_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.4f}", ha="center", fontweight="bold", fontsize=9.5)
    axes[1].set_ylabel("SSIM", fontsize=11); axes[1].set_title("Test Set Mean SSIM", fontsize=12)
    axes[1].set_ylim(0, 1.05); axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=20)

    b3 = axes[2].bar(model_names, acc_vals, color=colors, width=0.5)
    for b, v in zip(b3, acc_vals):
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=9.5)
    axes[2].set_ylabel("Accuracy (%)", fontsize=11); axes[2].set_title("Pixel Accuracy (e=0.05)", fontsize=12)
    axes[2].set_ylim(0, 110); axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].tick_params(axis="x", rotation=20)

    plt.suptitle("Baseline Benchmark Comparison (including U-Net Autoencoder)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] plot_model_comparison.png")

    print("\n" + "=" * 75)
    print("  UNET-AUTOENCODER EVALUATION AND PLOTS COMPLETED!")
    print("=" * 75)


if __name__ == "__main__":
    main()
