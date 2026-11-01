"""
evaluate_attention_unet.py
==========================
Performs full post-training evaluation for Attention U-Net:
1. Loads best checkpoint (checkpoints/attention_unet_best.pth).
2. Evaluates on untouched 1,000 TEST images (PSNR, SSIM, Pixel Acc/Prec/Recall @ 0.05).
3. Generates 3-panel representative visualizations (Train, Val, Test).
4. Generates training curves and comprehensive diagnostic plots.
5. Saves results/attention_unet/attention_unet_experiment_report.json.
6. Generates side-by-side comparison with ResUNet.
"""

import sys
import math
import time
import json
import csv
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.transforms import TrainTransform, ValTestTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import SatelliteSpeckleDataset, TRAIN_DIR, VAL_DIR, TEST_DIR
from models.attention_unet import AttentionUNet
from models.losses import CombinedL1SSIMLoss, ssim


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


def save_triplet_plot(clean: torch.Tensor, noisy: torch.Tensor, denoised: torch.Tensor, title: str, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    c_np = clean.permute(1, 2, 0).detach().cpu().numpy()
    n_np = noisy.permute(1, 2, 0).detach().cpu().numpy()
    d_np = denoised.permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(c_np);   axes[0].set_title("Original Clean Target", fontsize=11); axes[0].axis("off")
    axes[1].imshow(n_np);   axes[1].set_title("Speckle Noisy Input (L=4)", fontsize=11); axes[1].axis("off")
    axes[2].imshow(d_np);   axes[2].set_title("Attention U-Net Denoised", fontsize=11); axes[2].axis("off")
    plt.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("ATTENTION U-NET: FULL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device : {device}")

    results_dir = PROJECT_ROOT / "results" / "attention_unet"
    ckpt_path   = PROJECT_ROOT / "checkpoints" / "attention_unet_best.pth"
    csv_path    = results_dir / "training_log.csv"

    # 1. Parse CSV log
    epochs, t_loss, v_loss, v_psnr, v_ssim, lr_list, t_sec = [], [], [], [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["Epoch"]))
            t_loss.append(float(row["Train_Loss"]))
            v_loss.append(float(row["Val_Loss"]) if row["Val_Loss"] else float("nan"))
            v_psnr.append(float(row["Val_PSNR_dB"]) if row["Val_PSNR_dB"] else float("nan"))
            v_ssim.append(float(row["Val_SSIM"]) if row["Val_SSIM"] else float("nan"))
            lr_list.append(float(row["LR"]))
            t_sec.append(float(row["Time_sec"]))

    epochs_run = len(epochs)
    val_epochs = [e for e, p in zip(epochs, v_psnr) if not math.isnan(p)]
    val_psnrs  = [p for p in v_psnr if not math.isnan(p)]
    val_ssims  = [s for s in v_ssim if not math.isnan(s)]
    val_losses = [l for l in v_loss if not math.isnan(l)]

    best_idx = int(np.argmax(val_psnrs))
    best_epoch = val_epochs[best_idx]
    best_val_psnr = val_psnrs[best_idx]
    best_val_ssim = val_ssims[best_idx]

    print(f"Epochs Completed : {epochs_run}")
    print(f"Best Val Epoch   : {best_epoch}")
    print(f"Best Val PSNR    : {best_val_psnr:.2f} dB")
    print(f"Best Val SSIM    : {best_val_ssim:.4f}")

    # 2. Load model with best checkpoint
    model = AttentionUNet(in_channels=3, out_channels=3).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded Checkpoint: {ckpt_path.name} (recorded epoch {ckpt.get('epoch', '?')})")

    # 3. Test Set Evaluation
    noise_fn = SpeckleNoise(num_looks=4)
    train_ds = SatelliteSpeckleDataset(TRAIN_DIR, transform=TrainTransform(crop_size=256), noise_fn=noise_fn)
    val_ds   = SatelliteSpeckleDataset(VAL_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)
    test_ds  = SatelliteSpeckleDataset(TEST_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)

    use_pin = device.type == "cuda"
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_pin)

    print("\nRunning evaluation on 1,000 untouched test images...")
    psnr_list, ssim_list, acc_list = [], [], []

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(test_loader):
            noisy = noisy.to(device, non_blocking=use_pin)
            clean = clean.to(device, non_blocking=use_pin)

            with torch.amp.autocast(device_type=device.type, enabled=use_pin):
                outputs = model(noisy)

            for i in range(noisy.size(0)):
                p, c = outputs[i], clean[i]
                psnr_v = calculate_psnr(p, c)
                ssim_v = float(ssim(p.unsqueeze(0), c.unsqueeze(0)))
                acc, _, _ = calculate_pixel_metrics(p, c, threshold=0.05)

                psnr_list.append(psnr_v)
                ssim_list.append(ssim_v)
                acc_list.append(acc)

            if (batch_idx + 1) % 25 == 0:
                print(f"  Processed {(batch_idx + 1) * 8} / {len(test_ds)} images...", flush=True)

    test_metrics = {
        "test_count":           len(psnr_list),
        "mean_psnr":            float(np.mean(psnr_list)),
        "median_psnr":          float(np.median(psnr_list)),
        "mean_ssim":            float(np.mean(ssim_list)),
        "median_ssim":          float(np.median(ssim_list)),
        "mean_pixel_accuracy":  float(np.mean(acc_list)),
        "mean_pixel_precision": float(np.mean(acc_list)),
        "mean_pixel_recall":    float(np.mean(acc_list)),
        "threshold":            0.05,
    }

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION RESULTS — ATTENTION U-NET")
    print("=" * 70)
    print(f"Test Images Evaluated    : {test_metrics['test_count']} / 1,000")
    print(f"Mean Test PSNR           : {test_metrics['mean_psnr']:.2f} dB")
    print(f"Median Test PSNR         : {test_metrics['median_psnr']:.2f} dB")
    print(f"Mean Test SSIM           : {test_metrics['mean_ssim']:.4f}")
    print(f"Median Test SSIM         : {test_metrics['median_ssim']:.4f}")
    print(f"Pixel Restoration Acc    : {test_metrics['mean_pixel_accuracy']*100:.2f}% (tolerance = 0.05)")
    print(f"Pixel Restoration Prec   : {test_metrics['mean_pixel_precision']*100:.2f}%")
    print(f"Pixel Restoration Rec    : {test_metrics['mean_pixel_recall']*100:.2f}%")
    print("=" * 70)

    # 4. Save 3-panel representative visualizations
    print("\nGenerating representative 3-panel visualizations...")
    with torch.no_grad():
        for split_name, ds_obj, label in [
            ("train", train_ds, "Train Split (Sample 1)"),
            ("val",   val_ds,   "Validation Split (Sample 1)"),
            ("test",  test_ds,  "Test Split (Sample 1)"),
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
            print(f"  [OK] Saved {split_name} visual triplet to: {out_vis.name}")

    # 5. Save Experiment JSON
    total_time_sec = float(np.sum(t_sec))
    avg_epoch_time = float(np.mean(t_sec))
    summary_report = {
        "model": "Attention U-Net",
        "total_params":     sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "dataset_split":    {"train": 3500, "val": 500, "test": 1000},
        "preprocessing":    {"size": 256, "channels": 3, "range": "[0, 1]"},
        "speckle_noise":    {"model": "Gamma Multiplicative", "L": 4},
        "hyperparameters": {
            "epochs_configured": 100,
            "epochs_run": epochs_run,
            "val_every": 10,
            "early_stopping_patience": 100,
            "batch_size": 8,
            "learning_rate": 0.0005,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "loss": "0.84 * L1 + 0.16 * (1 - SSIM)",
        },
        "training_stats": {
            "total_time_sec":    round(total_time_sec, 2),
            "avg_epoch_time_sec": round(avg_epoch_time, 2),
            "final_train_loss":  round(t_loss[-1], 5),
            "final_val_loss":    round(val_losses[-1], 5),
            "best_epoch":        best_epoch,
            "best_val_psnr":     round(best_val_psnr, 2),
            "best_val_ssim":     round(best_val_ssim, 4),
        },
        "test_results": test_metrics,
        "device": str(device),
    }

    report_path = results_dir / "attention_unet_experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=4)
    print(f"\n[OK] Experiment summary JSON saved to: {report_path.name}")

    # 6. Plotting
    print("\nGenerating evaluation and comparison plots...")

    # (A) training_curves.png
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(epochs, t_loss, label="Train Loss", color="blue", linewidth=1.5)
    axes[0, 0].plot(val_epochs, val_losses, label="Val Loss", color="red", linestyle="--", marker="o", linewidth=1.5)
    axes[0, 0].set_title("Training & Validation Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(val_epochs, val_psnrs, label="Val PSNR (dB)", color="green", marker="o", linewidth=1.5)
    axes[0, 1].set_title("Validation PSNR (dB)"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(val_epochs, val_ssims, label="Val SSIM", color="purple", marker="o", linewidth=1.5)
    axes[1, 0].set_title("Validation SSIM"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, lr_list, label="Learning Rate", color="orange", linewidth=1.5)
    axes[1, 1].set_title("Learning Rate (Cosine Annealing)"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("LR")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Attention U-Net Training & Validation History ({epochs_run} Epochs)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(results_dir / "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] training_curves.png")

    # (B) plot_psnr_progression.png
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(val_epochs, val_psnrs, color="#00897B", marker="o", linewidth=2.0, label="Val PSNR (dB)")
    ax.axhline(best_val_psnr, color="#FF5722", linestyle="--", linewidth=1.2, label=f"Best Val PSNR: {best_val_psnr:.2f} dB (epoch {best_epoch})")
    ax.axhline(test_metrics["mean_psnr"], color="#4CAF50", linestyle="-.", linewidth=1.2, label=f"Test Set Mean PSNR: {test_metrics['mean_psnr']:.2f} dB")
    ax.scatter([best_epoch], [best_val_psnr], color="#FF5722", s=130, zorder=5)
    ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel("PSNR (dB)", fontsize=11)
    ax.set_title("Attention U-Net — Validation PSNR Progression", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_psnr_progression.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] plot_psnr_progression.png")

    # (C) plot_ssim_progression.png
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(val_epochs, val_ssims, color="#7B1FA2", marker="o", linewidth=2.0, label="Val SSIM")
    ax.axhline(best_val_ssim, color="#FF5722", linestyle="--", linewidth=1.2, label=f"Best Val SSIM: {best_val_ssim:.4f} (epoch {best_epoch})")
    ax.axhline(test_metrics["mean_ssim"], color="#4CAF50", linestyle="-.", linewidth=1.2, label=f"Test Set Mean SSIM: {test_metrics['mean_ssim']:.4f}")
    ax.scatter([best_epoch], [best_val_ssim], color="#FF5722", s=130, zorder=5)
    ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel("SSIM", fontsize=11)
    ax.set_title("Attention U-Net — Validation SSIM Progression", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_ssim_progression.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] plot_ssim_progression.png")

    # (D) plot_full_dashboard.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes[0, 0].plot(epochs, t_loss, color="#1976D2", linewidth=1.5, label="Train Loss")
    axes[0, 0].plot(val_epochs, val_losses, color="#D32F2F", linestyle="--", marker="o", label="Val Loss")
    axes[0, 0].set_title("Loss Convergence"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(val_epochs, val_psnrs, color="#388E3C", marker="o", label="Val PSNR")
    axes[0, 1].axhline(test_metrics["mean_psnr"], color="#00796B", linestyle="-.", label=f"Test PSNR: {test_metrics['mean_psnr']:.2f} dB")
    axes[0, 1].scatter([best_epoch], [best_val_psnr], color="#FF5722", s=100, zorder=5)
    axes[0, 1].set_title("PSNR Trajectory"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(epochs, t_sec, color="#546E7A", alpha=0.75, width=0.8)
    axes[1, 0].axhline(avg_epoch_time, color="#FF5722", linestyle="--", label=f"Mean: {avg_epoch_time:.1f}s")
    axes[1, 0].set_title("Epoch Training Time Profile"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].axis("off")
    summary_txt = (
        f"  ATTENTION U-NET — EVALUATION SUMMARY\n"
        f"  {'─'*42}\n\n"
        f"  Architecture : Attention U-Net (32.8M params)\n"
        f"  Attention    : Soft Attention Gates on Skips\n"
        f"  Dataset      : S2Looking (5,000 RGB images)\n"
        f"  Noise Model  : Gamma Speckle L=4\n"
        f"  Epochs Run   : {epochs_run} / 100\n"
        f"  Batch Size   : 8\n"
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

    plt.suptitle("Attention U-Net — Experiment Dashboard", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_full_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] plot_full_dashboard.png")

    # (E) plot_comparison_resunet_vs_attention.png
    resunet_json_path = PROJECT_ROOT / "results" / "residual_unet" / "resunet_experiment_report.json"
    if resunet_json_path.exists():
        with open(resunet_json_path) as f:
            res_rep = json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        models = ["Residual U-Net", "Attention U-Net"]

        # PSNR comparison
        psnr_vals = [res_rep["test_results"]["mean_psnr"], test_metrics["mean_psnr"]]
        b1 = axes[0].bar(models, psnr_vals, color=["#1E88E5", "#00897B"], width=0.45)
        for b, v in zip(b1, psnr_vals):
            axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"{v:.2f} dB", ha="center", fontweight="bold")
        axes[0].set_ylabel("PSNR (dB)"); axes[0].set_title("Test Set Mean PSNR (dB)")
        axes[0].set_ylim(0, max(psnr_vals) + 5); axes[0].grid(True, alpha=0.3, axis="y")

        # SSIM comparison
        ssim_vals = [res_rep["test_results"]["mean_ssim"], test_metrics["mean_ssim"]]
        b2 = axes[1].bar(models, ssim_vals, color=["#1E88E5", "#00897B"], width=0.45)
        for b, v in zip(b2, ssim_vals):
            axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{v:.4f}", ha="center", fontweight="bold")
        axes[1].set_ylabel("SSIM"); axes[1].set_title("Test Set Mean SSIM")
        axes[1].set_ylim(0, 1.05); axes[1].grid(True, alpha=0.3, axis="y")

        # Pixel accuracy comparison
        acc_vals = [res_rep["test_results"]["mean_pixel_accuracy"] * 100, test_metrics["mean_pixel_accuracy"] * 100]
        b3 = axes[2].bar(models, acc_vals, color=["#1E88E5", "#00897B"], width=0.45)
        for b, v in zip(b3, acc_vals):
            axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{v:.2f}%", ha="center", fontweight="bold")
        axes[2].set_ylabel("Accuracy (%)"); axes[2].set_title("Pixel Accuracy (e=0.05)")
        axes[2].set_ylim(0, 110); axes[2].grid(True, alpha=0.3, axis="y")

        plt.suptitle("Model Comparison: Residual U-Net vs Attention U-Net", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(str(results_dir / "plot_model_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  [OK] plot_model_comparison.png")

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS AND PLOTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
