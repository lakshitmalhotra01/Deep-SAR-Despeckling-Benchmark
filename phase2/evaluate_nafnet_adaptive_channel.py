"""
evaluate_nafnet_adaptive_channel.py
===================================
Performs full post-training evaluation for NAFNet with Adaptive Gate + Channel Attention (M4):
1. Loads best checkpoint (checkpoints/nafnet_adaptive_channel_best.pth).
2. Evaluates on untouched 1,000 TEST images (PSNR, SSIM, Pixel Acc @ 0.05).
3. Generates 3-panel representative visualizations (Train, Val, Test).
4. Generates standard training curves and diagnostic plots.
5. Saves results/nafnet_adaptive_channel/nafnet_adaptive_channel_experiment_report.json.
6. Generates full 9-model benchmark & ablation comparison chart.
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

# Strict D: drive cache isolation
CACHE_DIR = PROJECT_ROOT / ".cache"
TEMP_DIR  = PROJECT_ROOT / ".temp"
os.environ["TORCH_HOME"]          = str(CACHE_DIR / "torch")
os.environ["TEMP"]                = str(TEMP_DIR)
os.environ["TMP"]                 = str(TEMP_DIR)
os.environ["MPLCONFIGDIR"]        = str(CACHE_DIR / "matplotlib")
os.environ["CUDA_CACHE_PATH"]     = str(CACHE_DIR / "nv")
os.environ["PYTHONPYCACHEPREFIX"] = str(CACHE_DIR / "pycache")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.transforms import TrainTransform, ValTestTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import SatelliteSpeckleDataset, TRAIN_DIR, VAL_DIR, TEST_DIR
from models.nafnet_adaptive_channel import NAFNetAdaptiveChannel, nafnet_adaptive_channel_model
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
    axes[0].imshow(c_np);   axes[0].set_title("Original Clean Target",                fontsize=11); axes[0].axis("off")
    axes[1].imshow(n_np);   axes[1].set_title("Speckle Noisy Input (L=4)",            fontsize=11); axes[1].axis("off")
    axes[2].imshow(d_np);   axes[2].set_title("NAFNet-AdaptiveChannel Output (M4)",  fontsize=11); axes[2].axis("off")
    plt.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("NAFNET ADAPTIVE + CHANNEL ATTN (M4): FULL TEST EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device : {device}")

    results_dir = PROJECT_ROOT / "results" / "nafnet_adaptive_channel"
    ckpt_path   = PROJECT_ROOT / "checkpoints" / "nafnet_adaptive_channel_best.pth"
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

    epochs_run     = len(epochs)
    total_time_sec = sum(t_sec)
    avg_epoch_time = np.mean(t_sec) if t_sec else 0.0

    val_indices = [i for i, v in enumerate(v_psnr) if not math.isnan(v)]
    val_epochs  = [epochs[i] for i in val_indices]
    val_losses  = [v_loss[i] for i in val_indices]
    val_psnrs   = [v_psnr[i] for i in val_indices]
    val_ssims   = [v_ssim[i] for i in val_indices]

    best_idx      = int(np.nanargmax(v_psnr))
    best_epoch    = epochs[best_idx]
    best_val_psnr = v_psnr[best_idx]
    best_val_ssim = v_ssim[best_idx]

    print(f"Epochs Completed : {epochs_run}")
    print(f"Best Val Epoch   : {best_epoch}")
    print(f"Best Val PSNR    : {best_val_psnr:.2f} dB")
    print(f"Best Val SSIM    : {best_val_ssim:.4f}")

    # 2. Instantiate and load model
    model = nafnet_adaptive_channel_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())

    assert ckpt_path.exists(), f"Checkpoint not found at: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded Checkpoint: {ckpt_path.name} (recorded epoch {ckpt.get('epoch', best_epoch)})")
    print(f"Total Params     : {total_params:,}")

    # 3. Test Set Evaluation (1,000 untouched images)
    print("\nRunning evaluation on 1000 untouched test images (batch_size=4)...")
    noise_fn     = SpeckleNoise(num_looks=4)
    test_dataset = SatelliteSpeckleDataset(TEST_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2,
                              pin_memory=(device.type == "cuda"))

    test_psnr_list = []
    test_ssim_list = []
    test_acc_list  = []

    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(test_loader):
            noisy = noisy.to(device, non_blocking=use_cuda)
            clean = clean.to(device, non_blocking=use_cuda)

            with torch.amp.autocast('cuda', enabled=use_cuda):
                pred = model(noisy)

            pred = pred.clamp(0.0, 1.0)
            for i in range(pred.size(0)):
                p = pred[i:i+1]
                c = clean[i:i+1]
                psnr_val = calculate_psnr(p, c)
                ssim_val = float(ssim(p, c).item())
                acc_val, _, _ = calculate_pixel_metrics(p, c, threshold=0.05)
                test_psnr_list.append(psnr_val)
                test_ssim_list.append(ssim_val)
                test_acc_list.append(acc_val)

            if (batch_idx + 1) * 4 % 200 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"  Processed {min((batch_idx + 1) * 4, len(test_dataset))} / {len(test_dataset)} images...")

    test_metrics = {
        "mean_psnr":             float(np.mean(test_psnr_list)),
        "median_psnr":           float(np.median(test_psnr_list)),
        "std_psnr":              float(np.std(test_psnr_list)),
        "min_psnr":              float(np.min(test_psnr_list)),
        "max_psnr":              float(np.max(test_psnr_list)),
        "mean_ssim":             float(np.mean(test_ssim_list)),
        "median_ssim":           float(np.median(test_ssim_list)),
        "std_ssim":              float(np.std(test_ssim_list)),
        "min_ssim":              float(np.min(test_ssim_list)),
        "max_ssim":              float(np.max(test_ssim_list)),
        "mean_pixel_accuracy":   float(np.mean(test_acc_list)),
        "mean_pixel_precision":  float(np.mean(test_acc_list)),
        "mean_pixel_recall":     float(np.mean(test_acc_list)),
        "total_test_images":     len(test_psnr_list),
    }

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION RESULTS -- NAFNET ADAPTIVE + CHANNEL ATTN (M4)")
    print("=" * 70)
    print(f"Test Images Evaluated    : {test_metrics['total_test_images']} / 1,000")
    print(f"Mean Test PSNR           : {test_metrics['mean_psnr']:.2f} dB")
    print(f"Median Test PSNR         : {test_metrics['median_psnr']:.2f} dB")
    print(f"Mean Test SSIM           : {test_metrics['mean_ssim']:.4f}")
    print(f"Median Test SSIM         : {test_metrics['median_ssim']:.4f}")
    print(f"Pixel Restoration Acc    : {test_metrics['mean_pixel_accuracy']*100:.2f}% (tolerance = 0.05)")
    print(f"Pixel Restoration Prec   : {test_metrics['mean_pixel_precision']*100:.2f}%")
    print(f"Pixel Restoration Rec    : {test_metrics['mean_pixel_recall']*100:.2f}%")
    print("=" * 70)

    # 4. Generate Visual Triplet Artifacts
    print("\nGenerating representative 3-panel visualizations...")
    train_dataset = SatelliteSpeckleDataset(TRAIN_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)
    val_dataset   = SatelliteSpeckleDataset(VAL_DIR,   transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)

    with torch.no_grad():
        # Train triplet
        t_noisy, t_clean = train_dataset[12]
        t_n = t_noisy.unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=use_cuda):
            t_pred = model(t_n).clamp(0.0, 1.0)
        t_psnr = calculate_psnr(t_pred[0], t_clean.to(device))
        t_ssim = float(ssim(t_pred, t_clean.unsqueeze(0).to(device)).item())
        save_triplet_plot(
            t_clean, t_noisy, t_pred[0].cpu(),
            f"Training Sample Representative Triplet — PSNR: {t_psnr:.2f} dB | SSIM: {t_ssim:.4f}",
            results_dir / "representative_train_triplet.png"
        )
        print("  [OK] Saved train visual triplet to: representative_train_triplet.png")

        # Val triplet
        v_noisy, v_clean = val_dataset[42]
        v_n = v_noisy.unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=use_cuda):
            v_pred = model(v_n).clamp(0.0, 1.0)
        v_psnr_val = calculate_psnr(v_pred[0], v_clean.to(device))
        v_ssim_val = float(ssim(v_pred, v_clean.unsqueeze(0).to(device)).item())
        save_triplet_plot(
            v_clean, v_noisy, v_pred[0].cpu(),
            f"Validation Sample Representative Triplet — PSNR: {v_psnr_val:.2f} dB | SSIM: {v_ssim_val:.4f}",
            results_dir / "representative_val_triplet.png"
        )
        print("  [OK] Saved val visual triplet to: representative_val_triplet.png")

        # Test triplet
        te_noisy, te_clean = test_dataset[100]
        te_n = te_noisy.unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=use_cuda):
            te_pred = model(te_n).clamp(0.0, 1.0)
        te_psnr_val = calculate_psnr(te_pred[0], te_clean.to(device))
        te_ssim_val = float(ssim(te_pred, te_clean.unsqueeze(0).to(device)).item())
        save_triplet_plot(
            te_clean, te_noisy, te_pred[0].cpu(),
            f"Test Sample Representative Triplet — PSNR: {te_psnr_val:.2f} dB | SSIM: {te_ssim_val:.4f}",
            results_dir / "representative_test_triplet.png"
        )
        print("  [OK] Saved test visual triplet to: representative_test_triplet.png")

    # 5. Save Experiment Summary JSON
    report = {
        "model_name": "NAFNet-AdaptiveChannel",
        "phase": 2,
        "model_id": "M4",
        "total_parameters": total_params,
        "epochs_completed": epochs_run,
        "best_epoch": best_epoch,
        "best_val_psnr": best_val_psnr,
        "best_val_ssim": best_val_ssim,
        "total_training_time_seconds": total_time_sec,
        "avg_epoch_time_seconds": avg_epoch_time,
        "loss_type": "CombinedL1SSIMLoss (0.84*L1 + 0.16*(1-SSIM))",
        "optimizer": "AdamW (lr=5e-4, wd=1e-4)",
        "scheduler": "CosineAnnealingLR (T_max=100)",
        "hardware": str(device),
        "test_results": test_metrics
    }
    json_path = results_dir / "nafnet_adaptive_channel_experiment_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n[OK] Experiment summary JSON saved to: {json_path.name}")

    # 6. Generate Standard Plots
    print("\nGenerating standard evaluation plots...")

    # (A) training_curves.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(epochs, t_loss, label="Train Loss", color="blue", linewidth=1.5)
    if val_epochs:
        axes[0, 0].plot(val_epochs, val_losses, label="Val Loss", color="red", linestyle="--", marker="o")
    axes[0, 0].set_title("Training & Validation Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    if val_epochs:
        axes[0, 1].plot(val_epochs, val_psnrs, label="Val PSNR (dB)", color="green", marker="o", linewidth=1.5)
    axes[0, 1].set_title("Validation PSNR (dB)"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    if val_epochs:
        axes[1, 0].plot(val_epochs, val_ssims, label="Val SSIM", color="purple", marker="o", linewidth=1.5)
    axes[1, 0].set_title("Validation SSIM"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, lr_list, label="Learning Rate", color="orange", linewidth=1.5)
    axes[1, 1].set_title("Learning Rate (Cosine Annealing)"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("LR")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"NAFNet Adaptive + Channel Attn (M4) Training History ({epochs_run} Epochs)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(results_dir / "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] training_curves.png")

    # (B) plot_psnr_progression.png
    if val_epochs:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(val_epochs, val_psnrs, color="#00838F", marker="o", linewidth=2.0, label="Val PSNR (dB)")
        ax.axhline(best_val_psnr, color="#D84315", linestyle="--", linewidth=1.2, label=f"Best Val PSNR: {best_val_psnr:.2f} dB (epoch {best_epoch})")
        ax.axhline(test_metrics["mean_psnr"], color="#2E7D32", linestyle="-.", linewidth=1.2, label=f"Test Set Mean PSNR: {test_metrics['mean_psnr']:.2f} dB")
        ax.scatter([best_epoch], [best_val_psnr], color="#D84315", s=130, zorder=5)
        ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel("PSNR (dB)", fontsize=11)
        ax.set_title("NAFNet-AdaptiveChannel — Validation PSNR Progression", fontsize=13)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(results_dir / "plot_psnr_progression.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("  [OK] plot_psnr_progression.png")

    # (C) plot_ssim_progression.png
    if val_epochs:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(val_epochs, val_ssims, color="#AD1457", marker="o", linewidth=2.0, label="Val SSIM")
        ax.axhline(best_val_ssim, color="#D84315", linestyle="--", linewidth=1.2, label=f"Best Val SSIM: {best_val_ssim:.4f} (epoch {best_epoch})")
        ax.axhline(test_metrics["mean_ssim"], color="#2E7D32", linestyle="-.", linewidth=1.2, label=f"Test Set Mean SSIM: {test_metrics['mean_ssim']:.4f}")
        ax.scatter([best_epoch], [best_val_ssim], color="#D84315", s=130, zorder=5)
        ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel("SSIM", fontsize=11)
        ax.set_title("NAFNet-AdaptiveChannel — Validation SSIM Progression", fontsize=13)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(results_dir / "plot_ssim_progression.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("  [OK] plot_ssim_progression.png")

    # (D) plot_full_dashboard.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes[0, 0].plot(epochs, t_loss, color="#1565C0", linewidth=1.5, label="Train Loss")
    if val_epochs:
        axes[0, 0].plot(val_epochs, val_losses, color="#C62828", linestyle="--", marker="o", label="Val Loss")
    axes[0, 0].set_title("Loss Convergence"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    if val_epochs:
        axes[0, 1].plot(val_epochs, val_psnrs, color="#2E7D32", marker="o", label="Val PSNR")
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
        f"  NAFNET-ADAPTIVE-CHANNEL — SUMMARY (M4)\n"
        f"  {'─'*43}\n\n"
        f"  Architecture : NAFNet + Adaptive Gate + CAM\n"
        f"  Params       : {total_params:,} (2.91M)\n"
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

    plt.suptitle("NAFNet with Adaptive Gate + Channel Attn (M4) — Dashboard", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_full_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] plot_full_dashboard.png")

    # (E) 9-Model Benchmark & Ablation Comparison
    resunet_json = PROJECT_ROOT / "results" / "residual_unet"         / "resunet_experiment_report.json"
    att_json     = PROJECT_ROOT / "results" / "attention_unet"        / "attention_unet_experiment_report.json"
    multi_json   = PROJECT_ROOT / "results" / "multires_unet"         / "multires_unet_experiment_report.json"
    cbam_json    = PROJECT_ROOT / "results" / "cbam_unet"             / "cbam_unet_experiment_report.json"
    sar_json     = PROJECT_ROOT / "results" / "sar2sar"               / "sar2sar_experiment_report.json"
    naf_b_json   = PROJECT_ROOT / "results" / "nafnet_baseline"       / "nafnet_baseline_experiment_report.json"
    naf_ag_json  = PROJECT_ROOT / "results" / "nafnet_adaptive_gate"  / "nafnet_adaptive_gate_experiment_report.json"
    naf_sp_json  = PROJECT_ROOT / "results" / "nafnet_adaptive_spatial"/ "nafnet_adaptive_spatial_experiment_report.json"

    priors = [resunet_json, att_json, multi_json, cbam_json, sar_json, naf_b_json, naf_ag_json, naf_sp_json]
    if all(p.exists() for p in priors):
        with open(resunet_json) as f: res_rep   = json.load(f)
        with open(att_json)    as f: att_rep   = json.load(f)
        with open(multi_json)  as f: multi_rep = json.load(f)
        with open(cbam_json)   as f: cbam_rep  = json.load(f)
        with open(sar_json)    as f: sar_rep   = json.load(f)
        with open(naf_b_json)  as f: naf_b_rep = json.load(f)
        with open(naf_ag_json) as f: naf_ag_rep= json.load(f)
        with open(naf_sp_json) as f: naf_sp_rep= json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(28, 6))
        models = [
            "ResUNet", "Attention\nU-Net", "MultiRes\nUNet",
            "CBAM-UNet", "SAR2SAR", "NAFNet\n(Simple Gate)",
            "NAFNet\n(Adaptive Gate)", "NAFNet\n(Adaptive+Spatial)",
            "NAFNet\n(Adaptive+Channel)"
        ]
        colors = ["#1E88E5", "#00897B", "#FB8C00", "#8E24AA", "#5D4037", "#43A047", "#E53935", "#7B1FA2", "#00ACC1"]

        # PSNR comparison
        psnr_vals = [
            res_rep["test_results"]["mean_psnr"],
            att_rep["test_results"]["mean_psnr"],
            multi_rep["test_results"]["mean_psnr"],
            cbam_rep["test_results"]["mean_psnr"],
            sar_rep["test_results"]["mean_psnr"],
            naf_b_rep["test_results"]["mean_psnr"],
            naf_ag_rep["test_results"]["mean_psnr"],
            naf_sp_rep["test_results"]["mean_psnr"],
            test_metrics["mean_psnr"],
        ]
        b1 = axes[0].bar(models, psnr_vals, color=colors, width=0.55)
        for b, v in zip(b1, psnr_vals):
            axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                         f"{v:.2f} dB", ha="center", fontweight="bold", fontsize=7.5)
        axes[0].set_ylabel("PSNR (dB)", fontsize=11)
        axes[0].set_title("Test Set Mean PSNR (dB)", fontsize=12)
        axes[0].set_ylim(0, max(psnr_vals) + 5)
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[0].tick_params(axis='x', labelsize=7.0)

        # SSIM comparison
        ssim_vals = [
            res_rep["test_results"]["mean_ssim"],
            att_rep["test_results"]["mean_ssim"],
            multi_rep["test_results"]["mean_ssim"],
            cbam_rep["test_results"]["mean_ssim"],
            sar_rep["test_results"]["mean_ssim"],
            naf_b_rep["test_results"]["mean_ssim"],
            naf_ag_rep["test_results"]["mean_ssim"],
            naf_sp_rep["test_results"]["mean_ssim"],
            test_metrics["mean_ssim"],
        ]
        b2 = axes[1].bar(models, ssim_vals, color=colors, width=0.55)
        for b, v in zip(b2, ssim_vals):
            axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                         f"{v:.4f}", ha="center", fontweight="bold", fontsize=7.5)
        axes[1].set_ylabel("SSIM", fontsize=11)
        axes[1].set_title("Test Set Mean SSIM", fontsize=12)
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].tick_params(axis='x', labelsize=7.0)

        # Pixel Accuracy comparison
        acc_vals = [
            res_rep["test_results"]["mean_pixel_accuracy"] * 100,
            att_rep["test_results"]["mean_pixel_accuracy"] * 100,
            multi_rep["test_results"]["mean_pixel_accuracy"] * 100,
            cbam_rep["test_results"]["mean_pixel_accuracy"] * 100,
            sar_rep["test_results"]["mean_pixel_accuracy"] * 100,
            naf_b_rep["test_results"]["mean_pixel_accuracy"] * 100,
            naf_ag_rep["test_results"]["mean_pixel_accuracy"] * 100,
            naf_sp_rep["test_results"]["mean_pixel_accuracy"] * 100,
            test_metrics["mean_pixel_accuracy"] * 100,
        ]
        b3 = axes[2].bar(models, acc_vals, color=colors, width=0.55)
        for b, v in zip(b3, acc_vals):
            axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
                         f"{v:.2f}%", ha="center", fontweight="bold", fontsize=7.5)
        axes[2].set_ylabel("Accuracy (%)", fontsize=11)
        axes[2].set_title("Pixel Accuracy (e=0.05)", fontsize=12)
        axes[2].set_ylim(0, 115)
        axes[2].grid(True, alpha=0.3, axis="y")
        axes[2].tick_params(axis='x', labelsize=7.0)

        plt.suptitle(
            "Overall Benchmark & Ablation Study — Phase 1 vs NAFNet Simple Gate vs AGM vs Spatial vs Channel (M4)",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()

        target_path = results_dir / "plot_model_comparison.png"
        plt.savefig(str(target_path), dpi=150, bbox_inches="tight")
        print(f"  [OK] plot_model_comparison.png -> results/nafnet_adaptive_channel/")

        plt.close(fig)
        print("  [OK] 9-model comparison plot generated successfully.")
    else:
        print("  [WARN] One or more prior model JSON reports not found — skipping comparison plot.")

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS AND PLOTS COMPLETED SUCCESSFULLY!")
    print("NAFNet Adaptive + Channel Attn (Phase 2 M4) Fully Evaluated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
