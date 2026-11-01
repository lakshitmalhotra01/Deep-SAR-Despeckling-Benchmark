"""
evaluate_nafnet_adaptive_spatial.py
==================================
Performs full post-training evaluation for NAFNet with Adaptive Gate + Spatial Attention (M3):
1. Loads best checkpoint (checkpoints/nafnet_adaptive_spatial_best.pth).
2. Evaluates on untouched 1,000 TEST images (PSNR, SSIM, Pixel Acc @ 0.05).
3. Generates 3-panel representative visualizations (Train, Val, Test).
4. Generates standard training curves and diagnostic plots.
5. Saves results/nafnet_adaptive_spatial/nafnet_adaptive_spatial_experiment_report.json.
6. Generates full 8-model benchmark & ablation comparison chart.
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
from models.nafnet_adaptive_spatial import NAFNetAdaptiveSpatial
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
    axes[2].imshow(d_np);   axes[2].set_title("NAFNet-AdaptiveSpatial Output (M3)",  fontsize=11); axes[2].axis("off")
    plt.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("NAFNET ADAPTIVE + SPATIAL ATTN (M3): FULL TEST EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device : {device}")

    results_dir = PROJECT_ROOT / "results" / "nafnet_adaptive_spatial"
    ckpt_path   = PROJECT_ROOT / "checkpoints" / "nafnet_adaptive_spatial_best.pth"
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
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    base_ch    = config.get("base_ch", 32)
    num_blocks = config.get("num_blocks", [1, 1, 1, 4])
    dw_expand  = config.get("dw_expand", 2)
    ffn_expand = config.get("ffn_expand", 2)

    model = NAFNetAdaptiveSpatial(
        in_channels=3,
        out_channels=3,
        base_ch=base_ch,
        num_blocks=num_blocks,
        dw_expand=dw_expand,
        ffn_expand=ffn_expand
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded Checkpoint: {ckpt_path.name} (recorded epoch {ckpt.get('epoch', '?')})")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params     : {total_params:,}")

    # 3. Test Set Evaluation
    noise_fn = SpeckleNoise(num_looks=4)
    train_ds = SatelliteSpeckleDataset(TRAIN_DIR, transform=TrainTransform(crop_size=256), noise_fn=noise_fn)
    val_ds   = SatelliteSpeckleDataset(VAL_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)
    test_ds  = SatelliteSpeckleDataset(TEST_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)

    use_pin = device.type == "cuda"
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=use_pin)

    print(f"\nRunning evaluation on {len(test_ds)} untouched test images (batch_size=4)...")
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

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {min((batch_idx + 1) * 4, len(test_ds))} / {len(test_ds)} images...", flush=True)

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
    print("FINAL TEST EVALUATION RESULTS — NAFNET ADAPTIVE + SPATIAL ATTN (M3)")
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
            with torch.amp.autocast(device_type=device.type, enabled=use_pin):
                denoised_s = model(noisy_s.unsqueeze(0).to(device)).squeeze(0).cpu()
            pv = calculate_psnr(denoised_s, clean_s)
            sv = float(ssim(denoised_s.unsqueeze(0), clean_s.unsqueeze(0)))
            out_vis = results_dir / f"representative_{split_name}_triplet.png"
            save_triplet_plot(
                clean=clean_s, noisy=noisy_s, denoised=denoised_s,
                title=f"NAFNet-AdaptiveSpatial: {label} | PSNR: {pv:.2f} dB | SSIM: {sv:.4f}",
                save_path=out_vis,
            )
            print(f"  [OK] Saved {split_name} visual triplet to: {out_vis.name}")

    # 5. Save Experiment JSON
    total_time_sec = float(np.sum(t_sec))
    avg_epoch_time = float(np.mean(t_sec))
    summary_report = {
        "model": "NAFNet-AdaptiveSpatial (AGM + SAM — Phase 2 M3)",
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "dataset_split":    {"train": 3500, "val": 500, "test": 1000},
        "preprocessing":    {"size": 256, "channels": 3, "range": "[0, 1]"},
        "speckle_noise":    {"model": "Gamma Multiplicative", "L": 4},
        "hyperparameters": {
            "epochs_configured": 100,
            "epochs_run": epochs_run,
            "val_every": 10,
            "early_stopping_patience": 100,
            "batch_size": 4,
            "grad_accum": 2,
            "effective_batch_size": 8,
            "learning_rate": 0.0005,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "loss": "0.84 * L1 + 0.16 * (1 - SSIM)",
            "base_ch": base_ch,
            "num_blocks": num_blocks,
        },
        "training_stats": {
            "total_time_sec":     round(total_time_sec, 2),
            "avg_epoch_time_sec": round(avg_epoch_time, 2),
            "final_train_loss":   round(t_loss[-1], 5),
            "final_val_loss":     round(val_losses[-1], 5) if val_losses else None,
            "best_epoch":         best_epoch,
            "best_val_psnr":      round(best_val_psnr, 2),
            "best_val_ssim":      round(best_val_ssim, 4),
        },
        "test_results": test_metrics,
        "device": str(device),
    }

    report_path = results_dir / "nafnet_adaptive_spatial_experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=4)
    print(f"\n[OK] Experiment summary JSON saved to: {report_path.name}")

    # 6. Generate Standard Plots
    print("\nGenerating standard evaluation plots...")

    # (A) training_curves.png
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(epochs, t_loss, label="Train Loss", color="blue", linewidth=1.5)
    if val_epochs:
        axes[0, 0].plot(val_epochs, val_losses, label="Val Loss", color="red", linestyle="--", marker="o", linewidth=1.5)
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

    plt.suptitle(f"NAFNet Adaptive + Spatial Attn (M3) Training History ({epochs_run} Epochs)", fontsize=14)
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
        ax.set_title("NAFNet-AdaptiveSpatial — Validation PSNR Progression", fontsize=13)
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
        ax.set_title("NAFNet-AdaptiveSpatial — Validation SSIM Progression", fontsize=13)
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
        f"  NAFNET-ADAPTIVE-SPATIAL — SUMMARY (M3)\n"
        f"  {'─'*43}\n\n"
        f"  Architecture : NAFNet + Adaptive Gate + SAM\n"
        f"  Params       : {total_params:,} (3.06M)\n"
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

    plt.suptitle("NAFNet with Adaptive Gate + Spatial Attn (M3) — Dashboard", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_full_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] plot_full_dashboard.png")

    # (E) 8-Model Benchmark & Ablation Comparison
    resunet_json = PROJECT_ROOT / "results" / "residual_unet"        / "resunet_experiment_report.json"
    att_json     = PROJECT_ROOT / "results" / "attention_unet"       / "attention_unet_experiment_report.json"
    multi_json   = PROJECT_ROOT / "results" / "multires_unet"        / "multires_unet_experiment_report.json"
    cbam_json    = PROJECT_ROOT / "results" / "cbam_unet"            / "cbam_unet_experiment_report.json"
    sar_json     = PROJECT_ROOT / "results" / "sar2sar"              / "sar2sar_experiment_report.json"
    naf_b_json   = PROJECT_ROOT / "results" / "nafnet_baseline"      / "nafnet_baseline_experiment_report.json"
    naf_ag_json  = PROJECT_ROOT / "results" / "nafnet_adaptive_gate" / "nafnet_adaptive_gate_experiment_report.json"

    if all(p.exists() for p in [resunet_json, att_json, multi_json, cbam_json, sar_json, naf_b_json, naf_ag_json]):
        with open(resunet_json) as f: res_rep   = json.load(f)
        with open(att_json)    as f: att_rep   = json.load(f)
        with open(multi_json)  as f: multi_rep = json.load(f)
        with open(cbam_json)   as f: cbam_rep  = json.load(f)
        with open(sar_json)    as f: sar_rep   = json.load(f)
        with open(naf_b_json)  as f: naf_b_rep = json.load(f)
        with open(naf_ag_json) as f: naf_ag_rep= json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(26, 6))
        models = [
            "ResUNet", "Attention\nU-Net", "MultiRes\nUNet",
            "CBAM-UNet", "SAR2SAR", "NAFNet\n(Simple Gate)",
            "NAFNet\n(Adaptive Gate)", "NAFNet\n(Adaptive+Spatial)"
        ]
        colors = ["#1E88E5", "#00897B", "#FB8C00", "#8E24AA", "#5D4037", "#43A047", "#E53935", "#7B1FA2"]

        # PSNR comparison
        psnr_vals = [
            res_rep["test_results"]["mean_psnr"],
            att_rep["test_results"]["mean_psnr"],
            multi_rep["test_results"]["mean_psnr"],
            cbam_rep["test_results"]["mean_psnr"],
            sar_rep["test_results"]["mean_psnr"],
            naf_b_rep["test_results"]["mean_psnr"],
            naf_ag_rep["test_results"]["mean_psnr"],
            test_metrics["mean_psnr"],
        ]
        b1 = axes[0].bar(models, psnr_vals, color=colors, width=0.55)
        for b, v in zip(b1, psnr_vals):
            axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                         f"{v:.2f} dB", ha="center", fontweight="bold", fontsize=8.0)
        axes[0].set_ylabel("PSNR (dB)", fontsize=11)
        axes[0].set_title("Test Set Mean PSNR (dB)", fontsize=12)
        axes[0].set_ylim(0, max(psnr_vals) + 5)
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[0].tick_params(axis='x', labelsize=7.5)

        # SSIM comparison
        ssim_vals = [
            res_rep["test_results"]["mean_ssim"],
            att_rep["test_results"]["mean_ssim"],
            multi_rep["test_results"]["mean_ssim"],
            cbam_rep["test_results"]["mean_ssim"],
            sar_rep["test_results"]["mean_ssim"],
            naf_b_rep["test_results"]["mean_ssim"],
            naf_ag_rep["test_results"]["mean_ssim"],
            test_metrics["mean_ssim"],
        ]
        b2 = axes[1].bar(models, ssim_vals, color=colors, width=0.55)
        for b, v in zip(b2, ssim_vals):
            axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                         f"{v:.4f}", ha="center", fontweight="bold", fontsize=8.0)
        axes[1].set_ylabel("SSIM", fontsize=11)
        axes[1].set_title("Test Set Mean SSIM", fontsize=12)
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].tick_params(axis='x', labelsize=7.5)

        # Pixel Accuracy comparison
        acc_vals = [
            res_rep["test_results"]["mean_pixel_accuracy"] * 100,
            att_rep["test_results"]["mean_pixel_accuracy"] * 100,
            multi_rep["test_results"]["mean_pixel_accuracy"] * 100,
            cbam_rep["test_results"]["mean_pixel_accuracy"] * 100,
            sar_rep["test_results"]["mean_pixel_accuracy"] * 100,
            naf_b_rep["test_results"]["mean_pixel_accuracy"] * 100,
            naf_ag_rep["test_results"]["mean_pixel_accuracy"] * 100,
            test_metrics["mean_pixel_accuracy"] * 100,
        ]
        b3 = axes[2].bar(models, acc_vals, color=colors, width=0.55)
        for b, v in zip(b3, acc_vals):
            axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
                         f"{v:.2f}%", ha="center", fontweight="bold", fontsize=8.0)
        axes[2].set_ylabel("Accuracy (%)", fontsize=11)
        axes[2].set_title("Pixel Accuracy (e=0.05)", fontsize=12)
        axes[2].set_ylim(0, 115)
        axes[2].grid(True, alpha=0.3, axis="y")
        axes[2].tick_params(axis='x', labelsize=7.5)

        plt.suptitle(
            "Overall Benchmark & Ablation Study — Phase 1 vs NAFNet Simple Gate vs Adaptive Gate vs Adaptive+Spatial (M3)",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()

        target_path = results_dir / "plot_model_comparison.png"
        plt.savefig(str(target_path), dpi=150, bbox_inches="tight")
        print(f"  [OK] plot_model_comparison.png -> results/nafnet_adaptive_spatial/")

        plt.close(fig)
        print("  [OK] 8-model comparison plot generated successfully.")
    else:
        print("  [WARN] One or more prior model JSON reports not found — skipping comparison plot.")

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS AND PLOTS COMPLETED SUCCESSFULLY!")
    print("NAFNet Adaptive + Spatial Attn (Phase 2 M3) Fully Evaluated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
