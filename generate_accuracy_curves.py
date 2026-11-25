"""
generate_accuracy_curves.py
===========================
Generates research-paper-ready accuracy and convergence curves for all 11 models
and saves them into results/testing_evaluation/.

Outputs:
  1. training_validation_curves_all_models.png
     - Validation PSNR (dB) vs Epoch (Phase 1 vs Phase 2 + all models)
     - Validation SSIM vs Epoch
     - Training Loss vs Epoch
     - Validation Loss vs Epoch
  2. accuracy_curves_all_models.png
     - Pixel Accuracy (%) vs Error Tolerance Threshold (epsilon) on test set
     - Cumulative PSNR Accuracy Curve (% of test images >= PSNR threshold)
"""

import csv
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.transforms import ValTestTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import SatelliteSpeckleDataset, TEST_DIR
from unified_testing_evaluation import get_model_registry, load_model, calculate_psnr

OUTPUT_DIR = PROJECT_ROOT / "results" / "testing_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping from registry model name -> results/<folder>/training_log.csv
LOG_FOLDER_MAP = {
    "ResUNet": "residual_unet",
    "Attention U-Net": "attention_unet",
    "MultiResUNet": "multires_unet",
    "CBAM U-Net": "cbam_unet",
    "SAR2SAR": "sar2sar",
    "U-Net Autoencoder": "unet_autoencoder",
    "M1: NAFNet Baseline": "nafnet_baseline",
    "M2: NAFNet+AGM": "nafnet_adaptive_gate",
    "M3: NAFNet+AGM+SAM": "nafnet_adaptive_spatial",
    "M4: NAFNet+AGM+CAM": "nafnet_adaptive_channel",
    "M5: AdaptiveDual (Ours)": "nafnet_adaptive_dual",
}

STYLE_MAP = {
    "ResUNet": ("#2E7D32", "--", 1.6),
    "Att-UNet": ("#FB8C00", "--", 1.5),
    "MRUNet": ("#8E24AA", "--", 1.5),
    "CBAM": ("#00897B", "--", 1.5),
    "SAR2SAR": ("#6D4C41", "--", 1.5),
    "UNet-AE": ("#78909C", ":", 1.5),
    "M1": ("#90CAF9", "-", 1.7),
    "M2": ("#42A5F5", "-", 1.8),
    "M3": ("#1E88E5", "-", 1.9),
    "M4": ("#1565C0", "-", 2.0),
    "M5": ("#E53935", "-", 2.6),
}


def load_training_logs() -> Dict[str, Dict[str, Tuple[List[int], List[float]]]]:
    """Load Epoch, Train_Loss, Val_Loss, Val_PSNR_dB, Val_SSIM from each model's training_log.csv."""
    logs = {}
    for model_name, folder in LOG_FOLDER_MAP.items():
        csv_path = PROJECT_ROOT / "results" / folder / "training_log.csv"
        if not csv_path.exists():
            print(f"  [WARN] Missing training log: {csv_path}")
            continue
        data = {
            "train_loss": ([], []),
            "val_loss": ([], []),
            "val_psnr": ([], []),
            "val_ssim": ([], []),
        }
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_l = {k.strip().lower(): v for k, v in row.items() if k is not None}
                ep = int(row_l["epoch"])
                for key, cols in [
                    ("train_loss", ("train_loss",)),
                    ("val_loss", ("val_loss",)),
                    ("val_psnr", ("val_psnr_db", "val_psnr")),
                    ("val_ssim", ("val_ssim",)),
                ]:
                    val_str = ""
                    for col in cols:
                        if col in row_l and row_l[col] is not None:
                            val_str = row_l[col].strip()
                            if val_str != "":
                                break
                    if val_str != "":
                        data[key][0].append(ep)
                        data[key][1].append(float(val_str))
        logs[model_name] = data
    return logs


def plot_training_validation_curves(logs: Dict, registry: List):
    """Plot 2x2 grid of Train Loss, Val Loss, Val PSNR, Val SSIM across all 11 models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short_map = {name: short for name, short, _, _ in registry}

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    panels = [
        (axes[0, 0], "val_psnr", "Validation PSNR (dB)", "Validation PSNR Convergence Across Epochs", False),
        (axes[0, 1], "val_ssim", "Validation SSIM", "Validation SSIM Convergence Across Epochs", False),
        (axes[1, 0], "train_loss", "Training Loss", "Training Loss Curves Across Epochs", True),
        (axes[1, 1], "val_loss", "Validation Loss", "Validation Loss Curves Across Epochs", True),
    ]

    for ax, metric_key, ylabel, title, use_log in panels:
        for name, short, _, _ in registry:
            if name not in logs:
                continue
            eps, vals = logs[name][metric_key]
            if not eps:
                continue
            color, ls, lw = STYLE_MAP.get(short, ("#333333", "-", 1.5))
            zorder = 10 if short == "M5" else 3
            ax.plot(eps, vals, label=short, color=color, linestyle=ls, linewidth=lw, alpha=0.9, zorder=zorder)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if use_log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Single shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Training & Validation Accuracy / Convergence Curves — All 11 Models",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_path = OUTPUT_DIR / "training_validation_curves_all_models.png"
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")


def plot_test_accuracy_curves(registry: List, num_images: int = 250):
    """
    Compute and plot:
      1. Pixel Accuracy (%) vs Error Tolerance Threshold (epsilon)
      2. Cumulative PSNR Accuracy Curve (% of images >= PSNR threshold)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise_fn = SpeckleNoise(num_looks=4)
    test_ds = SatelliteSpeckleDataset(TEST_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)
    indices = random.sample(range(len(test_ds)), min(num_images, len(test_ds)))
    subset = Subset(test_ds, indices)
    loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    epsilons = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10])
    psnr_thresholds = np.linspace(30.0, 44.0, 57)

    results = {}
    for name, short, model, ckpt_path in registry:
        if not ckpt_path.exists():
            continue
        model = load_model(model, ckpt_path, device)
        eps_counts = np.zeros(len(epsilons), dtype=np.float64)
        total_pixels = 0
        psnrs = []

        with torch.no_grad():
            for noisy, clean in loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    out = model(noisy)
                out = torch.clamp(out.float(), 0.0, 1.0)
                abs_diff = torch.abs(out - clean.float())

                total_pixels += abs_diff.numel()
                for idx_e, eps in enumerate(epsilons):
                    eps_counts[idx_e] += float((abs_diff <= eps).sum().item())

                for i in range(noisy.size(0)):
                    psnrs.append(calculate_psnr(out[i], clean[i].float()))

        acc_curve = (eps_counts / total_pixels) * 100.0
        psnr_arr = np.array(psnrs)
        cdf_curve = np.array([(psnr_arr >= t).mean() * 100.0 for t in psnr_thresholds])
        results[short] = (acc_curve, cdf_curve)
        print(f"  [OK] Computed accuracy curves for {short} (Acc@0.02={acc_curve[3]:.2f}%, Acc@0.05={acc_curve[6]:.2f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    for _, short, _, _ in registry:
        if short not in results:
            continue
        acc_curve, cdf_curve = results[short]
        color, ls, lw = STYLE_MAP.get(short, ("#333333", "-", 1.5))
        zorder = 10 if short == "M5" else 3
        axes[0].plot(epsilons, acc_curve, label=short, color=color, linestyle=ls, linewidth=lw, marker='o', markersize=3.5, zorder=zorder)
        axes[1].plot(psnr_thresholds, cdf_curve, label=short, color=color, linestyle=ls, linewidth=lw, zorder=zorder)

    axes[0].axvline(0.05, color="gray", linestyle=":", linewidth=1.2, label="Standard ε=0.05")
    axes[0].set_xlabel("Pixel Error Tolerance Threshold (ε)", fontsize=12)
    axes[0].set_ylabel("Pixel Accuracy (%)", fontsize=12)
    axes[0].set_title("Pixel Accuracy vs Error Tolerance Threshold (ε)", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("PSNR Threshold τ (dB)", fontsize=12)
    axes[1].set_ylabel("Test Images Achieving ≥ τ (%)", fontsize=12)
    axes[1].set_title("Cumulative PSNR Accuracy Curve (Success Rate)", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.04))

    plt.suptitle(f"Test-Set Accuracy Curves — All 11 Models ({len(subset)} Synthetic Test Images)",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out_path = OUTPUT_DIR / "accuracy_curves_all_models.png"
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")


def main():
    print("=" * 70)
    print("  GENERATING TRAINING & TEST ACCURACY CURVES (ALL 11 MODELS)")
    print("=" * 70)
    registry = get_model_registry()
    logs = load_training_logs()
    plot_training_validation_curves(logs, registry)
    plot_test_accuracy_curves(registry, num_images=250)
    print("  ✓ ACCURACY CURVES COMPLETE!")


if __name__ == "__main__":
    main()
