"""
unified_testing_evaluation.py
==============================
Comprehensive testing evaluation of ALL 11 trained models on the synthetic test set.
Generates research-paper-quality visuals in results/testing_evaluation/

Outputs:
  1. Per-model test metrics (PSNR, SSIM, Pixel Accuracy) in JSON + CSV
  2. Side-by-side denoising comparison grids on random test images (all 11 models)
  3. Bar chart comparison of all models (PSNR, SSIM, Pixel Accuracy)
  4. Box plots showing metric distributions across test set
  5. Radar/spider chart for multi-metric comparison
  6. Per-image PSNR distribution histograms
  7. Sorted model ranking table as an image
  8. Visual grid: same image denoised by all models (for paper figure)
"""

import sys
import os
import math
import json
import csv
import random
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# ── Setup paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.transforms import ValTestTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import SatelliteSpeckleDataset, TEST_DIR
from models.losses import ssim

# ── Output directory ───────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "results" / "testing_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fix random seed for reproducibility ────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Device ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY: All 11 trained models
# ══════════════════════════════════════════════════════════════════════════

def get_model_registry():
    """Return ordered list of (name, short_label, model_instance, checkpoint_path)."""
    from models.resunet import ResUNet
    from models.attention_unet import AttentionUNet
    from models.multires_unet import MultiResUNet
    from models.cbam_unet import CBAMUNet
    from models.sar2sar import SAR2SARNet
    from models.nafnet_baseline import NAFNetBaseline
    from models.nafnet_adaptive_gate import NAFNetAdaptiveGate
    from models.nafnet_adaptive_spatial import NAFNetAdaptiveSpatial
    from models.nafnet_adaptive_channel import NAFNetAdaptiveChannel
    from models.nafnet_adaptive_dual import NAFNetAdaptiveDual
    from models.unet_autoencoder import UNetAutoencoder

    ckpt = PROJECT_ROOT / "checkpoints"

    registry = [
        # Phase 1: Baselines
        ("ResUNet",             "ResUNet",    ResUNet(in_channels=3, out_channels=3),              ckpt / "resunet_best.pth"),
        ("Attention U-Net",     "Att-UNet",   AttentionUNet(in_channels=3, out_channels=3),        ckpt / "attention_unet_best.pth"),
        ("MultiResUNet",        "MRUNet",     MultiResUNet(in_channels=3, out_channels=3),         ckpt / "multires_unet_best.pth"),
        ("CBAM U-Net",          "CBAM",       CBAMUNet(in_channels=3, out_channels=3),             ckpt / "cbam_unet_best.pth"),
        ("SAR2SAR",             "SAR2SAR",    SAR2SARNet(in_channels=3, out_channels=3),           ckpt / "sar2sar_best.pth"),
        ("U-Net Autoencoder",   "UNet-AE",    UNetAutoencoder(in_channels=3, out_channels=3),      ckpt / "unet_autoencoder_best.pth"),
        # Phase 2: NAFNet ablations
        ("M1: NAFNet Baseline", "M1",         NAFNetBaseline(),                                    ckpt / "nafnet_baseline_best.pth"),
        ("M2: NAFNet+AGM",      "M2",         NAFNetAdaptiveGate(),                                ckpt / "nafnet_adaptive_gate_best.pth"),
        ("M3: NAFNet+AGM+SAM",  "M3",         NAFNetAdaptiveSpatial(),                             ckpt / "nafnet_adaptive_spatial_best.pth"),
        ("M4: NAFNet+AGM+CAM",  "M4",         NAFNetAdaptiveChannel(),                             ckpt / "nafnet_adaptive_channel_best.pth"),
        ("M5: AdaptiveDual (Ours)", "M5",      NAFNetAdaptiveDual(),                                ckpt / "nafnet_adaptive_dual_best.pth"),
    ]
    return registry


# ══════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = float(torch.mean((pred.float() - target.float()) ** 2))
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(ssim(pred.unsqueeze(0), target.unsqueeze(0)))


def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.05) -> float:
    abs_diff = torch.abs(pred.float() - target.float())
    return float((abs_diff <= threshold).float().mean())


# ══════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════

def load_model(model: nn.Module, ckpt_path: Path, device: torch.device) -> nn.Module:
    """Load checkpoint weights into model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════
# EVALUATE ONE MODEL ON FULL TEST SET
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(model: nn.Module, test_loader: DataLoader, model_name: str) -> Dict:
    """Run model on all test images, return per-image metrics."""
    psnr_list, ssim_list, acc_list = [], [], []
    use_pin = device.type == "cuda"

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(test_loader):
            noisy = noisy.to(device, non_blocking=use_pin)
            clean = clean.to(device, non_blocking=use_pin)

            with torch.amp.autocast(device_type=device.type, enabled=use_pin):
                outputs = model(noisy)

            for i in range(noisy.size(0)):
                p, c = outputs[i].float(), clean[i].float()
                p = torch.clamp(p, 0.0, 1.0)
                psnr_list.append(calculate_psnr(p, c))
                ssim_list.append(calculate_ssim(p, c))
                acc_list.append(calculate_pixel_accuracy(p, c))

            if (batch_idx + 1) % 25 == 0:
                print(f"    [{model_name}] Processed {(batch_idx + 1) * test_loader.batch_size} / {len(test_loader.dataset)} images", flush=True)

    return {
        "psnr_list": psnr_list,
        "ssim_list": ssim_list,
        "acc_list": acc_list,
        "mean_psnr": float(np.mean(psnr_list)),
        "median_psnr": float(np.median(psnr_list)),
        "std_psnr": float(np.std(psnr_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "median_ssim": float(np.median(ssim_list)),
        "std_ssim": float(np.std(ssim_list)),
        "mean_pixel_accuracy": float(np.mean(acc_list)),
        "std_pixel_accuracy": float(np.std(acc_list)),
        "num_images": len(psnr_list),
    }


# ══════════════════════════════════════════════════════════════════════════
# GENERATE DENOISED SAMPLES (for visual comparison)
# ══════════════════════════════════════════════════════════════════════════

def generate_denoised_samples(models_dict: Dict, test_dataset, sample_indices: List[int]):
    """
    For each sample index, run all models and collect (clean, noisy, {model: denoised}).
    Returns list of dicts.
    """
    results = []
    for idx in sample_indices:
        noisy, clean = test_dataset[idx]
        noisy_dev = noisy.unsqueeze(0).to(device)
        sample = {
            "index": idx,
            "clean": clean.cpu(),
            "noisy": noisy.cpu(),
            "denoised": {},
        }
        with torch.no_grad():
            for name, model in models_dict.items():
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    out = model(noisy_dev)
                sample["denoised"][name] = torch.clamp(out.squeeze(0).float(), 0, 1).cpu()
        results.append(sample)
    return results


# ══════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def plot_bar_comparison(all_metrics: Dict, output_dir: Path):
    """Bar charts: PSNR, SSIM, Pixel Accuracy for all 11 models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(all_metrics.keys())
    short_names = [all_metrics[n]["short_label"] for n in names]
    psnrs = [all_metrics[n]["mean_psnr"] for n in names]
    ssims = [all_metrics[n]["mean_ssim"] for n in names]
    accs  = [all_metrics[n]["mean_pixel_accuracy"] * 100 for n in names]

    # Sort by PSNR for clean ordering
    order = np.argsort(psnrs)[::-1]
    short_names = [short_names[i] for i in order]
    psnrs = [psnrs[i] for i in order]
    ssims = [ssims[i] for i in order]
    accs  = [accs[i] for i in order]

    # Color: proposed model (M5) gets a highlight
    colors = ['#E53935' if 'M5' in n else '#1E88E5' if n.startswith('M') else '#43A047' for n in short_names]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # PSNR
    bars = axes[0].barh(short_names[::-1], psnrs[::-1], color=colors[::-1], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, psnrs[::-1]):
        axes[0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                     va='center', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('Test Set Mean PSNR (dB)', fontsize=13, fontweight='bold')
    axes[0].set_xlim(34, max(psnrs) + 1.5)
    axes[0].grid(True, alpha=0.3, axis='x')

    # SSIM
    bars = axes[1].barh(short_names[::-1], ssims[::-1], color=colors[::-1], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, ssims[::-1]):
        axes[1].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                     va='center', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('SSIM', fontsize=12)
    axes[1].set_title('Test Set Mean SSIM', fontsize=13, fontweight='bold')
    axes[1].set_xlim(0.90, max(ssims) + 0.01)
    axes[1].grid(True, alpha=0.3, axis='x')

    # Pixel Accuracy
    bars = axes[2].barh(short_names[::-1], accs[::-1], color=colors[::-1], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, accs[::-1]):
        axes[2].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
                     va='center', fontsize=9, fontweight='bold')
    axes[2].set_xlabel('Pixel Accuracy (%)', fontsize=12)
    axes[2].set_title('Test Set Pixel Accuracy (ε=0.05)', fontsize=13, fontweight='bold')
    axes[2].set_xlim(97, max(accs) + 0.5)
    axes[2].grid(True, alpha=0.3, axis='x')

    plt.suptitle('All 11 Models — Synthetic Test Set Performance Comparison',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'bar_comparison_all_models.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] bar_comparison_all_models.png")


def plot_boxplots(all_metrics: Dict, output_dir: Path):
    """Box plots of PSNR and SSIM distributions across test images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(all_metrics.keys())
    short_names = [all_metrics[n]["short_label"] for n in names]

    # Sort by mean PSNR
    mean_psnrs = [all_metrics[n]["mean_psnr"] for n in names]
    order = np.argsort(mean_psnrs)[::-1]
    short_names_sorted = [short_names[i] for i in order]
    names_sorted = [names[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # PSNR box plot
    psnr_data = [all_metrics[n]["psnr_list"] for n in names_sorted]
    bp1 = axes[0].boxplot(psnr_data, labels=short_names_sorted, patch_artist=True, vert=True,
                          showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
    colors_bp = ['#FFCDD2' if 'M5' in n else '#BBDEFB' if n.startswith('M') else '#C8E6C9' for n in short_names_sorted]
    for patch, color in zip(bp1['boxes'], colors_bp):
        patch.set_facecolor(color)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('PSNR Distribution Across Test Images', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')

    # SSIM box plot
    ssim_data = [all_metrics[n]["ssim_list"] for n in names_sorted]
    bp2 = axes[1].boxplot(ssim_data, labels=short_names_sorted, patch_artist=True, vert=True,
                          showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
    for patch, color in zip(bp2['boxes'], colors_bp):
        patch.set_facecolor(color)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM Distribution Across Test Images', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Per-Image Metric Distributions — All 11 Models on 1,000 Test Images',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(output_dir / 'boxplot_distributions.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] boxplot_distributions.png")


def plot_psnr_histograms(all_metrics: Dict, output_dir: Path):
    """Overlaid PSNR histograms for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(all_metrics.keys())
    short_names = [all_metrics[n]["short_label"] for n in names]

    # Separate Phase 1 and Phase 2 for clarity
    phase1 = [(n, s) for n, s in zip(names, short_names) if not s.startswith('M')]
    phase2 = [(n, s) for n, s in zip(names, short_names) if s.startswith('M')]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Phase 1 baselines
    colors_p1 = ['#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#F44336', '#00ACC1']
    for idx, (name, short) in enumerate(phase1):
        axes[0].hist(all_metrics[name]["psnr_list"], bins=40, alpha=0.5,
                     label=f'{short} (μ={all_metrics[name]["mean_psnr"]:.1f})',
                     color=colors_p1[idx % len(colors_p1)], edgecolor='white', linewidth=0.3)
    axes[0].set_xlabel('PSNR (dB)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Phase 1: Baseline Models — PSNR Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Phase 2 ablations
    colors_p2 = ['#90CAF9', '#64B5F6', '#42A5F5', '#1E88E5', '#E53935']
    for idx, (name, short) in enumerate(phase2):
        axes[1].hist(all_metrics[name]["psnr_list"], bins=40, alpha=0.5,
                     label=f'{short} (μ={all_metrics[name]["mean_psnr"]:.1f})',
                     color=colors_p2[idx % len(colors_p2)], edgecolor='white', linewidth=0.3)
    axes[1].set_xlabel('PSNR (dB)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Phase 2: NAFNet Ablations — PSNR Distribution', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Per-Image PSNR Distributions on 1,000 Test Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(output_dir / 'psnr_histograms.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] psnr_histograms.png")


def plot_radar_chart(all_metrics: Dict, output_dir: Path):
    """Radar chart comparing key metrics across all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    names = list(all_metrics.keys())
    short_names = [all_metrics[n]["short_label"] for n in names]

    # Metrics to compare (normalized to [0, 1] for radar)
    psnrs = np.array([all_metrics[n]["mean_psnr"] for n in names])
    ssims = np.array([all_metrics[n]["mean_ssim"] for n in names])
    accs  = np.array([all_metrics[n]["mean_pixel_accuracy"] for n in names])
    # Inverse of std (consistency) — lower std = better
    psnr_stds = np.array([all_metrics[n]["std_psnr"] for n in names])
    consistency = 1.0 - (psnr_stds - psnr_stds.min()) / (psnr_stds.max() - psnr_stds.min() + 1e-8)

    # Normalize each metric to [0, 1]
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    metrics = {
        'PSNR': norm(psnrs),
        'SSIM': norm(ssims),
        'Pixel Acc': norm(accs),
        'Consistency': consistency,
    }

    categories = list(metrics.keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Only plot selected models for clarity
    selected = ['M5', 'M1', 'ResUNet', 'Att-UNet', 'SAR2SAR']
    colors_radar = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

    for idx, short in enumerate(selected):
        if short not in short_names:
            continue
        model_idx = short_names.index(short)
        values = [metrics[cat][model_idx] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=short, color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.1, color=colors_radar[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('Multi-Metric Radar Comparison (Selected Models)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'radar_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] radar_comparison.png")


def plot_ranking_table(all_metrics: Dict, output_dir: Path):
    """Generate a visual ranking table image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(all_metrics.keys())
    # Sort by PSNR descending
    sorted_names = sorted(names, key=lambda n: all_metrics[n]["mean_psnr"], reverse=True)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    headers = ['Rank', 'Model', 'Type', 'Test PSNR (dB)', 'Test SSIM', 'Pixel Acc (%)', 'PSNR Std']
    table_data = []
    for rank, name in enumerate(sorted_names, 1):
        m = all_metrics[name]
        model_type = "Proposed" if "M5" in m["short_label"] else ("Ablation" if m["short_label"].startswith("M") else "Baseline")
        table_data.append([
            str(rank),
            name,
            model_type,
            f'{m["mean_psnr"]:.2f}',
            f'{m["mean_ssim"]:.4f}',
            f'{m["mean_pixel_accuracy"]*100:.2f}',
            f'{m["std_psnr"]:.2f}',
        ])

    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colWidths=[0.06, 0.25, 0.1, 0.13, 0.11, 0.13, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header
    for col in range(len(headers)):
        cell = table[0, col]
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')

    # Style rows
    for row in range(1, len(table_data) + 1):
        model_type = table_data[row - 1][2]
        if model_type == "Proposed":
            bg = '#FFCDD2'
        elif model_type == "Ablation":
            bg = '#BBDEFB'
        else:
            bg = '#C8E6C9'
        for col in range(len(headers)):
            table[row, col].set_facecolor(bg)
        if row == 1:  # Top model
            for col in range(len(headers)):
                table[row, col].set_text_props(fontweight='bold')

    plt.title('Model Ranking — Synthetic Test Set Evaluation (1,000 Images)',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(str(output_dir / 'model_ranking_table.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] model_ranking_table.png")


def plot_visual_comparison_grid(samples: List[Dict], all_metrics: Dict, output_dir: Path):
    """
    For each sample image, create a grid showing: Clean | Noisy | All 11 model outputs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names_sorted = sorted(all_metrics.keys(), key=lambda n: all_metrics[n]["mean_psnr"], reverse=True)

    for sample in samples:
        idx = sample["index"]
        clean = sample["clean"].permute(1, 2, 0).numpy()
        noisy = sample["noisy"].permute(1, 2, 0).numpy()

        n_models = len(model_names_sorted)
        total_cols = 2 + n_models  # clean + noisy + 11 models
        # Use 2 rows: top row = clean + noisy + first 5 models, bottom = remaining 6 + padding
        cols = 7
        rows = 2

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3))

        # Row 0, Col 0: Clean
        axes[0, 0].imshow(np.clip(clean, 0, 1))
        axes[0, 0].set_title('Clean\n(Ground Truth)', fontsize=9, fontweight='bold')
        axes[0, 0].axis('off')

        # Row 0, Col 1: Noisy
        axes[0, 1].imshow(np.clip(noisy, 0, 1))
        noisy_psnr = calculate_psnr(sample["noisy"], sample["clean"])
        axes[0, 1].set_title(f'Noisy (L=4)\nPSNR: {noisy_psnr:.1f} dB', fontsize=9, fontweight='bold', color='red')
        axes[0, 1].axis('off')

        # Fill models across both rows
        for m_idx, m_name in enumerate(model_names_sorted):
            denoised = sample["denoised"][m_name].permute(1, 2, 0).numpy()
            m_psnr = calculate_psnr(sample["denoised"][m_name], sample["clean"])
            m_ssim = calculate_ssim(sample["denoised"][m_name], sample["clean"])

            # Position in grid
            flat_pos = m_idx + 2  # offset by clean + noisy
            r = flat_pos // cols
            c = flat_pos % cols

            if r < rows and c < cols:
                axes[r, c].imshow(np.clip(denoised, 0, 1))
                short = all_metrics[m_name]["short_label"]
                color = '#E53935' if 'M5' in short else 'black'
                weight = 'bold' if 'M5' in short else 'normal'
                axes[r, c].set_title(f'{short}\n{m_psnr:.1f}dB | {m_ssim:.3f}',
                                     fontsize=8, fontweight=weight, color=color)
                axes[r, c].axis('off')

        # Hide any remaining empty cells
        for r in range(rows):
            for c in range(cols):
                flat_pos_check = r * cols + c
                if flat_pos_check >= 2 + n_models:
                    axes[r, c].axis('off')
                    axes[r, c].set_visible(False)

        plt.suptitle(f'Test Image #{idx} — All 11 Models Denoising Comparison',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_dir / f'visual_comparison_image_{idx}.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] visual_comparison_image_{idx}.png")


def plot_improvement_heatmap(all_metrics: Dict, output_dir: Path):
    """Pairwise PSNR improvement heatmap between models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(all_metrics.keys())
    short_names = [all_metrics[n]["short_label"] for n in names]

    # Sort by PSNR
    order = np.argsort([all_metrics[n]["mean_psnr"] for n in names])[::-1]
    names_sorted = [names[i] for i in order]
    short_sorted = [short_names[i] for i in order]

    n = len(names_sorted)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = all_metrics[names_sorted[i]]["mean_psnr"] - all_metrics[names_sorted[j]]["mean_psnr"]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=3)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_sorted, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(short_sorted, fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = 'white' if abs(val) > 1.5 else 'black'
            ax.text(j, i, f'{val:+.2f}', ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, label='PSNR Difference (dB)')
    plt.title('Pairwise PSNR Difference (Row − Column) in dB', fontsize=14, fontweight='bold')
    plt.xlabel('Model (Column)', fontsize=12)
    plt.ylabel('Model (Row)', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'psnr_improvement_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] psnr_improvement_heatmap.png")


def plot_scatter_psnr_vs_ssim(all_metrics: Dict, output_dir: Path):
    """Scatter plot: Mean PSNR vs Mean SSIM for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(all_metrics.keys())
    short_names = [all_metrics[n]["short_label"] for n in names]
    psnrs = [all_metrics[n]["mean_psnr"] for n in names]
    ssims = [all_metrics[n]["mean_ssim"] for n in names]
    params = []
    for n in names:
        p = all_metrics[n].get("total_params", 10e6)
        params.append(p / 1e6)  # in millions

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (p, s, short, par) in enumerate(zip(psnrs, ssims, short_names, params)):
        color = '#E53935' if 'M5' in short else '#1E88E5' if short.startswith('M') else '#43A047'
        size = max(50, par * 3)
        ax.scatter(p, s, s=size, c=color, edgecolors='white', linewidth=1, zorder=5, alpha=0.8)
        ax.annotate(short, (p, s), textcoords="offset points", xytext=(8, 8), fontsize=10,
                    fontweight='bold' if 'M5' in short else 'normal')

    ax.set_xlabel('Mean PSNR (dB)', fontsize=13)
    ax.set_ylabel('Mean SSIM', fontsize=13)
    ax.set_title('PSNR vs SSIM — All 11 Models (bubble size ∝ parameters)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legend for model types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#43A047', markersize=12, label='Phase 1 Baselines'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E88E5', markersize=12, label='Phase 2 Ablations'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E53935', markersize=12, label='M5 (Proposed)'),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='lower right')

    plt.tight_layout()
    plt.savefig(str(output_dir / 'scatter_psnr_vs_ssim.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] scatter_psnr_vs_ssim.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  UNIFIED TESTING EVALUATION — ALL 11 MODELS ON SYNTHETIC TEST SET")
    print("=" * 80)
    print(f"Device           : {device}")
    print(f"Output Directory : {OUTPUT_DIR}")
    print()

    # ── 1. Prepare test dataset ────────────────────────────────────────────
    noise_fn = SpeckleNoise(num_looks=4)
    test_ds = SatelliteSpeckleDataset(TEST_DIR, transform=ValTestTransform(crop_size=256), noise_fn=noise_fn)
    use_pin = device.type == "cuda"
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_pin)
    print(f"Test Dataset     : {len(test_ds)} images from {TEST_DIR}")
    print()

    # ── 2. Load all models ─────────────────────────────────────────────────
    registry = get_model_registry()
    all_metrics = {}
    models_dict = {}  # name -> loaded model for visual comparison

    print("=" * 80)
    print("  EVALUATING ALL MODELS ON 1,000 TEST IMAGES")
    print("=" * 80)

    for name, short_label, model, ckpt_path in registry:
        print(f"\n{'─'*60}")
        print(f"  Loading: {name}")
        print(f"  Checkpoint: {ckpt_path.name}")
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        model = load_model(model, ckpt_path, device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        print(f"  Running test evaluation...")
        metrics = evaluate_model(model, test_loader, name)
        metrics["short_label"] = short_label
        metrics["total_params"] = total_params

        print(f"  ✓ Mean PSNR: {metrics['mean_psnr']:.2f} dB | Mean SSIM: {metrics['mean_ssim']:.4f} | Pixel Acc: {metrics['mean_pixel_accuracy']*100:.2f}%")

        all_metrics[name] = metrics
        models_dict[name] = model

    # ── 3. Save metrics ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SAVING METRICS")
    print(f"{'='*80}")

    # JSON (without per-image lists)
    json_data = {}
    for name, m in all_metrics.items():
        json_data[name] = {k: v for k, v in m.items() if k not in ("psnr_list", "ssim_list", "acc_list")}
    with open(OUTPUT_DIR / "all_models_test_metrics.json", "w") as f:
        json.dump(json_data, f, indent=4)
    print("  [OK] all_models_test_metrics.json")

    # CSV
    csv_path = OUTPUT_DIR / "all_models_test_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Short Label", "Parameters", "Mean PSNR (dB)", "Median PSNR (dB)",
                         "Std PSNR", "Mean SSIM", "Median SSIM", "Std SSIM",
                         "Pixel Accuracy (%)", "Num Test Images"])
        for name in all_metrics:
            m = all_metrics[name]
            writer.writerow([
                name, m["short_label"], m.get("total_params", ""),
                f'{m["mean_psnr"]:.4f}', f'{m["median_psnr"]:.4f}', f'{m["std_psnr"]:.4f}',
                f'{m["mean_ssim"]:.6f}', f'{m["median_ssim"]:.6f}', f'{m["std_ssim"]:.6f}',
                f'{m["mean_pixel_accuracy"]*100:.4f}', m["num_images"],
            ])
    print("  [OK] all_models_test_metrics.csv")

    # ── 4. Generate visual comparison on random test images ────────────────
    print(f"\n{'='*80}")
    print("  GENERATING VISUAL COMPARISON ON RANDOM TEST IMAGES")
    print(f"{'='*80}")

    num_visual_samples = 5
    sample_indices = random.sample(range(len(test_ds)), num_visual_samples)
    print(f"  Selected sample indices: {sample_indices}")

    samples = generate_denoised_samples(models_dict, test_ds, sample_indices)

    plot_visual_comparison_grid(samples, all_metrics, OUTPUT_DIR)

    # ── 5. Generate all comparison plots ───────────────────────────────────
    print(f"\n{'='*80}")
    print("  GENERATING RESEARCH PAPER PLOTS")
    print(f"{'='*80}")

    plot_bar_comparison(all_metrics, OUTPUT_DIR)
    plot_boxplots(all_metrics, OUTPUT_DIR)
    plot_psnr_histograms(all_metrics, OUTPUT_DIR)
    plot_radar_chart(all_metrics, OUTPUT_DIR)
    plot_ranking_table(all_metrics, OUTPUT_DIR)
    plot_improvement_heatmap(all_metrics, OUTPUT_DIR)
    plot_scatter_psnr_vs_ssim(all_metrics, OUTPUT_DIR)

    # ── 6. Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  FINAL SUMMARY — MODEL RANKING BY TEST PSNR")
    print(f"{'='*80}")
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]["mean_psnr"], reverse=True)
    for rank, (name, m) in enumerate(sorted_models, 1):
        marker = " ★" if "M5" in m["short_label"] else ""
        print(f"  #{rank:2d}  {name:<30s}  PSNR: {m['mean_psnr']:.2f} dB  |  SSIM: {m['mean_ssim']:.4f}  |  Acc: {m['mean_pixel_accuracy']*100:.2f}%{marker}")

    print(f"\n{'='*80}")
    print(f"  ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
    print(f"{'='*80}")
    print("  Files generated:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"    • {f.name}")

    print("\n  ✓ UNIFIED TESTING EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
