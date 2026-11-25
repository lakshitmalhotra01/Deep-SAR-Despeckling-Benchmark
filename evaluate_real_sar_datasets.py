"""
evaluate_real_sar_datasets.py
=============================
Fixes and updates the no-reference SAR evaluation for all 11 models on:
  1. ALOS-1        -> results/alos1_evaluation/
  2. UAVSAR Panama -> results/uavsar_panama_evaluation/
  3. Radarsat-2    -> results/radarset2_evaluation/

Fixes over the previous scripts:
  - Excludes mask_valid_pixels and product_lut from evaluation channels
  - Properly reads ENVI dtype=6 (complex64) scattering matrix files (s11, s12, s21, s22)
    and converts to intensity |S|^2
  - Evaluates true SAR intensity channels (T11, T22, T33, C11, C22, |s11|^2..|s22|^2)
    instead of signed off-diagonal cross-terms
  - Uses valid-pixel masks to select valid 256x256 homogeneous & edge-rich patches
  - Normalizes each channel by its 99th percentile to [0, 1] before inference
  - Generates clean JSON, CSV, bar charts, ranking table, ENL-vs-EPI tradeoff plot,
    and visual denoising comparison grid in each dataset's results folder.
"""

import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import sobel

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model Registry (all 11 models) ───────────────────────────────────────────
def get_all_models():
    from models.resunet import ResUNet
    from models.attention_unet import AttentionUNet
    from models.multires_unet import MultiResUNet
    from models.cbam_unet import CBAMUNet
    from models.sar2sar import SAR2SARNet
    from models.nafnet_baseline import NAFNetBaseline
    from models.nafnet_adaptive_gate import NAFNetAdaptiveGate
    from models.nafnet_adaptive_spatial import NAFNetAdaptiveSpatial
    from models.nafnet_adaptive_channel import nafnet_adaptive_channel_model
    from models.nafnet_adaptive_dual import nafnet_adaptive_dual_model
    from models.unet_autoencoder import unet_autoencoder_model

    ckpt = PROJECT_ROOT / "checkpoints"
    return [
        ("ResUNet", ResUNet(), ckpt / "resunet_best.pth"),
        ("Attention-UNet", AttentionUNet(), ckpt / "attention_unet_best.pth"),
        ("MultiResUNet", MultiResUNet(), ckpt / "multires_unet_best.pth"),
        ("CBAM-UNet", CBAMUNet(), ckpt / "cbam_unet_best.pth"),
        ("SAR2SAR", SAR2SARNet(), ckpt / "sar2sar_best.pth"),
        ("M1-NAFNet-Base", NAFNetBaseline(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1, 1, 1, 4]), ckpt / "nafnet_baseline_best.pth"),
        ("M2-NAFNet-AGM", NAFNetAdaptiveGate(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1, 1, 1, 4]), ckpt / "nafnet_adaptive_gate_best.pth"),
        ("M3-NAFNet-SAM", NAFNetAdaptiveSpatial(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1, 1, 1, 4]), ckpt / "nafnet_adaptive_spatial_best.pth"),
        ("M4-NAFNet-CAM", nafnet_adaptive_channel_model(), ckpt / "nafnet_adaptive_channel_best.pth"),
        ("M5-NAFNet-Dual*", nafnet_adaptive_dual_model(), ckpt / "nafnet_adaptive_dual_best.pth"),
        ("UNet-Autoencoder", unet_autoencoder_model(), ckpt / "unet_autoencoder_best.pth"),
    ]


def load_checkpoint(model, ckpt_path: Path):
    state = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ── ENVI Header & Memmap Loading ─────────────────────────────────────────────
def parse_envi_hdr(bin_path: Path) -> Tuple[int, int, np.dtype]:
    hdr = bin_path.with_suffix(".hdr")
    if not hdr.exists():
        hdr = bin_path.with_name(bin_path.name + ".hdr")
    if not hdr.exists():
        raise FileNotFoundError(f"Header not found for {bin_path}")
    samples = lines = dtype_code = None
    for line in hdr.read_text(errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if k == "samples":
                samples = int(v)
            elif k == "lines":
                lines = int(v)
            elif k == "data type":
                dtype_code = int(v)
    np_dtype = np.complex64 if dtype_code == 6 else np.float32
    return lines, samples, np_dtype


def load_patch_pair(bin_path: Path, patch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-map a PolSAR .bin file, find one valid homogeneous 256x256 patch and
    one valid edge-rich 256x256 patch, and normalize to [0, 1] using 99th percentile.
    """
    lines, samples, np_dtype = parse_envi_hdr(bin_path)
    mm = np.memmap(str(bin_path), dtype=np_dtype, mode="r", shape=(lines, samples))

    # Sample a grid of candidate 256x256 patches across the image
    step_r = max(patch_size, (lines - patch_size) // 8)
    step_c = max(patch_size, (samples - patch_size) // 6)

    best_hom = None
    best_hom_cv = float("inf")
    best_edge = None
    best_edge_grad = -1.0

    # Also collect valid intensities for global p99 estimate
    sample_vals = []

    for r in range(0, lines - patch_size + 1, step_r):
        for c in range(0, samples - patch_size + 1, step_c):
            raw = np.array(mm[r : r + patch_size, c : c + patch_size])
            if np.iscomplexobj(raw):
                patch = (np.abs(raw) ** 2).astype(np.float32)
            else:
                patch = raw.astype(np.float32)

            valid = np.isfinite(patch) & (patch > 1e-10)
            if valid.mean() < 0.90:
                continue

            sample_vals.append(patch[valid][::16])
            mu = float(patch[valid].mean())
            sd = float(patch[valid].std())
            if mu <= 1e-10:
                continue
            cv = sd / mu
            # Prefer non-dark homogeneous regions (avoid near-zero water/shadow borders)
            if cv < best_hom_cv and mu > np.percentile(patch[valid], 10):
                best_hom_cv = cv
                best_hom = patch

            gx = sobel(patch, axis=1)
            gy = sobel(patch, axis=0)
            grad_e = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
            if grad_e > best_edge_grad:
                best_edge_grad = grad_e
                best_edge = patch

    if best_hom is None or best_edge is None:
        raise ValueError(f"No valid 256x256 patches found in {bin_path}")

    all_s = np.concatenate(sample_vals)
    p99 = float(np.percentile(all_s, 99))
    if p99 <= 0:
        p99 = 1.0

    hom_norm = np.clip(np.nan_to_num(best_hom, nan=0.0) / p99, 0.0, 1.0).astype(np.float32)
    edge_norm = np.clip(np.nan_to_num(best_edge, nan=0.0) / p99, 0.0, 1.0).astype(np.float32)
    return hom_norm, edge_norm


# ── Inference & No-Reference SAR Metrics ─────────────────────────────────────
def run_inference(model, patch_2d: np.ndarray) -> np.ndarray:
    H, W = patch_2d.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    t = torch.from_numpy(patch_2d).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    t = t.to(DEVICE)
    with torch.no_grad():
        out = model(t)
    out = out[:, :1, :H, :W].squeeze().float().cpu().numpy()
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compute_enl(patch: np.ndarray, sub_size: int = 48) -> float:
    h, w = patch.shape
    cy, cx = h // 2, w // 2
    r = sub_size // 2
    sub = patch[cy - r : cy + r, cx - r : cx + r]
    sub = sub[sub > 1e-8]
    if len(sub) < 10:
        return 0.0
    mu = float(sub.mean())
    sigma = float(sub.std())
    if sigma < 1e-10:
        return 999.0
    return float((mu / sigma) ** 2)


def compute_epi(noisy: np.ndarray, denoised: np.ndarray) -> float:
    def grad_mag(img):
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        return np.sqrt(gx ** 2 + gy ** 2)

    gn = grad_mag(noisy)
    gd = grad_mag(denoised)
    num = float(np.sum(gn * gd))
    den = float(np.sqrt(np.sum(gn ** 2) * np.sum(gd ** 2)))
    if not np.isfinite(den) or den < 1e-10:
        return 0.0
    val = float(num / den)
    return val if np.isfinite(val) else 0.0


def compute_ssi(noisy: np.ndarray, denoised: np.ndarray) -> float:
    n_pos = noisy[noisy > 1e-8]
    d_pos = denoised[denoised > 1e-8]
    if len(n_pos) == 0 or len(d_pos) == 0:
        return 1.0
    mn, sn = float(np.mean(n_pos)), float(np.std(n_pos))
    md, sd = float(np.mean(d_pos)), float(np.std(d_pos))
    if mn < 1e-10 or md < 1e-10 or sn < 1e-10:
        return 1.0
    return float((sd / md) / (sn / mn))


def compute_smpi(noisy: np.ndarray, denoised: np.ndarray) -> float:
    enl_n = compute_enl(noisy)
    enl_d = compute_enl(denoised)
    if enl_n < 1e-6:
        return 1.0
    return float(enl_d / enl_n)


# ── Dataset Channel Specifications ───────────────────────────────────────────
def get_dataset_configs():
    return [
        {
            "key": "alos1",
            "title": "ALOS-1 PALSAR (L-Band PolSAR)",
            "out_dir": PROJECT_ROOT / "results" / "alos1_evaluation",
            "json_name": "all_models_alos1_metrics.json",
            "csv_name": "all_models_alos1_metrics.csv",
            "channels": {
                "T3_T11": PROJECT_ROOT / "data/ALOS1_Data/ALOS1_Data/T3_Matrix/T11.bin",
                "T3_T22": PROJECT_ROOT / "data/ALOS1_Data/ALOS1_Data/T3_Matrix/T22.bin",
                "T3_T33": PROJECT_ROOT / "data/ALOS1_Data/ALOS1_Data/T3_Matrix/T33.bin",
                "S2_s11": PROJECT_ROOT / "data/ALOS1_Data/ALOS1_Data/S2_Matrix/s11.bin",
                "S2_s12": PROJECT_ROOT / "data/ALOS1_Data/ALOS1_Data/S2_Matrix/s12.bin",
                "S2_s21": PROJECT_ROOT / "data/ALOS1_Data/ALOS1_Data/S2_Matrix/s21.bin",
                "S2_s22": PROJECT_ROOT / "data/ALOS1_Data/ALOS1_Data/S2_Matrix/s22.bin",
                "C2_HH_HV_C11": PROJECT_ROOT / "data/ALOS1_Data/ALOS1_Data/Compact_and_Dual/C2_HH_HV/C11.bin",
                "C2_HH_HV_C22": PROJECT_ROOT / "data/ALOS1_Data/ALOS1_Data/Compact_and_Dual/C2_HH_HV/C22.bin",
                "C2_VV_VH_C11": PROJECT_ROOT / "data/ALOS1_Data/ALOS1_Data/Compact_and_Dual/C2_VV_VH/C11.bin",
                "C2_VV_VH_C22": PROJECT_ROOT / "data/ALOS1_Data/ALOS1_Data/Compact_and_Dual/C2_VV_VH/C22.bin",
            },
            "vis_channel": "T3_T11",
        },
        {
            "key": "uavsar",
            "title": "UAVSAR Panama Canal (L-Band PolSAR)",
            "out_dir": PROJECT_ROOT / "results" / "uavsar_panama_evaluation",
            "json_name": "all_models_uavsar_metrics.json",
            "csv_name": "all_models_uavsar_metrics.csv",
            "channels": {
                "T3_T11": PROJECT_ROOT / "data/UAVSAR_Panama_Canal/T3/T11.bin",
                "T3_T22": PROJECT_ROOT / "data/UAVSAR_Panama_Canal/T3/T22.bin",
                "T3_T33": PROJECT_ROOT / "data/UAVSAR_Panama_Canal/T3/T33.bin",
            },
            "vis_channel": "T3_T11",
        },
        {
            "key": "radarset2",
            "title": "Radarsat-2 (C-Band Quad-Pol S2)",
            "out_dir": PROJECT_ROOT / "results" / "radarset2_evaluation",
            "json_name": "all_models_radarset2_metrics.json",
            "csv_name": "all_models_radarset2_metrics.csv",
            "channels": {
                "S2_s11": PROJECT_ROOT / "data/RAdarset_2/s11.bin",
                "S2_s12": PROJECT_ROOT / "data/RAdarset_2/s12.bin",
                "S2_s21": PROJECT_ROOT / "data/RAdarset_2/s21.bin",
                "S2_s22": PROJECT_ROOT / "data/RAdarset_2/s22.bin",
            },
            "vis_channel": "S2_s11",
        },
    ]


# ── Plotting Functions ───────────────────────────────────────────────────────
def generate_dataset_plots(title: str, out_dir: Path, results: Dict, vis_patches: Dict, vis_channel: str, hom_patch: np.ndarray):
    models = list(results.keys())

    def avg_metric(m, key):
        vals = [v[key] for v in results[m].values() if key in v and np.isfinite(v[key])]
        return float(np.mean(vals)) if vals else 0.0

    enl_in_avg = avg_metric(models[0], "ENL_in")
    enl_out = [avg_metric(m, "ENL_out") for m in models]
    epi = [avg_metric(m, "EPI") for m in models]
    ssi = [avg_metric(m, "SSI") for m in models]
    smpi = [avg_metric(m, "SMPI") for m in models]
    colors = ["#E53935" if "M5" in m else "#1E88E5" if m.startswith("M") else "#43A047" for m in models]

    # 1. all_models_comparison_bars.png
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    specs = [
        (axes[0, 0], enl_out, f"Mean ENL (Noisy Input = {enl_in_avg:.2f}) ↑", "ENL_out"),
        (axes[0, 1], epi, "Edge Preservation Index (EPI) ↑", "EPI"),
        (axes[1, 0], ssi, "Speckle Suppression Index (SSI) ↓", "SSI"),
        (axes[1, 1], smpi, "ENL Gain / SMPI (ENL_out / ENL_in) ↑", "SMPI"),
    ]
    for ax, vals, stitle, ylabel in specs:
        bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.015, f"{v:.3f}" if max(vals) <= 2 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title(stitle, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle(f"{title} — All 11 Models Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_dir / "all_models_comparison_bars.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. all_models_metrics_table.png
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.axis("off")
    headers = ["Model", "ENL (In)", "ENL (Out) ↑", "ENL Gain ↑", "EPI ↑", "SSI ↓", "SMPI ↑"]
    rows = []
    for m, eo, ep, ss, sm in zip(models, enl_out, epi, ssi, smpi):
        gain = eo / max(enl_in_avg, 1e-6)
        rows.append([m, f"{enl_in_avg:.3f}", f"{eo:.3f}", f"{gain:.2f}x", f"{ep:.4f}", f"{ss:.4f}", f"{sm:.3f}"])
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)
    for col in range(len(headers)):
        tbl[0, col].set_facecolor("#1565C0")
        tbl[0, col].set_text_props(color="white", fontweight="bold")
    for r_idx, m in enumerate(models, 1):
        bg = "#FFCDD2" if "M5" in m else "#BBDEFB" if m.startswith("M") else "#C8E6C9"
        for col in range(len(headers)):
            tbl[r_idx, col].set_facecolor(bg)
            if "M5" in m:
                tbl[r_idx, col].set_text_props(fontweight="bold")
    plt.title(f"{title} — Summary Metrics Across All {len(results[models[0]])} Intensity Channels", fontsize=13, fontweight="bold", pad=16)
    plt.savefig(str(out_dir / "all_models_metrics_table.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. enl_gain_and_tradeoff.png
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    gains = [eo / max(enl_in_avg, 1e-6) for eo in enl_out]
    order = np.argsort(gains)[::-1]
    m_ord = [models[i] for i in order]
    g_ord = [gains[i] for i in order]
    c_ord = [colors[i] for i in order]
    bars = axes[0].barh(m_ord[::-1], g_ord[::-1], color=c_ord[::-1])
    for b, v in zip(bars, g_ord[::-1]):
        axes[0].text(b.get_width() + 0.03, b.get_y() + b.get_height() / 2, f"{v:.2f}x", va="center", fontsize=9, fontweight="bold")
    axes[0].set_xlabel("ENL Gain (ENL_out / ENL_in)", fontsize=11)
    axes[0].set_title("Speckle Smoothing Gain (Higher = Smoother)", fontsize=12, fontweight="bold")
    axes[0].set_xlim(0, max(g_ord) * 1.18)
    axes[0].grid(True, alpha=0.3, axis="x")

    speckle_supp = [1.0 - s for s in ssi]
    for m, ss_val, ep_val, col in zip(models, speckle_supp, epi, colors):
        axes[1].scatter(ss_val, ep_val, s=130 if "M5" in m else 85, color=col, edgecolors="white", linewidth=1, zorder=5)
        axes[1].annotate(m, (ss_val, ep_val), textcoords="offset points", xytext=(6, 5), fontsize=8.5,
                         fontweight="bold" if "M5" in m else "normal")
    axes[1].set_xlabel("Speckle Reduction (1 - SSI) ↑", fontsize=11)
    axes[1].set_ylabel("Edge Preservation Index (EPI) ↑", fontsize=11)
    axes[1].set_title("Edge Preservation vs. Speckle Suppression Trade-off", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    plt.suptitle(f"{title} — Quantitative Trade-off Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_dir / "enl_gain_and_tradeoff.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. visual_denoising_comparison.png (Noisy + all 11 models on vis_channel)
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    axes = axes.flatten()
    axes[0].imshow(hom_patch, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Noisy ({vis_channel})\nENL={compute_enl(hom_patch):.2f}", fontsize=9, fontweight="bold", color="red")
    axes[0].axis("off")
    for idx_m, m in enumerate(models, 1):
        den = vis_patches[m]
        axes[idx_m].imshow(den, cmap="gray", vmin=0, vmax=1)
        m_enl = compute_enl(den)
        m_epi = compute_epi(hom_patch, den)
        axes[idx_m].set_title(f"{m}\nENL={m_enl:.2f} | EPI={m_epi:.3f}", fontsize=8.5,
                              fontweight="bold" if "M5" in m else "normal",
                              color="#E53935" if "M5" in m else "black")
        axes[idx_m].axis("off")
    plt.suptitle(f"{title} — Visual Despeckling Comparison ({vis_channel})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_dir / "visual_denoising_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 75)
    print("  FIXING & UPDATING REAL SAR EVALUATIONS (ALOS-1, UAVSAR, RADARSAT-2)")
    print("=" * 75)
    print(f"Device: {DEVICE}")

    registry = get_all_models()
    loaded_models = []
    for name, model, ckpt_path in registry:
        loaded_models.append((name, load_checkpoint(model, ckpt_path)))
    print(f"Loaded {len(loaded_models)} models.")

    for cfg in get_dataset_configs():
        print(f"\n{'─'*70}")
        print(f"  Dataset: {cfg['title']}")
        print(f"{'─'*70}")
        out_dir = cfg["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-extract valid 256x256 patches for each channel
        channel_patches = {}
        for ch_name, bin_path in cfg["channels"].items():
            hom_p, edge_p = load_patch_pair(bin_path, patch_size=256)
            channel_patches[ch_name] = (hom_p, edge_p)
            print(f"  [DATA] {ch_name}: ENL_in={compute_enl(hom_p):.2f}")

        all_results = {}
        vis_patches = {}
        vis_ch = cfg["vis_channel"]

        for name, model in loaded_models:
            model_res = {}
            for ch_name, (hom_p, edge_p) in channel_patches.items():
                hom_out = run_inference(model, hom_p)
                edge_out = run_inference(model, edge_p)
                if ch_name == vis_ch:
                    vis_patches[name] = hom_out
                model_res[ch_name] = {
                    "ENL_in": round(compute_enl(hom_p), 3),
                    "ENL_out": round(compute_enl(hom_out), 3),
                    "EPI": round(compute_epi(edge_p, edge_out), 4),
                    "SSI": round(compute_ssi(hom_p, hom_out), 4),
                    "SMPI": round(compute_smpi(hom_p, hom_out), 3),
                }
            all_results[name] = model_res
            mean_eo = np.mean([v["ENL_out"] for v in model_res.values()])
            mean_ep = np.mean([v["EPI"] for v in model_res.values()])
            mean_ss = np.mean([v["SSI"] for v in model_res.values()])
            print(f"  [OK] {name:<18s} | Mean ENL_out={mean_eo:.2f} | EPI={mean_ep:.4f} | SSI={mean_ss:.4f}")

        # Save JSON
        with open(out_dir / cfg["json_name"], "w") as f:
            json.dump(all_results, f, indent=4)

        # Save CSV
        with open(out_dir / cfg["csv_name"], "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Channel", "ENL_in", "ENL_out", "EPI", "SSI", "SMPI"])
            for m_name, chs in all_results.items():
                for c_name, vals in chs.items():
                    writer.writerow([m_name, c_name, vals["ENL_in"], vals["ENL_out"], vals["EPI"], vals["SSI"], vals["SMPI"]])

        # Generate Plots
        generate_dataset_plots(cfg["title"], out_dir, all_results, vis_patches, vis_ch, channel_patches[vis_ch][0])
        print(f"  Saved all JSON, CSV, and 4 PNG plots to {out_dir}")

    print("\n  ✓ ALL THREE REAL SAR EVALUATIONS UPDATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
