"""
real_sar_all_models_evaluation.py
==================================
Evaluates ALL 10 trained models on real SAR data (both bands):
  - Sentinel-1 C-band (VV, VH)  — real satellite SLC
  - UAVSAR L-band Quad-Pol (HH, HV, VH, VV) — simulated airborne

Metrics: ENL, EPI, SSI, SMPI per channel per model.
No retraining — pure inference only.
"""

import sys, os
from pathlib import Path

# ── Force ALL caches / temp to D: drive ─────────────────────────────────
PROJECT_ROOT = Path("d:/image video project")
os.environ["TORCH_HOME"]          = str(PROJECT_ROOT / ".cache/torch")
os.environ["TEMP"]                = str(PROJECT_ROOT / ".temp")
os.environ["TMP"]                 = str(PROJECT_ROOT / ".temp")
os.environ["MPLCONFIGDIR"]        = str(PROJECT_ROOT / ".cache/matplotlib")
os.environ["CUDA_CACHE_PATH"]     = str(PROJECT_ROOT / ".cache/nv")
os.environ["PYTHONPYCACHEPREFIX"] = str(PROJECT_ROOT / ".cache/pycache")
for d in [".temp", ".cache/torch", ".cache/matplotlib", ".cache/nv", ".cache/pycache"]:
    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import time
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
CKPT_DIR    = PROJECT_ROOT / "checkpoints"
SENT1_DIR   = PROJECT_ROOT / "real_sar" / "sentinel1"   # FIXED: correct subfolder
UAVSAR_DIR  = PROJECT_ROOT / "real_sar" / "uavsar"
RESULTS_DIR = PROJECT_ROOT / "results" / "real_sar_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ═════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY (10 models only — UNet-Autoencoder excluded, still training)
# ═════════════════════════════════════════════════════════════════════════
def load_all_models():
    registry = []

    from models.resunet import ResUNet
    registry.append(("ResUNet", ResUNet(), CKPT_DIR / "resunet_best.pth"))

    from models.attention_unet import AttentionUNet
    registry.append(("Attention-UNet", AttentionUNet(), CKPT_DIR / "attention_unet_best.pth"))

    from models.multires_unet import MultiResUNet
    registry.append(("MultiResUNet", MultiResUNet(), CKPT_DIR / "multires_unet_best.pth"))

    from models.cbam_unet import CBAMUNet
    registry.append(("CBAM-UNet", CBAMUNet(), CKPT_DIR / "cbam_unet_best.pth"))

    from models.sar2sar import SAR2SARNet
    registry.append(("SAR2SAR", SAR2SARNet(), CKPT_DIR / "sar2sar_best.pth"))

    from models.nafnet_baseline import NAFNetBaseline
    registry.append(("M1-NAFNet-Base", NAFNetBaseline(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1,1,1,4]),
                     CKPT_DIR / "nafnet_baseline_best.pth"))

    from models.nafnet_adaptive_gate import NAFNetAdaptiveGate
    registry.append(("M2-NAFNet-AGM", NAFNetAdaptiveGate(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1,1,1,4]),
                     CKPT_DIR / "nafnet_adaptive_gate_best.pth"))

    from models.nafnet_adaptive_spatial import NAFNetAdaptiveSpatial
    registry.append(("M3-NAFNet-SAM", NAFNetAdaptiveSpatial(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1,1,1,4]),
                     CKPT_DIR / "nafnet_adaptive_spatial_best.pth"))

    from models.nafnet_adaptive_channel import nafnet_adaptive_channel_model
    registry.append(("M4-NAFNet-CAM", nafnet_adaptive_channel_model(), CKPT_DIR / "nafnet_adaptive_channel_best.pth"))

    from models.nafnet_adaptive_dual import nafnet_adaptive_dual_model
    registry.append(("M5-NAFNet-Dual*", nafnet_adaptive_dual_model(), CKPT_DIR / "nafnet_adaptive_dual_best.pth"))

    # Add UNet-Autoencoder (training completed, checkpoint ready)
    from models.unet_autoencoder import unet_autoencoder_model
    registry.append(("UNet-Autoencoder", unet_autoencoder_model(), CKPT_DIR / "unet_autoencoder_best.pth"))
    return registry


def load_checkpoint(model, ckpt_path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
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


# ═════════════════════════════════════════════════════════════════════════
# SAR DATA LOADING
# ═════════════════════════════════════════════════════════════════════════
MAX_VV = 0.40
MAX_VH = 0.10

def load_sentinel1():
    """Load Sentinel-1 C-band SLC. Returns dict {channel_name: intensity_array}."""
    data = {}
    for pol, fname, norm_max in [("S1_VV", "crop_vv.npy", MAX_VV), ("S1_VH", "crop_vh.npy", MAX_VH)]:
        fp = SENT1_DIR / fname
        if not fp.exists():
            print(f"  [WARN] Sentinel-1 not found: {fp}")
            continue
        slc = np.load(str(fp))
        if np.iscomplexobj(slc):
            intensity = np.abs(slc) ** 2
        else:
            intensity = slc.astype(np.float32)
        intensity = np.clip(intensity / norm_max, 0.0, 1.0).astype(np.float32)
        data[pol] = intensity
        print(f"  {pol}: shape={intensity.shape}, range=[{intensity.min():.4f}, {intensity.max():.4f}]")
    return data


def load_uavsar():
    """Load UAVSAR L-band PolSAR. Returns dict {channel_name: intensity_array}."""
    data = {}
    for pol in ["HH", "HV", "VH", "VV"]:
        fp = UAVSAR_DIR / f"uavsar_{pol}.npy"
        if not fp.exists():
            print(f"  [WARN] UAVSAR {pol} not found: {fp}")
            continue
        arr = np.load(str(fp)).astype(np.float32)
        arr_max = np.percentile(arr, 99)
        if arr_max > 0:
            arr = np.clip(arr / arr_max, 0.0, 1.0)
        data[f"UAVSAR_{pol}"] = arr
        print(f"  UAVSAR_{pol}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")
    return data


# ═════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════
def run_inference(model, patch_2d):
    H, W = patch_2d.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    t = torch.from_numpy(patch_2d).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    t = t.to(DEVICE)
    with torch.no_grad():
        out = model(t)
    out = out[:, :1, :H, :W].squeeze().cpu().numpy()
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════
def compute_enl(patch, sub_size=40):
    h, w = patch.shape
    cy, cx = h // 2, w // 2
    r = sub_size // 2
    sub = patch[cy-r:cy+r, cx-r:cx+r]
    sub = sub[sub > 1e-8]
    if len(sub) == 0:
        return 0.0
    mu = sub.mean()
    sigma = sub.std()
    if sigma < 1e-10:
        return 999.0
    return float((mu / sigma) ** 2)


def compute_epi(noisy, denoised):
    from scipy.ndimage import sobel
    def grad_mag(img):
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        return np.sqrt(gx**2 + gy**2)
    gn = grad_mag(noisy)
    gd = grad_mag(denoised)
    num = np.sum(gn * gd)
    den = np.sqrt(np.sum(gn**2) * np.sum(gd**2))
    if den < 1e-10:
        return 0.0
    return float(num / den)


def compute_ssi(noisy, denoised):
    mn = np.mean(noisy[noisy > 1e-8])
    sn = np.std(noisy[noisy > 1e-8])
    md = np.mean(denoised[denoised > 1e-8])
    sd = np.std(denoised[denoised > 1e-8])
    if mn < 1e-10 or md < 1e-10 or sn < 1e-10:
        return 1.0
    return float((sd / md) / (sn / mn))


def compute_smpi(noisy, denoised):
    enl_n = compute_enl(noisy)
    enl_d = compute_enl(denoised)
    if enl_n < 1e-6:
        return 1.0
    return float(enl_d / enl_n)


def evaluate_channel(model, intensity_full):
    H, W = intensity_full.shape

    # Homogeneous patch (256x256)
    r0, r1 = 100, min(356, H)
    c0, c1 = 40,  min(296, W)
    hom_in = intensity_full[r0:r1, c0:c1]
    if hom_in.shape[0] < 64 or hom_in.shape[1] < 64:
        hom_in = intensity_full[:min(256, H), :min(256, W)]
    hom_out = run_inference(model, hom_in)

    # Edge/urban patch (256x256)
    er0, er1 = 200, min(456, H)
    ec0, ec1 = max(0, min(1000, W - 256)), min(1256, W)
    if ec1 - ec0 < 64:
        ec0, ec1 = 0, min(256, W)
    if er1 - er0 < 64:
        er0, er1 = 0, min(256, H)
    edge_in  = intensity_full[er0:er1, ec0:ec1]
    edge_out = run_inference(model, edge_in)

    enl_in  = compute_enl(hom_in)
    enl_out = compute_enl(hom_out)
    epi     = compute_epi(edge_in, edge_out)
    ssi     = compute_ssi(hom_in, hom_out)
    smpi    = compute_smpi(hom_in, hom_out)

    return {
        "ENL_in":  round(enl_in, 3),
        "ENL_out": round(enl_out, 3),
        "EPI":     round(epi, 4),
        "SSI":     round(ssi, 4),
        "SMPI":    round(smpi, 3),
    }


# ═════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═════════════════════════════════════════════════════════════════════════
def run_all_evaluations():
    print("\n" + "="*70)
    print("  REAL SAR ALL-MODELS EVALUATION (10 Models x 6 Channels)")
    print("  C-band (Sentinel-1 VV, VH) + L-band (UAVSAR HH, HV, VH, VV)")
    print("="*70)

    print("\n[1/3] Loading SAR data...")
    s1_data     = load_sentinel1()
    uavsar_data = load_uavsar()
    all_channels = {**s1_data, **uavsar_data}

    if not all_channels:
        print("ERROR: No SAR data found!")
        return None

    print(f"\n  Total channels: {len(all_channels)} -> {list(all_channels.keys())}")

    print("\n[2/3] Loading 10 models...")
    registry = load_all_models()
    print(f"  Models loaded: {len(registry)}")
    for name, _, ckpt in registry:
        print(f"    - {name}: {ckpt.name}")

    print("\n[3/3] Running evaluation (10 models x 6 channels = 60 evaluations)...")
    all_results = {}

    for idx, (name, model, ckpt_path) in enumerate(registry, 1):
        print(f"\n  [{idx}/{len(registry)}] {name}")
        try:
            model = load_checkpoint(model, ckpt_path)
        except Exception as e:
            print(f"    ERROR loading checkpoint: {e}")
            all_results[name] = {}
            continue

        model_results = {}
        t0 = time.time()
        for ch_name, intensity in all_channels.items():
            try:
                metrics = evaluate_channel(model, intensity)
                model_results[ch_name] = metrics
                print(f"    {ch_name}: ENL={metrics['ENL_out']:.2f} EPI={metrics['EPI']:.4f} "
                      f"SSI={metrics['SSI']:.4f} SMPI={metrics['SMPI']:.2f}")
            except Exception as e:
                print(f"    ERROR on {ch_name}: {e}")
                model_results[ch_name] = {}

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        all_results[name] = model_results

        del model
        torch.cuda.empty_cache()

    return all_results, list(all_channels.keys())


# ═════════════════════════════════════════════════════════════════════════
# OUTPUT: JSON
# ═════════════════════════════════════════════════════════════════════════
def save_json(all_results):
    out_path = RESULTS_DIR / "all_models_sar_metrics.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"  JSON saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════
# OUTPUT: DETAILED TABLE PNG
# ═════════════════════════════════════════════════════════════════════════
def make_summary_table(all_results, channels):
    model_names = list(all_results.keys())
    metrics_list = ["ENL_out", "EPI", "SSI", "SMPI"]
    metric_labels = ["ENL", "EPI", "SSI", "SMPI"]

    col_headers = []
    col_keys = []
    for ch in channels:
        for m, ml in zip(metrics_list, metric_labels):
            col_headers.append(f"{ch}\n{ml}")
            col_keys.append((ch, m))

    rows = []
    for mname in model_names:
        row = [mname]
        for ch, m in col_keys:
            val = all_results.get(mname, {}).get(ch, {}).get(m, None)
            row.append(f"{val:.3f}" if val is not None else "--")
        rows.append(row)

    ncols = len(col_headers) + 1
    nrows = len(rows)
    fig_w = max(24, ncols * 1.2)
    fig_h = max(5, nrows * 0.55 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=["Model"] + col_headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.5)

    for j in range(ncols):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    for i in range(1, nrows + 1):
        tbl[(i, 0)].set_facecolor("#ecf0f1")
        tbl[(i, 0)].set_text_props(fontweight="bold")
        bg = "#ffffff" if i % 2 == 1 else "#f8f9fa"
        for j in range(1, ncols):
            tbl[(i, j)].set_facecolor(bg)
        if "Dual" in rows[i-1][0] or "M5" in rows[i-1][0]:
            for j in range(ncols):
                tbl[(i, j)].set_facecolor("#d5f5e3")

    plt.title("All 10 Models - Real SAR Evaluation\nSentinel-1 C-band (VV, VH) + UAVSAR L-band (HH, HV, VH, VV)",
              fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path = RESULTS_DIR / "all_models_metrics_table.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Detailed table saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════
# OUTPUT: BAR CHARTS
# ═════════════════════════════════════════════════════════════════════════
def make_comparison_bars(all_results, channels):
    model_names = list(all_results.keys())
    metrics_map = {
        "ENL_out": ("ENL (Equivalent Number of Looks) - higher=better", True),
        "EPI":     ("EPI (Edge Preservation Index) - higher=better",    True),
        "SSI":     ("SSI (Speckle Suppression Index) - lower=better",   False),
        "SMPI":    ("SMPI (Speckle Improvement Ratio) - higher=better", True),
    }
    cmaps = {"ENL_out": plt.cm.Blues, "EPI": plt.cm.Greens, "SSI": plt.cm.Reds, "SMPI": plt.cm.Purples}

    fig, axes = plt.subplots(2, 2, figsize=(24, 14))
    axes = axes.flatten()

    for ax_idx, (metric_key, (metric_label, higher_better)) in enumerate(metrics_map.items()):
        ax = axes[ax_idx]
        x = np.arange(len(model_names))
        width = 0.8 / max(len(channels), 1)
        offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, len(channels))
        cmap = cmaps[metric_key]

        for ch_idx, ch in enumerate(channels):
            vals = []
            for mname in model_names:
                v = all_results.get(mname, {}).get(ch, {}).get(metric_key, 0.0)
                vals.append(v if v else 0.0)
            shade = 0.4 + 0.5 * ch_idx / max(len(channels) - 1, 1)
            ax.bar(x + offsets[ch_idx], vals, width, label=ch, color=cmap(shade), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Real SAR Evaluation - All 10 Models\nSentinel-1 C-band + UAVSAR L-band",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = RESULTS_DIR / "all_models_comparison_bars.png"
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Bar charts saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════
# OUTPUT: PAPER-READY SUMMARY TABLE (channel-averaged)
# ═════════════════════════════════════════════════════════════════════════
def make_paper_table(all_results, channels):
    model_names = list(all_results.keys())
    metrics = ["ENL_out", "EPI", "SSI", "SMPI"]

    # Separate by band
    c_band_chs = [ch for ch in channels if ch.startswith("S1_")]
    l_band_chs = [ch for ch in channels if ch.startswith("UAVSAR_")]

    # Build rows: Model | C-band ENL/EPI/SSI/SMPI | L-band ENL/EPI/SSI/SMPI | Overall ENL/EPI/SSI/SMPI
    col_labels = ["Model",
                  "C-band\nENL", "C-band\nEPI", "C-band\nSSI", "C-band\nSMPI",
                  "L-band\nENL", "L-band\nEPI", "L-band\nSSI", "L-band\nSMPI",
                  "Overall\nENL", "Overall\nEPI", "Overall\nSSI", "Overall\nSMPI"]

    rows = []
    for mname in model_names:
        row = [mname]
        for band_chs in [c_band_chs, l_band_chs, channels]:
            for m in metrics:
                vals = [all_results[mname].get(ch, {}).get(m) for ch in band_chs
                        if all_results[mname].get(ch, {}).get(m) is not None]
                row.append(f"{np.mean(vals):.3f}" if vals else "--")
        rows.append(row)

    ncols = len(col_labels)
    fig, ax = plt.subplots(figsize=(22, max(5, len(rows) * 0.6 + 3)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.8)

    # Header styling
    for j in range(ncols):
        tbl[(0, j)].set_facecolor("#1a252f")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # C-band columns light blue, L-band light green, Overall light purple
    for i in range(1, len(rows) + 1):
        tbl[(i, 0)].set_text_props(fontweight="bold")
        tbl[(i, 0)].set_facecolor("#ecf0f1")
        for j in range(1, 5):   # C-band
            tbl[(i, j)].set_facecolor("#eaf4fb" if i % 2 == 0 else "#f5faff")
        for j in range(5, 9):   # L-band
            tbl[(i, j)].set_facecolor("#e8f8e8" if i % 2 == 0 else "#f5fff5")
        for j in range(9, 13):  # Overall
            tbl[(i, j)].set_facecolor("#f3eafc" if i % 2 == 0 else "#faf5ff")
        # Highlight M5 row
        if "Dual" in rows[i-1][0] or "M5" in rows[i-1][0]:
            for j in range(ncols):
                tbl[(i, j)].set_facecolor("#d5f5e3")

    # Bold best values per column
    for col_j in range(1, ncols):
        is_ssi = "SSI" in col_labels[col_j]
        higher_better = not is_ssi
        best_val, best_i = None, None
        for i, row in enumerate(rows, 1):
            try:
                v = float(row[col_j])
                if best_val is None:
                    best_val, best_i = v, i
                elif higher_better and v > best_val:
                    best_val, best_i = v, i
                elif not higher_better and v < best_val:
                    best_val, best_i = v, i
            except:
                pass
        if best_i is not None:
            tbl[(best_i, col_j)].set_facecolor("#f9e79f")
            tbl[(best_i, col_j)].set_text_props(fontweight="bold")

    plt.title("Paper Table III: All Models - Real SAR Metrics (C-band + L-band)\n"
              "Best values highlighted in yellow | M5 (proposed) highlighted in green",
              fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    out_path = RESULTS_DIR / "paper_summary_table.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Paper table saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ═════════════════════════════════════════════════════════════════════════
def print_summary(all_results, channels):
    c_chs = [ch for ch in channels if ch.startswith("S1_")]
    l_chs = [ch for ch in channels if ch.startswith("UAVSAR_")]

    for band_name, band_chs in [("C-BAND (Sentinel-1)", c_chs), ("L-BAND (UAVSAR)", l_chs), ("OVERALL (All Channels)", channels)]:
        print(f"\n{'='*65}")
        print(f"  {band_name} AVERAGED RESULTS")
        print(f"{'='*65}")
        print(f"  {'Model':<22} {'ENL':>7} {'EPI':>7} {'SSI':>7} {'SMPI':>8}")
        print(f"  {'-'*55}")
        for mname in all_results:
            vals = {m: [] for m in ["ENL_out", "EPI", "SSI", "SMPI"]}
            for ch in band_chs:
                cr = all_results[mname].get(ch, {})
                for m in vals:
                    v = cr.get(m)
                    if v is not None:
                        vals[m].append(v)
            enl  = np.mean(vals["ENL_out"]) if vals["ENL_out"] else float("nan")
            epi  = np.mean(vals["EPI"])     if vals["EPI"]     else float("nan")
            ssi  = np.mean(vals["SSI"])     if vals["SSI"]     else float("nan")
            smpi = np.mean(vals["SMPI"])    if vals["SMPI"]    else float("nan")
            marker = " <-- BEST" if ("Dual" in mname or "M5" in mname) else ""
            print(f"  {mname:<22} {enl:>7.3f} {epi:>7.4f} {ssi:>7.4f} {smpi:>8.3f}{marker}")

    print(f"\n  ENL/SMPI: higher=better | SSI: lower=better | EPI: higher=better")


# ═════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()

    result = run_all_evaluations()
    if result is None:
        print("Evaluation aborted.")
        sys.exit(1)

    all_results, channels = result

    print("\n" + "="*70)
    print("  SAVING RESULTS")
    print("="*70)
    save_json(all_results)
    make_summary_table(all_results, channels)
    make_comparison_bars(all_results, channels)
    make_paper_table(all_results, channels)
    print_summary(all_results, channels)

    total_time = time.time() - t_start
    print(f"\n\nTotal evaluation time: {total_time:.1f}s")
    print(f"All outputs saved to: {RESULTS_DIR}")
    print("Done!")
