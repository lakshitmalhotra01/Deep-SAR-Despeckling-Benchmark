"""evaluate_alos1_all_models.py
================================
Runs the full evaluation of all trained SAR models on the provided ALOS-1 dataset.
The script mirrors `real_sar_all_models_evaluation.py` but loads data from the
`data/ALOS1_Data` folder that the user supplied.

Results (JSON metrics, summary table PNG, bar chart PNG) are written to:
`results/alos1_evaluation/`.
"""

import sys, os
from pathlib import Path
import json, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Project root ────────────────────────────────────────────────────────
PROJECT_ROOT = Path("d:/image video project")
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".cache/torch")
os.environ["TEMP"] = str(PROJECT_ROOT / ".temp")
os.environ["TMP"] = str(PROJECT_ROOT / ".temp")
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache/matplotlib")
os.environ["CUDA_CACHE_PATH"] = str(PROJECT_ROOT / ".cache/nv")
os.environ["PYTHONPYCACHEPREFIX"] = str(PROJECT_ROOT / ".cache/pycache")
for d in [".temp", ".cache/torch", ".cache/matplotlib", ".cache/nv", ".cache/pycache"]:
    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(PROJECT_ROOT))

# Import model definitions (same as the original script)
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

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "ALOS1_Data"
RESULTS_DIR = PROJECT_ROOT / "results" / "alos1_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Model registry (11 models) ────────────────────────────────────────────
def load_all_models():
    CKPT_DIR = PROJECT_ROOT / "checkpoints"
    registry = []
    registry.append(("ResUNet", ResUNet(), CKPT_DIR / "resunet_best.pth"))
    registry.append(("Attention-UNet", AttentionUNet(), CKPT_DIR / "attention_unet_best.pth"))
    registry.append(("MultiResUNet", MultiResUNet(), CKPT_DIR / "multires_unet_best.pth"))
    registry.append(("CBAM-UNet", CBAMUNet(), CKPT_DIR / "cbam_unet_best.pth"))
    registry.append(("SAR2SAR", SAR2SARNet(), CKPT_DIR / "sar2sar_best.pth"))
    registry.append(("M1-NAFNet-Base", NAFNetBaseline(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1,1,1,4]),
                    CKPT_DIR / "nafnet_baseline_best.pth"))
    registry.append(("M2-NAFNet-AGM", NAFNetAdaptiveGate(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1,1,1,4]),
                    CKPT_DIR / "nafnet_adaptive_gate_best.pth"))
    registry.append(("M3-NAFNet-SAM", NAFNetAdaptiveSpatial(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1,1,1,4]),
                    CKPT_DIR / "nafnet_adaptive_spatial_best.pth"))
    registry.append(("M4-NAFNet-CAM", nafnet_adaptive_channel_model(), CKPT_DIR / "nafnet_adaptive_channel_best.pth"))
    registry.append(("M5-NAFNet-Dual*", nafnet_adaptive_dual_model(), CKPT_DIR / "nafnet_adaptive_dual_best.pth"))
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

# ── Data loading (ALOS‑1) ────────────────────────────────────────────────
def load_alos1():
    """Loads all `.bin` files under the ALOS1 data directory.
    For each `.bin` we look for a sibling `.hdr` file that (if present)
    defines `samples` (width) and `lines` (height).  Some headers are named
    `<name>.bin.hdr`; we check both `<name>.hdr` and `<name>.bin.hdr`.
    If the header is missing or dimensions don’t match, we fall back to a
    square reshape.  Returns a dict mapping a short name to a NumPy array.
    """
    data = {}
    for bin_path in DATA_DIR.rglob("*.bin"):
        # Derive a channel name from the relative path, strip extension
        rel = bin_path.relative_to(DATA_DIR)
        name = rel.with_suffix("").as_posix().replace("/", "_")
        try:
            arr = np.fromfile(str(bin_path), dtype=np.float32)
            # Look for possible header files
            hdr_path = bin_path.with_suffix('.hdr')  # e.g., mask_valid_pixels.hdr
            if not hdr_path.exists():
                # Try the .bin.hdr naming convention
                alt_hdr = bin_path.with_name(bin_path.name + '.hdr')
                if alt_hdr.exists():
                    hdr_path = alt_hdr
            if hdr_path.exists():
                samples = None
                lines = None
                with open(hdr_path, 'r') as hf:
                    for line in hf:
                        line = line.strip().lower()
                        if line.startswith('samples'):
                            try:
                                samples = int(line.split('=')[1].strip())
                            except Exception:
                                pass
                        elif line.startswith('lines'):
                            try:
                                lines = int(line.split('=')[1].strip())
                            except Exception:
                                pass
                if samples and lines and samples * lines == arr.size:
                    arr = arr.reshape(lines, samples)
                else:
                    side = int(np.sqrt(arr.size))
                    if side * side == arr.size:
                        arr = arr.reshape(side, side)
            else:
                side = int(np.sqrt(arr.size))
                if side * side == arr.size:
                    arr = arr.reshape(side, side)
            data[name] = arr
            print(f"  Loaded {name}: shape={arr.shape}")
        except Exception as e:
            print(f"  [WARN] Failed to load {bin_path}: {e}")
    return data
    """Loads all `.bin` files under the ALOS1 data directory.
    For each `.bin` we look for a sibling `.hdr` file that (if present)
    defines `samples` (width) and `lines` (height).  If the header is missing
    we fall back to a square reshape.
    The function returns a dict mapping a short name to a NumPy array.
    """
    data = {}
    for bin_path in DATA_DIR.rglob("*.bin"):
        # Derive a channel name from the relative path, strip extension
        rel = bin_path.relative_to(DATA_DIR)
        name = rel.with_suffix("").as_posix().replace("/", "_")
        try:
            arr = np.fromfile(str(bin_path), dtype=np.float32)
            # Try to locate a matching .hdr file for shape info
            hdr_path = bin_path.with_suffix('.hdr')
            if hdr_path.exists():
                # Very simple header parser – look for lines like "samples = N" and "lines   = M"
                samples = None
                lines = None
                with open(hdr_path, 'r') as hf:
                    for line in hf:
                        line = line.strip().lower()
                        if line.startswith('samples'):
                            try:
                                samples = int(line.split('=')[1].strip())
                            except Exception:
                                pass
                        elif line.startswith('lines'):
                            try:
                                lines = int(line.split('=')[1].strip())
                            except Exception:
                                pass
                if samples and lines and samples * lines == arr.size:
                    arr = arr.reshape(lines, samples)
                else:
                    # fallback to square if header dims don't match
                    side = int(np.sqrt(arr.size))
                    if side * side == arr.size:
                        arr = arr.reshape(side, side)
            else:
                # fallback to square reshape
                side = int(np.sqrt(arr.size))
                if side * side == arr.size:
                    arr = arr.reshape(side, side)
            data[name] = arr
            print(f"  Loaded {name}: shape={arr.shape}")
        except Exception as e:
            print(f"  [WARN] Failed to load {bin_path}: {e}")
    return data
    """Loads all `.bin` files under the ALOS1 data directory.
    For each file we assume a single‑band intensity image stored as float32.
    The function returns a dict mapping a short name to a NumPy array.
    """
    data = {}
    for bin_path in DATA_DIR.rglob("*.bin"):
        # Derive a channel name from the relative path
        rel = bin_path.relative_to(DATA_DIR)
        name = rel.with_suffix("").as_posix().replace("/", "_")
        try:
            arr = np.fromfile(str(bin_path), dtype=np.float32)
            # Attempt a square reshape; if not possible keep flat
            side = int(np.sqrt(arr.size))
            if side * side == arr.size:
                arr = arr.reshape(side, side)
            data[name] = arr
            print(f"  Loaded {name}: shape={arr.shape}")
        except Exception as e:
            print(f"  [WARN] Failed to load {bin_path}: {e}")
    return data

# ── Inference helper ────────────────────────────────────────────────────
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

# ── Metric implementations (same as original) ───────────────────────────────
def compute_enl(patch, sub_size=40):
    h, w = patch.shape
    cy, cx = h // 2, w // 2
    r = sub_size // 2
    sub = patch[cy - r : cy + r, cx - r : cx + r]
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
        return np.sqrt(gx ** 2 + gy ** 2)
    gn = grad_mag(noisy)
    gd = grad_mag(denoised)
    num = np.sum(gn * gd)
    den = np.sqrt(np.sum(gn ** 2) * np.sum(gd ** 2))
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
    # Homogeneous patch (center region)
    r0, r1 = 100, min(356, H)
    c0, c1 = 40, min(296, W)
    hom_in = intensity_full[r0:r1, c0:c1]
    if hom_in.shape[0] < 64 or hom_in.shape[1] < 64:
        hom_in = intensity_full[: min(256, H), : min(256, W)]
    hom_out = run_inference(model, hom_in)
    # Edge/urban patch (offset region)
    er0, er1 = 200, min(456, H)
    ec0, ec1 = max(0, min(1000, W - 256)), min(1256, W)
    if ec1 - ec0 < 64:
        ec0, ec1 = 0, min(256, W)
    if er1 - er0 < 64:
        er0, er1 = 0, min(256, H)
    edge_in = intensity_full[er0:er1, ec0:ec1]
    edge_out = run_inference(model, edge_in)
    enl_in = compute_enl(hom_in)
    enl_out = compute_enl(hom_out)
    epi = compute_epi(edge_in, edge_out)
    ssi = compute_ssi(hom_in, hom_out)
    smpi = compute_smpi(hom_in, hom_out)
    return {"ENL_in": round(enl_in, 3), "ENL_out": round(enl_out, 3), "EPI": round(epi, 4), "SSI": round(ssi, 4), "SMPI": round(smpi, 3)}

# ── Main evaluation ────────────────────────────────────────────────────────
def run_all_evaluations():
    print("\n" + "=" * 70)
    print("  ALOS‑1 ALL‑MODELS EVALUATION (11 Models x N Channels)")
    print("=" * 70)
    # Load data
    print("\n[1/3] Loading ALOS‑1 data ...")
    all_channels = load_alos1()
    # Keep only 2‑D intensity channels (skip 1‑D placeholder or mask)
    all_channels = {k:v for k,v in all_channels.items() if v.ndim == 2}
    if not all_channels:
        print("ERROR: No ALOS‑1 data found!")
        return None
    print(f"  Total channels: {len(all_channels)}")
    # Load models
    print("\n[2/3] Loading 11 models ...")
    registry = load_all_models()
    print(f"  Models loaded: {len(registry)}")
    # Evaluate
    print("\n[3/3] Running evaluations ...")
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
                print(f"    {ch_name}: ENL={metrics['ENL_out']:.2f} EPI={metrics['EPI']:.4f} SSI={metrics['SSI']:.4f} SMPI={metrics['SMPI']:.2f}")
            except Exception as e:
                print(f"    ERROR on {ch_name}: {e}")
                model_results[ch_name] = {}
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        all_results[name] = model_results
        del model
        torch.cuda.empty_cache()
    return all_results, list(all_channels.keys())

# ── Output helpers ────────────────────────────────────────────────────────
def save_json(all_results):
    out_path = RESULTS_DIR / "all_models_alos1_metrics.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"  JSON saved: {out_path}")

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
    plt.title("All 11 Models - ALOS‑1 Evaluation", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path = RESULTS_DIR / "all_models_metrics_table.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Detailed table saved: {out_path}")

def make_comparison_bars(all_results, channels):
    model_names = list(all_results.keys())
    metrics_map = {
        "ENL_out": ("ENL (higher=better)", True),
        "EPI": ("EPI (higher=better)", True),
        "SSI": ("SSI (lower=better)", False),
        "SMPI": ("SMPI (higher=better)", True),
    }
    cmaps = {"ENL_out": plt.cm.Blues, "EPI": plt.cm.Greens, "SSI": plt.cm.Reds, "SMPI": plt.cm.Purples}
    fig, axes = plt.subplots(2, 2, figsize=(24, 14))
    for ax, (metric, (title, higher)) in zip(axes.flat, metrics_map.items()):
        values = []
        for name in model_names:
            # aggregate across channels (mean)
            ch_vals = [all_results[name].get(ch, {}).get(metric, np.nan) for ch in channels]
            ch_vals = [v for v in ch_vals if not np.isnan(v)]
            values.append(np.mean(ch_vals) if ch_vals else np.nan)
        ax.bar(model_names, values, color=cmaps[metric](0.6))
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    out_path = RESULTS_DIR / "all_models_comparison_bars.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar charts saved: {out_path}")

if __name__ == "__main__":
    results, channels = run_all_evaluations()
    if results is None:
        sys.exit(1)
    save_json(results)
    make_summary_table(results, channels)
    make_comparison_bars(results, channels)
