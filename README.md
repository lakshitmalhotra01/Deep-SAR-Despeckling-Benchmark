# AdaptiveDual-NAFNet: Context-Adaptive Gated Network with Dual Attention for Satellite Optical & SAR Despeckling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![CUDA Ready](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Target Venue](https://img.shields.io/badge/Target-IEEE%20GRSL-blue)](https://mtt.org/)

Official PyTorch implementation, 11-model benchmarking suite, progressive ablation framework, and multi-sensor real Synthetic Aperture Radar (SAR) evaluation pipeline for **AdaptiveDual-NAFNet**.

---

## 📌 Key Highlights & Novelty

1. **Nonlinear Activation-Free Backbone (NAFNet)**: Replaces compute-heavy non-linear activations (ReLU/GELU) with lightweight computational gating.
2. **Adaptive Gate Mechanism (AGM)**: Novel context-aware gating mechanism replacing static 50/50 Simple Gates, dynamically modulating structural features vs. multiplicative speckle noise based on local radar scattering statistics.
3. **Dual Attention Module**:
   - **Spatial Attention Module (SAM)**: Preserves sharp building boundaries, road edges, and coastlines.
   - **Dual-Pool Channel Attention Module (CAM)**: Captures inter-channel correlations and speckle covariance.
4. **Comprehensive 11-Model Benchmark**:
   - **6 Phase 1 Baselines**: ResUNet, Attention U-Net, MultiResUNet, CBAM U-Net, SAR2SAR, and UNet-Autoencoder.
   - **5 Phase 2 Progressive Ablation Variants**: M1 (Simple Gate) → M2 (+AGM) → M3 (+SAM) → M4 (+CAM) → **M5: AdaptiveDual (Proposed Full Model)**.
5. **Multi-Sensor Real SAR Validation**:
   Evaluated with no-reference quality metrics (**ENL, EPI, SSI, SMPI**) on real complex polarimetric datasets:
   - **Sentinel-1** (C-band SLC Dual-Pol: VV, VH)
   - **NASA/JPL UAVSAR** (Airborne L-band PolSAR: HH, HV, VH, VV)
   - **ALOS-1 PALSAR** (Spaceborne L-band PolSAR: T3, S2, C2 matrices across 11 intensity channels)
   - **UAVSAR Panama Canal** (Airborne L-band PolSAR T3 matrix)
   - **Radarsat-2** (Spaceborne C-band Quad-Pol S2 scattering matrix)

---

## 🏆 Quantitative Benchmark Results

### 1. Synthetic Benchmark (1,000 Untouched Test Images)
Trained on 5,000 synthetic satellite images (3,500 train / 500 val / 1,000 test) degraded with multiplicative Gamma speckle noise ($L = 4$). Evaluated across all 11 models:

| Rank | Model | Type | Parameters | Test PSNR (dB) ↑ | Test SSIM ↑ | Pixel Acc ($\epsilon=0.05$) ↑ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 1 | **M5: AdaptiveDual (Ours)** | **Proposed** | **2.91M** | **38.36** | **0.9389** | **99.21%** |
| 🥈 2 | **M4: NAFNet + AGM + CAM** | Ablation | 2.91M | 38.36 | 0.9388 | 99.20% |
| 🥉 3 | **M3: NAFNet + AGM + SAM** | Ablation | 3.06M | 38.24 | 0.9379 | 99.17% |
| 4 | **M2: NAFNet + AGM** | Ablation | 3.06M | 38.23 | 0.9377 | 99.16% |
| 5 | **M1: NAFNet Baseline** | Ablation | 2.41M | 38.09 | 0.9370 | 99.12% |
| 6 | **ResUNet** | Baseline | 32.44M | 37.22 | 0.9279 | 98.89% |
| 7 | **MultiResUNet** | Baseline | 14.35M | 36.53 | 0.9265 | 98.76% |
| 8 | **Attention U-Net** | Baseline | 32.79M | 36.24 | 0.9229 | 98.64% |
| 9 | **SAR2SAR** | Baseline | 40.83M | 36.01 | 0.9173 | 98.28% |
| 10 | **UNet-Autoencoder** | Baseline | 31.04M | 36.00 | 0.9207 | 98.29% |
| 11 | **CBAM U-Net** | Baseline | 32.66M | 35.90 | 0.9165 | 98.04% |

> **Key takeaway:** The proposed **M5: AdaptiveDual** achieves the highest PSNR (38.36 dB) and SSIM (0.9389) while requiring **less than 10% of the parameter budget** of classic U-Net baselines (2.91M vs 32M+).

---

### 2. Multi-Sensor Real SAR No-Reference Evaluation Summary

Evaluated on genuine complex-valued SAR data using Equivalent Number of Looks (ENL), Edge Preservation Index (EPI), Speckle Suppression Index (SSI), and Speckle Mean Preservation Index (SMPI):

| Dataset / Sensor | Frequency | Mode / Product | Noisy Input ENL | M5 Denoised ENL | ENL Gain ↑ | EPI (Edge) ↑ | SSI (Speckle) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sentinel-1** | C-band | SLC Dual-Pol (VV, VH) | 1.10 – 1.32 | **2.11 – 2.41** | **1.83× – 1.91×** | 0.838 | 0.748 |
| **UAVSAR** | L-band | PolSAR (HH, HV, VH, VV) | 0.74 – 1.29 | **2.13 – 3.08** | **2.20× – 2.92×** | 0.780 | 0.627 |
| **ALOS-1 PALSAR** | L-band | PolSAR (T3, S2, C2 matrices) | 0.95 – 1.05 | **2.61 – 6.86** | **2.50× – 6.50×** | **0.936** | **0.546** |
| **UAVSAR Panama** | L-band | PolSAR ($T_{11}, T_{22}, T_{33}$) | 7.77 – 11.35 | **14.03 – 20.44** | **1.81× – 2.63×** | **0.949** | **0.787** |
| **Radarsat-2** | C-band | Quad-Pol S2 ($s_{11}, s_{12}, s_{21}, s_{22}$) | 0.94 – 0.99 | **1.91 – 3.71** | **1.95× – 3.75×** | **0.953** | **0.761** |

---

## 📊 Visuals & Research Paper Artifacts

All evaluation artifacts and high-resolution figures are organized in `results/`:

| Directory | Key Contents |
| :--- | :--- |
| **`results/Synthetic_data_testing_evaluation/`** | • `bar_comparison_all_models.png` — Test PSNR, SSIM, and Accuracy comparison<br>• `accuracy_curves_all_models.png` — Pixel accuracy vs tolerance threshold $\epsilon$ and cumulative PSNR curve<br>• `training_validation_curves_all_models.png` — 4-panel convergence progression<br>• `boxplot_distributions.png` — Distribution of metrics across 1,000 images<br>• `radar_comparison.png` — Multi-metric radar chart<br>• `psnr_improvement_heatmap.png` — Pairwise model difference matrix<br>• `visual_comparison_image_*.png` — Ground truth vs noisy vs all 11 model outputs |
| **`results/alos1_evaluation/`** | • `all_models_comparison_bars.png`<br>• `all_models_metrics_table.png`<br>• `enl_gain_and_tradeoff.png`<br>• `visual_denoising_comparison.png` |
| **`results/uavsar_panama_evaluation/`** | • `all_models_comparison_bars.png`<br>• `all_models_metrics_table.png`<br>• `enl_gain_and_tradeoff.png`<br>• `visual_denoising_comparison.png` |
| **`results/radarset2_evaluation/`** | • `all_models_comparison_bars.png`<br>• `all_models_metrics_table.png`<br>• `enl_gain_and_tradeoff.png`<br>• `visual_denoising_comparison.png` |
| **`results/unet_autoencoder/`** | • Full dashboard, training curves, PSNR/SSIM progressions, and visual triplets |

---

## 📁 Repository Structure

```
├── models/                           # Neural network architectures
│   ├── nafnet_adaptive_dual.py       # Proposed M5 (NAFNet + AGM + Dual Attention)
│   ├── nafnet_adaptive_channel.py    # M4 (NAFNet + AGM + CAM)
│   ├── nafnet_adaptive_spatial.py    # M3 (NAFNet + AGM + SAM)
│   ├── nafnet_adaptive_gate.py       # M2 (NAFNet + AGM)
│   ├── nafnet_baseline.py            # M1 (NAFNet with Simple Gate)
│   ├── resunet.py                    # Residual U-Net baseline
│   ├── attention_unet.py             # Attention U-Net baseline
│   ├── multires_unet.py              # MultiResUNet baseline
│   ├── cbam_unet.py                  # CBAM U-Net baseline
│   ├── sar2sar.py                    # SAR2SAR baseline
│   ├── unet_autoencoder.py           # U-Net Autoencoder baseline
│   └── losses.py                     # Combined 0.84*L1 + 0.16*(1-SSIM) loss
├── datasets/                         # Dataset loaders and transforms
│   ├── satellite_dataset.py          # S2Looking speckle simulation dataset
│   └── ...
├── preprocessing/                    # Geometric and photometric transforms
├── noise/                            # Multiplicative Gamma speckle noise generator
├── checkpoints/                      # Pretrained weights (see checkpoints/README.md)
├── results/                          # Evaluation outputs and paper figures
│   ├── Synthetic_data_testing_evaluation/  # Unified 11-model synthetic test suite
│   ├── alos1_evaluation/             # ALOS-1 PALSAR evaluation
│   ├── uavsar_panama_evaluation/     # UAVSAR Panama Canal evaluation
│   ├── radarset2_evaluation/         # Radarsat-2 evaluation
│   ├── real_sar_evaluation/          # Sentinel-1 & UAVSAR evaluation
│   ├── unet_autoencoder/             # UNet-AE individual evaluation suite
│   └── ...
├── unified_testing_evaluation.py     # Unified 11-model test evaluation script
├── generate_accuracy_curves.py       # Test & validation accuracy curves generator
├── evaluate_real_sar_datasets.py     # Multi-sensor real SAR evaluation pipeline
├── evaluate_unet_autoencoder.py      # UNet-AE evaluation script
└── README.md
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n sar_denoise python=3.10 -y
conda activate sar_denoise
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install opencv-python matplotlib numpy scipy pillow
```

### 2. Run Unified 11-Model Synthetic Benchmark
Evaluates all 11 models on the 1,000 synthetic test images and generates paper-ready visual grids and metric tables:
```bash
python unified_testing_evaluation.py
python generate_accuracy_curves.py
```

### 3. Run Real SAR Multi-Sensor Evaluation
Evaluates all 11 models on ALOS-1, UAVSAR Panama Canal, and Radarsat-2 polarimetric datasets:
```bash
python evaluate_real_sar_datasets.py
```

### 4. Evaluate Individual Baselines
```bash
python evaluate_unet_autoencoder.py
python real_sar_evaluation.py
```

---

## 📄 Citation & License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
