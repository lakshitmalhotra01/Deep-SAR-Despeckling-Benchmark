# AdaptiveDual-NAFNet: Context-Adaptive Gated Network with Dual Attention for Satellite Optical & SAR Despeckling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Target Venue](https://img.shields.io/badge/Target-IEEE%20GRSL-blue)](https://mtt.org/)

This repository contains the official PyTorch implementation, baseline benchmarking suite, progressive ablation framework, and real Synthetic Aperture Radar (SAR) evaluation pipeline for our **AdaptiveDual-NAFNet** speckle reduction framework.

---

## 📌 Key Highlights & Novelty

1. **Nonlinear Activation-Free Backbone (NAFNet)**: Replaces traditional heavy non-linear activations (ReLU/GELU) with lightweight computational gating.
2. **Adaptive Gate Mechanism (AGM)**: A novel context-aware gating mechanism replacing static 50/50 Simple Gates, dynamically modulating structural features vs. multiplicative speckle noise based on local radar scattering statistics.
3. **Dual Attention Module**:
   - **Spatial Attention Module (SAM)**: Preserves sharp building boundaries, road edges, and coastlines.
   - **Dual-Pool Channel Attention Module (CAM)**: Captures inter-channel correlations and speckle covariance.
4. **Comprehensive Benchmarking**:
   - **5 Phase 1 SOTA Baselines**: ResUNet, Attention U-Net, MultiResUNet, CBAM U-Net, and SAR2SAR.
   - **5 Phase 2 Progressive Ablation Variants**: M1 (Simple Gate) → M2 (AGM) → M3 (+SAM) → M4 (+CAM) → M5 (Proposed Full Dual Attention).
5. **Real Satellite SAR Validation**: Evaluated with no-reference quality metrics (**ENL, EPI, SSI, SMPI**) on:
   - **Sentinel-1** (C-band SLC Dual-Pol: VV, VH)
   - **NASA/JPL UAVSAR** (Airborne L-band PolSAR: HH, HV, VH, VV)

---

## 🏆 Quantitative Benchmark Results

### 1. Optical Synthetic Benchmark (1,000 Untouched Test Images)
| Model | Type | Params | Test PSNR (dB) | Test SSIM | Pixel Acc ($\epsilon=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResUNet** | Baseline | 32.4M | 37.21 | 0.9278 | 98.89% |
| **MultiResUNet** | Baseline | 15.3M | 36.52 | 0.9264 | 98.76% |
| **Attention U-Net** | Baseline | 34.8M | 36.24 | 0.9229 | 98.64% |
| **SAR2SAR** | Baseline | 43.4M | 36.01 | 0.9173 | 98.29% |
| **CBAM U-Net** | Baseline | 34.7M | 35.90 | 0.9164 | 98.03% |
| **M1: NAFNet Baseline** | Ablation | 2.58M | 38.09 | 0.9371 | 99.12% |
| **M2: NAFNet + AGM** | Ablation | 3.28M | 38.23 | 0.9377 | 99.16% |
| **M3: NAFNet + AGM + SAM** | Ablation | 3.29M | 38.23 | 0.9378 | 99.17% |
| **M4: NAFNet + AGM + CAM** | Ablation | 3.12M | 38.26 | 0.9381 | 99.18% |
| **M5: NAFNet-AdaptiveDual (Ours)** | **Proposed** | **2.91M** | **38.36** | **0.9389** | **99.21%** |

### 2. Real SAR Data No-Reference Evaluation
| Channel | Sensor | ENL (Noisy) | ENL (M5 Denoised) | ENL Gain | EPI (Edge) | SSI (Speckle) | SMPI (Mean) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sentinel-1 VV** | C-band SLC | 1.32 | **2.41** | **1.83×** | 0.8404 | 0.7652 | 1.0339 |
| **Sentinel-1 VH** | C-band SLC | 1.10 | **2.11** | **1.91×** | 0.8357 | 0.7319 | 1.0127 |
| **UAVSAR HH** | L-band PolSAR | 1.29 | **2.83** | **2.20×** | 0.8072 | 0.6887 | 1.0218 |
| **UAVSAR HV** | L-band PolSAR | 0.79 | **2.13** | **2.70×** | 0.7598 | 0.6015 | 0.9888 |
| **UAVSAR VH** | L-band PolSAR | 0.74 | **2.17** | **2.92×** | 0.7587 | 0.5806 | 0.9925 |
| **UAVSAR VV** | L-band PolSAR | 1.23 | **3.08** | **2.50×** | 0.7964 | 0.6384 | 1.0098 |

---

## 📁 Repository Structure

```
├── models/                     # Deep learning architectures
│   ├── nafnet_adaptive_dual.py # Proposed M5 (NAFNet + AGM + Dual Attention)
│   ├── nafnet_adaptive_gate.py # M2 (NAFNet + AGM)
│   ├── nafnet_adaptive_spatial.py # M3 (NAFNet + AGM + Spatial Attention)
│   ├── nafnet_adaptive_channel.py # M4 (NAFNet + AGM + Channel Attention)
│   ├── nafnet_baseline.py      # M1 (NAFNet with Simple Gate)
│   ├── resunet.py              # Residual U-Net Baseline
│   ├── attention_unet.py       # Attention U-Net Baseline
│   ├── multires_unet.py        # MultiResUNet Baseline
│   ├── cbam_unet.py            # CBAM U-Net Baseline
│   ├── sar2sar.py              # SAR2SAR Baseline
│   └── losses.py               # Combined L1 + SSIM Loss Function
├── phase2/                     # Phase 2 training and evaluation scripts (M1 - M5)
├── noise/                      # Multiplicative Gamma speckle noise generator
├── preprocessing/              # Transforms and dataset loaders
├── datasets/                   # PyTorch dataset implementations
├── real_sar/                   # Real Sentinel-1 & UAVSAR polarimetric arrays
├── results/                    # Metric summaries, CSV logs, and visual comparison figures
│   ├── real_sar_evaluation/    # Real SAR evaluation figures & tables
│   ├── nafnet_adaptive_dual/   # M5 benchmark & diagnostic plots
│   └── ...                     # Baselines evaluation artifacts
├── checkpoints/                # Model checkpoint directory (see checkpoints/README.md)
├── real_sar_evaluation.py      # Automated real SAR evaluation pipeline
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup
```bash
conda create -n sar_denoise python=3.10 -y
conda activate sar_denoise
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib numpy pillow
```

### 2. Run Real SAR Evaluation
To run the automated no-reference evaluation of the proposed model on Sentinel-1 and UAVSAR:
```bash
python real_sar_evaluation.py
```

### 3. Evaluate Proposed Model on Benchmark
```bash
python phase2/evaluate_nafnet_adaptive_dual.py
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
