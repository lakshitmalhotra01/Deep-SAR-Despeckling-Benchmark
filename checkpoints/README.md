# Model Checkpoints Directory

This directory stores the trained model weights (.pth files) for:

### Phase 1: Baseline Benchmarks
1. `resunet_best.pth` (Residual U-Net - 32.4M params)
2. `attention_unet_best.pth` (Attention U-Net - 34.8M params)
3. `multires_unet_best.pth` (MultiResUNet - 15.3M params)
4. `cbam_unet_best.pth` (CBAM U-Net - 34.7M params)
5. `sar2sar_best.pth` (SAR2SAR - 43.4M params)

### Phase 2: Proposed Architecture & Progressive Ablation
1. `nafnet_baseline_best.pth` (M1: NAFNet with Simple Gate - 2.58M params)
2. `nafnet_adaptive_gate_best.pth` (M2: NAFNet + Adaptive Gate Mechanism [AGM] - 3.28M params)
3. `nafnet_adaptive_spatial_best.pth` (M3: NAFNet + AGM + Spatial Attention - 3.29M params)
4. `nafnet_adaptive_channel_best.pth` (M4: NAFNet + AGM + Channel Attention - 3.12M params)
5. `nafnet_adaptive_dual_best.pth` (M5: Proposed NAFNet + AGM + Dual Attention - 2.91M params)

> Note: Large model checkpoints (>100MB) are ignored by git to comply with GitHub file size restrictions. Download trained checkpoints from the release page or train using the provided training scripts.
