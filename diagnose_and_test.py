"""
diagnose_and_test.py
====================
Diagnosis and verification script for Residual U-Net (ResUNet) output ranges,
RGB channel ordering, loss minimization, and visualization checks.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.transforms import TrainTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import SatelliteSpeckleDataset, TRAIN_DIR
from models.resunet import ResUNet
from models.losses import CombinedL1SSIMLoss, ssim


def run_diagnosis():
    print("=" * 70, flush=True)
    print("STEP 1 & 2 & 3: RESUNET ARCHITECTURE & DIAGNOSTIC VERIFICATION", flush=True)
    print("=" * 70, flush=True)

    # 1. Architecture details
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet(in_channels=3, out_channels=3).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Architecture         : Residual U-Net (ResUNet)")
    print(f"Encoder Structure    : 4 Residual Levels [64, 128, 256, 512] + MaxPool2d")
    print(f"Bottleneck Structure : 1024-channel Residual Block")
    print(f"Decoder Structure    : 4 ConvTranspose2d levels [512, 256, 128, 64] + Skip Connections")
    print(f"Final Output Layer   : Conv2d(64, 3, 1) + Sigmoid()")
    print(f"Input Channels       : 3 (RGB)")
    print(f"Output Channels      : 3 (RGB)")
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Target Device        : {device}\n", flush=True)

    # 2. Inspect Dataset Item
    ds = SatelliteSpeckleDataset(
        image_dir=TRAIN_DIR,
        transform=TrainTransform(crop_size=256),
        noise_fn=SpeckleNoise(num_looks=4)
    )
    noisy, clean = ds[0]

    # Print ranges before model pass
    print("--- Tensor Range & Stats Diagnosis ---", flush=True)
    print(f"Noisy Input : min={noisy.min():.4f}, max={noisy.max():.4f}, mean={noisy.mean():.4f}, shape={tuple(noisy.shape)}")
    print(f"Clean Target: min={clean.min():.4f}, max={clean.max():.4f}, mean={clean.mean():.4f}, shape={tuple(clean.shape)}")

    # 3. Model Untrained Output Diagnosis
    noisy_batch = noisy.unsqueeze(0).to(device)
    clean_batch = clean.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        untrained_output = model(noisy_batch).squeeze(0).cpu()

    print(f"Untrained Model Output: min={untrained_output.min():.4f}, max={untrained_output.max():.4f}, mean={untrained_output.mean():.4f}")
    print("  Note: Untrained Sigmoid output centers around 0.5 (neutral grey/purple in raw matplotlib view).")

    # 4. Verify Loss Minimization Formula
    criterion = CombinedL1SSIMLoss(alpha=0.84)
    l1_val = torch.nn.functional.l1_loss(untrained_output.to(device), clean_batch.squeeze(0))
    ssim_val = ssim(untrained_output.unsqueeze(0).to(device), clean_batch)
    loss_val = criterion(untrained_output.unsqueeze(0).to(device), clean_batch)

    print(f"\n--- Loss Function Verification ---")
    print(f"  Formula      : Loss = 0.84 * L1 + 0.16 * (1 - SSIM)")
    print(f"  L1 Term      : {l1_val.item():.4f}")
    print(f"  SSIM Term    : {ssim_val.item():.4f}  -> (1 - SSIM) = {1.0 - ssim_val.item():.4f}")
    print(f"  Combined Loss: {loss_val.item():.4f}")
    print(f"  Minimization Check: As SSIM approaches 1.0, (1 - SSIM) approaches 0.0 -> Minimization maximizes SSIM!")

    # 5. Run 10-step Gradient Verification
    print("\n--- 10-Step Quick Training Sanity Check ---", flush=True)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    losses = []
    for step in range(1, 11):
        optimizer.zero_grad()
        out = model(noisy_batch)
        loss = criterion(out, clean_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  Step  1 Loss: {losses[0]:.5f}")
    print(f"  Step  5 Loss: {losses[4]:.5f}")
    print(f"  Step 10 Loss: {losses[-1]:.5f}")

    model.eval()
    with torch.no_grad():
        trained_output = model(noisy_batch).squeeze(0).cpu()
    print(f"Trained Model Output (10 steps): min={trained_output.min():.4f}, max={trained_output.max():.4f}, mean={trained_output.mean():.4f}")

    # 6. Save Diagnostic Visualization
    results_dir = PROJECT_ROOT / "results" / "diagnosis"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(noisy.permute(1, 2, 0).numpy())
    axes[0].set_title("Noisy Input (Speckle L=4)")
    axes[0].axis("off")

    axes[1].imshow(trained_output.permute(1, 2, 0).numpy())
    axes[1].set_title("ResUNet Denoised (10 steps)")
    axes[1].axis("off")

    axes[2].imshow(clean.permute(1, 2, 0).numpy())
    axes[2].set_title("Clean Target (Original)")
    axes[2].axis("off")

    plt.suptitle("ResUNet Diagnosis: RGB Rendering & Loss Verification", fontsize=14)
    plt.tight_layout()
    diag_out = results_dir / "resunet_diagnosis_triplet.png"
    plt.savefig(str(diag_out), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Diagnosis triplet saved to: {diag_out.name}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_diagnosis()
