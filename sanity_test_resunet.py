"""
sanity_test_resunet.py
======================
Small-Scale Sanity Test for Residual U-Net (ResUNet).

Tests:
  1. Instantiates ResUNet model on available device (GPU/CPU).
  2. Runs 5 training epochs on a small 16-image subset of train/Image1.
  3. Verifies gradient flow and loss reduction across epochs.
  4. Saves visual comparisons: [Noisy Input | ResUNet Denoised | Clean Target] to results/sanity_test/.
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.transforms import TrainTransform
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import SatelliteSpeckleDataset, TRAIN_DIR
from models.resunet import ResUNet
from models.losses import CombinedL1SSIMLoss


def save_triplet_comparison(noisy: torch.Tensor, denoised: torch.Tensor, clean: torch.Tensor, save_path: Path, title: str):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_np = noisy.permute(1, 2, 0).detach().cpu().numpy()
    d_np = denoised.permute(1, 2, 0).detach().cpu().numpy()
    c_np = clean.permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(n_np)
    axes[0].set_title("Input (Speckle Noisy L=4)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(d_np)
    axes[1].set_title("ResUNet Denoised Output", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(c_np)
    axes[2].set_title("Target (Original Clean)", fontsize=11)
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_sanity_test():
    print("=" * 65, flush=True)
    print("RESUNET SMALL-SCALE SANITY TEST (16 IMAGES, 5 EPOCHS)", flush=True)
    print("=" * 65, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}", flush=True)

    # 1. Dataset & DataLoader (subset of 16 images)
    full_ds = SatelliteSpeckleDataset(
        image_dir=TRAIN_DIR,
        transform=TrainTransform(crop_size=256),
        noise_fn=SpeckleNoise(num_looks=4),
    )
    subset_ds = Subset(full_ds, range(16))
    loader = DataLoader(subset_ds, batch_size=4, shuffle=True)

    # 2. Model, Loss, Optimizer
    model = ResUNet(in_channels=3, out_channels=3).to(device)
    criterion = CombinedL1SSIMLoss(alpha=0.84)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {param_count:,}", flush=True)
    print("Starting 5-Epoch Sanity Training Loop...", flush=True)

    epoch_losses = []
    start_time = time.time()

    for epoch in range(1, 6):
        model.train()
        running_loss = 0.0

        for batch_idx, (noisy, clean) in enumerate(loader):
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)

        epoch_loss = running_loss / len(subset_ds)
        epoch_losses.append(epoch_loss)
        print(f"  Epoch [{epoch}/5] - Loss: {epoch_loss:.5f}", flush=True)

    elapsed = time.time() - start_time
    print(f"\nSanity training completed in {elapsed:.1f}s!", flush=True)

    # Loss reduction check
    initial_loss = epoch_losses[0]
    final_loss   = epoch_losses[-1]
    print(f"Initial Loss : {initial_loss:.5f}")
    print(f"Final Loss   : {final_loss:.5f}")
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    print("[OK] Gradient flow & loss convergence verified!")

    # 3. Save Visual Evaluation
    model.eval()
    results_dir = PROJECT_ROOT / "results" / "sanity_test"
    with torch.no_grad():
        noisy_sample, clean_sample = subset_ds[0]
        noisy_in = noisy_sample.unsqueeze(0).to(device)
        denoised_out = model(noisy_in).squeeze(0)

        save_triplet_comparison(
            noisy=noisy_sample,
            denoised=denoised_out,
            clean=clean_sample,
            save_path=results_dir / "resunet_sanity_triplet_comparison.png",
            title="ResUNet Sanity Test: Noisy Input vs Denoised vs Clean Target"
        )
        print(f"[OK] Visual comparison triplet saved to: results/sanity_test/resunet_sanity_triplet_comparison.png", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("RESUNET SANITY TEST PASSED! READY FOR FULL-DATASET TRAINING.", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    run_sanity_test()
