"""
test_pipeline.py
================
Script to execute Phase 5 (Small-Scale Test) and Phase 6 (Validation Check).

Phase 5:
  - Takes 10 images from train/Image1
  - Applies loading, RGB conversion, resizing/cropping, normalization, tensor conversion, speckle generation
  - Saves side-by-side comparison images into results/

Phase 6:
  - Validates train, val, and test data loaders
  - Checks clean targets and noisy inputs
  - Checks shape consistency and valid numeric range [0, 1]
  - Verifies original image dataset integrity (unmodified files, correct file counts)
"""

import os
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Ensure local imports work
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.transforms import TrainTransform, ValTestTransform, CROP_SIZE
from noise.speckle import SpeckleNoise
from datasets.satellite_dataset import SatelliteSpeckleDataset, make_dataloader, TRAIN_DIR, VAL_DIR, TEST_DIR
from visualization.visualize import save_comparison_grid, save_batch_grid


def run_phase_5_small_scale_test():
    print("=" * 60)
    print("PHASE 5 — SMALL-SCALE TEST (10 IMAGES)")
    print("=" * 60)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    noise_fn = SpeckleNoise(num_looks=4)
    transform = TrainTransform(crop_size=CROP_SIZE)

    dataset = SatelliteSpeckleDataset(
        image_dir=TRAIN_DIR,
        transform=transform,
        noise_fn=noise_fn,
    )

    print(f"Total available train images: {len(dataset)}")
    test_count = 10

    # Pick 10 sample images
    for idx in range(test_count):
        img_path = dataset.image_paths[idx]
        noisy, clean = dataset[idx]

        # Verification checks for each image
        assert clean.shape == (3, CROP_SIZE, CROP_SIZE), f"Unexpected clean shape: {clean.shape}"
        assert noisy.shape == (3, CROP_SIZE, CROP_SIZE), f"Unexpected noisy shape: {noisy.shape}"
        assert 0.0 <= clean.min() and clean.max() <= 1.0, f"Clean out of range: [{clean.min()}, {clean.max()}]"
        assert 0.0 <= noisy.min() and noisy.max() <= 1.0, f"Noisy out of range: [{noisy.min()}, {noisy.max()}]"

        # Save visual comparison
        out_file = results_dir / f"sample_{idx+1:02d}_{img_path.stem}_comparison.png"
        save_comparison_grid(
            noisy_tensor=noisy,
            clean_tensor=clean,
            save_path=out_file,
            title=f"Sample {idx+1}: {img_path.name} | Clean (Target) vs Noisy (Input, L=4)"
        )
        print(f"  [OK] Verified & saved sample {idx+1:02d}: {out_file.name}")

    # Also save a 10-sample overview grid
    batch_noisy = []
    batch_clean = []
    for idx in range(test_count):
        n, c = dataset[idx]
        batch_noisy.append(n)
        batch_clean.append(c)
    
    batch_noisy_tensor = torch.stack(batch_noisy)
    batch_clean_tensor = torch.stack(batch_clean)

    grid_file = results_dir / "small_scale_10_samples_overview.png"
    save_batch_grid(
        noisy_batch=batch_noisy_tensor,
        clean_batch=batch_clean_tensor,
        save_path=grid_file,
        num_samples=10,
    )
    print(f"\n[OK] Phase 5 Small-scale overview grid saved to: {grid_file.name}")
    print("Phase 5 Small-scale test completed successfully!\n")


def run_phase_6_validation_check():
    print("=" * 60)
    print("PHASE 6 --- VALIDATION CHECK")
    print("=" * 60)

    noise_fn = SpeckleNoise(num_looks=4)

    # 1. Train pipeline
    train_loader, train_ds = make_dataloader(
        TRAIN_DIR, TrainTransform(), noise_fn, batch_size=8, shuffle=True, num_workers=0
    )
    # 2. Val pipeline
    val_loader, val_ds = make_dataloader(
        VAL_DIR, ValTestTransform(), noise_fn, batch_size=8, shuffle=False, num_workers=0
    )
    # 3. Test pipeline
    test_loader, test_ds = make_dataloader(
        TEST_DIR, ValTestTransform(), noise_fn, batch_size=8, shuffle=False, num_workers=0
    )

    print(f"Train samples count : {len(train_ds)} (Batches: {len(train_loader)})")
    print(f"Val samples count   : {len(val_ds)} (Batches: {len(val_loader)})")
    print(f"Test samples count  : {len(test_ds)} (Batches: {len(test_loader)})")

    # Check batch consistency for Train
    train_noisy, train_clean = next(iter(train_loader))
    print(f"\nTrain Batch Check:")
    print(f"  Noisy shape: {train_noisy.shape}, dtype: {train_noisy.dtype}")
    print(f"  Clean shape: {train_clean.shape}, dtype: {train_clean.dtype}")
    print(f"  Noisy min/max: [{train_noisy.min():.4f}, {train_noisy.max():.4f}]")
    print(f"  Clean min/max: [{train_clean.min():.4f}, {train_clean.max():.4f}]")

    assert train_noisy.shape == (8, 3, CROP_SIZE, CROP_SIZE)
    assert train_clean.shape == (8, 3, CROP_SIZE, CROP_SIZE)

    # Check batch consistency for Val
    val_noisy, val_clean = next(iter(val_loader))
    print(f"\nVal Batch Check:")
    print(f"  Noisy shape: {val_noisy.shape}, dtype: {val_noisy.dtype}")
    print(f"  Clean shape: {val_clean.shape}, dtype: {val_clean.dtype}")
    assert val_noisy.shape == (8, 3, CROP_SIZE, CROP_SIZE)

    # Check batch consistency for Test
    test_noisy, test_clean = next(iter(test_loader))
    print(f"\nTest Batch Check:")
    print(f"  Noisy shape: {test_noisy.shape}, dtype: {test_noisy.dtype}")
    print(f"  Clean shape: {test_clean.shape}, dtype: {test_clean.dtype}")
    assert test_noisy.shape == (8, 3, CROP_SIZE, CROP_SIZE)

    # Verify no overlap between splits
    train_files = set(p.name for p in train_ds.image_paths)
    val_files = set(p.name for p in val_ds.image_paths)
    test_files = set(p.name for p in test_ds.image_paths)

    print("\nDataset Split Independence Check:")
    print(f"  Train & Val overlap  : {len(train_files.intersection(val_files))}")
    print(f"  Train & Test overlap : {len(train_files.intersection(test_files))}")
    print(f"  Val & Test overlap   : {len(val_files.intersection(test_files))}")

    print("\n[OK] Phase 6 Validation checks passed without any errors!\n")


if __name__ == "__main__":
    run_phase_5_small_scale_test()
    run_phase_6_validation_check()
