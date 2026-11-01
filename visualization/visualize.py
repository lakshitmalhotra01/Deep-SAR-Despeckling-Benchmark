"""
visualization/visualize.py
===========================
Visualization utilities for clean vs. noisy satellite images.
"""

import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt


def save_comparison_grid(
    noisy_tensor: torch.Tensor,
    clean_tensor: torch.Tensor,
    save_path: str | Path,
    title: str = "Clean vs Speckle-Corrupted Noisy Image",
):
    """
    Save a side-by-side comparison of a clean image and its synthetic speckle counterpart.

    Parameters
    ----------
    noisy_tensor : torch.Tensor
        (C, H, W) tensor, float32, range [0, 1]
    clean_tensor : torch.Tensor
        (C, H, W) tensor, float32, range [0, 1]
    save_path : str | Path
        Path to save output image file.
    title : str
        Figure title.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensors (C, H, W) -> numpy (H, W, C)
    clean_np = clean_tensor.permute(1, 2, 0).cpu().numpy()
    noisy_np = noisy_tensor.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(clean_np)
    axes[0].set_title("Clean Target (Original)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(noisy_np)
    axes[1].set_title("Noisy Input (Speckle L=4)", fontsize=12)
    axes[1].axis("off")

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    
    target_str = str(save_path)
    try:
        plt.savefig(target_str, dpi=150, bbox_inches="tight")
    except (PermissionError, OSError) as err:
        # If file is locked by Windows Photos viewer, save to fallback name
        fallback_str = str(save_path.with_name(f"{save_path.stem}_new{save_path.suffix}"))
        print(f"  [Note] '{save_path.name}' locked by viewer; saving to '{fallback_str}'")
        plt.savefig(fallback_str, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def save_batch_grid(
    noisy_batch: torch.Tensor,
    clean_batch: torch.Tensor,
    save_path: str | Path,
    num_samples: int = 5,
):
    """
    Save a grid of multiple clean vs noisy pairs.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = min(num_samples, noisy_batch.shape[0])
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 3 * num_samples))

    if num_samples == 1:
        axes = [axes]

    for idx in range(num_samples):
        clean_np = clean_batch[idx].permute(1, 2, 0).cpu().numpy()
        noisy_np = noisy_batch[idx].permute(1, 2, 0).cpu().numpy()

        axes[idx][0].imshow(clean_np)
        axes[idx][0].set_title(f"Sample {idx+1}: Clean", fontsize=10)
        axes[idx][0].axis("off")

        axes[idx][1].imshow(noisy_np)
        axes[idx][1].set_title(f"Sample {idx+1}: Speckle Noisy (L=4)", fontsize=10)
        axes[idx][1].axis("off")

    plt.tight_layout()
    target_str = str(save_path)
    try:
        plt.savefig(target_str, dpi=150, bbox_inches="tight")
    except (PermissionError, OSError):
        fallback_str = str(save_path.with_name(f"{save_path.stem}_new{save_path.suffix}"))
        plt.savefig(fallback_str, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
