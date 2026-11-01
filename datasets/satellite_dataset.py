"""
datasets/satellite_dataset.py
==============================
PyTorch Dataset and DataLoader factory for satellite optical image denoising.

Pipeline:
    Clean Image  →  [Transform]  →  Clean Tensor  →  [SpeckleNoise]  →  Noisy Tensor

At training time:
    INPUT  = Noisy Tensor   (speckle-corrupted)
    TARGET = Clean Tensor   (original, untouched)

Design decisions:
    - Noise is generated DYNAMICALLY per sample — no pre-saved noisy copies.
    - The SpeckleNoise transform is injected at construction time, so the
      Number of Looks (L) can be changed by passing a different noise_fn.
    - Augmentation (random crop, flips) is in the image transform, not here.
    - Three separate dataset instances share the same code but different paths
      and transforms (TrainTransform vs. ValTestTransform).
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Default paths (relative to project root)
DEFAULT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = DEFAULT_ROOT / "train" / "Image1"
VAL_DIR   = DEFAULT_ROOT / "val"   / "Image1"
TEST_DIR  = DEFAULT_ROOT / "test"  / "Image1"

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SatelliteSpeckleDataset(Dataset):
    """
    Dataset for synthetic speckle-noise denoising on satellite optical images.

    Each item returns:
        noisy  : torch.Tensor  (C, H, W)  — speckle-corrupted input
        clean  : torch.Tensor  (C, H, W)  — original clean target

    Parameters
    ----------
    image_dir : str | Path
        Directory containing the clean images (e.g. train/Image1).
    transform : Callable, optional
        Transform applied to the PIL image before noise.
        Typically TrainTransform() or ValTestTransform().
        If None, images are converted to [0,1] tensors without cropping.
    noise_fn : Callable, optional
        A callable that receives a clean tensor and returns a noisy tensor.
        Typically a SpeckleNoise(num_looks=L) instance.
        If None, noisy == clean (useful for debugging).
    """

    def __init__(
        self,
        image_dir: str | Path,
        transform: Optional[Callable] = None,
        noise_fn: Optional[Callable] = None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.noise_fn = noise_fn

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXT
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No supported images found in: {self.image_dir}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]

        # ── Load ──────────────────────────────────────────────────────
        img = Image.open(img_path)

        # ── RGB conversion ────────────────────────────────────────────
        if img.mode != "RGB":
            img = img.convert("RGB")

        # ── Transform (crop / flip / to_tensor / normalize) ───────────
        if self.transform is not None:
            clean = self.transform(img)          # → (C, H, W) float32 [0,1]
        else:
            import torchvision.transforms.functional as TF
            clean = TF.to_tensor(img)            # fallback: no crop

        # ── Speckle noise (dynamic, in-memory) ────────────────────────
        if self.noise_fn is not None:
            noisy = self.noise_fn(clean)
        else:
            noisy = clean.clone()

        return noisy, clean                      # (input, target)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n={len(self)}, "
            f"dir='{self.image_dir.name}', "
            f"transform={self.transform}, "
            f"noise_fn={self.noise_fn})"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    image_dir: str | Path,
    transform: Optional[Callable],
    noise_fn: Optional[Callable],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, SatelliteSpeckleDataset]:
    """
    Convenience factory that creates a Dataset + DataLoader in one call.

    Returns
    -------
    (DataLoader, SatelliteSpeckleDataset)
    """
    dataset = SatelliteSpeckleDataset(
        image_dir=image_dir,
        transform=transform,
        noise_fn=noise_fn,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(shuffle),   # drop last incomplete batch during training
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# Quick usage example (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(DEFAULT_ROOT))

    from preprocessing.transforms import TrainTransform, ValTestTransform
    from noise.speckle import SpeckleNoise

    noise_fn = SpeckleNoise(num_looks=4)

    train_loader, train_ds = make_dataloader(
        TRAIN_DIR, TrainTransform(), noise_fn, batch_size=4, num_workers=0
    )
    val_loader,   val_ds   = make_dataloader(
        VAL_DIR,   ValTestTransform(), noise_fn, batch_size=4, shuffle=False, num_workers=0
    )
    test_loader,  test_ds  = make_dataloader(
        TEST_DIR,  ValTestTransform(), noise_fn, batch_size=4, shuffle=False, num_workers=0
    )

    print(f"Train dataset : {len(train_ds)} images")
    print(f"Val   dataset : {len(val_ds)}   images")
    print(f"Test  dataset : {len(test_ds)}  images")

    noisy_batch, clean_batch = next(iter(train_loader))
    print(f"Batch shapes  : noisy={tuple(noisy_batch.shape)}  clean={tuple(clean_batch.shape)}")
    print(f"Noisy range   : [{noisy_batch.min():.4f}, {noisy_batch.max():.4f}]")
    print(f"Clean range   : [{clean_batch.min():.4f}, {clean_batch.max():.4f}]")
    print("DataLoader factory smoke-test PASSED.")
