"""
preprocessing/transforms.py
============================
Image preprocessing transforms for satellite optical image denoising.

Crop Size Choice — 256 × 256
------------------------------
Original images: 1024 × 1024 px

Reasons for 256 × 256 random crops:
  1. GPU memory efficiency: allows larger batch sizes during training
     (e.g. batch=8 at 256 fits in ~8GB VRAM; 512 would need ~32GB).
  2. Patch diversity: each 1024×1024 image can yield ~16 non-overlapping
     256×256 patches, giving richer data augmentation.
  3. Receptive field: deep U-Net/NAFNet models can still cover the full
     crop with 4 down-sampling levels.
  4. Community standard: 256 is the de-facto standard for image restoration
     benchmarks (SIDD, DND, etc.).

Normalization — [0, 1] min-max (per-image)
-------------------------------------------
PNG images already store uint8 values in [0, 255].
We divide by 255.0 → float32 in [0, 1].
This is compatible with the multiplicative speckle noise model
(noise mean = 1 ensures the noisy image stays in the same range).
We do NOT apply ImageNet mean/std normalization because we are doing
pixel-level reconstruction, not classification.
"""

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
import random


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CROP_SIZE = 256          # Training patch size (H × W)
IMAGE_SIZE = (1024, 1024)  # Native image size — kept for validation/test


# ---------------------------------------------------------------------------
# Core preprocessing helpers
# ---------------------------------------------------------------------------

def pil_to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a float32 tensor in [0, 1].

    Steps:
        1. Ensure RGB (convert if needed)
        2. PIL → torch.Tensor  via ToTensor()  [0,255] uint8 → [0,1] float32
        3. Shape: (C=3, H, W)

    Parameters
    ----------
    img : PIL.Image.Image
        Input image (any mode).

    Returns
    -------
    torch.Tensor
        Float32 tensor, shape (3, H, W), values in [0, 1].
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    tensor = TF.to_tensor(img)   # handles uint8→float32, divides by 255
    return tensor                 # shape: (3, H, W), range: [0, 1]


# ---------------------------------------------------------------------------
# Train transform — random crop + optional horizontal flip
# ---------------------------------------------------------------------------

class TrainTransform:
    """
    Preprocessing pipeline for training images.

    Applied in order:
        1. Convert to RGB
        2. Random crop (256 × 256)
        3. Random horizontal flip (p=0.5) — optional augmentation
        4. Convert to float32 tensor normalized to [0, 1]

    The same random parameters are applied to both the clean image
    (which becomes the TARGET) and the noisy counterpart (generated later
    by the Dataset class using the SpeckleNoise module).
    """

    def __init__(
        self,
        crop_size: int = CROP_SIZE,
        hflip: bool = True,
        vflip: bool = True,
    ):
        self.crop_size = crop_size
        self.hflip = hflip
        self.vflip = vflip

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Parameters
        ----------
        img : PIL.Image.Image

        Returns
        -------
        torch.Tensor  shape (3, crop_size, crop_size), float32, [0,1]
        """
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Random crop
        i, j, h, w = T.RandomCrop.get_params(
            img, output_size=(self.crop_size, self.crop_size)
        )
        img = TF.crop(img, i, j, h, w)

        # Random flips
        if self.hflip and random.random() > 0.5:
            img = TF.hflip(img)
        if self.vflip and random.random() > 0.5:
            img = TF.vflip(img)

        return TF.to_tensor(img)   # (3, 256, 256), float32, [0, 1]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"crop_size={self.crop_size}, "
            f"hflip={self.hflip}, vflip={self.vflip})"
        )


# ---------------------------------------------------------------------------
# Val / Test transform — centre crop (no randomness)
# ---------------------------------------------------------------------------

class ValTestTransform:
    """
    Deterministic preprocessing pipeline for validation and test images.

    Applied in order:
        1. Convert to RGB
        2. Centre crop (256 × 256)  — deterministic, no randomness
        3. Convert to float32 tensor normalized to [0, 1]

    Using a centre crop (rather than the full 1024×1024) keeps validation
    tensors the same shape as training crops, simplifying metric computation.
    For final evaluation you can switch to the full image by setting
    crop_size=1024 or using no-crop mode.
    """

    def __init__(self, crop_size: int = CROP_SIZE):
        self.crop_size = crop_size
        self._transform = T.Compose([
            T.CenterCrop(crop_size),
            T.ToTensor(),          # (3, H, W), float32, [0, 1]
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self._transform(img)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_size={self.crop_size})"
