"""
noise/speckle.py
================
Synthetic multiplicative speckle noise module for satellite optical images.

Model:
    Noisy Image = Clean Image × Speckle Noise
    where Speckle Noise ~ Gamma(L, 1/L)  →  mean=1, variance=1/L

The Number of Looks (L) controls noise intensity:
    - Small L  (e.g. L=1) → strong speckle
    - Large L  (e.g. L=16) → mild speckle
    - L=4 is the default — a realistic single-look average

References:
    Goodman, J.W. (1976). "Some fundamental properties of speckle."
    Oliver, C. & Quegan, S. (2004). "Understanding Synthetic Aperture Radar Images."
"""

import torch
import numpy as np
from typing import Union


def generate_speckle_noise(
    shape: tuple,
    num_looks: int = 4,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate a Gamma-distributed speckle noise map.

    The Gamma distribution used here has:
        shape (alpha) = num_looks  (L)
        scale (theta) = 1 / num_looks   →  mean = 1, variance = 1/L

    Parameters
    ----------
    shape : tuple
        Shape of the output noise tensor, e.g. (C, H, W) or (B, C, H, W).
    num_looks : int
        Number of Looks (L). Higher L → less speckle. Default = 4.
    device : torch.device, optional
        Target device. Defaults to CPU.
    dtype : torch.dtype
        Tensor dtype. Default = float32.

    Returns
    -------
    torch.Tensor
        Speckle noise tensor with the same shape as `shape`.
        Values are positive with mean ≈ 1.0.
    """
    if num_looks < 1:
        raise ValueError(f"num_looks must be >= 1, got {num_looks}")

    # Generate using NumPy (Gamma) then move to tensor
    noise_np = np.random.gamma(
        shape=num_looks,          # alpha = L
        scale=1.0 / num_looks,   # theta = 1/L  → mean = 1
        size=shape,
    ).astype(np.float32)

    noise = torch.from_numpy(noise_np)

    if device is not None:
        noise = noise.to(device)

    return noise.to(dtype)


def apply_speckle_noise(
    clean: torch.Tensor,
    num_looks: int = 4,
    clip: bool = True,
) -> torch.Tensor:
    """
    Apply multiplicative speckle noise to a clean (normalized) image tensor.

    Parameters
    ----------
    clean : torch.Tensor
        Clean image tensor in range [0, 1]. Shape: (C, H, W) or (B, C, H, W).
    num_looks : int
        Number of Looks (L). Default = 4.
    clip : bool
        If True, clamp the noisy output to [0, 1]. Default = True.

    Returns
    -------
    torch.Tensor
        Speckle-corrupted image tensor, same shape and dtype as `clean`.
    """
    noise = generate_speckle_noise(
        shape=clean.shape,
        num_looks=num_looks,
        device=clean.device,
        dtype=clean.dtype,
    )
    noisy = clean * noise

    if clip:
        noisy = torch.clamp(noisy, 0.0, 1.0)

    return noisy


class SpeckleNoise:
    """
    Callable speckle noise transform — compatible with torchvision transforms.

    Usage
    -----
    >>> transform = SpeckleNoise(num_looks=4)
    >>> noisy_tensor = transform(clean_tensor)
    """

    def __init__(self, num_looks: int = 4, clip: bool = True):
        """
        Parameters
        ----------
        num_looks : int
            Number of Looks (L). Default = 4.
        clip : bool
            Clamp output to [0, 1]. Default = True.
        """
        self.num_looks = num_looks
        self.clip = clip

    def __call__(self, clean: torch.Tensor) -> torch.Tensor:
        return apply_speckle_noise(clean, num_looks=self.num_looks, clip=self.clip)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_looks={self.num_looks}, clip={self.clip})"
