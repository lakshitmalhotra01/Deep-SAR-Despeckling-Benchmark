"""
models/losses.py
================
Loss functions for image denoising:
  - Combined L1 + SSIM Loss
  - MAE (L1) Loss
  - MSE (L2) Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.exp(
        torch.tensor([-(x - window_size // 2) ** 2 / (2 * sigma ** 2) for x in range(window_size)])
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, val_range: float = 1.0) -> torch.Tensor:
    img1 = img1.float()
    img2 = img2.float()
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device, dtype=torch.float32)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class CombinedL1SSIMLoss(nn.Module):
    """
    Combined Loss = alpha * L1_loss + (1 - alpha) * (1 - SSIM)
    """

    def __init__(self, alpha: float = 0.84, window_size: int = 11):
        super(CombinedL1SSIMLoss, self).__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.l1_loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_loss(pred, target)
        ssim_val = ssim(pred, target, window_size=self.window_size)
        loss = self.alpha * l1 + (1 - self.alpha) * (1.0 - ssim_val)
        return loss
