"""
models/unet_autoencoder.py
==========================
U-Net Denoising Autoencoder (UNet-Autoencoder) Architecture for Satellite Optical
and Synthetic Aperture Radar (SAR) Speckle Denoising.

Key Architecture Features:
  1. Contracting Path (Encoder):
     - Repeated application of two 3x3 convolutions (unpadded / padded to preserve dimensions),
       each followed by Batch Normalization and ReLU activation.
     - 2x2 Max Pooling with stride 2 for downsampling at each level.
     - Progressively doubles feature channels: 64 -> 128 -> 256 -> 512.
  2. Latent Bottleneck:
     - Deepest feature representation capturing global spatial context and semantic scene structure
       at 1024 feature channels.
  3. Expanding Path (Decoder):
     - 2x2 Transposed Convolution (stride 2) for upsampling at each level.
     - Skip connections: Concatenation with corresponding high-resolution feature maps from
       the contracting encoder path to recover fine-grained spatial and edge details.
     - Two 3x3 convolutions + BatchNorm + ReLU after each concatenation.
  4. Output Reconstruction Head:
     - 1x1 convolution mapping 64 channels to 3 output channels.
     - Final Sigmoid activation strictly bounding output reflectance to [0, 1].

Reference:
  Ronneberger, O., Fischer, P., & Brox, T. (2015).
  "U-Net: Convolutional Networks for Biomedical Image Segmentation."
  MICCAI 2015. arXiv:1505.04597.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling then DoubleConv with skip connection concatenation"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super(UpBlock, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle potential padding discrepancies for arbitrary input dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channel axis: [Encoder_Skip, Decoder_Upsampled]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetAutoencoder(nn.Module):
    """
    Standard U-Net Denoising Autoencoder Architecture.

    Parameters:
      in_channels  : int, default 3 (RGB optical / multi-channel SAR)
      out_channels : int, default 3 (Clean reconstructed output)
      base_channels: int, default 64 (Starting channel depth)
      bilinear     : bool, default False (Uses learnable ConvTranspose2d)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64, bilinear: bool = False):
        super(UNetAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # ── Encoder (Contracting Path) ──────────────────────────────────────────
        self.inc = DoubleConv(in_channels, base_channels)                       # 3 -> 64
        self.down1 = DownBlock(base_channels, base_channels * 2)               # 64 -> 128
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)           # 128 -> 256
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)           # 256 -> 512

        # ── Latent Bottleneck ───────────────────────────────────────────────────
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(base_channels * 8, (base_channels * 16) // factor)  # 512 -> 1024 (or 512)

        # ── Decoder (Expanding Path with Skip Connections) ──────────────────────
        self.up1 = UpBlock(base_channels * 16, (base_channels * 8) // factor, bilinear)
        self.up2 = UpBlock(base_channels * 8, (base_channels * 4) // factor, bilinear)
        self.up3 = UpBlock(base_channels * 4, (base_channels * 2) // factor, bilinear)
        self.up4 = UpBlock(base_channels * 2, base_channels, bilinear)

        # ── Reconstruction Head ────────────────────────────────────────────────
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x5 = self.down4(x4)

        # Decoder forward pass with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Reconstructed output in range [0, 1]
        logits = self.outc(x)
        return self.sigmoid(logits)


def unet_autoencoder_model(in_channels: int = 3, out_channels: int = 3) -> UNetAutoencoder:
    """Factory function for UNetAutoencoder"""
    return UNetAutoencoder(in_channels=in_channels, out_channels=out_channels, base_channels=64)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = unet_autoencoder_model().to(device)
    dummy = torch.randn(2, 3, 256, 256).to(device)
    out = model(dummy)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNetAutoencoder successfully initialized on {device}.")
    print(f"Input Shape : {dummy.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Total Trainable Parameters: {params:,} (~{params/1e6:.2f}M)")
    assert out.shape == dummy.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    print("Self-test passed!")
