"""
models/resunet.py
=================
Residual U-Net (ResUNet) architecture for satellite optical image speckle denoising.

Key Features:
  - Residual Blocks in both Encoder and Decoder paths to improve gradient flow.
  - Skip connections between corresponding Encoder and Decoder levels to preserve spatial detail.
  - 4-level deep encoder-decoder structure.
  - Final activation: Sigmoid / Identity (bounded to [0, 1]).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Two-layer Convolutional Residual Block with Batch Normalization and LeakyReLU.
    Includes identity shortcut connection (with 1x1 conv projection if channel count changes).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Shortcut projection if in_channels != out_channels
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + res)
        return out


class ResUNet(nn.Module):
    """
    Residual U-Net Architecture for Speckle Denoising.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (default = 3 for RGB).
    out_channels : int
        Number of target output channels (default = 3 for RGB).
    features : list of int
        Channel depths for each encoder level (default = [64, 128, 256, 512]).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: list = [64, 128, 256, 512],
    ):
        super(ResUNet, self).__init__()

        self.features = features
        self.encoders = nn.ModuleList()
        self.poolers  = nn.ModuleList()

        # --- Encoder Path ---
        curr_in = in_channels
        for feat in features:
            self.encoders.append(ResidualBlock(curr_in, feat))
            self.poolers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_in = feat

        # --- Bottleneck ---
        bottleneck_dim = features[-1] * 2
        self.bottleneck = ResidualBlock(features[-1], bottleneck_dim)

        # --- Decoder Path ---
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_dim = bottleneck_dim
        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(prev_dim, feat, kernel_size=2, stride=2))
            # Input to decoder block: concatenated skip (feat) + upsampled (feat) = feat * 2
            self.decoders.append(ResidualBlock(feat * 2, feat))
            prev_dim = feat

        # --- Final Output Layer ---
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid(),  # Ensures output values strictly in [0, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder forward pass
        out = x
        for encoder, pooler in zip(self.encoders, self.poolers):
            out = encoder(out)
            skip_connections.append(out)
            out = pooler(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder forward pass
        skip_connections = skip_connections[::-1]  # reverse for top-down decoding
        for idx in range(len(self.features)):
            out = self.upconvs[idx](out)
            skip = skip_connections[idx]

            # Handle potential shape mismatch due to odd dimensions
            if out.shape != skip.shape:
                out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)

            out = torch.cat([skip, out], dim=1)
            out = self.decoders[idx](out)

        # Final projection to RGB image
        output = self.final_conv(out)
        return output


if __name__ == "__main__":
    # Smoke test architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet(in_channels=3, out_channels=3).to(device)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResUNet Smoke Test PASSED:")
    print(f"  Input Shape  : {dummy_input.shape}")
    print(f"  Output Shape : {output.shape}")
    print(f"  Total Params : {total_params:,}")
