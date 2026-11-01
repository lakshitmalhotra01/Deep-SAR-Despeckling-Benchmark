"""
models/cbam_unet.py
===================
Convolutional Block Attention Module U-Net (CBAM-UNet) architecture
for satellite optical image speckle denoising.

Key Features (Woo et al., ECCV 2018):
  1. Channel Attention Module (CAM):
     - Uses BOTH Global Average Pooling and Global Max Pooling.
     - Shared MLP with channel reduction ratio r=16.
     - Learns 'WHAT' feature channels to emphasize for speckle reduction.
  2. Spatial Attention Module (SAM):
     - Uses channel-wise Average and Max pooling.
     - 7x7 convolution with Sigmoid activation.
     - Learns 'WHERE' to attend spatially (edges, textures vs noisy flats).
  3. Integrated CBAM Blocks:
     - Applied after each Residual Block in both Encoder and Decoder paths.
     - Preserves residual gradient flow while adding adaptive attention.
  4. Standard 4-level U-Net depth:
     - Input  : (B, 3, 256, 256) [noisy RGB satellite image]
     - Output : (B, 3, 256, 256) [denoised RGB image in range [0, 1]]

Reference:
  Woo, S., Park, J., Lee, J.Y., & Kweon, I.S., "CBAM: Convolutional Block
  Attention Module", ECCV 2018. arXiv:1807.06521
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Channel Attention Module (CAM)
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """
    Channel Attention Module:
    Explores cross-channel relationships using both AvgPool and MaxPool
    passed through a shared multi-layer perceptron (MLP).
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        mid_channels = max(1, in_channels // reduction_ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP implemented via 1x1 convs
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# Spatial Attention Module (SAM)
# ─────────────────────────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module:
    Explores inter-spatial relationships by computing channel-wise
    average and max features, concatenated and convolved with a 7x7 kernel.
    """

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat_out = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.bn(self.conv(cat_out)))
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# CBAM (Channel Attention + Spatial Attention)
# ─────────────────────────────────────────────────────────────────────────────
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module:
    Sequentially applies Channel Attention then Spatial Attention.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CBAM Residual Block
# ─────────────────────────────────────────────────────────────────────────────
class CBAMResidualBlock(nn.Module):
    """
    Two-layer Residual Block integrated with CBAM attention.
    Flow: Conv -> BN -> Act -> Conv -> BN -> CBAM -> (+ Shortcut) -> Act
    """

    def __init__(self, in_channels: int, out_channels: int, reduction_ratio: int = 16):
        super(CBAMResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.cbam  = CBAM(out_channels, reduction_ratio=reduction_ratio)

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
        out = self.cbam(out)
        out = self.act(out + res)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CBAM U-Net
# ─────────────────────────────────────────────────────────────────────────────
class CBAMUNet(nn.Module):
    """
    CBAM U-Net Architecture for Speckle Denoising.

    Parameters
    ----------
    in_channels  : int
        Input channels (default = 3 for RGB).
    out_channels : int
        Output channels (default = 3 for RGB).
    features     : list of int
        Channel depths per encoder level (default = [64, 128, 256, 512]).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: list = [64, 128, 256, 512],
    ):
        super(CBAMUNet, self).__init__()

        self.features = features
        self.encoders = nn.ModuleList()
        self.poolers  = nn.ModuleList()

        # ── Encoder Path ─────────────────────────────────────────────────────
        curr_in = in_channels
        for feat in features:
            self.encoders.append(CBAMResidualBlock(curr_in, feat))
            self.poolers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_in = feat

        # ── Bottleneck ───────────────────────────────────────────────────────
        bottleneck_dim = features[-1] * 2  # 1024
        self.bottleneck = CBAMResidualBlock(features[-1], bottleneck_dim)

        # ── Decoder Path ─────────────────────────────────────────────────────
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_dim = bottleneck_dim
        for feat in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(prev_dim, feat, kernel_size=2, stride=2)
            )
            # Concatenation of skip (feat) + upsampled (feat) = feat * 2
            self.decoders.append(CBAMResidualBlock(feat * 2, feat))
            prev_dim = feat

        # ── Final Output Layer ────────────────────────────────────────────────
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid(),  # Output values strictly bounded to [0, 1]
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
        skips = skip_connections[::-1]
        for idx in range(len(self.features)):
            out = self.upconvs[idx](out)
            skip = skips[idx]

            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)

            out = torch.cat([skip, out], dim=1)
            out = self.decoders[idx](out)

        # Final projection to RGB image
        return self.final_conv(out)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CBAMUNet(in_channels=3, out_channels=3).to(device)
    dummy  = torch.randn(2, 3, 256, 256).to(device)

    with torch.no_grad():
        out = model(dummy)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("CBAM U-Net Smoke Test PASSED")
    print(f"  Device           : {device}")
    print(f"  Input Shape      : {list(dummy.shape)}")
    print(f"  Output Shape     : {list(out.shape)}")
    print(f"  Total Params     : {total:,}")
    print(f"  Trainable Params : {trainable:,}")
    print(f"  Output Range     : [{out.min():.4f}, {out.max():.4f}]")
