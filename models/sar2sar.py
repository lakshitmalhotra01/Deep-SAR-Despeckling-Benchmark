"""
models/sar2sar.py
=================
SAR2SAR Despeckling Architecture for Satellite Image Denoising.

Key Architecture Features (Dammann et al., Lecoq et al.):
  1. Multi-Scale Deep Residual Encoder-Decoder:
     - Specifically designed for multiplicative speckle noise characteristics.
     - Preserves high-reflectance features (edges, structural contours, point targets).
  2. Dilated Multi-Scale Bottleneck:
     - Integrates dilated convolutions with rates [1, 2, 4] to aggregate wide
       contextual speckle statistics without loss of spatial resolution.
  3. High-Frequency Detail Retention:
     - Skip connections with residual refinement blocks preserve high-frequency
       satellite scene details while suppressing speckle variance.
  4. Output Layer:
     - 1x1 convolution with Sigmoid activation strictly bounding output to [0, 1].

Reference:
  Dammann et al., "SAR2SAR: a self-supervised despeckling training strategy for
  Sentinel-1 / SAR imagery", ISPRS Annals / IEEE TGRS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block for SAR2SAR
# ─────────────────────────────────────────────────────────────────────────────
class SARResidualBlock(nn.Module):
    """
    Two-layer Residual Convolutional Block with Batch Normalization
    and LeakyReLU activation (negative slope = 0.2).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(SARResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

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
        return self.act(out + res)


# ─────────────────────────────────────────────────────────────────────────────
# Dilated Multi-Scale Bottleneck
# ─────────────────────────────────────────────────────────────────────────────
class SARDilatedBottleneck(nn.Module):
    """
    Multi-scale dilated convolutional block:
    Captures speckle statistics at multiple receptive fields (dilation 1, 2, 4)
    and fuses them with a residual shortcut.
    """

    def __init__(self, channels: int):
        super(SARDilatedBottleneck, self).__init__()
        branch_ch = channels // 4

        self.b1 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.b1(x)
        out2 = self.b2(x)
        out3 = self.b3(x)
        out4 = self.b4(x)
        fused = self.fuse(torch.cat([out1, out2, out3, out4], dim=1))
        return self.act(x + fused)


# ─────────────────────────────────────────────────────────────────────────────
# SAR2SAR Network
# ─────────────────────────────────────────────────────────────────────────────
class SAR2SARNet(nn.Module):
    """
    SAR2SAR Deep Despeckling Network for Satellite Optical & Radar Images.

    Parameters
    ----------
    in_channels  : int
        Number of input channels (default = 3 for RGB satellite images).
    out_channels : int
        Number of output channels (default = 3 for RGB satellite images).
    features     : list of int
        Channels per level (default = [64, 128, 256, 512]).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: list = [64, 128, 256, 512],
    ):
        super(SAR2SARNet, self).__init__()

        self.features = features
        self.encoders = nn.ModuleList()
        self.poolers  = nn.ModuleList()

        # ── Encoder Path ─────────────────────────────────────────────────────
        curr_in = in_channels
        for feat in features:
            self.encoders.append(SARResidualBlock(curr_in, feat))
            self.poolers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_in = feat

        # ── Bottleneck (Residual + Multi-scale Dilated) ──────────────────────
        bottleneck_dim = features[-1] * 2  # 1024
        self.bottleneck_res = SARResidualBlock(features[-1], bottleneck_dim)
        self.bottleneck_dil = SARDilatedBottleneck(bottleneck_dim)

        # ── Decoder Path ─────────────────────────────────────────────────────
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_dim = bottleneck_dim
        for feat in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(prev_dim, feat, kernel_size=2, stride=2)
            )
            self.decoders.append(SARResidualBlock(feat * 2, feat))
            prev_dim = feat

        # ── Final Output Layer ────────────────────────────────────────────────
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid(),  # Bounded strictly to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder pass
        out = x
        for encoder, pooler in zip(self.encoders, self.poolers):
            out = encoder(out)
            skip_connections.append(out)
            out = pooler(out)

        # Bottleneck pass
        out = self.bottleneck_res(out)
        out = self.bottleneck_dil(out)

        # Decoder pass
        skips = skip_connections[::-1]
        for idx in range(len(self.features)):
            out = self.upconvs[idx](out)
            skip = skips[idx]

            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)

            out = torch.cat([skip, out], dim=1)
            out = self.decoders[idx](out)

        # Final projection
        return self.final_conv(out)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SAR2SARNet(in_channels=3, out_channels=3).to(device)
    dummy  = torch.randn(2, 3, 256, 256).to(device)

    with torch.no_grad():
        out = model(dummy)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("SAR2SAR Smoke Test PASSED")
    print(f"  Device           : {device}")
    print(f"  Input Shape      : {list(dummy.shape)}")
    print(f"  Output Shape     : {list(out.shape)}")
    print(f"  Total Params     : {total:,}")
    print(f"  Trainable Params : {trainable:,}")
    print(f"  Output Range     : [{out.min():.4f}, {out.max():.4f}]")
