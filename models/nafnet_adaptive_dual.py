"""
nafnet_adaptive_dual.py
=======================
NAFNet with Adaptive Gate + Dual Attention (Spatial + Channel Attention) — Phase 2, Model 5 (M5).
This is the PROPOSED DUAL-ATTENTION MODEL for the IEEE GRSL Paper.

Ablation Study Culmination:
  M5 combines all three novel contributions:
    1. Adaptive Gate Mechanism (AGM) — context-aware gating for speckle statistics
    2. Spatial Attention Module (SAM) — edge and structural preservation
    3. Dual-Pool Channel Attention Module (CAM) — inter-channel speckle correlation

Architecture:
  - Base: NAFNet 4-level U-Net
  - Gating: Adaptive Gate Mechanism (AGM)
  - Dual Attention:
      * SAM(F) = Sigmoid( Conv7x7( [AvgPool_c(F); MaxPool_c(F)] ) )
      * CAM(F) = Sigmoid( MLP(GAP(F)) + MLP(GMP(F)) )
  - Pipeline per block: DWConv -> AGM -> SAM -> CAM -> Projection
  - Balanced parameter footprint (~2.91M params) for rigorous fair comparison.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================

class LayerNormChannels(nn.Module):
    """Channels-first LayerNorm across C dimension."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.unsqueeze(-1).unsqueeze(-1) * x_norm \
             + self.bias.unsqueeze(-1).unsqueeze(-1)


class AdaptiveGate(nn.Module):
    """
    Adaptive Gate Mechanism (AGM):
    Conditions gating on depthwise + pointwise local speckle statistics.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.dw_gate = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1,
            groups=num_channels, bias=True
        )
        self.pw_gate = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        gate = self.sigmoid(self.dw_gate(x2) + self.pw_gate(x2))
        return x1 * (self.alpha * gate + self.beta) + self.gamma * x1


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM):
    Focuses on edge boundaries, linear textures, and high-contrast features
    by extracting spatial context via channel-wise pooling + convolution.

      SAM(F) = Sigmoid( Conv7x7( [AvgPool_c(F); MaxPool_c(F)] ) )
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class DualPoolChannelAttention(nn.Module):
    """
    Dual-Pool Channel Attention Module (CAM):
    Captures inter-channel dependencies via joint Average + Max global pooling
    through a shared bottleneck MLP.

      CAM(F) = Sigmoid( MLP(GAP(F)) + MLP(GMP(F)) )
    """
    def __init__(self, num_channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_ch = max(4, num_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(num_channels, reduced_ch, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(reduced_ch, num_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)
        scale = self.sigmoid(self.mlp(avg_feat) + self.mlp(max_feat))
        return x * scale


class NAFAdaptiveDualBlock(nn.Module):
    """
    NAFBlock with AGM + Dual Attention (SAM + CAM):
      1. LayerNorm -> 1x1 expansion -> DWConv 3x3
      2. Adaptive Gate (AGM)
      3. Spatial Attention (SAM)
      4. Dual-Pool Channel Attention (CAM)
      5. 1x1 projection + residual
      6. FFN branch with Adaptive Gate
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch  = int(c * dw_expand)
        ffn_ch = int(c * ffn_expand)

        # Depthwise branch: AGM -> SAM -> CAM
        self.norm1    = LayerNormChannels(c)
        self.conv1    = nn.Conv2d(c, dw_ch, kernel_size=1, bias=True)
        self.dwconv   = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1,
                                  groups=dw_ch, bias=True)
        self.ag1      = AdaptiveGate(dw_ch // 2)
        self.sam      = SpatialAttention(kernel_size=7)
        self.cam      = DualPoolChannelAttention(dw_ch // 2, reduction_ratio=4)
        self.conv2    = nn.Conv2d(dw_ch // 2, c, kernel_size=1, bias=True)
        self.beta     = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

        # FFN branch with Adaptive Gate
        self.norm2    = LayerNormChannels(c)
        self.conv3    = nn.Conv2d(c, ffn_ch, kernel_size=1, bias=True)
        self.ag2      = AdaptiveGate(ffn_ch // 2)
        self.conv4    = nn.Conv2d(ffn_ch // 2, c, kernel_size=1, bias=True)
        self.gamma    = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise branch
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.ag1(x)          # Adaptive Gate
        x = self.sam(x)          # Spatial Attention
        x = self.cam(x)          # Channel Attention
        x = self.conv2(x)
        x = x * self.beta + residual

        # FFN branch
        residual = x
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.ag2(x)          # Adaptive Gate
        x = self.conv4(x)
        x = x * self.gamma + residual

        return x


# =============================================================================
# Down / Up sampling modules
# =============================================================================

class Downsample(nn.Module):
    """Pixel-unshuffle downsampling (2x): C -> 2C at half spatial resolution."""
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Pixel-shuffle upsampling (2x): 2C -> C at double spatial resolution."""
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# =============================================================================
# Full Network — NAFNet Adaptive Dual Attention
# =============================================================================

class NAFNetAdaptiveDual(nn.Module):
    """
    NAFNet with Adaptive Gate + Dual Attention (Phase 2, Model 5 - M5).

    Structure:
      - 4-level U-Net hierarchy
      - Blocks per stage: [1, 1, 1, 4]
      - Channel width: 32 -> 64 -> 128 -> 256
      - Adaptive Gate Mechanism (AGM)
      - Spatial Attention Module (SAM)
      - Dual-Pool Channel Attention Module (CAM)
    """
    def __init__(
        self,
        in_channels:  int  = 3,
        out_channels: int  = 3,
        base_ch:      int  = 32,
        num_blocks:   list = None,
        dw_expand:    int  = 2,
        ffn_expand:   int  = 2,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1, 4]

        self.stem = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=True)

        # Encoder
        ch = base_ch
        self.enc1 = nn.Sequential(*[NAFAdaptiveDualBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[0])])
        self.down1 = Downsample(ch)

        ch = base_ch * 2
        self.enc2 = nn.Sequential(*[NAFAdaptiveDualBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[1])])
        self.down1_2 = Downsample(ch)

        ch = base_ch * 4
        self.enc3 = nn.Sequential(*[NAFAdaptiveDualBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[2])])
        self.down3 = Downsample(ch)

        ch = base_ch * 8
        self.bottleneck = nn.Sequential(*[NAFAdaptiveDualBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[3])])

        # Decoder
        self.up3   = Upsample(ch)
        self.fuse3 = nn.Conv2d(ch // 2 + base_ch * 4, base_ch * 4, 1, bias=True)
        self.dec3  = nn.Sequential(*[NAFAdaptiveDualBlock(base_ch * 4, dw_expand, ffn_expand) for _ in range(num_blocks[2])])

        self.up2   = Upsample(base_ch * 4)
        self.fuse2 = nn.Conv2d(base_ch * 2 + base_ch * 2, base_ch * 2, 1, bias=True)
        self.dec2  = nn.Sequential(*[NAFAdaptiveDualBlock(base_ch * 2, dw_expand, ffn_expand) for _ in range(num_blocks[1])])

        self.up1   = Upsample(base_ch * 2)
        self.fuse1 = nn.Conv2d(base_ch + base_ch, base_ch, 1, bias=True)
        self.dec1  = nn.Sequential(*[NAFAdaptiveDualBlock(base_ch, dw_expand, ffn_expand) for _ in range(num_blocks[0])])

        self.output = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x

        # Stem
        x = self.stem(x)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down1_2(e2))
        x  = self.bottleneck(self.down3(e3))

        # Decoder
        x = self.up3(x)
        x = self.fuse3(torch.cat([x, e3], dim=1))
        x = self.dec3(x)

        x = self.up2(x)
        x = self.fuse2(torch.cat([x, e2], dim=1))
        x = self.dec2(x)

        x = self.up1(x)
        x = self.fuse1(torch.cat([x, e1], dim=1))
        x = self.dec1(x)

        # Output + global residual
        x = self.output(x) + inp
        return torch.clamp(x, 0.0, 1.0)


def nafnet_adaptive_dual_model(in_channels: int = 3, out_channels: int = 3, **kwargs) -> NAFNetAdaptiveDual:
    """Factory constructor for NAFNet with AGM + Dual Attention (M5)."""
    return NAFNetAdaptiveDual(
        in_channels=in_channels,
        out_channels=out_channels,
        base_ch=32,
        num_blocks=[1, 1, 1, 4],
        dw_expand=2,
        ffn_expand=2,
        **kwargs
    )


if __name__ == "__main__":
    model = nafnet_adaptive_dual_model()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 65)
    print("NAFNet + AGM + Dual Attention (M5) Sanity Test")
    print("=" * 65)
    print(f"Input Shape       : {tuple(x.shape)}")
    print(f"Output Shape      : {tuple(out.shape)}")
    print(f"Total Parameters  : {total_params:,}")
    print(f"Trainable Params  : {trainable_params:,}")
    print("Forward pass successful!")
