"""
nafnet_adaptive_channel.py
==========================
NAFNet with Adaptive Gate + Channel Attention — Phase 2, Model 4 (M4).
Ablation Study for IEEE GRSL Paper:
Evaluates the contribution of explicit Channel Attention (CAM with joint Average + Max
global context pooling) combined with the Adaptive Gating Mechanism (AGM) for
speckle noise suppression.

Architecture:
  - Base: NAFNet 4-level U-Net
  - Gating: Adaptive Gate Mechanism (AGM) with depthwise + pointwise context gating
  - Attention: Dual-Pool Channel Attention Module (CAM):
      CAM(F) = Sigmoid( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
      F_attended = F * CAM(F)
  - Balanced parameter footprint (~3.06M - 3.10M params) for rigorous fair comparison.
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
    Adaptive Gate Mechanism (AGM) from M2:
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


class DualPoolChannelAttention(nn.Module):
    """
    Dual-Pool Channel Attention Module (CAM):
    Captures inter-channel dependencies by combining global Average Pooling (smooth
    background and distributed radar clutter) and global Max Pooling (prominent point
    scatterers and hard specular reflections) through a shared bottleneck MLP.

      Mc(x) = Sigmoid( W1 * GELU(W0 * AvgPool(x)) + W1 * GELU(W0 * MaxPool(x)) )
      output = x * Mc(x)
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


class NAFAdaptiveChannelBlock(nn.Module):
    """
    NAFBlock equipped with Adaptive Gate (AGM) + Dual-Pool Channel Attention (CAM):
      1. Depthwise 3x3 convolution
      2. Adaptive Gate (AGM)
      3. Dual-Pool Channel Attention (CAM)
      4. FFN with Adaptive Gate
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch  = int(c * dw_expand)
        ffn_ch = int(c * ffn_expand)

        # Depthwise branch: Adaptive Gate + Dual-Pool Channel Attention
        self.norm1    = LayerNormChannels(c)
        self.conv1    = nn.Conv2d(c, dw_ch, kernel_size=1, bias=True)
        self.dwconv   = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1,
                                  groups=dw_ch, bias=True)
        self.ag1      = AdaptiveGate(dw_ch // 2)
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
# Full Network
# =============================================================================

class NAFNetAdaptiveChannel(nn.Module):
    """
    NAFNet with Adaptive Gate + Channel Attention (Phase 2, Model 4 - M4).

    Structure:
      - 4-level U-Net hierarchy
      - Blocks per stage: [1, 1, 1, 4]
      - Channel width: 32 -> 64 -> 128 -> 256
      - Adaptive Gate Mechanism (AGM)
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
        self.enc1 = nn.Sequential(*[NAFAdaptiveChannelBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[0])])
        self.down1 = Downsample(ch)

        ch = base_ch * 2
        self.enc2 = nn.Sequential(*[NAFAdaptiveChannelBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[1])])
        self.down2 = Downsample(ch)

        ch = base_ch * 4
        self.enc3 = nn.Sequential(*[NAFAdaptiveChannelBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[2])])
        self.down3 = Downsample(ch)

        ch = base_ch * 8
        self.bottleneck = nn.Sequential(*[NAFAdaptiveChannelBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[3])])

        # Decoder
        self.up3   = Upsample(ch)
        self.fuse3 = nn.Conv2d(ch // 2 + base_ch * 4, base_ch * 4, 1, bias=True)
        self.dec3  = nn.Sequential(*[NAFAdaptiveChannelBlock(base_ch * 4, dw_expand, ffn_expand) for _ in range(num_blocks[2])])

        self.up2   = Upsample(base_ch * 4)
        self.fuse2 = nn.Conv2d(base_ch * 2 + base_ch * 2, base_ch * 2, 1, bias=True)
        self.dec2  = nn.Sequential(*[NAFAdaptiveChannelBlock(base_ch * 2, dw_expand, ffn_expand) for _ in range(num_blocks[1])])

        self.up1   = Upsample(base_ch * 2)
        self.fuse1 = nn.Conv2d(base_ch + base_ch, base_ch, 1, bias=True)
        self.dec1  = nn.Sequential(*[NAFAdaptiveChannelBlock(base_ch, dw_expand, ffn_expand) for _ in range(num_blocks[0])])

        self.output = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x

        # Stem
        x = self.stem(x)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
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

        # Output + residual
        x = self.output(x) + inp
        return torch.clamp(x, 0.0, 1.0)


def nafnet_adaptive_channel_model(in_channels: int = 3, out_channels: int = 3, **kwargs) -> NAFNetAdaptiveChannel:
    """Factory constructor for NAFNet with Adaptive Gate + Channel Attention (M4)."""
    return NAFNetAdaptiveChannel(
        in_channels=in_channels,
        out_channels=out_channels,
        base_ch=32,
        num_blocks=[1, 1, 1, 4],
        dw_expand=2,
        ffn_expand=2,
        **kwargs
    )


if __name__ == "__main__":
    model = nafnet_adaptive_channel_model()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("NAFNet + Adaptive Gate + Channel Attention (M4) Sanity Test")
    print("=" * 60)
    print(f"Input Shape       : {tuple(x.shape)}")
    print(f"Output Shape      : {tuple(out.shape)}")
    print(f"Total Parameters  : {total_params:,}")
    print(f"Trainable Params  : {trainable_params:,}")
    print("Forward pass successful!")
