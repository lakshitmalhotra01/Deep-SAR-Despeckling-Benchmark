"""
nafnet_adaptive_spatial.py
==========================
NAFNet with Adaptive Gate + Spatial Attention — Phase 2, Model 3 (M3).
Ablation Study for IEEE GRSL Paper:
Evaluates the contribution of explicit Spatial Attention combined with the
Adaptive Gating Mechanism (AGM) for speckle noise suppression.

Architecture:
  - Base: NAFNet 4-level U-Net
  - Gating: Adaptive Gate Mechanism (AGM)
  - Attention: Spatial Attention Module (SAM) using channel-pooled spatial statistics:
      SAM(F) = Sigmoid( Conv7x7( [AvgPool(F); MaxPool(F)] ) )
      F_attended = F * SAM(F)
  - Channel Attention: Simplified Channel Attention (SCA)
"""

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


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM):
    Extracts spatial context by concatenating channel-wise Average-Pool and Max-Pool,
    followed by a 7x7 spatial convolution and Sigmoid activation.
    Focuses the network on edge boundaries, linear textures, and high-contrast features.
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


class SimplifiedChannelAttention(nn.Module):
    """Simplified Channel Attention (SCA)."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.gap(x))


class NAFAdaptiveSpatialBlock(nn.Module):
    """
    NAFBlock equipped with Adaptive Gate (AGM) + Spatial Attention (SAM):
      1. Depthwise 3x3 convolution
      2. Adaptive Gate (AGM)
      3. Spatial Attention (SAM) — selectively enhances spatial structures
      4. Simplified Channel Attention (SCA)
      5. FFN with Adaptive Gate
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch  = int(c * dw_expand)
        ffn_ch = int(c * ffn_expand)

        # Depthwise branch: Adaptive Gate + Spatial Attention + SCA
        self.norm1    = LayerNormChannels(c)
        self.conv1    = nn.Conv2d(c, dw_ch, kernel_size=1, bias=True)
        self.dwconv   = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1,
                                  groups=dw_ch, bias=True)
        self.ag1      = AdaptiveGate(dw_ch // 2)
        self.sam      = SpatialAttention(kernel_size=7)
        self.sca      = SimplifiedChannelAttention(dw_ch // 2)
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
        x = self.sca(x)          # Channel Attention
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
# NAFNet with Adaptive Gate + Spatial Attention (Full U-Net)
# =============================================================================

class NAFNetAdaptiveSpatial(nn.Module):
    """
    NAFNet with Adaptive Gate + Spatial Attention — Phase 2 Model 3 (M3).

    Structure:
      Stem Conv (3 -> base_ch)
      Encoder:
        Level 1: num_blocks[0] NAFAdaptiveSpatialBlocks -> Downsample
        Level 2: num_blocks[1] NAFAdaptiveSpatialBlocks -> Downsample
        Level 3: num_blocks[2] NAFAdaptiveSpatialBlocks -> Downsample
        Level 4: num_blocks[3] NAFAdaptiveSpatialBlocks (bottleneck)
      Decoder:
        Level 4: Upsample -> concat skip L3 -> fuse -> NAFAdaptiveSpatialBlock*[3]
        Level 3: Upsample -> concat skip L2 -> fuse -> NAFAdaptiveSpatialBlock*[2]
        Level 2: Upsample -> concat skip L1 -> fuse -> NAFAdaptiveSpatialBlock*[1]
        Level 1: Upsample -> concat skip stem -> fuse -> NAFAdaptiveSpatialBlock*[0]
      Output Conv (base_ch -> 3)
      Global Residual: add input x
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
        self.enc1 = nn.Sequential(*[NAFAdaptiveSpatialBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[0])])
        self.down1 = Downsample(ch)

        ch = base_ch * 2
        self.enc2 = nn.Sequential(*[NAFAdaptiveSpatialBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[1])])
        self.down2 = Downsample(ch)

        ch = base_ch * 4
        self.enc3 = nn.Sequential(*[NAFAdaptiveSpatialBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[2])])
        self.down3 = Downsample(ch)

        ch = base_ch * 8
        self.bottleneck = nn.Sequential(*[NAFAdaptiveSpatialBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[3])])

        # Decoder
        self.up3   = Upsample(ch)
        self.fuse3 = nn.Conv2d(ch // 2 + base_ch * 4, base_ch * 4, 1, bias=True)
        self.dec3  = nn.Sequential(*[NAFAdaptiveSpatialBlock(base_ch * 4, dw_expand, ffn_expand) for _ in range(num_blocks[2])])

        self.up2   = Upsample(base_ch * 4)
        self.fuse2 = nn.Conv2d(base_ch * 2 + base_ch * 2, base_ch * 2, 1, bias=True)
        self.dec2  = nn.Sequential(*[NAFAdaptiveSpatialBlock(base_ch * 2, dw_expand, ffn_expand) for _ in range(num_blocks[1])])

        self.up1   = Upsample(base_ch * 2)
        self.fuse1 = nn.Conv2d(base_ch + base_ch, base_ch, 1, bias=True)
        self.dec1  = nn.Sequential(*[NAFAdaptiveSpatialBlock(base_ch, dw_expand, ffn_expand) for _ in range(num_blocks[0])])

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


# =============================================================================
# Sanity check
# =============================================================================
if __name__ == "__main__":
    model = NAFNetAdaptiveSpatial(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1, 1, 1, 4])
    dummy = torch.randn(2, 3, 256, 256)
    out   = model(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Total params : {total:,}")
    print("NAFNet with Adaptive Gate + Spatial Attention OK!")
