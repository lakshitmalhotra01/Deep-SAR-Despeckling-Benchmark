"""
nafnet_adaptive_gate.py
=======================
NAFNet with Adaptive Gate — Phase 2, Model 2 (M2).
Core Research Novelty for IEEE GRSL Paper:
Replaces static Simple Gate (X1 * X2) with an Adaptive Gating Mechanism (AGM)
specifically designed for spatially heterogeneous SAR & satellite optical speckle noise.

Formulation:
  Input X is split into X1 (feature representation) and X2 (gating control).
  AdaptiveGate(X):
      Gate = Sigmoid( DWConv_3x3(X2) + PointwiseConv(X2) )
      Output = X1 * (alpha * Gate + beta) + gamma * X1
  where:
    - DWConv_3x3 captures local contextual speckle statistics around each pixel.
    - alpha, beta, gamma are learnable per-channel modulation parameters.
    - When speckle variance is high, Gate adaptively attenuates noise while preserving edges.
    - When speckle is low, gamma provides a direct identity bypass preserving sharp structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================

class LayerNormChannels(nn.Module):
    """
    Layer Normalization applied in channels-first (BCHW) format.
    Normalizes across C dimension at each spatial location.
    """
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
    Adaptive Gate Mechanism (AGM) — Phase 2 Core Research Novelty (M2).

    Unlike static SimpleGate (X1 * X2), AdaptiveGate:
    1. Splits input into feature content X1 and gating descriptor X2.
    2. Applies a local depthwise spatial filter + pointwise projection to X2 to compute
       spatially-aware gating coefficients conditioned on local speckle patterns.
    3. Uses learnable channel parameters (alpha, beta, gamma) to adaptively balance
       non-linear noise attenuation vs. structural identity pass-through.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        # Depthwise 3x3 context filter for local speckle spatial statistics
        self.dw_gate = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1,
            groups=num_channels, bias=True
        )
        self.pw_gate = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Learnable modulation parameters (initialized for smooth stable training)
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        # Compute spatially adaptive gating mask
        gate = self.sigmoid(self.dw_gate(x2) + self.pw_gate(x2))
        # Adaptive modulation
        return x1 * (self.alpha * gate + self.beta) + self.gamma * x1


class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA):
    Global Average Pool -> Conv1x1 -> Channel scale.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.gap(x))


class NAFAdaptiveBlock(nn.Module):
    """
    NAFBlock equipped with Adaptive Gate Mechanism (AGM):

        Input
          |
        LayerNorm (channels-first)
          |
        Conv1x1 (expand: C -> 2*DW_C)
          |
        DWConv 3x3 (depthwise, 2*DW_C ch)
          |
        AdaptiveGate (2*DW_C -> DW_C)        <-- PROPOSED NOVELTY
          |
        SCA (DW_C)
          |
        Conv1x1 (DW_C -> C)
          |
        * beta (learnable scale)
          |
        + residual skip
          |
        LayerNorm
          |
        Conv1x1 (C -> 2*FFN_C)
          |
        AdaptiveGate (2*FFN_C -> FFN_C)      <-- PROPOSED NOVELTY
          |
        Conv1x1 (FFN_C -> C)
          |
        * gamma (learnable scale)
          |
        + residual skip
          |
        Output
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch  = int(c * dw_expand)
        ffn_ch = int(c * ffn_expand)

        # ---------- Depthwise branch with Adaptive Gate ----------
        self.norm1    = LayerNormChannels(c)
        self.conv1    = nn.Conv2d(c, dw_ch, kernel_size=1, bias=True)
        self.dwconv   = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1,
                                  groups=dw_ch, bias=True)
        self.ag1      = AdaptiveGate(dw_ch // 2)
        self.sca      = SimplifiedChannelAttention(dw_ch // 2)
        self.conv2    = nn.Conv2d(dw_ch // 2, c, kernel_size=1, bias=True)
        self.beta     = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

        # ---------- FFN branch with Adaptive Gate ----------
        self.norm2    = LayerNormChannels(c)
        self.conv3    = nn.Conv2d(c, ffn_ch, kernel_size=1, bias=True)
        self.ag2      = AdaptiveGate(ffn_ch // 2)
        self.conv4    = nn.Conv2d(ffn_ch // 2, c, kernel_size=1, bias=True)
        self.gamma    = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise attention branch
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.ag1(x)          # Adaptive Gate
        x = self.sca(x)
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
# NAFNet with Adaptive Gate (Full U-Net)
# =============================================================================

class NAFNetAdaptiveGate(nn.Module):
    """
    NAFNet with Adaptive Gate — Phase 2 Model 2 (M2).

    Structure:
      Stem Conv (3 -> base_ch)
      Encoder:
        Level 1: num_blocks[0] NAFAdaptiveBlocks -> Downsample
        Level 2: num_blocks[1] NAFAdaptiveBlocks -> Downsample
        Level 3: num_blocks[2] NAFAdaptiveBlocks -> Downsample
        Level 4: num_blocks[3] NAFAdaptiveBlocks (bottleneck)
      Decoder:
        Level 4: Upsample -> concat skip L3 -> fuse -> NAFAdaptiveBlock*[3]
        Level 3: Upsample -> concat skip L2 -> fuse -> NAFAdaptiveBlock*[2]
        Level 2: Upsample -> concat skip L1 -> fuse -> NAFAdaptiveBlock*[1]
        Level 1: Upsample -> concat skip stem -> fuse -> NAFAdaptiveBlock*[0]
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

        # Stem
        self.stem = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=True)

        # Encoder
        ch = base_ch
        self.enc1 = nn.Sequential(*[NAFAdaptiveBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[0])])
        self.down1 = Downsample(ch)

        ch = base_ch * 2
        self.enc2 = nn.Sequential(*[NAFAdaptiveBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[1])])
        self.down2 = Downsample(ch)

        ch = base_ch * 4
        self.enc3 = nn.Sequential(*[NAFAdaptiveBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[2])])
        self.down3 = Downsample(ch)

        ch = base_ch * 8
        self.bottleneck = nn.Sequential(*[NAFAdaptiveBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[3])])

        # Decoder
        self.up3   = Upsample(ch)
        self.fuse3 = nn.Conv2d(ch // 2 + base_ch * 4, base_ch * 4, 1, bias=True)
        self.dec3  = nn.Sequential(*[NAFAdaptiveBlock(base_ch * 4, dw_expand, ffn_expand) for _ in range(num_blocks[2])])

        self.up2   = Upsample(base_ch * 4)
        self.fuse2 = nn.Conv2d(base_ch * 2 + base_ch * 2, base_ch * 2, 1, bias=True)
        self.dec2  = nn.Sequential(*[NAFAdaptiveBlock(base_ch * 2, dw_expand, ffn_expand) for _ in range(num_blocks[1])])

        self.up1   = Upsample(base_ch * 2)
        self.fuse1 = nn.Conv2d(base_ch + base_ch, base_ch, 1, bias=True)
        self.dec1  = nn.Sequential(*[NAFAdaptiveBlock(base_ch, dw_expand, ffn_expand) for _ in range(num_blocks[0])])

        # Output + Global residual
        self.output = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x

        # Stem
        x = self.stem(x)

        # Encoder + save skips
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        x  = self.bottleneck(self.down3(e3))

        # Decoder + fuse skips
        x = self.up3(x)
        x = self.fuse3(torch.cat([x, e3], dim=1))
        x = self.dec3(x)

        x = self.up2(x)
        x = self.fuse2(torch.cat([x, e2], dim=1))
        x = self.dec2(x)

        x = self.up1(x)
        x = self.fuse1(torch.cat([x, e1], dim=1))
        x = self.dec1(x)

        # Output + global identity bypass
        x = self.output(x) + inp
        return torch.clamp(x, 0.0, 1.0)


# =============================================================================
# Sanity check
# =============================================================================
if __name__ == "__main__":
    model = NAFNetAdaptiveGate(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1, 1, 1, 4])
    dummy = torch.randn(2, 3, 256, 256)
    out   = model(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Total params : {total:,}")
    print("NAFNet with Adaptive Gate OK!")
