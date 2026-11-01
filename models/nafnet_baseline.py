"""
nafnet_baseline.py
==================
NAFNet Baseline Model — Phase 2, Model 1 (M1).
Architecture: Deep U-Net with Simple Gate nonlinearity (no ReLU/GELU anywhere).

Key components:
  - LayerNorm (channels-first, applied before each block)
  - Depthwise Convolution (3x3, grouped=in_channels)
  - Simple Gate: splits channels in half, X1 * X2 (element-wise product)
  - Simplified Channel Attention (SCA): GAP -> Conv1x1 -> channel scale
  - 4-level U-Net encoder-decoder with skip connections
  - Global residual: output = model(x) + x

Reference: NAFNet (ECCV 2022) — "Simple Baselines for Image Restoration"
           https://arxiv.org/abs/2204.04676
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
        # x: (B, C, H, W)
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.unsqueeze(-1).unsqueeze(-1) * x_norm \
             + self.bias.unsqueeze(-1).unsqueeze(-1)


class SimpleGate(nn.Module):
    """
    Simple Gate nonlinearity: splits tensor in half along channel dim,
    then element-wise multiplies: SG(X) = X1 ⊙ X2.
    Input channels must be even; output channels = input_channels // 2.
    This is the ONLY nonlinearity in NAFNet — replaces all ReLU/GELU.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA):
    Global Average Pool -> Conv1x1 (no reduction ratio, no sigmoid) -> channel scale.
    Simpler than CBAM/SE: just one conv, no activation between.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.gap(x))


class NAFBlock(nn.Module):
    """
    Core NAFNet block:

        Input
          |
        LayerNorm (channels-first)
          |
        Conv1x1 (expand: C -> 2*DW_C)        # channel expansion
          |
        DWConv 3x3 (depthwise, 2*DW_C ch)    # spatial mixing
          |
        SimpleGate  (2*DW_C -> DW_C)         # the only nonlinearity
          |
        SCA (DW_C)                            # channel attention
          |
        Conv1x1 (DW_C -> C)                  # project back
          |
        * beta (learnable scalar)
          |
        + residual skip
          |
        LayerNorm
          |
        Conv1x1 (C -> 2*FFN_C)               # FFN expand
          |
        SimpleGate  (2*FFN_C -> FFN_C)       # second gate
          |
        Conv1x1 (FFN_C -> C)                 # FFN project
          |
        * gamma (learnable scalar)
          |
        + residual skip
          |
        Output

    DW_expand_ratio: expansion ratio for depthwise branch (default 2).
    FFN_expand_ratio: expansion ratio for FFN branch (default 2).
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch  = int(c * dw_expand)   # must be even for SimpleGate
        ffn_ch = int(c * ffn_expand)  # must be even for SimpleGate

        # ---------- Depthwise branch ----------
        self.norm1    = LayerNormChannels(c)
        self.conv1    = nn.Conv2d(c, dw_ch, kernel_size=1, bias=True)           # expand
        self.dwconv   = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1,
                                  groups=dw_ch, bias=True)                       # depthwise
        self.sg1      = SimpleGate()                                             # dw_ch -> dw_ch//2
        self.sca      = SimplifiedChannelAttention(dw_ch // 2)
        self.conv2    = nn.Conv2d(dw_ch // 2, c, kernel_size=1, bias=True)      # project back
        self.beta     = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)            # learnable scale

        # ---------- FFN branch ----------
        self.norm2    = LayerNormChannels(c)
        self.conv3    = nn.Conv2d(c, ffn_ch, kernel_size=1, bias=True)
        self.sg2      = SimpleGate()                                             # ffn_ch -> ffn_ch//2
        self.conv4    = nn.Conv2d(ffn_ch // 2, c, kernel_size=1, bias=True)
        self.gamma    = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Depthwise attention branch ---
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.sg1(x)          # Simple Gate: halves channels
        x = self.sca(x)
        x = self.conv2(x)
        x = x * self.beta + residual

        # --- FFN branch ---
        residual = x
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.sg2(x)          # Simple Gate: halves channels
        x = self.conv4(x)
        x = x * self.gamma + residual

        return x


# =============================================================================
# Down / Up sampling modules
# =============================================================================

class Downsample(nn.Module):
    """Pixel-unshuffle downsampling (2x): C -> 4C at half spatial resolution."""
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=1, bias=False),
            nn.PixelUnshuffle(2),          # (B, C//2, H, W) -> (B, C*2, H/2, W/2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Pixel-shuffle upsampling (2x): C -> C//4 at double spatial resolution."""
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=1, bias=False),
            nn.PixelShuffle(2),            # (B, C*2, H, W) -> (B, C//2, H*2, W*2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# =============================================================================
# NAFNet Baseline (Full U-Net)
# =============================================================================

class NAFNetBaseline(nn.Module):
    """
    NAFNet U-Net Baseline — Phase 2 Model 1.

    Structure:
      Stem Conv (3 -> base_ch)
      Encoder:
        Level 1: num_blocks[0] NAFBlocks  -> Downsample (base_ch -> base_ch*2)
        Level 2: num_blocks[1] NAFBlocks  -> Downsample (base_ch*2 -> base_ch*4)
        Level 3: num_blocks[2] NAFBlocks  -> Downsample (base_ch*4 -> base_ch*8)
        Level 4: num_blocks[3] NAFBlocks  (bottleneck, base_ch*8)
      Decoder:
        Level 4: Upsample -> concat skip L3 -> NAFBlock*[3]
        Level 3: Upsample -> concat skip L2 -> NAFBlock*[2]
        Level 2: Upsample -> concat skip L1 -> NAFBlock*[1]
        Level 1: Upsample -> concat skip stem -> NAFBlock*[0]
      Output Conv (base_ch -> 3)
      Global Residual: add input x

    Default: base_ch=32, num_blocks=[1,1,1,4], dw_expand=2, ffn_expand=2
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

        # --- Stem ---
        self.stem = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=True)

        # --- Encoder ---
        ch = base_ch
        self.enc1 = nn.Sequential(*[NAFBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[0])])
        self.down1 = Downsample(ch)          # ch -> ch*2

        ch = base_ch * 2
        self.enc2 = nn.Sequential(*[NAFBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[1])])
        self.down2 = Downsample(ch)          # ch -> ch*2

        ch = base_ch * 4
        self.enc3 = nn.Sequential(*[NAFBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[2])])
        self.down3 = Downsample(ch)          # ch -> ch*2

        ch = base_ch * 8
        self.bottleneck = nn.Sequential(*[NAFBlock(ch, dw_expand, ffn_expand) for _ in range(num_blocks[3])])

        # --- Decoder ---
        # After each Upsample + skip concat, channels halve then we project
        self.up3    = Upsample(ch)                                                     # ch -> ch//2
        self.fuse3  = nn.Conv2d(ch // 2 + base_ch * 4, base_ch * 4, 1, bias=True)    # fuse skip
        self.dec3   = nn.Sequential(*[NAFBlock(base_ch * 4, dw_expand, ffn_expand) for _ in range(num_blocks[2])])

        self.up2    = Upsample(base_ch * 4)
        self.fuse2  = nn.Conv2d(base_ch * 2 + base_ch * 2, base_ch * 2, 1, bias=True)
        self.dec2   = nn.Sequential(*[NAFBlock(base_ch * 2, dw_expand, ffn_expand) for _ in range(num_blocks[1])])

        self.up1    = Upsample(base_ch * 2)
        self.fuse1  = nn.Conv2d(base_ch + base_ch, base_ch, 1, bias=True)
        self.dec1   = nn.Sequential(*[NAFBlock(base_ch, dw_expand, ffn_expand) for _ in range(num_blocks[0])])

        # --- Output ---
        self.output = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x  # save for global residual

        # Stem
        x = self.stem(x)

        # Encoder + skip save
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        x  = self.bottleneck(self.down3(e3))

        # Decoder + skip fuse
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


# =============================================================================
# Quick sanity check
# =============================================================================
if __name__ == "__main__":
    model = NAFNetBaseline(in_channels=3, out_channels=3, base_ch=32, num_blocks=[1, 1, 1, 4])
    dummy = torch.randn(1, 3, 256, 256)
    out   = model(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Total params : {total:,}")
    print("NAFNet Baseline OK")
