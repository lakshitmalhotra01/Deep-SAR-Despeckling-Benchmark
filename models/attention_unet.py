"""
models/attention_unet.py
========================
Attention U-Net architecture for satellite optical image speckle denoising.

Key Features over Residual U-Net:
  - Attention Gates on EVERY skip connection before concatenation.
  - Each gate learns to suppress irrelevant spatial regions and amplify
    task-relevant features (denoising-relevant textures/edges).
  - Same Residual Blocks in encoder/decoder as ResUNet for fair comparison.
  - Same 4-level encoder-decoder depth.
  - Input  : (B, 3, 256, 256)  [noisy RGB satellite image]
  - Output : (B, 3, 256, 256)  [denoised RGB image, values in [0, 1]]

Reference:
  Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
  MIDL 2018. arXiv:1804.03999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block  (identical to ResUNet for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """
    Two-layer Convolutional Residual Block with Batch Normalization and LeakyReLU.
    Includes identity shortcut (with 1×1 conv projection if channels change).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()

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
# Attention Gate
# ─────────────────────────────────────────────────────────────────────────────
class AttentionGate(nn.Module):
    """
    Soft attention gate (Oktay et al. 2018).

    Parameters
    ----------
    F_g : int   Channels in the gating signal  (from decoder / coarser scale).
    F_l : int   Channels in the skip connection (from encoder / finer scale).
    F_int : int  Intermediate channel count for the attention computation.

    Forward
    -------
    g : (B, F_g, H, W)  -- gating signal (already upsampled to match x spatially)
    x : (B, F_l, H, W)  -- encoder skip feature map
    Returns weighted skip:  x * alpha,  where alpha in [0, 1]^{H×W}
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()

        # 1×1 conv to collapse gating signal to F_int channels
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # 1×1 conv to collapse skip connection to F_int channels
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # 1×1 conv to produce a single-channel attention map, then sigmoid
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Project gating signal and skip to shared F_int space
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Align spatial dims if necessary (gating may be slightly smaller)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)

        # Element-wise sum → ReLU → 1-channel attention map → Sigmoid
        alpha = self.psi(self.relu(g1 + x1))   # (B, 1, H, W)

        # Broadcast and weight the skip feature map
        return x * alpha                         # (B, F_l, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Attention U-Net
# ─────────────────────────────────────────────────────────────────────────────
class AttentionUNet(nn.Module):
    """
    Attention U-Net for Speckle Denoising on RGB Satellite Imagery.

    Parameters
    ----------
    in_channels  : int        Input channels  (3 for RGB).
    out_channels : int        Output channels (3 for RGB).
    features     : list[int]  Channel depths per encoder level.
                              Default = [64, 128, 256, 512]  (same as ResUNet).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: list = [64, 128, 256, 512],
    ):
        super(AttentionUNet, self).__init__()

        self.features = features

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.poolers  = nn.ModuleList()

        curr_in = in_channels
        for feat in features:
            self.encoders.append(ResidualBlock(curr_in, feat))
            self.poolers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_in = feat

        # ── Bottleneck ───────────────────────────────────────────────────────
        bottleneck_dim = features[-1] * 2          # 1024
        self.bottleneck = ResidualBlock(features[-1], bottleneck_dim)

        # ── Decoder + Attention Gates ─────────────────────────────────────────
        self.upconvs     = nn.ModuleList()
        self.att_gates   = nn.ModuleList()
        self.decoders    = nn.ModuleList()

        prev_dim = bottleneck_dim
        for feat in reversed(features):
            # Transpose conv: upsample prev_dim → feat
            self.upconvs.append(
                nn.ConvTranspose2d(prev_dim, feat, kernel_size=2, stride=2)
            )
            # Attention gate:
            #   g  = upsampled feature  (feat channels)
            #   x  = encoder skip      (feat channels)
            #   F_int = feat // 2
            self.att_gates.append(
                AttentionGate(F_g=feat, F_l=feat, F_int=feat // 2)
            )
            # After concat(attended_skip, upsampled): feat + feat = feat*2 → feat
            self.decoders.append(ResidualBlock(feat * 2, feat))
            prev_dim = feat

        # ── Final Output Layer ────────────────────────────────────────────────
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid(),   # output strictly in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # ── Encoder pass ─────────────────────────────────────────────────────
        out = x
        for encoder, pooler in zip(self.encoders, self.poolers):
            out = encoder(out)
            skip_connections.append(out)
            out = pooler(out)

        # ── Bottleneck ────────────────────────────────────────────────────────
        out = self.bottleneck(out)

        # ── Decoder pass (with attention gates on every skip) ─────────────────
        skips = skip_connections[::-1]   # reverse: deepest skip first
        for idx in range(len(self.features)):
            # 1. Upsample
            out = self.upconvs[idx](out)
            skip = skips[idx]

            # 2. Align spatial dims (handles odd input sizes gracefully)
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)

            # 3. Apply attention gate → weighted skip
            attended_skip = self.att_gates[idx](g=out, x=skip)

            # 4. Concatenate attended skip + upsampled feature
            out = torch.cat([attended_skip, out], dim=1)

            # 5. Residual decoder block
            out = self.decoders[idx](out)

        # ── Final 1×1 projection ──────────────────────────────────────────────
        return self.final_conv(out)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AttentionUNet(in_channels=3, out_channels=3).to(device)
    dummy  = torch.randn(2, 3, 256, 256).to(device)
    out    = model(dummy)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Attention U-Net Smoke Test PASSED")
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {out.shape}")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Output range     : [{out.min():.4f}, {out.max():.4f}]")
