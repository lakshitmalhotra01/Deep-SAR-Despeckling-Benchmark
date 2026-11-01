"""
models/multires_unet.py
=======================
Multi-Resolution U-Net (MultiResUNet) architecture for satellite optical
image speckle denoising.

Key Innovations (Ibtehaz & Rahman, 2020):
  1. MultiRes Blocks:
     - Replaces standard convolutional/residual blocks.
     - Factorises larger receptive fields via compact chained 3x3 convolutions
       (3x3, 5x5, 7x7 equivalents).
     - Concatenates multi-scale features together with a 1x1 residual shortcut.
  2. ResPaths:
     - Replaces plain skip connections between encoder and decoder.
     - Bridges the semantic gap between lower-level encoder features and
       higher-level decoder featur es using non-linear residual paths
       .
     - Deeper ResPaths for shallow levels (4, 3, 2, 1 blocks for stages 1..4).
  3. Consistent 4-level depth:
     - Matches ResUNet and Attention U-Net for scientific comparison.
     - Input  : (B, 3, 256, 256) [noisy RGB satellite image]
     - Output : (B, 3, 256, 256) [denoised RGB image, values in [0, 1]]

Reference:
  Ibtehaz, N. & Rahman, M.S., "MultiResUNet : Rethinking the U-Net architecture
  for multimodal biomedical image segmentation", Neural Networks 121 (2020) 74-87.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# MultiRes Block
# ─────────────────────────────────────────────────────────────────────────────
class MultiResBlock(nn.Module):
    """
    Multi-Resolution Block:
    Factorises multi-scale features (3x3, 5x5, 7x7 receptive fields)
    via three cascaded 3x3 convolutions and adds a 1x1 residual shortcut.

    Channel allocation for total output channels U:
      W1 = int(U * 0.167)  (~1/6 of channels for 3x3)
      W2 = int(U * 0.333)  (~1/3 of channels for 5x5 equivalent)
      W3 = U - W1 - W2     (~1/2 of channels for 7x7 equivalent)
    Total concatenated channels = W1 + W2 + W3 = U.
    """

    def __init__(self, in_channels: int, out_channels: int, alpha: float = 1.0):
        super(MultiResBlock, self).__init__()

        # Allocate channel widths for the 3 resolution branches
        w1 = max(1, int(out_channels * 0.167 * alpha))
        w2 = max(1, int(out_channels * 0.333 * alpha))
        w3 = out_channels - (w1 + w2)
        if w3 <= 0:
            w3 = max(1, out_channels // 2)

        self.w1, self.w2, self.w3 = w1, w2, w3

        # Stream 1: 3x3 conv
        self.conv3x3_1 = nn.Conv2d(in_channels, w1, kernel_size=3, padding=1, bias=False)
        self.bn1       = nn.BatchNorm2d(w1)

        # Stream 2: 3x3 conv (cascaded from stream 1 -> 5x5 receptive field)
        self.conv3x3_2 = nn.Conv2d(w1, w2, kernel_size=3, padding=1, bias=False)
        self.bn2       = nn.BatchNorm2d(w2)

        # Stream 3: 3x3 conv (cascaded from stream 2 -> 7x7 receptive field)
        self.conv3x3_3 = nn.Conv2d(w2, w3, kernel_size=3, padding=1, bias=False)
        self.bn3       = nn.BatchNorm2d(w3)

        # Concatenation batchnorm
        self.bn_cat = nn.BatchNorm2d(w1 + w2 + w3)

        # Residual shortcut: 1x1 conv + BN
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, w1 + w2 + w3, kernel_size=1, bias=False),
            nn.BatchNorm2d(w1 + w2 + w3),
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)

        # Cascaded multi-scale features
        out1 = self.act(self.bn1(self.conv3x3_1(x)))
        out2 = self.act(self.bn2(self.conv3x3_2(out1)))
        out3 = self.act(self.bn3(self.conv3x3_3(out2)))

        # Concatenate all three scales
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.bn_cat(out)

        # Add residual connection and activate
        out = self.act(out + res)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# ResPath Block & Sequence
# ─────────────────────────────────────────────────────────────────────────────
class ResPathBlock(nn.Module):
    """
    Single block inside a ResPath:
    Combines a 3x3 convolution with a 1x1 residual shortcut.
    """

    def __init__(self, channels: int):
        super(ResPathBlock, self).__init__()

        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(channels)

        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2     = nn.BatchNorm2d(channels)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.bn2(self.conv1x1(x))
        out = self.bn1(self.conv3x3(x))
        out = self.act(out + res)
        return out


class ResPath(nn.Module):
    """
    ResPath (Residual Path):
    Bridges the semantic gap along skip connections by running features
    through num_blocks residual blocks.
    """

    def __init__(self, channels: int, num_blocks: int):
        super(ResPath, self).__init__()
        self.blocks = nn.Sequential(
            *[ResPathBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# ─────────────────────────────────────────────────────────────────────────────
# MultiResUNet
# ─────────────────────────────────────────────────────────────────────────────
class MultiResUNet(nn.Module):
    """
    Multi-Resolution U-Net (MultiResUNet) for Satellite Image Speckle Denoising.

    Parameters
    ----------
    in_channels  : int
        Input RGB image channels (default = 3).
    out_channels : int
        Target output RGB channels (default = 3).
    features     : list of int
        Channels at each of the 4 encoder levels (default = [64, 128, 256, 512]).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: list = [64, 128, 256, 512],
    ):
        super(MultiResUNet, self).__init__()

        self.features = features
        num_levels = len(features)

        # ── Encoder Path ─────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.poolers  = nn.ModuleList()

        curr_in = in_channels
        for feat in features:
            self.encoders.append(MultiResBlock(curr_in, feat))
            self.poolers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_in = feat

        # ── Bottleneck ───────────────────────────────────────────────────────
        bottleneck_dim = features[-1] * 2  # 1024
        self.bottleneck = MultiResBlock(features[-1], bottleneck_dim)

        # ── ResPaths for Skip Connections ─────────────────────────────────────
        # Shallow levels get more blocks to bridge larger semantic gaps:
        # Level 0 (64 ch)  -> 4 blocks
        # Level 1 (128 ch) -> 3 blocks
        # Level 2 (256 ch) -> 2 blocks
        # Level 3 (512 ch) -> 1 block
        self.respaths = nn.ModuleList()
        for idx, feat in enumerate(features):
            num_blocks = num_levels - idx
            self.respaths.append(ResPath(channels=feat, num_blocks=num_blocks))

        # ── Decoder Path ─────────────────────────────────────────────────────
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_dim = bottleneck_dim
        for feat in reversed(features):
            # Upsample: prev_dim -> feat
            self.upconvs.append(
                nn.ConvTranspose2d(prev_dim, feat, kernel_size=2, stride=2)
            )
            # Concatenation of respath skip (feat) + upsampled (feat) = feat * 2
            self.decoders.append(MultiResBlock(feat * 2, feat))
            prev_dim = feat

        # ── Final Output Layer ────────────────────────────────────────────────
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid(),  # Output values strictly in [0, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # 1. Encoder forward pass
        out = x
        for idx in range(len(self.features)):
            out = self.encoders[idx](out)
            # Pass through corresponding ResPath before saving skip
            skip_processed = self.respaths[idx](out)
            skip_connections.append(skip_processed)
            out = self.poolers[idx](out)

        # 2. Bottleneck
        out = self.bottleneck(out)

        # 3. Decoder forward pass
        skips = skip_connections[::-1]  # Deepest skip first
        for idx in range(len(self.features)):
            out = self.upconvs[idx](out)
            skip = skips[idx]

            # Spatial alignment for odd dimensions if needed
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)

            out = torch.cat([skip, out], dim=1)
            out = self.decoders[idx](out)

        # 4. Final projection to RGB image [0, 1]
        return self.final_conv(out)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiResUNet(in_channels=3, out_channels=3).to(device)
    dummy = torch.randn(2, 3, 256, 256).to(device)

    with torch.no_grad():
        out = model(dummy)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("MultiResUNet Smoke Test PASSED")
    print(f"  Device           : {device}")
    print(f"  Input Shape      : {list(dummy.shape)}")
    print(f"  Output Shape     : {list(out.shape)}")
    print(f"  Total Params     : {total:,}")
    print(f"  Trainable Params : {trainable:,}")
    print(f"  Output Range     : [{out.min():.4f}, {out.max():.4f}]")
