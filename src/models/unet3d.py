"""
unet3d.py
---------
3D U-Net with optional attention gates for fetal brain MRI segmentation.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~
Encoder  : 4 levels of DoubleConv blocks, each followed by max-pool ×2.
Bottleneck: DoubleConv at the deepest level.
Decoder  : 4 upsampling levels using transposed convolution + skip connection
           (optionally gated by an attention block) + DoubleConv.
Output   : 1×1×1 convolution → ``out_channels`` logits.

Default configuration (features=[32,64,128,256]):
  Input : [B,  1, 128, 128, 128]
  Output: [B,  7, 128, 128, 128]
  Params: ~26.3 M

Reference
~~~~~~~~~
Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
Segmentation", MICCAI 2015.
Oktay et al., "Attention U-Net", MIDL 2018.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive Conv3d → InstanceNorm3d → LeakyReLU blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        layers: List[nn.Module] = [
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        layers += [
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Max-pool ×2 followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Transposed convolution upsample ×2 + skip-connection concatenation
    + DoubleConv.

    When ``use_attention=True`` an attention gate is applied to the skip
    features before concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate(
                gate_channels=in_channels // 2,
                feat_channels=skip_channels,
                inter_channels=skip_channels // 2,
            )
        # after concat: (in_channels // 2) + skip_channels
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Pad x if skip spatial dims are slightly larger (odd-sized inputs)
        diff = [skip.shape[i] - x.shape[i] for i in range(2, 5)]
        x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])

        if self.use_attention:
            skip = self.attention(gate=x, feat=skip)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Additive soft attention gate (Oktay et al., 2018).

    Gates the skip-connection features ``feat`` using the decoder signal
    ``gate``.  Both are projected to ``inter_channels``, summed, passed
    through a sigmoid, and multiplied element-wise with ``feat``.
    """

    def __init__(
        self,
        gate_channels: int,
        feat_channels: int,
        inter_channels: int,
    ) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(inter_channels, affine=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(feat_channels, inter_channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(inter_channels, affine=True),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(gate)
        x1 = self.W_x(feat)
        # Align spatial dims
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="trilinear", align_corners=True)
        psi = self.psi(self.relu(g1 + x1))
        return feat * psi


# ---------------------------------------------------------------------------
# Full 3-D U-Net
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """
    3-D U-Net with attention gates.

    Args:
        in_channels:   Number of input channels (1 for single-channel MRI).
        out_channels:  Number of output segmentation classes.
        features:      Number of feature maps at each encoder level.
                       Default ``[32, 64, 128, 256]`` → bottleneck at 512.
        dropout:       Dropout rate applied inside DoubleConv blocks.
        use_attention: Whether to use attention gates in the decoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 7,
        features: List[int] = None,
        dropout: float = 0.1,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        # ---- Encoder ----
        self.enc1 = DoubleConv(in_channels, features[0], dropout=dropout)
        self.enc2 = Down(features[0], features[1], dropout=dropout)
        self.enc3 = Down(features[1], features[2], dropout=dropout)
        self.enc4 = Down(features[2], features[3], dropout=dropout)

        # ---- Bottleneck ----
        bottleneck_channels = features[3] * 2
        self.bottleneck = Down(features[3], bottleneck_channels, dropout=dropout)

        # ---- Decoder ----
        self.dec4 = Up(bottleneck_channels, features[3], features[3], dropout, use_attention)
        self.dec3 = Up(features[3], features[2], features[2], dropout, use_attention)
        self.dec2 = Up(features[2], features[1], features[1], dropout, use_attention)
        self.dec1 = Up(features[1], features[0], features[0], dropout, use_attention)

        # ---- Output ----
        self.output_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)       # [B, F0, D,   H,   W  ]
        s2 = self.enc2(s1)      # [B, F1, D/2, H/2, W/2]
        s3 = self.enc3(s2)      # [B, F2, D/4, H/4, W/4]
        s4 = self.enc4(s3)      # [B, F3, D/8, H/8, W/8]

        # Bottleneck
        bn = self.bottleneck(s4)  # [B, F3*2, D/16, ...]

        # Decoder
        d4 = self.dec4(bn, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.output_conv(d1)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_unet3d(config: Dict) -> UNet3D:
    """
    Instantiate a UNet3D from a loaded YAML config dict.

    Example config section::

        model:
          in_channels: 1
          out_channels: 7
          features: [32, 64, 128, 256]
          dropout: 0.1
    """
    model_cfg = config["model"]
    return UNet3D(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 7),
        features=model_cfg.get("features", [32, 64, 128, 256]),
        dropout=model_cfg.get("dropout", 0.1),
        use_attention=model_cfg.get("use_attention", True),
    )