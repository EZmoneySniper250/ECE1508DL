"""
Lightweight CNN backbone for Student networks.

Produces two feature levels:
  - fine features:   1/2 resolution, 64-dim   (from stage 1)
  - coarse features: 1/8 resolution, 128-dim  (from stage 3)

Significantly smaller than LoFTR's ResNet-18 FPN backbone (~1M vs ~11M params).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StudentCNNConfig


class BasicResBlock(nn.Module):
    """Simple residual block with two 3×3 convolutions."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class LightweightBackbone(nn.Module):
    """
    Lightweight CNN backbone for student networks.

    Architecture:
        Stem:   1 → 32 channels, stride=1  (full resolution)
        Stage1: 32 → 64, stride=2          (1/2 resolution)  → fine features
        Stage2: 64 → 64, stride=2          (1/4 resolution)
        Stage3: 64 → 128, stride=2         (1/8 resolution)  → coarse features

    For 480×640 input:
        fine features:   (B, 64,  240, 320)
        coarse features: (B, 128, 60,  80)
    """

    def __init__(self, config: StudentCNNConfig = None):
        super().__init__()
        if config is None:
            config = StudentCNNConfig()

        c = config.backbone_channels  # [32, 64, 128]

        # Stem: grayscale → 32 channels, no spatial reduction
        self.stem = nn.Sequential(
            nn.Conv2d(1, c[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32 → 64, stride=2 → 1/2 resolution (fine features)
        self.stage1 = nn.Sequential(
            BasicResBlock(c[0], c[1], stride=2),
            BasicResBlock(c[1], c[1]),
        )

        # Stage 2: 64 → 64, stride=2 → 1/4 resolution
        self.stage2 = nn.Sequential(
            BasicResBlock(c[1], c[1], stride=2),
            BasicResBlock(c[1], c[1]),
        )

        # Stage 3: 64 → 128, stride=2 → 1/8 resolution (coarse features)
        self.stage3 = nn.Sequential(
            BasicResBlock(c[1], c[2], stride=2),
            BasicResBlock(c[2], c[2]),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, H, W) grayscale image tensor, H and W must be multiples of 8.

        Returns:
            coarse_feat: (B, 128, H/8, W/8)
            fine_feat:   (B, 64,  H/2, W/2)
        """
        x = self.stem(x)           # (B, 32,  H,   W)
        fine_feat = self.stage1(x)  # (B, 64,  H/2, W/2)
        x = self.stage2(fine_feat)  # (B, 64,  H/4, W/4)
        coarse_feat = self.stage3(x)  # (B, 128, H/8, W/8)

        return coarse_feat, fine_feat
