"""
Dilated Interaction Module — replaces attention in Student-CNN.

Uses stacked dilated convolutions (dilation rates 1, 2, 4, 8) to enlarge the
receptive field within each image independently, then L2-normalizes features
for stable dot-product cross-correlation in the matching stage.

This is the core architectural difference from Student-Hybrid: no attention
mechanism is used, only purely local (convolutional) operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StudentCNNConfig


class DilatedInteractionModule(nn.Module):
    """
    Purely convolutional feature interaction module.

    Pipeline:
        1. Dilated Conv Stack (shared weights for both images):
           4 layers of dilated convolutions with rates [1, 2, 4, 8]
           → effective receptive field diameter ≈ 31 pixels at coarse level
           → equivalent to ~248 pixels at original resolution

        2. Residual connection: enhanced = dilated(feat) + feat

        3. L2 normalization: ensures stable dot-product correlation
    """

    def __init__(self, config: StudentCNNConfig = None):
        super().__init__()
        if config is None:
            config = StudentCNNConfig()

        dim = config.dilated_channels    # 128
        rates = config.dilation_rates    # [1, 2, 4, 8]

        # Dilated conv stack — shared for both images
        layers = []
        for d in rates:
            layers.extend([
                nn.Conv2d(dim, dim, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ])
        self.dilated_stack = nn.Sequential(*layers)

        # Final 1×1 projection
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(
        self, feat0: torch.Tensor, feat1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat0: (B, C, H0, W0) coarse features from image 0
            feat1: (B, C, H1, W1) coarse features from image 1

        Returns:
            enhanced0: (B, C, H0, W0) L2-normalized enhanced features
            enhanced1: (B, C, H1, W1) L2-normalized enhanced features
        """
        # Apply dilated convolutions independently to each image
        enhanced0 = self.proj(self.dilated_stack(feat0))
        enhanced1 = self.proj(self.dilated_stack(feat1))

        # Residual connection
        enhanced0 = enhanced0 + feat0
        enhanced1 = enhanced1 + feat1

        # L2 normalize for stable dot-product correlation
        enhanced0 = F.normalize(enhanced0, p=2, dim=1)
        enhanced1 = F.normalize(enhanced1, p=2, dim=1)

        return enhanced0, enhanced1
