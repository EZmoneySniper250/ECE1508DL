"""
MobileNetV3-Small backbone for student networks.

Produces two feature levels:
  - fine features:   1/2 resolution, 32-dim   (from MobileNetV3 stage 0)
  - coarse features: 1/8 resolution, 128-dim  (from MobileNetV3 stages 1–5)

Significantly smaller than LoFTR's ResNet-18 FPN backbone.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV3Backbone(nn.Module):
    """
    Truncated MobileNetV3-Small as shared backbone.
    Extracts coarse (1/8) and fine (1/2) features from grayscale input.
    """

    def __init__(self, coarse_channels=128):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        features = list(mobilenet.features.children())

        # Modify first layer: 3-channel RGB → 1-channel grayscale
        orig_conv = features[0][0]
        features[0][0] = nn.Conv2d(
            1, 16, kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride, padding=orig_conv.padding, bias=False
        )
        # Initialize grayscale channel with mean of pretrained RGB weights
        with torch.no_grad():
            features[0][0].weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))

        # MobileNetV3-Small features structure (verified empirically on 480×480):
        # [0]  ConvBNActivation  stride=2  → 1/2   (16 ch)
        # [1]  InvertedResidual  stride=2  → 1/4   (16 ch)
        # [2]  InvertedResidual  stride=2  → 1/8   (24 ch)  ← coarse target
        # [3]  InvertedResidual  stride=1  → 1/8   (24 ch)
        # [4]  InvertedResidual  stride=2  → 1/16  (40 ch)  ← too deep
        # ...

        # Fine features: up to 1/2 resolution
        self.fine_layers = nn.Sequential(*features[:1])    # → 1/2, 16 ch

        # Coarse features: up to 1/8 resolution
        self.coarse_layers = nn.Sequential(*features[1:4])  # → 1/8, 24 ch

        # Project to unified channel counts
        self.fine_proj = nn.Conv2d(16, 32, 1)
        self.coarse_proj = nn.Conv2d(24, coarse_channels, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, H, W) grayscale image, H and W must be divisible by 8.

        Returns:
            coarse_feat: (B, coarse_channels, H/8, W/8)
            fine_feat:   (B, 32, H/2, W/2)
        """
        feat_fine = self.fine_layers(x)                   # (B, 16, H/2, W/2)
        feat_coarse = self.coarse_layers(feat_fine)        # (B, 40, H/8, W/8)

        feat_fine = self.fine_proj(feat_fine)              # (B, 32, H/2, W/2)
        feat_coarse = self.coarse_proj(feat_coarse)        # (B, coarse_channels, H/8, W/8)

        return feat_coarse, feat_fine
