"""
Student-CNN: Pure Convolutional Student Network for LoFTR Distillation.

Key architecture:
    - MobileNetV3-Small backbone (shared with Student-Hybrid)
    - Dilated convolution feature interaction (NO attention)
    - Dual-softmax coarse matching
    - Local window fine refinement

The only difference from Student-Hybrid is the feature interaction module:
    Student-CNN uses dilated convs + cross-correlation
    Student-Hybrid uses shallow cross-attention layers
"""

import torch
import torch.nn as nn

from .config import StudentCNNConfig
from .backbone import MobileNetV3Backbone
from .dilated_interaction import DilatedInteractionModule
from .matching import CoarseMatching, FineRefinement


class StudentCNN(nn.Module):
    """
    Student-CNN: fully convolutional student for LoFTR distillation.

    Pipeline:
        1. Backbone: extract coarse (1/8) + fine (1/2) features from both images
        2. Dilated Interaction: enlarge receptive field with dilated convs (no attention)
        3. Coarse Matching: dual-softmax on dot-product correlation → confidence matrix
        4. Fine Refinement: local window correlation → sub-pixel coordinates

    During training, also outputs projected features for distillation.
    """

    def __init__(self, config: StudentCNNConfig = None):
        super().__init__()
        if config is None:
            config = StudentCNNConfig()
        self.config = config

        # ─── Shared backbone ────────────────────────────────────────────────
        self.backbone = MobileNetV3Backbone(coarse_channels=config.coarse_dim)

        # ─── Feature interaction: dilated convolutions (replaces attention) ─
        self.interaction = DilatedInteractionModule(config)

        # ─── Matching heads ─────────────────────────────────────────────────
        self.coarse_matching = CoarseMatching(config)
        self.fine_refinement = FineRefinement(config)

        # ─── 1×1 conv projector for feature distillation ────────────────────
        # Maps student coarse features (128-dim) → teacher dim (256-dim)
        # Only used during training; discarded at inference
        self.feat_projector = nn.Conv2d(
            config.coarse_dim, config.teacher_coarse_dim, 1, bias=False
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data: dict) -> dict:
        """
        Args:
            data: dict with:
                - 'image0': (B, 1, H, W) grayscale tensor in [0, 1]
                - 'image1': (B, 1, H, W) grayscale tensor in [0, 1]

        Returns:
            data: updated dict with:
              Training mode:
                - 'conf_matrix':  (B, N0, N1) coarse confidence matrix
                - 'sim_matrix':   (B, N0, N1) raw similarity scores
                - 'coarse_feat0':      (B, 128, H/8, W/8)
                - 'coarse_feat1':      (B, 128, H/8, W/8)
                - 'coarse_feat0_proj': (B, 256, H/8, W/8)
                - 'coarse_feat1_proj': (B, 256, H/8, W/8)

              Inference mode (additionally):
                - 'keypoints0':  (M, 2) matched keypoint coords in image 0
                - 'keypoints1':  (M, 2) matched keypoint coords in image 1
                - 'confidence':  (M,)   match confidence scores
        """
        img0 = data["image0"]
        img1 = data["image1"]

        # ── 1. Feature extraction ───────────────────────────────────────────
        coarse0, fine0 = self.backbone(img0)  # (B,128,H/8,W/8), (B,32,H/2,W/2)
        coarse1, fine1 = self.backbone(img1)

        # Store raw coarse features for distillation
        data["coarse_feat0"] = coarse0
        data["coarse_feat1"] = coarse1

        # Project coarse features for feature-level distillation loss
        if self.training:
            data["coarse_feat0_proj"] = self.feat_projector(coarse0)
            data["coarse_feat1_proj"] = self.feat_projector(coarse1)

        # ── 2. Feature interaction via dilated convolutions ─────────────────
        enhanced0, enhanced1 = self.interaction(coarse0, coarse1)

        # ── 3. Coarse matching ──────────────────────────────────────────────
        data = self.coarse_matching(enhanced0, enhanced1, data)

        # ── 4. Fine refinement (inference only) ─────────────────────────────
        if not self.training:
            data = self.fine_refinement(fine0, fine1, data)

            # Output interface compatible with LoFTR
            data["keypoints0"] = data.get("mkpts0_f", torch.zeros(0, 2))
            data["keypoints1"] = data.get("mkpts1_f", torch.zeros(0, 2))
            data["confidence"] = data.get("mconf", torch.zeros(0))

        return data

    def count_parameters(self) -> dict:
        """Count parameters in each component."""
        components = {
            "backbone": self.backbone,
            "interaction": self.interaction,
            "coarse_matching": self.coarse_matching,
            "fine_refinement": self.fine_refinement,
            "feat_projector": self.feat_projector,
        }
        counts = {}
        total = 0
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            counts[name] = n
            total += n
        counts["total"] = total
        return counts
