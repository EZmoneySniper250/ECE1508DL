"""
Student-Hybrid: MobileNetV3-Small backbone + shallow cross-attention
for feature matching with knowledge distillation from LoFTR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .backbone import MobileNetV3Backbone
from .config import StudentHybridConfig


# Shallow Cross-Attention Module

class ShallowCrossAttention(nn.Module):
    """
    Lightweight self-attention + cross-attention, repeated N rounds.
    This is the simplified version of LoFTR's Transformer module.

    LoFTR uses:  4 rounds, 8 heads, d_model=256
    We use:      2 rounds, 4 heads, d_model=128
    nrounds: number of self+cross attention rounds (half of LoFTR's rounds)
    nhead:   number of attention heads (half of LoFTR's heads)
    """

    def __init__(self, d_model=128, nhead=4, n_rounds=2):
        super().__init__()
        self.n_rounds = n_rounds
        self.d_model = d_model

        # self-attention + cross-attention for each round
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(n_rounds)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(n_rounds)
        ])

        #LayerNorm + FFN (after each attention)
        self.self_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_rounds)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_rounds)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(n_rounds)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_rounds)
        ])

    def forward(self, f0, f1):
        """
        Args:
            f0: (B, C, H, W) coarse features from image 0
            f1: (B, C, H, W) coarse features from image 1
        Returns:
            f0_out: (B, C, H, W) transformed features
            f1_out: (B, C, H, W) transformed features
        """
        B, C, H, W = f0.shape

        # Flatten: (B, C, H, W) → (B, H*W, C)
        f0 = f0.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        f1 = f1.flatten(2).permute(0, 2, 1)

        for i in range(self.n_rounds):
            # ── Self-attention ──
            # f0 attends to itself, f1 attends to itself
            f0_sa, _ = self.self_attn_layers[i](f0, f0, f0)
            f1_sa, _ = self.self_attn_layers[i](f1, f1, f1)
            f0 = self.self_norms[i](f0 + f0_sa)  # residual + norm
            f1 = self.self_norms[i](f1 + f1_sa)

            # ── Cross-attention ──
            # f0 queries f1, f1 queries f0
            f0_ca, _ = self.cross_attn_layers[i](f0, f1, f1)
            f1_ca, _ = self.cross_attn_layers[i](f1, f0, f0)
            f0 = self.cross_norms[i](f0 + f0_ca)
            f1 = self.cross_norms[i](f1 + f1_ca)

            # ── FFN ──
            f0 = self.ffn_norms[i](f0 + self.ffn_layers[i](f0))
            f1 = self.ffn_norms[i](f1 + self.ffn_layers[i](f1))

        # Reshape back: (B, H*W, C) → (B, C, H, W)
        f0 = f0.permute(0, 2, 1).reshape(B, C, H, W)
        f1 = f1.permute(0, 2, 1).reshape(B, C, H, W)

        return f0, f1


# ─── Coarse Matching Head ────────────────────────────────────────────────────

class CoarseMatching(nn.Module):
    """
    Dual-softmax matching on coarse features.
    Produces a confidence matrix and selects mutual nearest neighbor matches.
    """

    def __init__(self, temperature=0.1, threshold=0.2):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold

    def forward(self, f0, f1):
        """
        Args:
            f0: (B, C, H0, W0) transformed coarse features from image 0
            f1: (B, C, H1, W1) transformed coarse features from image 1
        Returns:
            confidence_matrix: (B, H0*W0, H1*W1) matching confidence
            mkpts0: (N, 2) matched keypoint coordinates in image 0 (coarse grid)
            mkpts1: (N, 2) matched keypoint coordinates in image 1 (coarse grid)
            mconf:  (N,) confidence scores
        """
        B, C, H0, W0 = f0.shape
        _, _, H1, W1 = f1.shape

        # Flatten and L2 normalize
        f0_flat = f0.flatten(2).permute(0, 2, 1)  # (B, H0*W0, C)
        f1_flat = f1.flatten(2).permute(0, 2, 1)  # (B, H1*W1, C)
        f0_flat = F.normalize(f0_flat, dim=-1)
        f1_flat = F.normalize(f1_flat, dim=-1)

        # Similarity matrix
        S = torch.bmm(f0_flat, f1_flat.transpose(1, 2)) / self.temperature  # (B, N0, N1)

        # Dual-softmax
        P = F.softmax(S, dim=-1) * F.softmax(S, dim=-2)  # (B, N0, N1)

        # Extract matches (for inference / evaluation)
        # Mutual nearest neighbor
        max0 = P.max(dim=-1)   # best match in f1 for each f0 location
        max1 = P.max(dim=-2)   # best match in f0 for each f1 location

        # only dealling with batch_size=1 case (for evaluation)
        # during training we just return the confidence matrix for loss computation
        if B == 1:
            indices0 = max0.indices[0]  # (N0,) best f1 index for each f0
            indices1 = max1.indices[0]  # (N1,) best f0 index for each f1

            # Mutual nearest neighbor check
            mutual = indices1[indices0] == torch.arange(len(indices0), device=P.device)

            # Confidence threshold
            conf = max0.values[0]  # (N0,)
            valid = mutual & (conf > self.threshold)

            # Convert flat indices to 2D coordinates
            idx0 = torch.where(valid)[0]
            idx1 = indices0[valid]

            mkpts0 = torch.stack([idx0 % W0, idx0 // W0], dim=1).float()  # (N, 2) in coarse grid
            mkpts1 = torch.stack([idx1 % W1, idx1 // W1], dim=1).float()
            mconf = conf[valid]
        else:
            mkpts0, mkpts1, mconf = None, None, None

        return P, mkpts0, mkpts1, mconf


# ─── Fine Refinement ─────────────────────────────────────────────────────────

class FineRefinement(nn.Module):
    """
    Simplified fine-level refinement.
    For each coarse match, crops a local window from fine features
    and computes sub-pixel offset via correlation + expectation.
    """

    def __init__(self, fine_channels=32, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.proj = nn.Conv2d(fine_channels, fine_channels, 1)

    def forward(self, mkpts0_coarse, mkpts1_coarse, feat_fine0, feat_fine1, stride=8):
        """
        Args:
            mkpts0_coarse: (N, 2) coarse match coords in image 0 (coarse grid units)
            mkpts1_coarse: (N, 2) coarse match coords in image 1 (coarse grid units)
            feat_fine0: (B, C, H_fine, W_fine) fine features from image 0
            feat_fine1: (B, C, H_fine, W_fine) fine features from image 1
            stride: ratio between fine and coarse resolution (typically 4, since fine=1/2, coarse=1/8)
        Returns:
            mkpts0_fine: (N, 2) refined coords in original image resolution
            mkpts1_fine: (N, 2) refined coords in original image resolution
        """
        if mkpts0_coarse is None or len(mkpts0_coarse) == 0:
            return mkpts0_coarse, mkpts1_coarse

        feat_fine0 = self.proj(feat_fine0)
        feat_fine1 = self.proj(feat_fine1)

        W = self.window_size
        half_w = W // 2
        _, C, H_f, W_f = feat_fine0.shape

        # Convert coarse grid coords to fine feature coords
        # coarse grid (x, y) → fine feature coords: multiply by stride ratio (coarse_stride / fine_stride = 8/2 = 4)
        fine_ratio = stride // 2  # = 4 if fine=1/2, coarse=1/8
        center0 = (mkpts0_coarse * fine_ratio).long()  # (N, 2) in fine feature coords
        center1 = (mkpts1_coarse * fine_ratio).long()

        refined_pts0 = []
        refined_pts1 = []

        for n in range(len(center0)):
            cx0, cy0 = center0[n]
            cx1, cy1 = center1[n]

            # Bounds check
            if (cx0 - half_w < 0 or cx0 + half_w >= W_f or
                cy0 - half_w < 0 or cy0 + half_w >= H_f or
                cx1 - half_w < 0 or cx1 + half_w >= W_f or
                cy1 - half_w < 0 or cy1 + half_w >= H_f):
                # Out of bounds, keep coarse position
                refined_pts0.append(mkpts0_coarse[n] * stride + stride // 2)
                refined_pts1.append(mkpts1_coarse[n] * stride + stride // 2)
                continue

            # Crop local windows from fine features
            patch0 = feat_fine0[0, :, cy0 - half_w:cy0 + half_w + 1, cx0 - half_w:cx0 + half_w + 1]  # (C, W, W)
            patch1 = feat_fine1[0, :, cy1 - half_w:cy1 + half_w + 1, cx1 - half_w:cx1 + half_w + 1]

            # Correlation: center of patch0 vs all positions of patch1
            center_feat = patch0[:, half_w, half_w]  # (C,)
            corr = torch.einsum('c,chw->hw', center_feat, patch1)  # (W, W)

            # Softmax → expectation for sub-pixel offset
            corr_soft = F.softmax(corr.flatten(), dim=0).reshape(W, W)

            # Expected offset from center
            grid_y, grid_x = torch.meshgrid(
                torch.arange(W, device=corr.device).float() - half_w,
                torch.arange(W, device=corr.device).float() - half_w,
                indexing='ij'
            )
            offset_x = (corr_soft * grid_x).sum()
            offset_y = (corr_soft * grid_y).sum()

            # Final refined coordinates in original image resolution
            pt0 = mkpts0_coarse[n] * stride + stride // 2
            pt1_x = mkpts1_coarse[n, 0] * stride + stride // 2 + offset_x * (stride // fine_ratio)
            pt1_y = mkpts1_coarse[n, 1] * stride + stride // 2 + offset_y * (stride // fine_ratio)

            refined_pts0.append(pt0)
            refined_pts1.append(torch.stack([pt1_x, pt1_y]))

        mkpts0_fine = torch.stack(refined_pts0)
        mkpts1_fine = torch.stack(refined_pts1)

        return mkpts0_fine, mkpts1_fine


# ─── Full Student-Hybrid Model ───────────────────────────────────────────────

class StudentHybrid(nn.Module):
    """
    Complete Student-Hybrid model for feature matching.
    Architecture: MobileNetV3 backbone + shallow cross-attention + coarse matching + fine refinement.
    """

    def __init__(self, config: StudentHybridConfig = None):
        super().__init__()
        if config is None:
            config = StudentHybridConfig()
        self.config = config

        self.backbone = MobileNetV3Backbone(coarse_channels=config.coarse_dim)
        self.attention = ShallowCrossAttention(
            d_model=config.coarse_dim, nhead=config.nhead, n_rounds=config.n_rounds
        )
        self.coarse_matching = CoarseMatching(
            temperature=config.temperature, threshold=config.match_threshold
        )
        self.fine_refinement = FineRefinement(
            fine_channels=config.fine_dim, window_size=config.fine_window_size
        )

        # 1×1 conv projector for feature distillation (student → teacher dim)
        self.feat_projector = nn.Conv2d(
            config.coarse_dim, config.teacher_coarse_dim, 1, bias=False
        )

        self.coarse_stride = 8  # backbone coarse features are at 1/8 resolution

    def forward(self, data):
        """
        Args:
            data: dict with 'image0' (B, 1, H, W) and 'image1' (B, 1, H, W)
        Returns:
            dict with:
                'keypoints0': (N, 2) matched keypoints in image 0
                'keypoints1': (N, 2) matched keypoints in image 1
                'confidence': (N,)  matching confidence
                'conf_matrix': (B, H0*W0, H1*W1) coarse confidence matrix (for KD loss)
                'coarse_feat0': (B, C, H/8, W/8) coarse features (for feature KD loss)
                'coarse_feat1': (B, C, H/8, W/8)
        """
        img0, img1 = data['image0'], data['image1']

        # 1. Backbone: extract coarse + fine features
        coarse0, fine0 = self.backbone(img0)
        coarse1, fine1 = self.backbone(img1)

        # 2. Store pre-attention features (raw, for the training-loop projector)
        # Note: coarse_feat0_proj is set externally by the training loop via
        # project_student_feat_for_kd(), which uses the nn.Linear feature_projector.
        # The Conv2d feat_projector inside this model is kept for potential future use
        # but is NOT called here to avoid a dead computation branch.
        data['coarse_feat0'] = coarse0
        data['coarse_feat1'] = coarse1

        # 3. Feature interaction: shallow cross-attention
        coarse0, coarse1 = self.attention(coarse0, coarse1)

        # 4. Coarse matching
        conf_matrix, mkpts0_c, mkpts1_c, mconf = self.coarse_matching(coarse0, coarse1)
        data['conf_matrix'] = conf_matrix

        # 5. Fine refinement (only during inference with batch_size=1)
        device = img0.device
        if mkpts0_c is not None and len(mkpts0_c) > 0:
            mkpts0_f, mkpts1_f = self.fine_refinement(
                mkpts0_c, mkpts1_c, fine0, fine1, stride=self.coarse_stride
            )
        else:
            mkpts0_f = torch.zeros(0, 2, device=device)
            mkpts1_f = torch.zeros(0, 2, device=device)
            mconf = torch.zeros(0, device=device)

        data['keypoints0'] = mkpts0_f
        data['keypoints1'] = mkpts1_f
        data['confidence'] = mconf

        return data


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = StudentHybrid()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    attention_params = sum(p.numel() for p in model.attention.parameters())
    matching_params = sum(p.numel() for p in model.coarse_matching.parameters())
    fine_params = sum(p.numel() for p in model.fine_refinement.parameters())

    print(f"Total parameters:     {total_params / 1e6:.3f}M")
    print(f"  Backbone:           {backbone_params / 1e6:.3f}M")
    print(f"  Attention:          {attention_params / 1e6:.3f}M")
    print(f"  Coarse matching:    {matching_params / 1e6:.3f}M")
    print(f"  Fine refinement:    {fine_params / 1e6:.3f}M")

    dummy = {'image0': torch.rand(1, 1, 480, 480), 'image1': torch.rand(1, 1, 480, 480)}
    with torch.no_grad():
        out = model(dummy)

    print(f"\nForward pass OK")
    print(f"  conf_matrix:  {out['conf_matrix'].shape}")
    print(f"  coarse_feat0: {out['coarse_feat0'].shape}")
    print(f"  keypoints0:   {out['keypoints0'].shape}")
    print(f"  confidence:   {out['confidence'].shape}")