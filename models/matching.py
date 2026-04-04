"""
Coarse Matching and Fine Refinement modules.

CoarseMatching:
    Computes dual-softmax confidence matrix from dot-product correlation,
    then extracts mutual nearest neighbor matches above a confidence threshold.

FineRefinement:
    For each coarse match, crops a local window from fine features,
    computes correlation, and predicts sub-pixel offset via spatial expectation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StudentCNNConfig


class CoarseMatching(nn.Module):
    """
    Dual-softmax coarse matching — same method as LoFTR.

    Given enhanced coarse features from both images:
        1. Flatten spatial dims → (B, N, C)
        2. Dot-product similarity → (B, N0, N1)
        3. Dual-softmax → confidence matrix
        4. Mutual nearest neighbor + threshold → coarse matches
    """

    def __init__(self, config: StudentCNNConfig = None):
        super().__init__()
        if config is None:
            config = StudentCNNConfig()

        self.temperature = config.temperature
        self.match_threshold = config.match_threshold
        self.border_remove = config.border_remove

    def forward(
        self, feat0: torch.Tensor, feat1: torch.Tensor, data: dict
    ) -> dict:
        """
        Args:
            feat0: (B, C, H0, W0) enhanced coarse features from image 0
            feat1: (B, C, H1, W1) enhanced coarse features from image 1
            data:  dict to store outputs

        Returns:
            data: updated dict with 'conf_matrix', 'sim_matrix', and
                  (during inference) 'mkpts0_c', 'mkpts1_c', 'mconf', 'b_ids'
        """
        B, C, H0, W0 = feat0.shape
        _, _, H1, W1 = feat1.shape

        # Flatten spatial dimensions: (B, C, H, W) → (B, N, C)
        feat0_flat = feat0.reshape(B, C, -1).permute(0, 2, 1)  # (B, N0, C)
        feat1_flat = feat1.reshape(B, C, -1).permute(0, 2, 1)  # (B, N1, C)

        # Dot-product similarity scaled by temperature
        sim_matrix = torch.einsum(
            "bmc,bnc->bmn", feat0_flat, feat1_flat
        ) / self.temperature  # (B, N0, N1)

        # Dual-softmax → confidence matrix
        conf_matrix = F.softmax(sim_matrix, dim=2) * F.softmax(sim_matrix, dim=1)

        data["conf_matrix"] = conf_matrix
        data["sim_matrix"] = sim_matrix
        data["hw0_c"] = (H0, W0)
        data["hw1_c"] = (H1, W1)

        # During inference, extract explicit matches
        if not self.training:
            self._extract_matches(conf_matrix, data, H0, W0, H1, W1)

        return data

    @torch.no_grad()
    def _extract_matches(
        self,
        conf_matrix: torch.Tensor,
        data: dict,
        H0: int, W0: int, H1: int, W1: int,
    ):
        """Extract mutual nearest neighbor matches from the confidence matrix."""
        B = conf_matrix.shape[0]
        device = conf_matrix.device

        # Best match for each position
        max0_val, max0_idx = conf_matrix.max(dim=2)  # (B, N0)
        max1_val, max1_idx = conf_matrix.max(dim=1)  # (B, N1)

        # Mutual nearest neighbor check
        arange = torch.arange(max0_idx.shape[1], device=device).unsqueeze(0).expand(B, -1)
        mutual_mask = max1_idx.gather(1, max0_idx) == arange

        # Confidence threshold
        conf_mask = max0_val > self.match_threshold

        # Border removal
        if self.border_remove > 0:
            br = self.border_remove
            ids = arange[0]
            x0 = ids % W0
            y0 = ids // W0
            border_mask_0 = (x0 >= br) & (x0 < W0 - br) & (y0 >= br) & (y0 < H0 - br)
            border_mask_0 = border_mask_0.unsqueeze(0).expand(B, -1)

            ids1 = max0_idx
            x1 = ids1 % W1
            y1 = ids1 // W1
            border_mask_1 = (x1 >= br) & (x1 < W1 - br) & (y1 >= br) & (y1 < H1 - br)
        else:
            border_mask_0 = torch.ones_like(mutual_mask)
            border_mask_1 = torch.ones_like(mutual_mask)

        mask = mutual_mask & conf_mask & border_mask_0 & border_mask_1

        # Collect matches per batch
        all_mkpts0_c = []
        all_mkpts1_c = []
        all_mconf = []
        all_b_ids = []

        for b in range(B):
            m = mask[b]
            ids0 = torch.where(m)[0]
            ids1 = max0_idx[b][ids0]

            # Flat index → (x, y) coordinates
            mkpts0_c = torch.stack([ids0 % W0, ids0 // W0], dim=1).float()
            mkpts1_c = torch.stack([ids1 % W1, ids1 // W1], dim=1).float()

            all_mkpts0_c.append(mkpts0_c)
            all_mkpts1_c.append(mkpts1_c)
            all_mconf.append(max0_val[b][ids0])
            all_b_ids.append(torch.full((len(ids0),), b, device=device, dtype=torch.long))

        data["mkpts0_c"] = torch.cat(all_mkpts0_c) if all_mkpts0_c else torch.zeros(0, 2, device=device)
        data["mkpts1_c"] = torch.cat(all_mkpts1_c) if all_mkpts1_c else torch.zeros(0, 2, device=device)
        data["mconf"] = torch.cat(all_mconf) if all_mconf else torch.zeros(0, device=device)
        data["b_ids"] = torch.cat(all_b_ids) if all_b_ids else torch.zeros(0, dtype=torch.long, device=device)


class FineRefinement(nn.Module):
    """
    Sub-pixel refinement of coarse matches using fine-level features.

    For each coarse match (p0, p1):
        1. Scale p0, p1 to fine feature resolution (×4, since coarse=1/8, fine=1/2)
        2. Extract center feature at p0 from fine_feat0
        3. Crop a W×W window centered at p1 from fine_feat1
        4. Correlation = dot product between center feature and all window positions
        5. Spatial softmax → expected position → sub-pixel offset
    """

    def __init__(self, config: StudentCNNConfig = None):
        super().__init__()
        if config is None:
            config = StudentCNNConfig()

        self.W = config.fine_window_size  # 5
        self.fine_dim = config.fine_dim    # 64

        # Small 1×1 conv to project fine features before correlation
        self.feat_proj = nn.Conv2d(config.fine_dim, config.fine_dim, 1, bias=False)

    def forward(
        self,
        fine_feat0: torch.Tensor,
        fine_feat1: torch.Tensor,
        data: dict,
    ) -> dict:
        """
        Args:
            fine_feat0: (B, C, Hf, Wf) fine features from image 0
            fine_feat1: (B, C, Hf, Wf) fine features from image 1
            data: dict containing coarse matches

        Returns:
            data: updated with 'mkpts0_f', 'mkpts1_f' at original resolution
        """
        mkpts0_c = data.get("mkpts0_c")
        mkpts1_c = data.get("mkpts1_c")
        b_ids = data.get("b_ids")

        device = fine_feat0.device

        if mkpts0_c is None or len(mkpts0_c) == 0:
            data["mkpts0_f"] = torch.zeros(0, 2, device=device)
            data["mkpts1_f"] = torch.zeros(0, 2, device=device)
            return data

        # Project fine features
        fine_feat0 = self.feat_proj(fine_feat0)
        fine_feat1 = self.feat_proj(fine_feat1)

        B, C, Hf, Wf = fine_feat0.shape
        r = self.W // 2

        # Scale coarse coords → fine coords (×4: coarse=1/8, fine=1/2)
        pts0_f = (mkpts0_c * 4.0).long()
        pts1_f = (mkpts1_c * 4.0).long()

        offsets = []
        for idx in range(len(mkpts0_c)):
            b = b_ids[idx].long().item()
            x0 = pts0_f[idx, 0].clamp(0, Wf - 1).item()
            y0 = pts0_f[idx, 1].clamp(0, Hf - 1).item()
            x1 = pts1_f[idx, 0].item()
            y1 = pts1_f[idx, 1].item()

            # Center feature from image 0
            query = fine_feat0[b, :, y0, x0]  # (C,)

            # Crop local window from image 1
            y_lo = max(0, y1 - r)
            y_hi = min(Hf, y1 + r + 1)
            x_lo = max(0, x1 - r)
            x_hi = min(Wf, x1 + r + 1)

            window = fine_feat1[b, :, y_lo:y_hi, x_lo:x_hi]  # (C, h, w)
            h, w = window.shape[1], window.shape[2]

            if h == 0 or w == 0:
                offsets.append(torch.zeros(2, device=device))
                continue

            # Correlation between query and window
            corr = torch.einsum("c,chw->hw", query, window)  # (h, w)

            # Spatial softmax → expected position
            prob = F.softmax(corr.reshape(-1), dim=0).reshape(h, w)

            grid_y, grid_x = torch.meshgrid(
                torch.arange(y_lo, y_hi, device=device, dtype=torch.float32),
                torch.arange(x_lo, x_hi, device=device, dtype=torch.float32),
                indexing="ij",
            )

            exp_x = (prob * grid_x).sum()
            exp_y = (prob * grid_y).sum()

            # Offset from the coarse match center (in fine coords)
            offsets.append(torch.stack([exp_x - float(x1), exp_y - float(y1)]))

        offsets = torch.stack(offsets, dim=0)  # (M, 2)

        # Convert to original resolution:
        #   coarse_coord * 8 → original resolution
        #   fine offset * 2  → original resolution (fine is 1/2)
        data["mkpts0_f"] = mkpts0_c * 8.0
        data["mkpts1_f"] = mkpts1_c * 8.0 + offsets * 2.0
        data["fine_offsets"] = offsets

        return data
