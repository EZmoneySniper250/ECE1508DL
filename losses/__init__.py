"""
Multi-level distillation losses for knowledge distillation from LoFTR.

Four loss components:
    1. L_coarse_kd:  KL divergence on coarse confidence matrix
    2. L_feat_kd:    MSE on projected intermediate features
    3. L_fine_kd:    L1 on fine-level sub-pixel coordinates
    4. L_gt:         Ground-truth supervision (focal loss + L2)

Reference: Table 1 in the project report.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import StudentCNNConfig


class CoarseKDLoss(nn.Module):
    """KL divergence between student and teacher coarse confidence matrices."""

    def forward(
        self,
        student_conf: torch.Tensor,
        teacher_conf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_conf: (B, N0, N1) student's dual-softmax confidence matrix
            teacher_conf: (B, N0, N1) teacher's dual-softmax confidence matrix

        Returns:
            kl_loss: scalar KL divergence loss
        """
        # Clamp to avoid log(0)
        student_conf = student_conf.clamp(min=1e-8)
        teacher_conf = teacher_conf.clamp(min=1e-8)

        # Flatten spatial dims for KL computation
        B = student_conf.shape[0]
        s = student_conf.reshape(B, -1)
        t = teacher_conf.reshape(B, -1)

        # Normalize to valid distributions
        s = s / s.sum(dim=1, keepdim=True)
        t = t / t.sum(dim=1, keepdim=True)

        # KL(teacher || student) = sum(t * log(t/s))
        kl = F.kl_div(s.log(), t, reduction="batchmean")

        return kl


class FeatureKDLoss(nn.Module):
    """MSE between projected student features and teacher features."""

    def forward(
        self,
        student_feat_proj: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_feat_proj: (B, D_t, H, W) student features projected to teacher dim
            teacher_feat:      (B, D_t, H, W) teacher features

        Returns:
            mse_loss: scalar MSE loss
        """
        return F.mse_loss(student_feat_proj, teacher_feat)


class FineKDLoss(nn.Module):
    """L1 distance between student and teacher fine-level coordinates."""

    def forward(
        self,
        student_coords: torch.Tensor,
        teacher_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_coords: (M, 2) student's sub-pixel refined coordinates
            teacher_coords: (M, 2) teacher's sub-pixel refined coordinates

        Returns:
            l1_loss: scalar L1 loss
        """
        if len(student_coords) == 0 or len(teacher_coords) == 0:
            return torch.tensor(0.0, device=student_coords.device)

        return F.l1_loss(student_coords, teacher_coords)


class GTSupervisionLoss(nn.Module):
    """
    Ground-truth supervision loss (LoFTR's original loss).

    Components:
        - Focal loss on coarse matching confidence matrix
        - L2 on fine-level coordinate predictions

    This prevents the student from blindly replicating the teacher's errors.
    """

    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        conf_matrix: torch.Tensor,
        gt_conf_matrix: torch.Tensor,
        fine_coords: torch.Tensor = None,
        gt_fine_coords: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            conf_matrix:    (B, N0, N1) student's confidence matrix
            gt_conf_matrix: (B, N0, N1) ground-truth confidence matrix from correspondences
            fine_coords:    (M, 2) student's fine coords (optional)
            gt_fine_coords: (M, 2) ground-truth fine coords (optional)

        Returns:
            loss: scalar combined loss
        """
        # Focal loss on coarse matching
        pos_mask = gt_conf_matrix > 0.5
        neg_mask = ~pos_mask

        pos_conf = conf_matrix[pos_mask].clamp(min=1e-8)
        neg_conf = conf_matrix[neg_mask].clamp(min=1e-8)

        # Positive: focal loss encourages high confidence for correct matches
        alpha = self.focal_alpha
        gamma = self.focal_gamma

        pos_loss = -alpha * ((1 - pos_conf) ** gamma) * pos_conf.log()
        neg_loss = -(1 - alpha) * (neg_conf ** gamma) * (1 - neg_conf).clamp(min=1e-8).log()

        focal_loss = pos_loss.mean() + neg_loss.mean() if len(pos_loss) > 0 else neg_loss.mean()

        # L2 on fine coordinates
        fine_loss = torch.tensor(0.0, device=conf_matrix.device)
        if fine_coords is not None and gt_fine_coords is not None and len(fine_coords) > 0:
            fine_loss = F.mse_loss(fine_coords, gt_fine_coords)

        return focal_loss + fine_loss


class DistillationLoss(nn.Module):
    """
    Combined multi-level distillation loss.

    L_total = λ₁ · L_coarse_kd + λ₂ · L_feat_kd + λ₃ · L_fine_kd + λ₄ · L_gt

    Default weights from config:
        λ₁ = 1.0  (coarse KD)
        λ₂ = 0.5  (feature KD)
        λ₃ = 0.5  (fine KD)
        λ₄ = 1.0  (GT supervision)
    """

    def __init__(self, config: StudentCNNConfig = None):
        super().__init__()
        if config is None:
            config = StudentCNNConfig()

        self.lambda_coarse_kd = config.lambda_coarse_kd
        self.lambda_feat_kd = config.lambda_feat_kd
        self.lambda_fine_kd = config.lambda_fine_kd
        self.lambda_gt = config.lambda_gt

        self.coarse_kd = CoarseKDLoss()
        self.feat_kd = FeatureKDLoss()
        self.fine_kd = FineKDLoss()
        self.gt_sup = GTSupervisionLoss()

    def forward(
        self,
        student_data: dict,
        teacher_data: dict,
        gt_data: dict = None,
    ) -> dict:
        """
        Args:
            student_data: dict from student model forward pass
            teacher_data: dict from teacher model forward pass
            gt_data:      dict with GT correspondences (optional)

        Returns:
            losses: dict with individual losses and total loss
        """
        losses = {}

        # 1. Coarse-level KD: KL(student_conf || teacher_conf)
        l_coarse = self.coarse_kd(
            student_data["conf_matrix"],
            teacher_data["conf_matrix"],
        )
        losses["coarse_kd"] = l_coarse

        # 2. Feature-level KD: MSE(projector(student_feat), teacher_feat)
        l_feat0 = self.feat_kd(
            student_data["student_coarse_feat0_proj"],
            teacher_data["teacher_coarse_feat0"],
        )
        l_feat1 = self.feat_kd(
            student_data["student_coarse_feat1_proj"],
            teacher_data["teacher_coarse_feat1"],
        )
        l_feat = (l_feat0 + l_feat1) / 2
        losses["feat_kd"] = l_feat

        # 3. Fine-level KD: L1(student_fine_coords, teacher_fine_coords)
        l_fine = torch.tensor(0.0, device=l_coarse.device)
        if "fine_offsets" in student_data and "fine_offsets" in teacher_data:
            l_fine = self.fine_kd(
                student_data["fine_offsets"],
                teacher_data["fine_offsets"],
            )
        losses["fine_kd"] = l_fine

        # 4. GT supervision (if GT correspondences provided)
        l_gt = torch.tensor(0.0, device=l_coarse.device)
        if gt_data is not None and "gt_conf_matrix" in gt_data:
            l_gt = self.gt_sup(
                student_data["conf_matrix"],
                gt_data["gt_conf_matrix"],
                student_data.get("fine_offsets"),
                gt_data.get("gt_fine_coords"),
            )
        losses["gt_sup"] = l_gt

        # Total weighted loss
        total = (
            self.lambda_coarse_kd * l_coarse
            + self.lambda_feat_kd * l_feat
            + self.lambda_fine_kd * l_fine
            + self.lambda_gt * l_gt
        )
        losses["total"] = total

        return losses
