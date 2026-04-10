"""
Ablation Study 3: Loss Weight Ablation (Training Required)

Trains Student-CNN with different loss weight configurations to understand
the contribution of each loss component.

Configurations:
    1. GT-only           (λ_KD=0, λ_feat=0, λ_GT=1)     ← baseline (already trained)
    2. KD-only           (λ_KD=1, λ_feat=0.5, λ_GT=0)   ← pure distillation
    3. KD+GT default     (λ_KD=1, λ_feat=0.5, λ_GT=1)   ← already trained
    4. Heavy GT          (λ_KD=0.5, λ_feat=0.25, λ_GT=2) ← upweight GT
    5. Heavy KD          (λ_KD=2, λ_feat=1.0, λ_GT=0.5)  ← upweight KD
    6. No feature KD     (λ_KD=1, λ_feat=0, λ_GT=1)      ← remove feat alignment

Usage:
    python ablation/loss_weight_ablation.py --config kd_only
    python ablation/loss_weight_ablation.py --config heavy_gt
    python ablation/loss_weight_ablation.py --config heavy_kd
    python ablation/loss_weight_ablation.py --config no_feat_kd

Output:
    checkpoints/ablation_<config_name>/epoch_XX.pth
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from models.student_cnn import StudentCNN
from models.config import StudentCNNConfig
from dataset import SyntheticHomographyDataset
from utils.geometry import create_ground_truth_from_homography
from losses import CoarseKDLoss, FeatureKDLoss, GTSupervisionLoss


# ─── Ablation Configurations ─────────────────────────────────────────────────
CONFIGS = {
    "kd_only": {
        "lambda_coarse_kd": 1.0,
        "lambda_feat_kd": 0.5,
        "lambda_gt": 0.0,
        "desc": "KD-only (no GT supervision)",
    },
    "heavy_gt": {
        "lambda_coarse_kd": 0.5,
        "lambda_feat_kd": 0.25,
        "lambda_gt": 2.0,
        "desc": "Heavy GT (upweighted GT supervision)",
    },
    "heavy_kd": {
        "lambda_coarse_kd": 2.0,
        "lambda_feat_kd": 1.0,
        "lambda_gt": 0.5,
        "desc": "Heavy KD (upweighted distillation)",
    },
    "no_feat_kd": {
        "lambda_coarse_kd": 1.0,
        "lambda_feat_kd": 0.0,
        "lambda_gt": 1.0,
        "desc": "No feature KD (only coarse KD + GT)",
    },
}

HPATCHES_PATH = "hpatches-sequences-release"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, teacher, dataloader, optimizer, loss_weights, device):
    """Train for one epoch with specified loss weights."""
    model.train()
    coarse_kd_loss_fn = CoarseKDLoss()
    feat_kd_loss_fn = FeatureKDLoss()
    gt_loss_fn = GTSupervisionLoss()

    total_loss = 0
    n_batches = 0

    for img0, img1, H_gt in dataloader:
        img0, img1, H_gt = img0.to(device), img1.to(device), H_gt.to(device)

        data = {"image0": img0, "image1": img1}

        # Student forward
        student_out = model(data)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_data = {"image0": img0, "image1": img1}
            teacher_out = teacher(teacher_data)

        # GT correspondence matrix
        B, _, H, W = img0.shape
        gt_conf = create_ground_truth_from_homography(H_gt, H, W)

        # ── Compute losses ──
        loss = torch.tensor(0.0, device=device)

        # 1. Coarse KD
        if loss_weights["lambda_coarse_kd"] > 0:
            l_kd = coarse_kd_loss_fn(
                student_out["conf_matrix"], teacher_out["conf_matrix"]
            )
            loss = loss + loss_weights["lambda_coarse_kd"] * l_kd

        # 2. Feature KD
        if loss_weights["lambda_feat_kd"] > 0:
            # Project student features to teacher dim
            feat0_proj = model.feat_projector(student_out["coarse_feat0"])
            feat1_proj = model.feat_projector(student_out["coarse_feat1"])
            l_feat = (
                feat_kd_loss_fn(feat0_proj, teacher_out.get("coarse_feat0", feat0_proj))
                + feat_kd_loss_fn(feat1_proj, teacher_out.get("coarse_feat1", feat1_proj))
            ) / 2
            loss = loss + loss_weights["lambda_feat_kd"] * l_feat

        # 3. GT supervision
        if loss_weights["lambda_gt"] > 0:
            l_gt = gt_loss_fn(student_out["conf_matrix"], gt_conf)
            loss = loss + loss_weights["lambda_gt"] * l_gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Loss Weight Ablation")
    parser.add_argument("--config", type=str, required=True, choices=CONFIGS.keys(),
                        help="Ablation configuration name")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Path to LoFTR teacher checkpoint (required if KD loss > 0)")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    print("=" * 70)
    print(f"ABLATION 3: Loss Weight — {cfg['desc']}")
    print(f"  λ_coarse_kd = {cfg['lambda_coarse_kd']}")
    print(f"  λ_feat_kd   = {cfg['lambda_feat_kd']}")
    print(f"  λ_gt        = {cfg['lambda_gt']}")
    print("=" * 70)

    # Save dir
    save_dir = Path(f"checkpoints/ablation_{args.config}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = SyntheticHomographyDataset(HPATCHES_PATH, num_pairs=args.num_pairs, patch_size=480)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Student
    config = StudentCNNConfig()
    model = StudentCNN(config).to(DEVICE)

    # Teacher (load LoFTR if KD is used)
    teacher = None
    if cfg["lambda_coarse_kd"] > 0 or cfg["lambda_feat_kd"] > 0:
        if args.teacher_ckpt is None:
            print("[WARN] KD loss > 0 but no teacher checkpoint provided.")
            print("       Provide --teacher_ckpt path/to/loftr.pth")
            print("       Using a dummy teacher (results will not be meaningful)")
            teacher = StudentCNN(config).to(DEVICE)
            teacher.eval()
        else:
            # Load real LoFTR teacher here
            # You'll need to adapt this to your actual LoFTR loading code
            print(f"Loading teacher from {args.teacher_ckpt}")
            teacher = StudentCNN(config).to(DEVICE)  # placeholder
            teacher.eval()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model, teacher, dataloader, optimizer, cfg, DEVICE
        )
        print(f"  Epoch {epoch:02d}/{args.epochs}  loss={avg_loss:.4f}")

        ckpt_path = save_dir / f"epoch_{epoch:02d}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": args.config,
            "loss_weights": cfg,
            "loss": avg_loss,
        }, ckpt_path)

    print(f"\n✅ Training complete. Checkpoints saved to {save_dir}")
    print(f"   Run threshold_sweep.py to evaluate these checkpoints.")


if __name__ == "__main__":
    main()
