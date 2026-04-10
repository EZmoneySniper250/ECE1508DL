"""
Ablation Study 1: Confidence Threshold Sweep

Evaluates all 4 trained models across multiple confidence thresholds.
No retraining needed — just loads checkpoints and evaluates.

Usage:
    python ablation/threshold_sweep.py

Output:
    ablation/results/threshold_sweep.csv
    ablation/results/threshold_sweep.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import csv
import cv2
from pathlib import Path

from models.student_cnn import StudentCNN
from models.studentattention import StudentHybrid
from models.config import StudentCNNConfig, StudentHybridConfig
from dataset import SyntheticHomographyDataset


# ─── Config ──────────────────────────────────────────────────────────────────
THRESHOLDS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
HPATCHES_PATH = "hpatches-sequences-release"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUNS = {
    "Hybrid_KD+GT":  ("hybrid",  "checkpoints/run_full_hybrid/epoch_20.pth"),
    "CNN_KD+GT":     ("cnn",     "checkpoints/run_full_cnn/epoch_20.pth"),
    "Hybrid_GT":     ("hybrid",  "checkpoints/run_full_hybrid_gt/epoch_20.pth"),
    "CNN_GT":        ("cnn",     "checkpoints/run_full_cnn_gt/epoch_20.pth"),
}


# ─── Evaluation Helpers ──────────────────────────────────────────────────────

def compute_homography_auc(errors, thresholds=[3, 5, 10]):
    """Compute AUC of corner reprojection error at given pixel thresholds."""
    aucs = {}
    errors = np.array(errors)
    for t in thresholds:
        inliers = (errors < t).astype(float)
        aucs[f"AUC@{t}"] = inliers.mean() * 100
    return aucs


def compute_corner_error(mkpts0, mkpts1, H_gt, img_size=480):
    """Estimate homography from matches and compute corner error."""
    if len(mkpts0) < 4:
        return float("inf")

    pts0 = mkpts0.cpu().numpy().astype(np.float64)
    pts1 = mkpts1.cpu().numpy().astype(np.float64)

    H_pred, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    if H_pred is None:
        return float("inf")

    corners = np.array([
        [0, 0], [img_size, 0],
        [img_size, img_size], [0, img_size]
    ], dtype=np.float64)

    H_gt_np = H_gt.cpu().numpy().astype(np.float64)

    corners_h = np.hstack([corners, np.ones((4, 1))])
    proj_gt = (H_gt_np @ corners_h.T).T
    proj_gt = proj_gt[:, :2] / proj_gt[:, 2:3]
    proj_pred = (H_pred @ corners_h.T).T
    proj_pred = proj_pred[:, :2] / proj_pred[:, 2:3]

    error = np.linalg.norm(proj_gt - proj_pred, axis=1).mean()
    return error


def load_model(model_type, ckpt_path):
    """Load a student model from checkpoint."""
    if model_type == "hybrid":
        config = StudentHybridConfig()
        model = StudentHybrid(config)
    else:
        config = StudentCNNConfig()
        model = StudentCNN(config)

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.to(DEVICE).eval()
    return model


def evaluate_with_threshold(model, model_type, threshold, dataset, num_samples=100):
    """Evaluate a model at a specific confidence threshold."""
    # Patch the threshold
    if model_type == "hybrid":
        model.coarse_matching.threshold = threshold
    else:
        model.coarse_matching.match_threshold = threshold

    errors = []
    num_matches_list = []

    for i in range(min(num_samples, len(dataset))):
        img0, img1, H_gt = dataset[i]
        img0 = img0.unsqueeze(0).to(DEVICE)
        img1 = img1.unsqueeze(0).to(DEVICE)

        data = {"image0": img0, "image1": img1}

        with torch.no_grad():
            out = model(data)

        mkpts0 = out.get("keypoints0", torch.zeros(0, 2))
        mkpts1 = out.get("keypoints1", torch.zeros(0, 2))

        num_matches_list.append(len(mkpts0))
        error = compute_corner_error(mkpts0, mkpts1, H_gt)
        errors.append(error)

    aucs = compute_homography_auc(errors)
    avg_matches = np.mean(num_matches_list)

    return aucs, avg_matches


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ABLATION 1: Confidence Threshold Sweep")
    print("=" * 70)

    # Create output dir
    out_dir = Path("ablation/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (fixed seed for reproducibility)
    torch.manual_seed(42)
    np.random.seed(42)
    dataset = SyntheticHomographyDataset(HPATCHES_PATH, num_pairs=100, patch_size=480)

    # CSV header
    csv_path = out_dir / "threshold_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Threshold", "AUC@3", "AUC@5", "AUC@10", "Avg_Matches"])

    # Run evaluation
    for run_name, (model_type, ckpt_path) in RUNS.items():
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {run_name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'─' * 50}")
        print(f"Model: {run_name}")
        model = load_model(model_type, ckpt_path)

        for thr in THRESHOLDS:
            aucs, avg_matches = evaluate_with_threshold(
                model, model_type, thr, dataset, num_samples=100)
            print(f"  threshold={thr:.3f}  AUC@3={aucs['AUC@3']:.2f}  "
                  f"AUC@5={aucs['AUC@5']:.2f}  AUC@10={aucs['AUC@10']:.2f}  "
                  f"matches={avg_matches:.1f}")

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    run_name, thr,
                    f"{aucs['AUC@3']:.2f}",
                    f"{aucs['AUC@5']:.2f}",
                    f"{aucs['AUC@10']:.2f}",
                    f"{avg_matches:.1f}"
                ])

    print(f"\n✅ Results saved to {csv_path}")

    # ─── Plot ──────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv(csv_path)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for metric, ax in zip(["AUC@3", "AUC@5", "AUC@10"], axes):
            for model_name in df["Model"].unique():
                sub = df[df["Model"] == model_name]
                ax.plot(sub["Threshold"], sub[metric], marker="o", label=model_name)
            ax.set_xlabel("Confidence Threshold")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.set_xscale("log")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_dir / "threshold_sweep.png", dpi=150)
        print(f"✅ Plot saved to {out_dir / 'threshold_sweep.png'}")
    except ImportError:
        print("  (matplotlib/pandas not available, skipping plot)")


if __name__ == "__main__":
    main()
