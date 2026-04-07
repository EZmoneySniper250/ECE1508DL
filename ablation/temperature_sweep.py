"""
Ablation Study 2: Temperature Sweep

Evaluates how the dual-softmax temperature affects matching performance.
No retraining needed — temperature is applied at inference time.

Usage:
    python ablation/temperature_sweep.py

Output:
    ablation/results/temperature_sweep.csv
    ablation/results/temperature_sweep.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import csv
from pathlib import Path

from models.student_cnn import StudentCNN
from models.studentattention import StudentHybrid
from models.config import StudentCNNConfig, StudentHybridConfig
from dataset import SyntheticHomographyDataset
from ablation.threshold_sweep import (
    compute_homography_auc, compute_corner_error, load_model, DEVICE
)


# ─── Config ──────────────────────────────────────────────────────────────────
TEMPERATURES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
HPATCHES_PATH = "hpatches-sequences-release"

RUNS = {
    "Hybrid_KD+GT":  ("hybrid",  "checkpoints/run_full_hybrid/epoch_20.pth"),
    "CNN_KD+GT":     ("cnn",     "checkpoints/run_full_cnn/epoch_20.pth"),
    "Hybrid_GT":     ("hybrid",  "checkpoints/run_full_hybrid_gt/epoch_20.pth"),
    "CNN_GT":        ("cnn",     "checkpoints/run_full_cnn_gt/epoch_20.pth"),
}

# Use optimal thresholds per model type based on previous experiments
THRESHOLDS = {
    "Hybrid_KD+GT": 0.02,
    "CNN_KD+GT": 0.02,
    "Hybrid_GT": 0.2,
    "CNN_GT": 0.2,
}


def evaluate_with_temperature(model, model_type, temperature, threshold, dataset, num_samples=580):
    """Evaluate model at a specific temperature and threshold."""
    # Patch temperature and threshold
    if model_type == "hybrid":
        model.coarse_matching.temperature = temperature
        model.coarse_matching.threshold = threshold
    else:
        model.coarse_matching.temperature = temperature
        model.coarse_matching.match_threshold = threshold

    errors = []
    for i in range(min(num_samples, len(dataset))):
        img0, img1, H_gt = dataset[i]
        img0 = img0.unsqueeze(0).to(DEVICE)
        img1 = img1.unsqueeze(0).to(DEVICE)

        data = {"image0": img0, "image1": img1}
        with torch.no_grad():
            out = model(data)

        mkpts0 = out.get("keypoints0", torch.zeros(0, 2))
        mkpts1 = out.get("keypoints1", torch.zeros(0, 2))
        error = compute_corner_error(mkpts0, mkpts1, H_gt)
        errors.append(error)

    return compute_homography_auc(errors)


def main():
    print("=" * 70)
    print("ABLATION 2: Temperature Sweep")
    print("=" * 70)

    out_dir = Path("ablation/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    dataset = SyntheticHomographyDataset(HPATCHES_PATH, num_pairs=580, patch_size=480)

    csv_path = out_dir / "temperature_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Temperature", "Threshold", "AUC@3", "AUC@5", "AUC@10"])

    for run_name, (model_type, ckpt_path) in RUNS.items():
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {run_name}: checkpoint not found")
            continue

        print(f"\n{'─' * 50}")
        print(f"Model: {run_name}")
        model = load_model(model_type, ckpt_path)
        thr = THRESHOLDS[run_name]

        for temp in TEMPERATURES:
            aucs = evaluate_with_temperature(
                model, model_type, temp, thr, dataset, num_samples=580
            )
            print(f"  temp={temp:.3f}  AUC@3={aucs['AUC@3']:.2f}  "
                  f"AUC@5={aucs['AUC@5']:.2f}  AUC@10={aucs['AUC@10']:.2f}")

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    run_name, temp, thr,
                    f"{aucs['AUC@3']:.2f}",
                    f"{aucs['AUC@5']:.2f}",
                    f"{aucs['AUC@10']:.2f}",
                ])

    print(f"\n✅ Results saved to {csv_path}")

    # Plot
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
                ax.plot(sub["Temperature"], sub[metric], marker="o", label=model_name)
            ax.set_xlabel("Temperature")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs Temperature")
            ax.set_xscale("log")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_dir / "temperature_sweep.png", dpi=150)
        print(f"✅ Plot saved to {out_dir / 'temperature_sweep.png'}")
    except ImportError:
        print("  (matplotlib/pandas not available, skipping plot)")


if __name__ == "__main__":
    main()
