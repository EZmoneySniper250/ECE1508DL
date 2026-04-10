# Distilling LoFTR into Compact Matching Networks

ECE 1508 Applied Deep Learning — Winter 2026, University of Toronto

## Overview

This project distills LoFTR (a Transformer-based dense feature matcher) into two compact student networks and evaluates them on the HPatches homography benchmark. The two students share the same MobileNetV3-Small backbone and matching heads but differ in feature interaction:

- **Student-Hybrid**: shallow self/cross-attention (0.44M params)
- **Student-CNN**: dilated convolutions + SE channel attention (0.66M params)

Training uses multi-level knowledge distillation (coarse KL + feature MSE + GT focal loss) from a frozen LoFTR teacher.

## Setup

```bash
pip install -r requirements.txt
```

Download the HPatches dataset and place it so the path matches `hpatches-sequences-release/` under this directory (excluded via `.gitignore`).

## Repository Structure

```
main/
├── models/              # Student architectures and shared modules
│   ├── backbone.py      # MobileNetV3-Small backbone
│   ├── studentattention.py  # Student-Hybrid
│   ├── student_cnn.py   # Student-CNN
│   ├── dilated_interaction.py
│   ├── matching.py      # Coarse matching + fine refinement
│   └── config.py        # Hyperparameter configs
├── losses/              # Multi-level distillation losses
├── utils/               # Geometry utilities (GT from homography)
├── dataset.py           # Synthetic homography pair generator
├── notebooks/
│   ├── training.ipynb           # KD training pipeline
│   ├── evaluation.ipynb         # HPatches evaluation
│   └── teacher_baseline.ipynb   # LoFTR baseline reproduction
├── ablation/            # Threshold, temperature, speed sweeps
├── checkpoints/         # Saved model weights
├── visualization/       # Qualitative matching visualization
└── diagrams/            # Architecture and pipeline figures
```

## Training

Open `notebooks/training.ipynb` and run all cells. The notebook loads a frozen LoFTR teacher via kornia, generates synthetic pairs from HPatches, and trains the student with the combined KD+GT loss. Checkpoints are saved to `checkpoints/`.

## Evaluation

Open `notebooks/evaluation.ipynb`. It loads trained checkpoints, runs inference on all 580 HPatches pairs, estimates homographies via OpenCV RANSAC, and reports AUC@3/5/10, matching precision, and inlier ratio.

## Ablation

```bash
python -m ablation.threshold_sweep
python -m ablation.temperature_sweep
python -m ablation.speed_benchmark
```

Results are saved to `ablation/results/`.

## Authors

Jerry Chen, Spiro Li, Weijie Zhu
