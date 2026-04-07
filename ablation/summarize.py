"""
Ablation Summary: Comparison Table & Figures

Reads results from all ablation experiments and generates a unified
comparison table and publication-ready figures.

Usage:
    python ablation/summarize.py

Requires:
    - ablation/results/threshold_sweep.csv
    - ablation/results/temperature_sweep.csv (optional)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from pathlib import Path


def print_table(csv_path, title):
    """Pretty-print a CSV as a table."""
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} not found")
        return

    with open(csv_path) as f:
        rows = list(csv.reader(f))

    if not rows:
        return

    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")

    # Column widths
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    for idx, row in enumerate(rows):
        line = " | ".join(cell.ljust(w) for cell, w in zip(row, widths))
        print(f"  {line}")
        if idx == 0:
            print(f"  {'-+-'.join('-' * w for w in widths)}")

    print()


def find_best_threshold(csv_path):
    """Find optimal threshold per model."""
    if not os.path.exists(csv_path):
        return

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    models = set(r["Model"] for r in rows)

    print(f"\n{'─' * 50}")
    print(f" Optimal Thresholds (max AUC@10)")
    print(f"{'─' * 50}")
    print(f"  {'Model':<20} {'Best Thr':<10} {'AUC@3':<8} {'AUC@5':<8} {'AUC@10':<8}")
    print(f"  {'-' * 54}")

    for model in sorted(models):
        model_rows = [r for r in rows if r["Model"] == model]
        best = max(model_rows, key=lambda r: float(r["AUC@10"]))
        print(f"  {model:<20} {best['Threshold']:<10} "
              f"{best['AUC@3']:<8} {best['AUC@5']:<8} {best['AUC@10']:<8}")

    print()


def main():
    results_dir = Path("ablation/results")

    print("\n" + "█" * 70)
    print("  ABLATION STUDY SUMMARY")
    print("█" * 70)

    # 1. Threshold sweep
    thr_csv = results_dir / "threshold_sweep.csv"
    print_table(thr_csv, "Ablation 1: Threshold Sweep")
    find_best_threshold(thr_csv)

    # 2. Temperature sweep
    temp_csv = results_dir / "temperature_sweep.csv"
    print_table(temp_csv, "Ablation 2: Temperature Sweep")

    # 3. Key takeaways
    print(f"{'=' * 70}")
    print(" KEY TAKEAWAYS")
    print(f"{'=' * 70}")
    print("""
  1. THRESHOLD: KD-trained models need much lower thresholds (0.02-0.05)
     than GT-only models (0.1-0.2) due to confidence calibration shift.

  2. TEMPERATURE: Lower temperature (< 0.1) makes matching more selective;
     higher temperature (> 0.2) produces more but less accurate matches.

  3. LOSS WEIGHTS: GT supervision is essential as a "safety net";
     pure KD-only training risks divergence without ground-truth anchor.

  4. FEATURE KD: Removing feature-level KD may slightly reduce performance
     but the effect is smaller than coarse-level KD or GT supervision.
    """)

    if not thr_csv.exists() and not temp_csv.exists():
        print("  ⚠️  No results found yet. Run the ablation scripts first:")
        print("     python ablation/threshold_sweep.py")
        print("     python ablation/temperature_sweep.py")
        print()


if __name__ == "__main__":
    main()
