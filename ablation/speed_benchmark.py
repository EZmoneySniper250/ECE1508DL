"""
Speed Benchmark: Student vs Teacher Inference Speed

Measures inference latency, throughput (FPS), and parameter counts
for LoFTR teacher vs. Student-Hybrid vs. Student-CNN.

Usage:
    python ablation/speed_benchmark.py

Output:
    ablation/results/speed_benchmark.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path

from models.student_cnn import StudentCNN
from models.studentattention import StudentHybrid
from models.config import StudentCNNConfig, StudentHybridConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 480
WARMUP = 20
NUM_RUNS = 100


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(model, img_size=480, num_runs=100, warmup=20, device="cuda"):
    """Measure average inference time in milliseconds."""
    model.eval().to(device)

    dummy_input = {
        "image0": torch.randn(1, 1, img_size, img_size, device=device),
        "image1": torch.randn(1, 1, img_size, img_size, device=device),
    }

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if isinstance(device, torch.device) and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_input)

            if isinstance(device, torch.device) and device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1000)  # ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "median_ms": np.median(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "fps": 1000 / np.mean(times),
    }


def measure_memory(model, img_size=480, device="cuda"):
    """Measure peak GPU memory usage during inference."""
    if not torch.cuda.is_available():
        return {"peak_mb": 0, "allocated_mb": 0}

    model.eval().to(device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    dummy_input = {
        "image0": torch.randn(1, 1, img_size, img_size, device=device),
        "image1": torch.randn(1, 1, img_size, img_size, device=device),
    }

    with torch.no_grad():
        _ = model(dummy_input)

    torch.cuda.synchronize()

    return {
        "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
    }


def try_load_loftr():
    """Try to load LoFTR teacher model for comparison."""
    try:
        from kornia.feature import LoFTR as KorniaLoFTR
        teacher = KorniaLoFTR(pretrained="outdoor")
        print("  LoFTR loaded from kornia")
        return teacher, "kornia"
    except Exception:
        pass

    # Try loading from local checkpoint
    loftr_paths = [
        "checkpoints/loftr_outdoor.ckpt",
        "checkpoints/loftr.pth",
        "../LoFTR/weights/outdoor_ds.ckpt",
    ]
    for p in loftr_paths:
        if os.path.exists(p):
            print(f"  Found LoFTR checkpoint at {p}")
            return None, p  # Return path, let caller handle loading

    print("  [WARN] LoFTR teacher not found. Skipping teacher benchmark.")
    print("         Install kornia: pip install kornia")
    print("         Or place checkpoint at checkpoints/loftr_outdoor.ckpt")
    return None, None


def main():
    print("=" * 70)
    print("SPEED BENCHMARK: Student vs Teacher")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Warmup: {WARMUP}, Timed runs: {NUM_RUNS}")
    print("=" * 70)

    out_dir = Path("ablation/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # ─── Student-Hybrid ──────────────────────────────────────────────────
    print("\n[1/3] Student-Hybrid (0.44M)")
    hybrid = StudentHybrid(StudentHybridConfig())
    total, trainable = count_parameters(hybrid)
    print(f"  Parameters: {total / 1e6:.3f}M (trainable: {trainable / 1e6:.3f}M)")

    timing = measure_inference_time(hybrid, IMG_SIZE, NUM_RUNS, WARMUP, DEVICE)
    mem = measure_memory(hybrid, IMG_SIZE, DEVICE)
    print(f"  Latency: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
    print(f"  FPS: {timing['fps']:.1f}")
    print(f"  Peak GPU Memory: {mem['peak_mb']:.1f} MB")

    results.append({
        "model": "Student-Hybrid",
        "params_M": total / 1e6,
        **timing,
        **mem,
    })

    # ─── Student-CNN ─────────────────────────────────────────────────────
    print("\n[2/3] Student-CNN (0.66M)")
    cnn = StudentCNN(StudentCNNConfig())
    total, trainable = count_parameters(cnn)
    print(f"  Parameters: {total / 1e6:.3f}M (trainable: {trainable / 1e6:.3f}M)")

    timing = measure_inference_time(cnn, IMG_SIZE, NUM_RUNS, WARMUP, DEVICE)
    mem = measure_memory(cnn, IMG_SIZE, DEVICE)
    print(f"  Latency: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
    print(f"  FPS: {timing['fps']:.1f}")
    print(f"  Peak GPU Memory: {mem['peak_mb']:.1f} MB")

    results.append({
        "model": "Student-CNN",
        "params_M": total / 1e6,
        **timing,
        **mem,
    })

    # ─── LoFTR Teacher ───────────────────────────────────────────────────
    print("\n[3/3] LoFTR Teacher (~12M)")
    teacher, source = try_load_loftr()

    if teacher is not None:
        total, trainable = count_parameters(teacher)
        print(f"  Parameters: {total / 1e6:.3f}M")

        # LoFTR from kornia has different input format
        if source == "kornia":
            # Measure manually for kornia LoFTR
            teacher.eval().to(DEVICE)
            dummy = {
                "image0": torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=DEVICE),
                "image1": torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=DEVICE),
            }

            # Warmup
            with torch.no_grad():
                for _ in range(WARMUP):
                    _ = teacher(dummy)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            times = []
            with torch.no_grad():
                for _ in range(NUM_RUNS):
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = teacher(dummy)
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000)

            timing = {
                "mean_ms": np.mean(times),
                "std_ms": np.std(times),
                "median_ms": np.median(times),
                "fps": 1000 / np.mean(times),
            }
            mem = measure_memory(teacher, IMG_SIZE, DEVICE)
        else:
            timing = measure_inference_time(teacher, IMG_SIZE, NUM_RUNS, WARMUP, DEVICE)
            mem = measure_memory(teacher, IMG_SIZE, DEVICE)

        print(f"  Latency: {timing['mean_ms']:.2f} ± {timing.get('std_ms', 0):.2f} ms")
        print(f"  FPS: {timing['fps']:.1f}")
        print(f"  Peak GPU Memory: {mem['peak_mb']:.1f} MB")

        results.append({
            "model": "LoFTR Teacher",
            "params_M": total / 1e6,
            **timing,
            **mem,
        })
    else:
        # Use known LoFTR stats as reference
        print("  Using published reference numbers:")
        print("  Parameters: ~12M")
        print("  Latency: ~116ms (RTX 2080 Ti) — will be faster on H100")
        results.append({
            "model": "LoFTR Teacher (reference)",
            "params_M": 12.0,
            "mean_ms": 116.0,
            "fps": 8.6,
            "peak_mb": 0,
        })

    # ─── Summary Table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Params':<10} {'Latency(ms)':<15} {'FPS':<10} {'GPU Mem(MB)':<12}")
    print("-" * 70)

    baseline_ms = results[-1].get("mean_ms", 1)
    for r in results:
        speedup = baseline_ms / r.get("mean_ms", 1)
        print(f"{r['model']:<25} {r['params_M']:.2f}M     "
              f"{r.get('mean_ms', 0):.2f}            "
              f"{r.get('fps', 0):.1f}       "
              f"{r.get('peak_mb', 0):.0f}")

    # Speedup comparison
    if len(results) >= 2:
        print(f"\n{'─' * 50}")
        print("Speedup vs Teacher:")
        for r in results[:-1]:
            speedup = baseline_ms / r.get("mean_ms", 1)
            param_ratio = results[-1]["params_M"] / r["params_M"]
            print(f"  {r['model']}: {speedup:.1f}x faster, {param_ratio:.0f}x fewer params")

    # Save to file
    txt_path = out_dir / "speed_benchmark.txt"
    with open(txt_path, "w") as f:
        f.write(f"Speed Benchmark Results\n")
        f.write(f"Device: {DEVICE}")
        if torch.cuda.is_available():
            f.write(f" ({torch.cuda.get_device_name()})")
        f.write(f"\nImage: {IMG_SIZE}x{IMG_SIZE}, Runs: {NUM_RUNS}\n\n")
        f.write(f"{'Model':<25} {'Params':<10} {'Latency(ms)':<15} {'FPS':<10} {'Mem(MB)':<10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['model']:<25} {r['params_M']:.2f}M     "
                    f"{r.get('mean_ms', 0):.2f}            "
                    f"{r.get('fps', 0):.1f}       "
                    f"{r.get('peak_mb', 0):.0f}\n")

    print(f"\n✅ Results saved to {txt_path}")


if __name__ == "__main__":
    main()
