"""
Local Feature Matching Visualization

Loads epoch-20 checkpoints for all four models (fullcnn, fullcnngt,
fullhybrid, fullhybridgt) and visualizes their matches on an HPatches image
pair using OpenCV.  Also runs LoFTR (via kornia) and saves its output as a
separate image.

Each run always produces TWO output images:
    1. matches_<scene>_<pair>_thr<threshold>.png   — 2×2 grid of student models
    2. loftr_<scene>_<pair>.png                    — LoFTR teacher output

Usage:
    python visualization/visualize_matching.py
    python visualization/visualize_matching.py --threshold 0.3
    python visualization/visualize_matching.py --threshold 0.1 --scene i_castle
    python visualization/visualize_matching.py --threshold 0.2 --scene i_books --img_pair 1 3
    python visualization/visualize_matching.py --save

Config:
    --threshold   Match confidence threshold for student models (default: 0.2)
    --scene       HPatches scene name (default: random)
    --img_pair    Two image indices, e.g. 1 2 (default: 1 2)
    --resize      Resize images to this height (default: 480, 0 = no resize)
    --max_matches Maximum number of matches to draw (default: 200)
    --save        Save output images instead of displaying them
    --out_dir     Directory to save outputs (default: visualization/outputs)
"""

import sys
import os
import argparse
import random

import cv2
import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import StudentCNN, StudentHybrid, StudentCNNConfig, StudentHybridConfig

# ── constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
HPATCHES_DIR   = os.path.join(ROOT, "hpatches-sequences-release")
EPOCH          = 20

MODELS_META = {
    "fullcnn":      ("run_full_cnn",       "cnn"),
    "fullcnngt":    ("run_full_cnn_gt",    "cnn"),
    "fullhybrid":   ("run_full_hybrid",    "hybrid"),
    "fullhybridgt": ("run_full_hybrid_gt", "hybrid"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(run_name: str, arch: str, threshold: float, device: torch.device):
    ckpt_path = os.path.join(CHECKPOINT_DIR, run_name, f"epoch_{EPOCH:02d}.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if arch == "cnn":
        cfg = StudentCNNConfig()
        cfg.match_threshold = threshold
        model = StudentCNN(cfg)
    else:
        cfg = StudentHybridConfig()
        cfg.match_threshold = threshold
        model = StudentHybrid(cfg)

    state = torch.load(ckpt_path, map_location=device)
    # checkpoints may wrap state_dict under a key
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess(img_bgr: np.ndarray, resize_h: int) -> torch.Tensor:
    """BGR uint8 → (1,1,H,W) float32 tensor in [0,1]."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if resize_h > 0:
        h, w = gray.shape
        new_w = int(w * resize_h / h)
        gray = cv2.resize(gray, (new_w, resize_h))
    t = torch.from_numpy(gray.astype(np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


@torch.no_grad()
def run_inference(model, img0_t: torch.Tensor, img1_t: torch.Tensor, device):
    data = {
        "image0": img0_t.to(device),
        "image1": img1_t.to(device),
    }
    out = model(data)
    kp0 = out["keypoints0"].cpu().numpy()   # (N,2)
    kp1 = out["keypoints1"].cpu().numpy()   # (N,2)
    conf = out["confidence"].cpu().numpy()  # (N,)
    return kp0, kp1, conf


def draw_matches(img0_bgr, img1_bgr, kp0, kp1, conf, max_matches=200, title=""):
    """
    Place img0 and img1 side by side and draw connecting lines for each match.
    Lines are coloured green→red by descending confidence.
    """
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    h_out = max(h0, h1)

    # Pad shorter image vertically
    canvas0 = np.zeros((h_out, w0, 3), dtype=np.uint8)
    canvas1 = np.zeros((h_out, w1, 3), dtype=np.uint8)
    canvas0[:h0, :w0] = img0_bgr
    canvas1[:h1, :w1] = img1_bgr

    canvas = np.concatenate([canvas0, canvas1], axis=1)

    if len(kp0) == 0:
        cv2.putText(canvas, "No matches", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        # Sort by confidence and keep top-N
        order = np.argsort(conf)[::-1][:max_matches]
        kp0_s = kp0[order]
        kp1_s = kp1[order]
        conf_s = conf[order]
        n = len(kp0_s)

        for i in range(n):
            # colour: high conf → green, low conf → red  (BGR)
            t = float(conf_s[i])           # already in [0,1] range
            t = max(0.0, min(1.0, t))
            color = (int((1 - t) * 255), int(t * 255), 0)

            x0, y0 = int(kp0_s[i, 0]), int(kp0_s[i, 1])
            x1, y1 = int(kp1_s[i, 0]) + w0, int(kp1_s[i, 1])

            cv2.circle(canvas, (x0, y0), 3, color, -1)
            cv2.circle(canvas, (x1, y1), 3, color, -1)
            cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

        cv2.putText(canvas, f"{n} matches",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Title bar
    bar = np.full((36, canvas.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 255), 2)
    return np.concatenate([bar, canvas], axis=0)


def make_grid(panels, ncols=2):
    """Tile a list of panels (same width) into an ncols-wide grid."""
    rows = []
    for i in range(0, len(panels), ncols):
        group = panels[i:i + ncols]
        # Pad to same height within each row
        max_h = max(p.shape[0] for p in group)
        padded = []
        for p in group:
            if p.shape[0] < max_h:
                pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
                p = np.concatenate([p, pad], axis=0)
            padded.append(p)
        # Pad row if fewer than ncols panels
        while len(padded) < ncols:
            padded.append(np.zeros_like(padded[0]))
        rows.append(np.concatenate(padded, axis=1))
    return np.concatenate(rows, axis=0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local feature matching visualizer")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Match confidence threshold (default: 0.2)")
    parser.add_argument("scene", type=str, nargs="?", default=None,
                        help="HPatches scene name, e.g. i_brooklyn (default: random)")
    parser.add_argument("--img_pair", type=int, nargs=2, default=[1, 2],
                        metavar=("IMG0", "IMG1"),
                        help="Image indices within the scene (default: 1 2)")
    parser.add_argument("--resize", type=int, default=480,
                        help="Resize images to this height in pixels (0 = no resize)")
    parser.add_argument("--max_matches", type=int, default=200,
                        help="Maximum number of matches to draw (default: 200)")
    parser.add_argument("--save", action="store_true",
                        help="Save output image instead of displaying it")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(ROOT, "visualization", "outputs"),
                        help="Directory to save outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Match threshold: {args.threshold}")

    # ── pick scene ────────────────────────────────────────────────────────────
    all_scenes = sorted([
        d for d in os.listdir(HPATCHES_DIR)
        if os.path.isdir(os.path.join(HPATCHES_DIR, d))
    ])
    if not all_scenes:
        raise FileNotFoundError(f"No scenes found in {HPATCHES_DIR}")

    if args.scene is None:
        scene = random.choice(all_scenes)
        print(f"Randomly selected scene: {scene}")
    else:
        scene = args.scene
        if scene not in all_scenes:
            raise ValueError(f"Scene '{scene}' not found. Available: {all_scenes}")
        print(f"Scene: {scene}")

    scene_dir = os.path.join(HPATCHES_DIR, scene)
    idx0, idx1 = args.img_pair
    img0_path = os.path.join(scene_dir, f"{idx0}.ppm")
    img1_path = os.path.join(scene_dir, f"{idx1}.ppm")

    if not os.path.isfile(img0_path) or not os.path.isfile(img1_path):
        raise FileNotFoundError(
            f"Image files not found: {img0_path}, {img1_path}"
        )

    img0_bgr = cv2.imread(img0_path)
    img1_bgr = cv2.imread(img1_path)
    print(f"Image 0: {img0_path}  shape={img0_bgr.shape}")
    print(f"Image 1: {img1_path}  shape={img1_bgr.shape}")

    img0_t = preprocess(img0_bgr, args.resize)
    img1_t = preprocess(img1_bgr, args.resize)

    # Resize BGR images for display to match tensor size
    if args.resize > 0:
        h0_t = img0_t.shape[2]
        w0_t = img0_t.shape[3]
        h1_t = img1_t.shape[2]
        w1_t = img1_t.shape[3]
        img0_disp = cv2.resize(img0_bgr, (w0_t, h0_t))
        img1_disp = cv2.resize(img1_bgr, (w1_t, h1_t))
    else:
        img0_disp = img0_bgr.copy()
        img1_disp = img1_bgr.copy()

    # ── LoFTR inference ───────────────────────────────────────────────────────
    print("\nLoading LoFTR (kornia, pretrained=outdoor) ...")
    loftr_kp0, loftr_kp1, loftr_conf = np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    loftr_ok = False
    try:
        from kornia.feature import LoFTR as KorniaLoFTR
        loftr = KorniaLoFTR(pretrained="outdoor").to(device).eval()
        with torch.no_grad():
            loftr_out = loftr({
                "image0": img0_t.to(device),
                "image1": img1_t.to(device),
            })
        loftr_kp0  = loftr_out["keypoints0"].cpu().numpy()
        loftr_kp1  = loftr_out["keypoints1"].cpu().numpy()
        loftr_conf = loftr_out["confidence"].cpu().numpy()
        print(f"  LoFTR matches found: {len(loftr_kp0)}")
        loftr_ok = True
    except Exception as e:
        print(f"  WARNING: LoFTR not available — {e}")
        print("           Install with: pip install kornia")

    loftr_title = (f"LoFTR (teacher) | scene={scene} | pair={idx0}-{idx1} "
                   f"| matches={len(loftr_kp0)}")
    loftr_panel = draw_matches(img0_disp, img1_disp, loftr_kp0, loftr_kp1,
                               loftr_conf, max_matches=args.max_matches,
                               title=loftr_title)

    # ── load student models & run inference ───────────────────────────────────
    panels = []
    for label, (run_name, arch) in MODELS_META.items():
        print(f"\nLoading {label} ({run_name}, epoch {EPOCH}) ...")
        try:
            model = load_model(run_name, arch, args.threshold, device)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            blank = np.zeros((100, img0_disp.shape[1] + img1_disp.shape[1], 3), dtype=np.uint8)
            cv2.putText(blank, f"{label}: checkpoint not found",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            panels.append(blank)
            continue

        kp0, kp1, conf = run_inference(model, img0_t, img1_t, device)
        print(f"  Matches found: {len(kp0)}")

        title = (f"{label} | scene={scene} | pair={idx0}-{idx1} "
                 f"| matches={len(kp0)}")
        panel = draw_matches(img0_disp, img1_disp, kp0, kp1, conf,
                             max_matches=args.max_matches, title=title)
        panels.append(panel)

    # ── make sure all panels have the same width ───────────────────────────────
    max_w = max(p.shape[1] for p in panels)
    for i, p in enumerate(panels):
        if p.shape[1] < max_w:
            pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
            panels[i] = np.concatenate([p, pad], axis=1)

    grid = make_grid(panels, ncols=2)

    # ── main title bar (threshold shown here) ─────────────────────────────────
    main_title = (f"Student Models  |  threshold={args.threshold}  |  "
                  f"scene={scene}  |  pair={idx0}-{idx1}")
    title_bar = np.full((48, grid.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(title_bar, main_title, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 255), 2)
    grid = np.concatenate([title_bar, grid], axis=0)

    # ── output ────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    students_fname = f"matches_{scene}_{idx0}-{idx1}_thr{args.threshold:.2f}.png"
    loftr_fname    = f"loftr_{scene}_{idx0}-{idx1}.png"
    students_path  = os.path.join(args.out_dir, students_fname)
    loftr_path     = os.path.join(args.out_dir, loftr_fname)

    # Always save both images
    cv2.imwrite(students_path, grid)
    cv2.imwrite(loftr_path, loftr_panel)
    print(f"\nSaved student models image : {students_path}")
    print(f"Saved LoFTR image          : {loftr_path}")

    if not args.save:
        screen_w = 1600

        # Student models window
        scale = min(1.0, screen_w / grid.shape[1])
        cv2.namedWindow("Student Models", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Student Models",
                         int(grid.shape[1] * scale), int(grid.shape[0] * scale))
        cv2.imshow("Student Models", grid)

        # LoFTR window
        scale_l = min(1.0, screen_w / loftr_panel.shape[1])
        cv2.namedWindow("LoFTR (teacher)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LoFTR (teacher)",
                         int(loftr_panel.shape[1] * scale_l),
                         int(loftr_panel.shape[0] * scale_l))
        cv2.imshow("LoFTR (teacher)", loftr_panel)

        print("\nPress any key to close the windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
