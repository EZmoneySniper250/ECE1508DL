"""
Configuration for Student models.

StudentCNNConfig    — Student-CNN (dilated conv interaction)
StudentHybridConfig — Student-Hybrid (shallow cross-attention interaction)
"""


class StudentHybridConfig:
    """Default configuration for Student-Hybrid."""

    # ── Teacher dimensions (for distillation projector) ──────────────────────
    teacher_coarse_dim = 256   # LoFTR coarse feature channels
    teacher_fine_dim = 128     # LoFTR fine feature channels

    # ── Student backbone ─────────────────────────────────────────────────────
    coarse_dim = 128           # 1/8 resolution coarse feature channels
    fine_dim = 32              # 1/2 resolution fine feature channels

    # ── Shallow cross-attention ───────────────────────────────────────────────
    nhead = 4                  # number of attention heads
    n_rounds = 2               # number of self+cross attention rounds

    # ── Coarse matching ──────────────────────────────────────────────────────
    temperature = 0.1          # dual-softmax temperature
    match_threshold = 0.2      # confidence threshold for match extraction

    # ── Fine refinement ──────────────────────────────────────────────────────
    fine_window_size = 5       # local window size for sub-pixel refinement

    # ── Training ─────────────────────────────────────────────────────────────
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 30

    # ── Distillation loss weights ────────────────────────────────────────────
    lambda_coarse_kd = 1.0     # KL divergence on coarse confidence matrix
    lambda_feat_kd = 0.5       # MSE on projected features
    lambda_fine_kd = 0.5       # L1 on fine-level coordinates
    lambda_gt = 1.0            # ground-truth supervision loss


class StudentCNNConfig:
    """Default configuration for Student-CNN."""

    # ── Teacher dimensions (for distillation projector) ──────────────────────
    teacher_coarse_dim = 256   # LoFTR coarse feature channels
    teacher_fine_dim = 128     # LoFTR fine feature channels

    # ── Student backbone ─────────────────────────────────────────────────────
    coarse_dim = 128           # 1/8 resolution coarse feature channels
    fine_dim = 32              # 1/2 resolution fine feature channels

    # ── Dilated interaction module ───────────────────────────────────────────
    dilated_channels = 128     # same as coarse_dim
    dilation_rates = [1, 2, 4, 8]  # progressively larger receptive field

    # ── Coarse matching ──────────────────────────────────────────────────────
    temperature = 0.1          # dual-softmax temperature
    match_threshold = 0.2      # confidence threshold for match extraction
    border_remove = 2          # remove matches too close to image border

    # ── Fine refinement ──────────────────────────────────────────────────────
    fine_window_size = 5       # local window size for sub-pixel refinement

    # ── Training ─────────────────────────────────────────────────────────────
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 30

    # ── Distillation loss weights ────────────────────────────────────────────
    lambda_coarse_kd = 1.0     # KL divergence on coarse confidence matrix
    lambda_feat_kd = 0.5       # MSE on projected features
    lambda_fine_kd = 0.5       # L1 on fine-level coordinates
    lambda_gt = 1.0            # ground-truth supervision loss
