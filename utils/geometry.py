"""
Geometric utilities for handling ground-truth generations from homographies.
"""

import torch

def create_ground_truth_from_homography(H_gt, height, width, coarse_scale=8, match_threshold_pixels=4.0):
    """
    Coverts a ground-truth homography matrix H_gt into a dense confidence matrix.
    
    Args:
        H_gt: (B, 3, 3) ground truth homography mapping img0 coordinates to img1
        height, width: original image dimensions
        coarse_scale: downsampling factor of the feature map (default: 8)
        match_threshold_pixels: tolerance in original resolution pixels to consider a match valid
        
    Returns:
        gt_conf_matrix: (B, Hc * Wc, Hc * Wc) binary matrix of valid correspondence grid points.
                        1 if cell i in img0 matches to cell j in img1 optimally, else 0.
    """
    B = H_gt.shape[0]
    device = H_gt.device
    
    Hc, Wc = height // coarse_scale, width // coarse_scale
    
    # Generate centers of all coarse cells in img0 (original image coordinates)
    y0, x0 = torch.meshgrid(
        torch.arange(Hc, device=device).float() * coarse_scale + coarse_scale / 2,
        torch.arange(Wc, device=device).float() * coarse_scale + coarse_scale / 2,
        indexing='ij'
    )
    # Shape: (Hc, Wc)
    
    # Flatten and add homogeneous coordinate: (N, 3) where N = Hc * Wc
    pts0 = torch.stack([x0.flatten(), y0.flatten(), torch.ones_like(x0.flatten())], dim=1)
    N = pts0.shape[0]
    
    # Apply homography for all batches
    # H_gt: (B, 3, 3), pts0: (N, 3) -> transpose to (3, N) -> batch to (B, 3, N)
    pts0_b = pts0.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)
    pts1_proj = torch.bmm(H_gt, pts0_b) # (B, 3, N)
    
    # Normalize by z-coordinate
    z = pts1_proj[:, 2:3, :].clamp(min=1e-8)
    x1_proj = pts1_proj[:, 0, :] / z.squeeze(1) # (B, N)
    y1_proj = pts1_proj[:, 1, :] / z.squeeze(1) # (B, N)
    
    # Map back to coarse grid cell indices in img1
    col1 = (x1_proj / coarse_scale).floor().long()
    row1 = (y1_proj / coarse_scale).floor().long()
    
    # Validation mask: checks if projected point falls within img1 bounds
    valid_mask = (col1 >= 0) & (col1 < Wc) & (row1 >= 0) & (row1 < Hc) # (B, N)
    
    # Calculate projection distances to the center of the assigned target cell
    # The cell center in original pixels is:
    x1_center = col1.float() * coarse_scale + coarse_scale / 2
    y1_center = row1.float() * coarse_scale + coarse_scale / 2
    
    dist_sq = (x1_proj - x1_center)**2 + (y1_proj - y1_center)**2
    # Only keep matches that fall strongly (within radius) to the target cell's center
    dist_mask = dist_sq <= (match_threshold_pixels ** 2)
    
    valid_mask = valid_mask & dist_mask
    
    # Flatten index in img1 grid
    idx1 = row1 * Wc + col1 # (B, N)
    
    # Build sparse binary conf_matrix
    gt_conf_matrix = torch.zeros(B, N, N, device=device)
    
    for b in range(B):
        v = valid_mask[b]
        idx0_valid = torch.arange(N, device=device)[v]
        idx1_valid = idx1[b][v]
        
        # Set matching cells to 1.0
        gt_conf_matrix[b, idx0_valid, idx1_valid] = 1.0
        
    return gt_conf_matrix

def compute_ground_truth_fine_offsets(H_gt, mkpts0_c, b_ids, coarse_scale=8, fine_scale=4):
    """
    Computes sub-pixel offset labels for the fine matching stage given the exact Homography.
    """
    # This can be implemented when training fine-level loss without the teacher
    # returning zeros for now if required to not crash
    pass

