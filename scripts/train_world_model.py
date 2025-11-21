#!/usr/bin/env python3
"""
OmniVec2 World Model: Future LiDAR Prediction

Given current image + LiDAR at time t,
predict the LiDAR point cloud at time t+1.

This demonstrates the world model's ability to:
1. Understand scene dynamics from multimodal fusion
2. Predict future states (core of world modeling)
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer
from src.models.omnivec2_core import CrossModalBlock, OmniVec2Tiny
from src.data.nuscenes_loader import NuScenesMiniIter


class FutureLiDARDecoder(nn.Module):
    """
    Decoder: Fused embedding → Future LiDAR point cloud
    
    Architecture:
    1. Take fused tokens [B, 596, 96]
    2. Pool to scene-level embedding [B, 96]
    3. MLP to predict future LiDAR parameters
    4. Generate predicted point cloud
    """
    def __init__(self, embed_dim=96, num_pred_points=512):
        super().__init__()
        self.num_pred_points = num_pred_points
        
        # MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, num_pred_points * 4),  # Predict N points × 4 channels
        )
    
    def forward(self, fused):
        """
        fused: [B, 596, 96]
        returns: [B, num_pred_points, 4]  (predicted x, y, z, intensity)
        """
        # Pool fused tokens
        z_scene = fused.mean(dim=1)  # [B, 96]
        
        # Decode to point cloud
        pred_flat = self.decoder(z_scene)  # [B, num_pred_points * 4]
        pred_points = pred_flat.view(-1, self.num_pred_points, 4)  # [B, N, 4]
        
        return pred_points


def chamfer_distance(pred_points, gt_points):
    """
    Chamfer Distance: measures similarity between two point clouds.
    
    For each predicted point, find nearest GT point (forward).
    For each GT point, find nearest predicted point (backward).
    Average both directions.
    
    pred_points: [B, N_pred, 3]  (x, y, z only, ignore intensity)
    gt_points:   [B, N_gt, 3]
    """
    B = pred_points.shape[0]
    
    # Pairwise distances: [B, N_pred, N_gt]
    pred_expanded = pred_points.unsqueeze(2)  # [B, N_pred, 1, 3]
    gt_expanded = gt_points.unsqueeze(1)      # [B, 1, N_gt, 3]
    dists = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)  # [B, N_pred, N_gt]
    
    # Forward: for each pred point, find nearest GT
    min_dist_pred_to_gt, _ = torch.min(dists, dim=2)  # [B, N_pred]
    loss_forward = min_dist_pred_to_gt.mean()
    
    # Backward: for each GT point, find nearest pred
    min_dist_gt_to_pred, _ = torch.min(dists, dim=1)  # [B, N_gt]
    loss_backward = min_dist_gt_to_pred.mean()
    
    return (loss_forward + loss_backward) / 2


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    cfg_path = os.path.join("config", "dataset.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    nus_cfg = cfg.get("nuscenes", {})
    root = nus_cfg.get("dataroot")
    img_size_cfg = nus_cfg.get("image_size", [224, 384])
    H, W = img_size_cfg[-2], img_size_cfg[-1]
    
    print(f"Dataroot: {root}")
    print(f"Image size: H={H}, W={W}")
    
    # Model components
    embed_dim = 96
    num_pred_points = 512  # Predict 512 points for future LiDAR
    
    # Load pre-trained encoder (from InfoNCE checkpoint)
    print("\nLoading pre-trained encoder from InfoNCE training...")
    ckpt_path = "checkpoints/omnivec2_hybrid_96D.pt"
    
    img_tok = ImagePatchTokenizer(embed_dim=embed_dim, patch=32).to(device)
    lid_tok = PointTokenTokenizer(in_ch=4, embed_dim=embed_dim, num_tokens=512).to(device)
    cross_modal = CrossModalBlock(dim=embed_dim, heads=3, ff=192).to(device)
    backbone = OmniVec2Tiny(dim=embed_dim, heads=3, ff=192, depth=1).to(device)
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        img_tok.load_state_dict(ckpt['img_tok'])
        lid_tok.load_state_dict(ckpt['lid_tok'])
        cross_modal.load_state_dict(ckpt['cross_modal'])
        backbone.load_state_dict(ckpt['backbone'])
        print("✅ Loaded pre-trained encoder")
    else:
        print("⚠️  No checkpoint found, using random initialization")
    
    # Freeze encoder (optional: only train decoder)
    for param in img_tok.parameters():
        param.requires_grad = False
    for param in lid_tok.parameters():
        param.requires_grad = False
    for param in cross_modal.parameters():
        param.requires_grad = False
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Create decoder
    decoder = FutureLiDARDecoder(embed_dim=embed_dim, num_pred_points=num_pred_points).to(device)
    
    # Optimizer (only decoder parameters)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Data loader (shuffle=False to keep temporal order)
    ds = NuScenesMiniIter(
        root=root,
        batch_size=4,
        steps=200,
        img_size=(H, W),
        shuffle=False,  # Keep temporal sequence
    )
    
    # Training loop
    max_iters = 100
    print(f"\nTraining world model decoder for {max_iters} iterations...")
    print("Task: Predict future LiDAR from current image + LiDAR\n")
    
    decoder.train()
    img_tok.eval()
    lid_tok.eval()
    cross_modal.eval()
    backbone.eval()
    
    step = 0
    prev_pts = None  # Store previous timestep's LiDAR
    
    for imgs, pts in ds:
        if step >= max_iters:
            break
        
        # Skip first sample (no previous to predict from)
        if prev_pts is None:
            prev_pts = pts
            continue
        
        imgs = imgs.float().to(device) / 255.0
        pts_current = prev_pts.float().to(device)  # Use previous as input
        pts_future = pts.float().to(device)        # Current becomes "future" target
        
        # Forward pass through encoder (frozen)
        with torch.no_grad():
            ti = img_tok(imgs)
            tl = lid_tok(pts_current)
            ti_cross, tl_cross = cross_modal(ti, tl)
            fused = backbone(ti_cross, tl_cross)  # [B, 596, 96]
        
        # Decode to predicted future LiDAR
        pred_points = decoder(fused)  # [B, 512, 4]
        
        # Compute Chamfer Distance loss (only xyz, ignore intensity for simplicity)
        pred_xyz = pred_points[:, :, :3]  # [B, 512, 3]
        
        # Subsample ground truth to match prediction size
        N_gt = pts_future.shape[1]
        sample_indices = torch.randint(0, N_gt, (num_pred_points,))
        gt_xyz = pts_future[:, sample_indices, :3]  # [B, 512, 3]
        
        loss = chamfer_distance(pred_xyz, gt_xyz)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if (step + 1) % 10 == 0:
            print(f"step {step+1}/{max_iters}  loss={loss.item():.4f}")
        
        step += 1
        prev_pts = pts  # Update for next iteration
    
    # Save decoder checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    decoder_path = "checkpoints/omnivec2_world_model_decoder.pt"
    torch.save({
        'decoder': decoder.state_dict(),
    }, decoder_path)
    
    print(f"\n✅ Decoder checkpoint saved to {decoder_path}")
    print("\nWorld model training complete!")
    print("The model can now predict future LiDAR point clouds from current observations.")


if __name__ == "__main__":
    main()
