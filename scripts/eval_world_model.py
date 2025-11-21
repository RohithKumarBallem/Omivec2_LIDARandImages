#!/usr/bin/env python3
"""
Evaluation: Is the world model actually predicting future LiDAR correctly?

Compare:
1. Predicted future LiDAR
2. Ground truth future LiDAR
3. Baseline (just copy current LiDAR as "prediction")

If predictions are closer to GT than baseline, model is learning.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer
from src.models.omnivec2_core import CrossModalBlock, OmniVec2Tiny
from src.data.nuscenes_loader import NuScenesMiniIter


class FutureLiDARDecoder(nn.Module):
    def __init__(self, embed_dim=96, num_pred_points=512):
        super().__init__()
        self.num_pred_points = num_pred_points
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, num_pred_points * 4),
        )
    
    def forward(self, fused):
        z_scene = fused.mean(dim=1)
        pred_flat = self.decoder(z_scene)
        pred_points = pred_flat.view(-1, self.num_pred_points, 4)
        return pred_points


def chamfer_distance(pred_points, gt_points, return_separate=False):
    """
    Chamfer Distance between two point clouds.
    
    pred_points: [B, N_pred, 3]
    gt_points:   [B, N_gt, 3]
    """
    B = pred_points.shape[0]
    
    # Pairwise distances
    pred_expanded = pred_points.unsqueeze(2)  # [B, N_pred, 1, 3]
    gt_expanded = gt_points.unsqueeze(1)      # [B, 1, N_gt, 3]
    dists = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)  # [B, N_pred, N_gt]
    
    # Forward
    min_dist_pred_to_gt, _ = torch.min(dists, dim=2)
    loss_forward = min_dist_pred_to_gt.mean()
    
    # Backward
    min_dist_gt_to_pred, _ = torch.min(dists, dim=1)
    loss_backward = min_dist_gt_to_pred.mean()
    
    cd = (loss_forward + loss_backward) / 2
    
    if return_separate:
        return cd, loss_forward, loss_backward
    return cd


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Config
    cfg_path = os.path.join("config", "dataset.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    nus_cfg = cfg.get("nuscenes", {})
    root = nus_cfg.get("dataroot")
    img_size_cfg = nus_cfg.get("image_size", [224, 384])
    H, W = img_size_cfg[-2], img_size_cfg[-1]
    
    print("\n" + "=" * 80)
    print(" " * 20 + "WORLD MODEL EVALUATION")
    print("=" * 80)
    print("\nTest: Can the model predict future LiDAR better than baseline?")
    print("-" * 80)
    
    # Load encoder
    embed_dim = 96
    num_pred_points = 512
    
    img_tok = ImagePatchTokenizer(embed_dim=embed_dim, patch=32).to(device)
    lid_tok = PointTokenTokenizer(in_ch=4, embed_dim=embed_dim, num_tokens=512).to(device)
    cross_modal = CrossModalBlock(dim=embed_dim, heads=3, ff=192).to(device)
    backbone = OmniVec2Tiny(dim=embed_dim, heads=3, ff=192, depth=1).to(device)
    
    ckpt_path = "checkpoints/omnivec2_hybrid_96D.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        img_tok.load_state_dict(ckpt['img_tok'])
        lid_tok.load_state_dict(ckpt['lid_tok'])
        cross_modal.load_state_dict(ckpt['cross_modal'])
        backbone.load_state_dict(ckpt['backbone'])
    
    for param in [img_tok, lid_tok, cross_modal, backbone]:
        for p in param.parameters():
            p.requires_grad = False
    
    # Load decoder
    decoder = FutureLiDARDecoder(embed_dim=embed_dim, num_pred_points=num_pred_points).to(device)
    decoder_ckpt = "checkpoints/omnivec2_world_model_decoder.pt"
    if os.path.exists(decoder_ckpt):
        ckpt = torch.load(decoder_ckpt, map_location=device)
        decoder.load_state_dict(ckpt['decoder'])
        print("✅ Loaded decoder checkpoint")
    else:
        print("❌ Decoder checkpoint not found")
        return
    
    # Set to eval mode
    img_tok.eval()
    lid_tok.eval()
    cross_modal.eval()
    backbone.eval()
    decoder.eval()
    
    # Load data
    ds = NuScenesMiniIter(root=root, batch_size=4, steps=10, img_size=(H, W), shuffle=False)
    
    # Evaluation metrics
    cd_predicted_list = []  # Predicted → GT
    cd_baseline_list = []   # Baseline (copy current) → GT
    
    print("\nEvaluating on 40 samples...")
    
    step = 0
    prev_pts = None
    
    with torch.no_grad():
        for imgs, pts in ds:
            if step >= 10:
                break
            
            if prev_pts is None:
                prev_pts = pts
                step += 1
                continue
            
            imgs = imgs.float().to(device) / 255.0
            pts_current = prev_pts.float().to(device)
            pts_future = pts.float().to(device)
            
            # Forward through encoder
            ti = img_tok(imgs)
            tl = lid_tok(pts_current)
            ti_cross, tl_cross = cross_modal(ti, tl)
            fused = backbone(ti_cross, tl_cross)
            
            # Predict
            pred_points = decoder(fused)  # [B, 512, 4]
            pred_xyz = pred_points[:, :, :3]
            
            # Subsample GT
            N_gt = pts_future.shape[1]
            sample_indices = torch.randint(0, N_gt, (num_pred_points,))
            gt_xyz = pts_future[:, sample_indices, :3]
            
            # Baseline: just use current LiDAR as "prediction"
            pts_current_xyz = pts_current[:, sample_indices, :3]
            
            # Compute metrics
            cd_pred = chamfer_distance(pred_xyz, gt_xyz)
            cd_base = chamfer_distance(pts_current_xyz, gt_xyz)
            
            cd_predicted_list.append(cd_pred.item())
            cd_baseline_list.append(cd_base.item())
            
            step += 1
            prev_pts = pts
    
    # Results
    mean_cd_pred = np.mean(cd_predicted_list)
    mean_cd_base = np.mean(cd_baseline_list)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nMean Chamfer Distance (Predicted):  {mean_cd_pred:.4f}")
    print(f"Mean Chamfer Distance (Baseline):  {mean_cd_base:.4f}")
    print(f"\nDifference: {abs(mean_cd_pred - mean_cd_base):.4f}")
    
    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    if mean_cd_pred < mean_cd_base * 0.95:
        print("\n✅ WORLD MODEL IS LEARNING!")
        print(f"   Predicted predictions are {(mean_cd_base/mean_cd_pred - 1)*100:.1f}% better than baseline")
        print(f"   The model successfully predicts future LiDAR structure")
    elif mean_cd_pred < mean_cd_base * 1.05:
        print("\n⚠️  MARGINAL: Model performance ~same as baseline")
        print("   Consider training longer or using larger batch size")
    else:
        print("\n❌ MODEL NOT LEARNING")
        print("   Predictions are worse than just copying current LiDAR")
        print("   Potential issues:")
        print("   - Decoder not trained enough (only 100 iterations)")
        print("   - Learning rate too low")
        print("   - Future LiDAR too different from current (hard prediction task)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
