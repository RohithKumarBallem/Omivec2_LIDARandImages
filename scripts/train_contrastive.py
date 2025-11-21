#!/usr/bin/env python3
"""
Hybrid contrastive training: CrossModalBlock + OmniVec2Tiny.

Pipeline:
1. Tokenize → [B, 84, 96], [B, 512, 96]
2. CrossModalBlock (explicit cross-attention) → [B, 84, 96], [B, 512, 96]
3. OmniVec2Tiny (shared self-attention fusion) → [B, 596, 96]
4. Pool → [B, 96]
5. Project → [B, 128] for each modality
6. InfoNCE loss
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer
from src.models.omnivec2_core import CrossModalBlock, OmniVec2Tiny
from src.data.nuscenes_loader import NuScenesMiniIter


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    def __init__(self, in_dim=96, out_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)


def info_nce_loss(z_i, z_l, tau=0.1):
    """
    Symmetric InfoNCE loss.
    z_i, z_l: [B, D] normalized embeddings
    """
    B = z_i.size(0)
    
    # Normalize
    z_i = F.normalize(z_i, dim=1)
    z_l = F.normalize(z_l, dim=1)
    
    # Similarity matrix [B, B]
    sim = (z_i @ z_l.T) / tau
    
    # Positive pairs are on diagonal
    labels = torch.arange(B, device=z_i.device)
    
    # Cross-entropy loss both directions
    loss_i2l = F.cross_entropy(sim, labels)
    loss_l2i = F.cross_entropy(sim.T, labels)
    
    return (loss_i2l + loss_l2i) / 2


def main():
    # Device
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(dev)
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
    proj_dim = 128
    
    # Tokenizers
    img_tok = ImagePatchTokenizer(embed_dim=embed_dim, patch=32).to(device)
    lid_tok = PointTokenTokenizer(in_ch=4, embed_dim=embed_dim, num_tokens=512).to(device)
    
    # Stage 1: CrossModalBlock (explicit cross-attention)
    cross_modal = CrossModalBlock(dim=embed_dim, heads=3, ff=192).to(device)
    
    # Stage 2: OmniVec2Tiny (shared self-attention fusion)
    backbone = OmniVec2Tiny(dim=embed_dim, heads=3, ff=192, depth=1).to(device)
    
    # Projection heads (separate for image and lidar branches)
    proj_i = ProjectionHead(in_dim=embed_dim, out_dim=proj_dim).to(device)
    proj_l = ProjectionHead(in_dim=embed_dim, out_dim=proj_dim).to(device)
    
    # Optimizer (all parameters)
    params = (
        list(img_tok.parameters()) + 
        list(lid_tok.parameters()) + 
        list(cross_modal.parameters()) + 
        list(backbone.parameters()) +
        list(proj_i.parameters()) + 
        list(proj_l.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
    
    # Data loader
    ds = NuScenesMiniIter(
        root=root,
        batch_size=4,
        steps=200,
        img_size=(H, W),
        shuffle=True,
    )
    
    # Training loop
    max_iters = 100
    print(f"\nTraining for {max_iters} iterations with hybrid architecture...")
    print("  Stage 1: CrossModalBlock (cross-attention)")
    print("  Stage 2: OmniVec2Tiny (self-attention fusion)\n")
    
    cross_modal.train()
    backbone.train()
    step = 0
    
    for imgs, pts in ds:
        if step >= max_iters:
            break
        
        imgs = imgs.float().to(device) / 255.0
        pts = pts.float().to(device)
        
        # 1. Tokenize
        ti = img_tok(imgs)      # [B, 84, 96]
        tl = lid_tok(pts)       # [B, 512, 96]
        
        # 2. Stage 1: Cross-modal interaction (explicit cross-attention)
        ti_cross, tl_cross = cross_modal(ti, tl)  # [B, 84, 96], [B, 512, 96]
        
        # 3. Stage 2: Deep fusion (shared self-attention)
        fused = backbone(ti_cross, tl_cross)  # [B, 596, 96]  (84+512 tokens)
        
        # 4. Split back into modality-specific token groups
        T_img = ti.shape[1]  # 84
        ti_fused = fused[:, :T_img, :]      # [B, 84, 96]  (image tokens)
        tl_fused = fused[:, T_img:, :]      # [B, 512, 96] (lidar tokens)
        
        # 5. Pool each modality separately
        z_i = ti_fused.mean(dim=1)  # [B, 96]
        z_l = tl_fused.mean(dim=1)  # [B, 96]
        
        # 6. Project to contrastive space
        z_i_proj = proj_i(z_i)  # [B, 128]
        z_l_proj = proj_l(z_l)  # [B, 128]
        
        # 7. Compute InfoNCE loss
        loss = info_nce_loss(z_i_proj, z_l_proj, tau=0.1)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if (step + 1) % 5 == 0:
            with torch.no_grad():
                z_i_norm = F.normalize(z_i_proj, dim=1)
                z_l_norm = F.normalize(z_l_proj, dim=1)
                
                # Diagonal similarity (positives)
                sim_pos = (z_i_norm * z_l_norm).sum(dim=1).mean()
                
                # Embedding stats
                zi_std = z_i_proj.std(dim=0).mean()
                zl_std = z_l_proj.std(dim=0).mean()
                z_i_mag = z_i.norm(dim=1).mean()
                
                print(f"step {step+1}/{max_iters} loss {loss.item():.4f} "
                      f"sim+={sim_pos.item():.3f} zi_std={zi_std.item():.3f} "
                      f"zl_std={zl_std.item():.3f} z_mag={z_i_mag.item():.2f}")
        
        step += 1
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/omnivec2_hybrid_96D.pt"
    torch.save({
        'img_tok': img_tok.state_dict(),
        'lid_tok': lid_tok.state_dict(),
        'cross_modal': cross_modal.state_dict(),
        'backbone': backbone.state_dict(),
        'proj_i': proj_i.state_dict(),
        'proj_l': proj_l.state_dict(),
    }, ckpt_path)
    
    print(f"\nCheckpoint saved to {ckpt_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
