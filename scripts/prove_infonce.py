#!/usr/bin/env python3
"""
Proof that InfoNCE actually learned meaningful alignments.

Takes a random test image and retrieves its matching LiDAR scan.
Shows that:
1. Correct match has highest similarity
2. Incorrect matches have lower similarity
3. Ranking clearly separates them
"""

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer
from src.models.omnivec2_core import CrossModalBlock, OmniVec2Tiny
from src.data.nuscenes_loader import NuScenesMiniIter


class ProjectionHead(torch.nn.Module):
    """MLP projection head."""
    def __init__(self, in_dim=96, out_dim=128, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)


def load_model(ckpt_path, device):
    """Load trained model from checkpoint."""
    embed_dim = 96
    proj_dim = 128
    
    img_tok = ImagePatchTokenizer(embed_dim=embed_dim, patch=32).to(device)
    lid_tok = PointTokenTokenizer(in_ch=4, embed_dim=embed_dim, num_tokens=512).to(device)
    cross_modal = CrossModalBlock(dim=embed_dim, heads=3, ff=192).to(device)
    backbone = OmniVec2Tiny(dim=embed_dim, heads=3, ff=192, depth=1).to(device)
    proj_i = ProjectionHead(in_dim=embed_dim, out_dim=proj_dim).to(device)
    proj_l = ProjectionHead(in_dim=embed_dim, out_dim=proj_dim).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    img_tok.load_state_dict(ckpt['img_tok'])
    lid_tok.load_state_dict(ckpt['lid_tok'])
    cross_modal.load_state_dict(ckpt['cross_modal'])
    backbone.load_state_dict(ckpt['backbone'])
    proj_i.load_state_dict(ckpt['proj_i'])
    proj_l.load_state_dict(ckpt['proj_l'])
    
    return img_tok, lid_tok, cross_modal, backbone, proj_i, proj_l


def compute_all_embeddings(dataloader, img_tok, lid_tok, cross_modal, backbone, proj_i, proj_l, device, num_batches=10):
    """
    Compute all image and LiDAR embeddings from the dataset.
    Returns:
        z_imgs: [N, 128]  (projected image embeddings)
        z_lids: [N, 128]  (projected lidar embeddings)
    """
    z_imgs_list = []
    z_lids_list = []
    
    with torch.no_grad():
        for batch_idx, (imgs, pts) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            imgs = imgs.float().to(device) / 255.0
            pts = pts.float().to(device)
            
            # Forward pass
            ti = img_tok(imgs)
            tl = lid_tok(pts)
            ti_cross, tl_cross = cross_modal(ti, tl)
            fused = backbone(ti_cross, tl_cross)
            
            T_img = ti.shape[1]
            ti_fused = fused[:, :T_img, :]
            tl_fused = fused[:, T_img:, :]
            
            z_i = ti_fused.mean(dim=1)
            z_l = tl_fused.mean(dim=1)
            
            z_i_proj = proj_i(z_i)
            z_l_proj = proj_l(z_l)
            
            z_imgs_list.append(z_i_proj.cpu())
            z_lids_list.append(z_l_proj.cpu())
    
    z_imgs = torch.cat(z_imgs_list, dim=0)  # [N, 128]
    z_lids = torch.cat(z_lids_list, dim=0)  # [N, 128]
    
    return z_imgs, z_lids


def prove_infonce(ckpt_path, root, device_str="mps", test_sample_idx=0, num_batches=10):
    """
    Proof: Take one random test image and show it retrieves correct LiDAR.
    """
    device = torch.device(device_str)
    
    # Load config
    cfg_path = os.path.join("config", "dataset.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    nus_cfg = cfg.get("nuscenes", {})
    img_size_cfg = nus_cfg.get("image_size", [224, 384])
    H, W = img_size_cfg[-2], img_size_cfg[-1]
    
    print("=" * 70)
    print("PROOF: InfoNCE Loss is Actually Working")
    print("=" * 70)
    print(f"\nUsing checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading trained model...")
    img_tok, lid_tok, cross_modal, backbone, proj_i, proj_l = load_model(ckpt_path, device)
    img_tok.eval()
    lid_tok.eval()
    cross_modal.eval()
    backbone.eval()
    proj_i.eval()
    proj_l.eval()
    
    # Get dataloader
    ds = NuScenesMiniIter(root=root, batch_size=4, steps=num_batches, img_size=(H, W), shuffle=False)
    
    # Compute all embeddings
    print(f"\nComputing embeddings for {num_batches} batches...")
    z_imgs, z_lids = compute_all_embeddings(ds, img_tok, lid_tok, cross_modal, backbone, proj_i, proj_l, device, num_batches)
    
    N = z_imgs.shape[0]
    print(f"Total samples: {N}")
    
    # Prove on one random test image
    test_idx = test_sample_idx % N
    print(f"\nTest image index: {test_idx}")
    
    # Get test image embedding
    z_test_img = z_imgs[test_idx:test_idx+1]  # [1, 128]
    
    # Compute similarities to ALL LiDAR scans
    z_test_img_norm = F.normalize(z_test_img, dim=1)
    z_lids_norm = F.normalize(z_lids, dim=1)
    
    similarities = (z_test_img_norm @ z_lids_norm.T).squeeze()  # [N]
    
    # Get ranking
    ranked_indices = torch.argsort(similarities, descending=True)
    ranked_sims = similarities[ranked_indices]
    
    # Correct match is at position test_idx (same index for paired data)
    correct_rank = (ranked_indices == test_idx).nonzero(as_tuple=True)[0].item() + 1
    correct_sim = similarities[test_idx].item()
    
    print("\n" + "=" * 70)
    print("PROOF RESULTS")
    print("=" * 70)
    print(f"\nQuery: Image #{test_idx}")
    print(f"Correct match: LiDAR #{test_idx}")
    print(f"\nCorrect match similarity: {correct_sim:.4f}")
    print(f"Correct match RANK: {correct_rank} / {N}")
    
    if correct_rank == 1:
        print("✅ SUCCESS: Correct LiDAR is RANKED #1!")
    elif correct_rank <= 5:
        print(f"✅ GOOD: Correct LiDAR is in top-5 (rank {correct_rank})")
    elif correct_rank <= 10:
        print(f"⚠️  OKAY: Correct LiDAR is in top-10 (rank {correct_rank})")
    else:
        print(f"❌ FAIL: Correct LiDAR ranked {correct_rank} (model not learning well)")
    
    print("\nTop-5 Most Similar LiDAR Scans:")
    print("-" * 70)
    for rank, idx in enumerate(ranked_indices[:5], 1):
        idx_val = idx.item()
        sim_val = ranked_sims[rank-1].item()
        is_correct = "✓ CORRECT" if idx_val == test_idx else ""
        print(f"  Rank {rank}: LiDAR #{idx_val:3d}  |  Similarity: {sim_val:+.4f}  {is_correct}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("RETRIEVAL STATISTICS")
    print("=" * 70)
    print(f"Correct match similarity:    {correct_sim:.4f}")
    print(f"Top-1 similarity:            {ranked_sims[0].item():+.4f}")
    print(f"Mean top-5 similarity:       {ranked_sims[:5].mean().item():+.4f}")
    print(f"Mean bottom-5 similarity:    {ranked_sims[-5:].mean().item():+.4f}")
    print(f"Similarity gap (top vs bottom): {(ranked_sims[0] - ranked_sims[-1]).item():.4f}")
    
    # Visualization
    print("\nGenerating visualization...")
    os.makedirs("runs/proof", exist_ok=True)
    
    # Plot 1: Histogram of similarities
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(similarities.numpy(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(correct_sim, color='red', linestyle='--', linewidth=2, label=f'Correct match ({correct_sim:.4f})')
    ax.axvline(ranked_sims[0].item(), color='green', linestyle='--', linewidth=2, label=f'Top-1 ({ranked_sims[0].item():.4f})')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Similarity Distribution\n(Query: Image #{test_idx})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bar plot: Top-20 rankings
    ax = axes[1]
    top_20_sims = ranked_sims[:20].numpy()
    top_20_indices = ranked_indices[:20].numpy()
    colors = ['red' if idx == test_idx else 'blue' for idx in top_20_indices]
    bars = ax.barh(range(20), top_20_sims, color=colors, edgecolor='black')
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"Rank {i+1}\n(LiDAR #{idx})" for i, idx in enumerate(top_20_indices)], fontsize=9)
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_title('Top-20 Retrieved LiDAR Scans', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    
    # Highlight correct match
    if correct_rank <= 20:
        bars[correct_rank-1].set_linewidth(3)
        bars[correct_rank-1].set_edgecolor('gold')
    
    plt.tight_layout()
    plt.savefig('runs/proof/infonce_proof.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to runs/proof/infonce_proof.png")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if correct_rank <= 5:
        print("✅ InfoNCE is WORKING! The model learned to align modalities correctly.")
        print(f"   The correct LiDAR match ranks #{correct_rank}, proving the loss optimized")
        print("   the embeddings to be similar for paired samples.")
    else:
        print("⚠️  InfoNCE training may need improvement.")
        print(f"   Correct match ranked #{correct_rank}; consider longer training or larger batch size.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/omnivec2_hybrid_96D.pt")
    parser.add_argument("--root", default="/Users/rohithkumarballem/Downloads/nuscenes")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--test-idx", type=int, default=0, help="Which test image to use (0 for first)")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to compute embeddings")
    args = parser.parse_args()
    
    prove_infonce(args.ckpt, args.root, args.device, args.test_idx, args.num_batches)
