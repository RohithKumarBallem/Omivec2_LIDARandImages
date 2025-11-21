#!/usr/bin/env python3
"""
CONCRETE PROOF: Untrained vs Trained Model

This script:
1. Tests UNTRAINED model (random weights) → should be ~random performance
2. Tests TRAINED model → should clearly rank correct pairs higher
3. Shows the quantitative difference

No ambiguity. Pure numbers.
"""

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np

from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer
from src.models.omnivec2_core import CrossModalBlock, OmniVec2Tiny
from src.data.nuscenes_loader import NuScenesMiniIter


class ProjectionHead(torch.nn.Module):
    def __init__(self, in_dim=96, out_dim=128, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)


def load_trained_model(ckpt_path, device):
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


def create_untrained_model(device):
    """Create fresh model with random weights."""
    embed_dim = 96
    proj_dim = 128
    
    img_tok = ImagePatchTokenizer(embed_dim=embed_dim, patch=32).to(device)
    lid_tok = PointTokenTokenizer(in_ch=4, embed_dim=embed_dim, num_tokens=512).to(device)
    cross_modal = CrossModalBlock(dim=embed_dim, heads=3, ff=192).to(device)
    backbone = OmniVec2Tiny(dim=embed_dim, heads=3, ff=192, depth=1).to(device)
    proj_i = ProjectionHead(in_dim=embed_dim, out_dim=proj_dim).to(device)
    proj_l = ProjectionHead(in_dim=embed_dim, out_dim=proj_dim).to(device)
    
    return img_tok, lid_tok, cross_modal, backbone, proj_i, proj_l


def compute_retrieval_metrics(dataloader, img_tok, lid_tok, cross_modal, backbone, proj_i, proj_l, device, num_batches=10):
    """
    Compute Recall@K metrics by:
    1. Computing all image embeddings
    2. Computing all LiDAR embeddings
    3. For each image, rank all LiDAR by similarity
    4. Check if correct match is in top-K
    """
    z_imgs_list = []
    z_lids_list = []
    
    with torch.no_grad():
        for batch_idx, (imgs, pts) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            imgs = imgs.float().to(device) / 255.0
            pts = pts.float().to(device)
            
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
    N = z_imgs.shape[0]
    
    # Normalize
    z_imgs_norm = F.normalize(z_imgs, dim=1)
    z_lids_norm = F.normalize(z_lids, dim=1)
    
    # Compute all similarities
    similarities = z_imgs_norm @ z_lids_norm.T  # [N, N]
    
    # For each query image, check if correct LiDAR is in top-K
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    mean_rank = 0
    
    for query_idx in range(N):
        sim_row = similarities[query_idx]  # [N]
        ranked_indices = torch.argsort(sim_row, descending=True)  # [N]
        
        # Correct answer is at index query_idx (paired data)
        rank_of_correct = (ranked_indices == query_idx).nonzero(as_tuple=True)[0].item() + 1
        
        if rank_of_correct == 1:
            recall_at_1 += 1
        if rank_of_correct <= 5:
            recall_at_5 += 1
        if rank_of_correct <= 10:
            recall_at_10 += 1
        
        mean_rank += rank_of_correct
    
    recall_at_1 = recall_at_1 / N
    recall_at_5 = recall_at_5 / N
    recall_at_10 = recall_at_10 / N
    mean_rank = mean_rank / N
    
    return recall_at_1, recall_at_5, recall_at_10, mean_rank


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load config
    cfg_path = os.path.join("config", "dataset.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    nus_cfg = cfg.get("nuscenes", {})
    root = nus_cfg.get("dataroot")
    img_size_cfg = nus_cfg.get("image_size", [224, 384])
    H, W = img_size_cfg[-2], img_size_cfg[-1]
    
    print("\n" + "=" * 80)
    print(" " * 20 + "CONCRETE PROOF: InfoNCE IS WORKING")
    print("=" * 80)
    print("\nTest Setup:")
    print("  - 40 samples (10 batches)")
    print("  - For each image, rank all 40 LiDAR by similarity")
    print("  - Check: Is the correct LiDAR in top-1, top-5, or top-10?")
    print("=" * 80)
    
    # Load dataloader
    ds = NuScenesMiniIter(root=root, batch_size=4, steps=10, img_size=(H, W), shuffle=False)
    
    # ==================== TEST 1: UNTRAINED MODEL ====================
    print("\n\n" + "▶" * 40)
    print("TEST 1: UNTRAINED MODEL (Random Weights)")
    print("▶" * 40)
    
    print("\nLoading untrained model with random weights...")
    img_tok_u, lid_tok_u, cross_modal_u, backbone_u, proj_i_u, proj_l_u = create_untrained_model(device)
    img_tok_u.eval()
    lid_tok_u.eval()
    cross_modal_u.eval()
    backbone_u.eval()
    proj_i_u.eval()
    proj_l_u.eval()
    
    print("Computing retrieval metrics...")
    r1_u, r5_u, r10_u, mean_rank_u = compute_retrieval_metrics(
        ds, img_tok_u, lid_tok_u, cross_modal_u, backbone_u, proj_i_u, proj_l_u, device, num_batches=10
    )
    
    print("\n" + "─" * 80)
    print("UNTRAINED MODEL RESULTS:")
    print("─" * 80)
    print(f"  Recall@1:   {r1_u:.1%}  (random expectation: 2.5%)")
    print(f"  Recall@5:   {r5_u:.1%}  (random expectation: 12.5%)")
    print(f"  Recall@10:  {r10_u:.1%}  (random expectation: 25%)")
    print(f"  Mean Rank:  {mean_rank_u:.1f}  (random expectation: 20.5)")
    print("─" * 80)
    
    # ==================== TEST 2: TRAINED MODEL ====================
    print("\n\n" + "▶" * 40)
    print("TEST 2: TRAINED MODEL (100 iterations)")
    print("▶" * 40)
    
    ckpt_path = "checkpoints/omnivec2_hybrid_96D.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return
    
    print(f"\nLoading trained model from {ckpt_path}...")
    img_tok_t, lid_tok_t, cross_modal_t, backbone_t, proj_i_t, proj_l_t = load_trained_model(ckpt_path, device)
    img_tok_t.eval()
    lid_tok_t.eval()
    cross_modal_t.eval()
    backbone_t.eval()
    proj_i_t.eval()
    proj_l_t.eval()
    
    print("Computing retrieval metrics...")
    # Need fresh dataloader
    ds2 = NuScenesMiniIter(root=root, batch_size=4, steps=10, img_size=(H, W), shuffle=False)
    r1_t, r5_t, r10_t, mean_rank_t = compute_retrieval_metrics(
        ds2, img_tok_t, lid_tok_t, cross_modal_t, backbone_t, proj_i_t, proj_l_t, device, num_batches=10
    )
    
    print("\n" + "─" * 80)
    print("TRAINED MODEL RESULTS:")
    print("─" * 80)
    print(f"  Recall@1:   {r1_t:.1%}  (random expectation: 2.5%)")
    print(f"  Recall@5:   {r5_t:.1%}  (random expectation: 12.5%)")
    print(f"  Recall@10:  {r10_t:.1%}  (random expectation: 25%)")
    print(f"  Mean Rank:  {mean_rank_t:.1f}  (random expectation: 20.5)")
    print("─" * 80)
    
    # ==================== COMPARISON ====================
    print("\n\n" + "🔥" * 40)
    print(" " * 25 + "COMPARISON & PROOF")
    print("🔥" * 40)
    
    improvement_r1 = (r1_t - r1_u) / (r1_u + 1e-6) * 100
    improvement_r5 = (r5_t - r5_u) / (r5_u + 1e-6) * 100
    improvement_r10 = (r10_t - r10_u) / (r10_u + 1e-6) * 100
    rank_improvement = (mean_rank_u - mean_rank_t) / mean_rank_u * 100
    
    print(f"\nRecall@1 improvement:    {r1_u:.1%} → {r1_t:.1%}  ({improvement_r1:+.0f}%)")
    print(f"Recall@5 improvement:    {r5_u:.1%} → {r5_t:.1%}  ({improvement_r5:+.0f}%)")
    print(f"Recall@10 improvement:   {r10_u:.1%} → {r10_t:.1%}  ({improvement_r10:+.0f}%)")
    print(f"Mean Rank improvement:   {mean_rank_u:.1f} → {mean_rank_t:.1f}  ({rank_improvement:+.0f}% better)")
    
    # ==================== VERDICT ====================
    print("\n\n" + "=" * 80)
    print(" " * 30 + "FINAL VERDICT")
    print("=" * 80)
    
    if r1_t > r1_u * 2 and r5_t > r5_u * 1.5:
        print("\n✅✅✅ InfoNCE IS DEFINITIVELY WORKING ✅✅✅")
        print("\nEvidence:")
        print(f"  • Trained model Recall@5: {r5_t:.1%} (vs untrained {r5_u:.1%})")
        print(f"  • Trained model Mean Rank: {mean_rank_t:.1f} (vs untrained {mean_rank_u:.1f})")
        print(f"  • The model learned to rank correct pairs MUCH higher")
        print(f"  • This is NOT random chance—InfoNCE successfully optimized embeddings")
        print("\n✅ Proof complete. You can confidently tell your professor:")
        print("   'The trained model significantly outperforms random weights,")
        print("    proving that InfoNCE loss successfully learned meaningful")
        print("    image-LiDAR alignments.'")
    elif r5_t > r5_u:
        print("\n⚠️  Partial evidence of learning, but not conclusive.")
        print("    Consider training longer (500+ iterations) for stronger proof.")
    else:
        print("\n❌ Model not learning. Check training configuration.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
