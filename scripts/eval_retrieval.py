import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer
from src.models.omnivec2_core import OmniVec2Tiny
from src.data.nuscenes_loader import NuScenesMiniIter

def compute_embeddings_batch(loader, img_tok, lid_tok, backbone, device, num_batches=5):
    """Compute image and LiDAR embeddings for retrieval."""
    z_imgs, z_lids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pts) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            imgs = imgs.float().to(device) / 255.0
            pts = pts.float().to(device)
            
            ti = img_tok(imgs)      # [B, T_img, 96]
            tl = lid_tok(pts)       # [B, 512, 96]
            fused = backbone(ti, tl)  # [B, T_img+512, 96]
            
            # Pool per modality
            T_img = ti.shape[1]
            z_img_fused = fused[:, :T_img].mean(1)  # [B, 96]
            z_lid_fused = fused[:, T_img:].mean(1)  # [B, 96]
            
            # Normalize
            z_img_fused = torch.nn.functional.normalize(z_img_fused, dim=-1)
            z_lid_fused = torch.nn.functional.normalize(z_lid_fused, dim=-1)
            
            z_imgs.append(z_img_fused.cpu())
            z_lids.append(z_lid_fused.cpu())
    
    z_imgs = torch.cat(z_imgs, dim=0)  # [N, 96]
    z_lids = torch.cat(z_lids, dim=0)  # [N, 96]
    return z_imgs, z_lids

def retrieval_demo(ckpt_path, root, device_str="cpu", k=5, num_batches=5):
    device = torch.device(device_str)
    
    # Load model
    img_tok = ImagePatchTokenizer(embed_dim=96, patch=32).to(device)
    lid_tok = PointTokenTokenizer(in_ch=4, embed_dim=96, num_tokens=512).to(device)
    backbone = OmniVec2Tiny(dim=96, heads=3, ff=192, depth=1).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt.get("backbone", ckpt))
    backbone.eval()
    
    # Load data
    loader = NuScenesMiniIter(root, batch_size=4, steps=num_batches * 4)
    
    # Compute embeddings
    z_imgs, z_lids = compute_embeddings_batch(loader, img_tok, lid_tok, backbone, device, num_batches)
    
    # Compute similarity matrix
    sim = z_imgs @ z_lids.t()  # [N, N]
    
    print(f"\nRetrieval metrics on {z_imgs.shape[0]} samples:")
    diag = torch.diagonal(sim)
    off_diag = sim[~torch.eye(sim.shape[0], dtype=torch.bool)]
    print(f"  Diagonal (positive) sim: mean={diag.mean():.4f}, std={diag.std():.4f}")
    print(f"  Off-diagonal (negative) sim: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
    
    # Recall@K
    for k_val in [1, 5, 10]:
        topk_indices = torch.topk(sim, min(k_val, sim.shape[1]), dim=1)[1]
        recall = (torch.arange(sim.shape[0]).unsqueeze(1) == topk_indices).any(1).float().mean()
        print(f"  Recall@{k_val}: {recall:.4f}")
    
    os.makedirs("runs/retrieval", exist_ok=True)
    print(f"\nRetrieval demo saved to runs/retrieval/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/omnivec2_fused_contrastive_96D.pt")
    parser.add_argument("--root", default="/path/to/nuscenes_mini")
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    retrieval_demo(args.ckpt, args.root, args.device, num_batches=args.num_batches)
