import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer
from src.models.omnivec2_core import OmniVec2Tiny
from src.data.nuscenes_loader import NuScenesMiniIter


def demo_matching(ckpt_path, data_root, device_str="cpu"):
    """Demo: matching vs. mismatched pairs with visuals."""
    device = torch.device(device_str)
    
    # Load model
    img_tok = ImagePatchTokenizer(embed_dim=96, patch=32).to(device)
    lid_tok = PointTokenTokenizer(in_ch=4, embed_dim=96, num_tokens=512).to(device)
    backbone = OmniVec2Tiny(dim=96, heads=3, ff=192, depth=1).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt.get("backbone", ckpt))
    backbone.eval()
    
    # Load data
    loader = NuScenesMiniIter(root=data_root, batch_size=1, steps=4, shuffle=False)
    
    samples = []
    print("Loading samples...")
    with torch.no_grad():
        for imgs_batch, pts_batch in loader:
            if len(samples) >= 4:
                break
            imgs = imgs_batch[0].float().to(device) / 255.0  # [3, H, W]
            pts = pts_batch[0].float().to(device)  # [N, 4]
            
            ti = img_tok(imgs.unsqueeze(0))  # [1, T_img, 96]
            tl = lid_tok(pts.unsqueeze(0))  # [1, T_lid, 96]
            fused = backbone(ti, tl)  # [1, T_img+T_lid, 96]
            
            T_img = ti.shape[1]
            z_img = F.normalize(fused[0, :T_img].mean(0, keepdim=True), dim=-1)  # [1, 96]
            z_lid = F.normalize(fused[0, T_img:].mean(0, keepdim=True), dim=-1)  # [1, 96]
            
            samples.append({
                "img": imgs.cpu(),
                "pts": pts.cpu(),
                "z_img": z_img.cpu(),
                "z_lid": z_lid.cpu()
            })
    
    # Positive pairs (same scene)
    print("\n=== POSITIVE PAIRS (MATCHING IMAGE + LIDAR) ===\n")
    pos_sims = []
    for i, sample in enumerate(samples):
        sim = (sample["z_img"] * sample["z_lid"]).sum().item()
        pos_sims.append(sim)
        print(f"Sample {i}: Image <-> LiDAR (same scene) | Sim = {sim:+.4f}")
    
    # Negative pairs (shuffled)
    print("\n=== NEGATIVE PAIRS (MISMATCHED IMAGE + RANDOM LIDAR) ===\n")
    neg_sims = []
    for i in range(len(samples)):
        j = (i + len(samples) // 2) % len(samples)
        sim = (samples[i]["z_img"] * samples[j]["z_lid"]).sum().item()
        neg_sims.append(sim)
        print(f"Sample {i}: Image <-> LiDAR from sample {j} (random) | Sim = {sim:+.4f}")
    
    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)
    
    print(f"\n=== SUMMARY ===")
    print(f"Positive sims: mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}")
    print(f"Negative sims: mean={neg_sims.mean():.4f}, std={neg_sims.std():.4f}")
    print(f"Gap (pos - neg): {pos_sims.mean() - neg_sims.mean():.4f}")
    
    # Visualization
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5 * len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        j = (i + len(samples) // 2) % len(samples)
        
        # Query image
        img_np = sample["img"].permute(1, 2, 0).numpy().astype(np.uint8)
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Image {i}")
        axes[i, 0].axis("off")
        
        # Matching LiDAR BEV
        pts_np = sample["pts"].numpy()
        intensity = pts_np[:, 3] if pts_np.shape[1] >= 4 else np.ones(pts_np.shape[0])
        axes[i, 1].scatter(pts_np[:, 0], pts_np[:, 1], c=intensity, s=2, cmap='viridis')
        axes[i, 1].set_title(f"Matching LiDAR (Sim={pos_sims[i]:.3f})")
        axes[i, 1].set_xlabel("X (m)")
        axes[i, 1].set_ylabel("Y (m)")
        axes[i, 1].axis("equal")
        
        # Mismatched LiDAR BEV
        pts_np = samples[j]["pts"].numpy()
        intensity = pts_np[:, 3] if pts_np.shape[1] >= 4 else np.ones(pts_np.shape[0])
        axes[i, 2].scatter(pts_np[:, 0], pts_np[:, 1], c=intensity, s=2, cmap='viridis')
        axes[i, 2].set_title(f"Mismatched LiDAR (Sim={neg_sims[i]:.3f})")
        axes[i, 2].set_xlabel("X (m)")
        axes[i, 2].set_ylabel("Y (m)")
        axes[i, 2].axis("equal")
    
    plt.tight_layout()
    os.makedirs("runs", exist_ok=True)
    plt.savefig("runs/matching_demo.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to runs/matching_demo.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    demo_matching(args.ckpt, args.data, device_str=args.device)
