#!/usr/bin/env python3
"""
Full pipeline demo:

- Reads nuScenes config from config/dataset.yaml under 'nuscenes'.
- Uses NuScenesMiniIter to get one (images, LiDAR) batch.
- Tokenizes images and LiDAR with ImagePatchTokenizer and PointTokenTokenizer.
- Runs them through OmniVec2Tiny and prints token and fused shapes.
"""

import os
import yaml
import torch

from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer
from src.models.omnivec2_core import OmniVec2Tiny
from src.data.nuscenes_loader import NuScenesMiniIter


def main():
    # -----------------------------
    # 1. Device selection
    # -----------------------------
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(dev)
    print(f"Using device: {device}")

    # -----------------------------
    # 2. Load dataset config
    # -----------------------------
    cfg_path = os.path.join("config", "dataset.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    nus_cfg = cfg.get("nuscenes", {})
    root = nus_cfg.get("dataroot")
    if not root:
        raise ValueError("nuscenes.dataroot not set in config/dataset.yaml")

    img_size_cfg = nus_cfg.get("image_size", [224, 384])
    if isinstance(img_size_cfg, (list, tuple)) and len(img_size_cfg) >= 2:
        H, W = img_size_cfg[-2], img_size_cfg[-1]
    else:
        H, W = 224, 384

    print(f"Using dataroot: {root}")
    print(f"Using image size: H={H}, W={W}")

    # -----------------------------
    # 3. Build tiny model pieces
    # -----------------------------
    embed_dim = 96

    img_tok = ImagePatchTokenizer(
        embed_dim=embed_dim,
        patch=32,
    ).to(device)

    lid_tok = PointTokenTokenizer(
        in_ch=4,            # x, y, z, intensity
        embed_dim=embed_dim,
        num_tokens=512,     # or 1024 if you prefer
    ).to(device)

    backbone = OmniVec2Tiny(
        dim=embed_dim,
        heads=3,
        ff=192,
        depth=1,
    ).to(device)

    backbone.eval()

    # -----------------------------
    # 4. Fetch one batch from iterator
    # -----------------------------
    ds = NuScenesMiniIter(
        root=root,
        batch_size=1,
        steps=1,
        img_size=(H, W),
        shuffle=True,
    )

    imgs, pts = next(iter(ds))  # imgs: [1, 3, H, W], pts: [1, N, 4]
    print(f"Loaded images batch: {imgs.shape}, dtype={imgs.dtype}")
    print(f"Loaded LiDAR batch:  {pts.shape}, dtype={pts.dtype}")

    imgs = imgs.float().to(device) / 255.0
    pts = pts.float().to(device)

    # For this demo, just use the first image in the batch (still shape [1, 3, H, W])
    img = imgs  # already [1, 3, H, W]

    # -----------------------------
    # 5. Tokenize and fuse
    # -----------------------------
    with torch.no_grad():
        ti = img_tok(img)      # [1, T_img, 96]
        tl = lid_tok(pts)      # [1, num_tokens, 96]
        z = backbone(ti, tl)   # shape depends on OmniVec2Tiny, often [1, 96] or [1, T, 96]

    print("Image tokens shape:", ti.shape)
    print("LiDAR tokens shape:", tl.shape)
    print("Fused output shape:", z.shape)
    print("\nDemo complete: tokenization + fusion ran successfully.")


if __name__ == "__main__":
    main()
