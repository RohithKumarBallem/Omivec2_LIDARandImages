#!/usr/bin/env python3
"""
Simple nuScenes I/O smoke test using NuScenesMiniIter.

- Reads dataroot and image_size from config/dataset.yaml under the 'nuscenes' key.
- Instantiates NuScenesMiniIter, which expects:
    root/
      samples/CAM_FRONT/*.jpg
      samples/LIDAR_TOP/*.bin
      pairs.txt  (each line: image_path lidar_path)
- Prints batch shapes for images and LiDAR.
"""

import os
import yaml
import torch

from src.data.nuscenes_loader import NuScenesMiniIter


def main():
    # Load dataset config
    cfg_path = os.path.join("config", "dataset.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Access nested 'nuscenes' section
    nus_cfg = cfg.get("nuscenes", {})
    root = nus_cfg.get("dataroot")
    if not root:
        raise ValueError("nuscenes.dataroot not set in config/dataset.yaml")

    # Handle image_size from config; supports [H, W] or [C, H, W]
    img_size_cfg = nus_cfg.get("image_size", [224, 384])
    if isinstance(img_size_cfg, (list, tuple)) and len(img_size_cfg) >= 2:
        H, W = img_size_cfg[-2], img_size_cfg[-1]  # last two entries = (H, W)
    else:
        H, W = 224, 384

    print(f"Using dataroot: {root}")
    print(f"Using image size: H={H}, W={W}")

    # Create iterator over a few steps for sanity check
    ds = NuScenesMiniIter(
        root=root,
        batch_size=2,
        steps=3,
        img_size=(H, W),
        shuffle=True,
    )

    # Take the first batch and print shapes
    for imgs, pts in ds:
        # imgs: [B, 3, H, W], uint8
        # pts:  [B, N, 4], float32
        print(f"Images batch shape: {imgs.shape}")
        print(f"LiDAR batch shape:  {pts.shape}")
        print(f"Images dtype: {imgs.dtype}, LiDAR dtype: {pts.dtype}")
        break


if __name__ == "__main__":
    main()
