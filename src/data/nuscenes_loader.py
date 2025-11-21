import os
import numpy as np
import torch
from PIL import Image


class NuScenesMiniIter:
    """
    Minimal iterator for nuScenes mini from a prepared file list.

    Expected folder layout under `root`:
      root/
        samples/
          CAM_FRONT/*.jpg
          LIDAR_TOP/*.pcd.bin   # LiDAR point clouds
        pairs.txt   with lines: "<img_rel_path> <lidar_rel_path>"

    Example pairs.txt line:
      samples/CAM_FRONT/n015-...-CAM_FRONT-....jpg samples/LIDAR_TOP/n015-...-LIDAR_TOP-....pcd.bin
    """

    def __init__(self, root, batch_size=4, steps=200, img_size=(224, 384), shuffle=True):
        self.root = root
        self.batch_size = batch_size
        self.steps = steps

        # img_size is (H, W)
        self.H, self.W = img_size

        pairs_file = os.path.join(root, "pairs.txt")
        if not os.path.exists(pairs_file):
            raise FileNotFoundError(f"pairs.txt not found at {pairs_file}")

        with open(pairs_file, "r") as f:
            lines = [ln.strip().split() for ln in f if ln.strip()]

        items = []
        for a, b in lines:
            a_lower = a.lower()
            b_lower = b.lower()

            # Decide which of (a, b) is image vs LiDAR by extension
            if any(a_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                img_rel, lid_rel = a, b
            elif any(b_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                img_rel, lid_rel = b, a
            else:
                # Fallback: assume original order is [image, lidar]
                img_rel, lid_rel = a, b

            img_path = os.path.join(root, img_rel)
            lid_path = os.path.join(root, lid_rel)
            items.append((img_path, lid_path))

        self.items = items
        if shuffle:
            rng = np.random.RandomState(0)
            rng.shuffle(self.items)

        self._idx = 0

    def _load_image(self, path):
        """Load RGB image, resize to (H, W), return as uint8 CHW array."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.W, self.H), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)      # [H, W, 3]
        return arr.transpose(2, 0, 1)           # [3, H, W]

    def _load_lidar_xyz_i(self, path):
        """
        Load LiDAR point cloud as [N, 4] float32 (x, y, z, intensity).

        nuScenes `.pcd.bin` files pack multiple float32 values per point
        (e.g., x, y, z, intensity, ring, ...). We read the flat array and
        reshape to (-1, K) with a fixed K (e.g., 6), then keep the first 4
        channels. [x, y, z, intensity] are the common first 4 fields. [web:263][web:269][web:274]
        """
        data = np.fromfile(path, dtype=np.float32)
        if data.size % 6 == 0:
            pts = data.reshape(-1, 6)[:, :4]  # [N, 4]
        elif data.size % 5 == 0:
            pts = data.reshape(-1, 5)[:, :4]  # [N, 4]
        elif data.size % 4 == 0:
            pts = data.reshape(-1, 4)         # [N, 4]
        else:
            raise ValueError(
                f"Unexpected LiDAR binary size {data.size} in {path}; "
                "cannot reshape into Nx{4,5,6} float32 channels."
            )
        return pts.astype(np.float32)

    def __iter__(self):
        B = self.batch_size
        n = len(self.items)
        for _ in range(self.steps):
            imgs, clouds = [], []
            for _b in range(B):
                p_img, p_lid = self.items[self._idx % n]
                self._idx += 1

                imgs.append(self._load_image(p_img))
                clouds.append(self._load_lidar_xyz_i(p_lid))

            # Stack images: [B, 3, H, W], uint8
            imgs_b = torch.from_numpy(np.stack(imgs, axis=0))

            # Pad LiDAR to max N in batch: [B, N, 4], float32
            maxN = max(pc.shape[0] for pc in clouds)
            pts_pad = np.zeros((B, maxN, 4), dtype=np.float32)
            for i, pc in enumerate(clouds):
                pts_pad[i, :pc.shape[0]] = pc
            pts_b = torch.from_numpy(pts_pad)

            yield imgs_b, pts_b
