import os  # [attached_file:1]
import numpy as np  # [attached_file:1]
import torch  # [attached_file:1]
from PIL import Image  # [attached_file:1]

class NuScenesMiniIter:
    """
    Minimal iterator for nuScenes mini from a prepared file list.
    Folder layout:
      root/
        samples/
          CAM_FRONT/*.jpg
          LIDAR_TOP/*.bin  (float32 XYZI… at least 4 columns)
        pairs.txt   with lines: samples/CAM_FRONT/xxx.jpg samples/LIDAR_TOP/xxx.bin
    """  # [attached_file:1]
    def __init__(self, root, batch_size=4, steps=200, img_size=(224, 384), shuffle=True):
        self.root = root
        self.batch_size = batch_size
        self.steps = steps
        self.H, self.W = img_size
        pairs_file = os.path.join(root, "pairs.txt")
        with open(pairs_file, "r") as f:
            lines = [ln.strip().split() for ln in f if ln.strip()]
        self.items = [(os.path.join(root, a), os.path.join(root, b)) for a, b in lines]
        if shuffle:
            rng = np.random.RandomState(0)
            rng.shuffle(self.items)
        self._idx = 0  # [attached_file:1]

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.W, self.H), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)  # H, W, 3
        return arr.transpose(2, 0, 1)  # 3, H, W  # [attached_file:1]

    def _load_lidar_xyz_i(self, path):
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, -1)  # [N, K]  # [attached_file:1]
        if pts.shape[1] >= 4:
            pts = pts[:, :4]
        else:
            pad = np.zeros((pts.shape[0], 4), dtype=np.float32)
            pad[:, :pts.shape[1]] = pts
            pts = pad
        return pts  # [N,4]  # [attached_file:1]

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
            imgs_b = torch.from_numpy(np.stack(imgs, axis=0))  # [B,3,H,W] uint8  # [attached_file:1]
            maxN = max(pc.shape[0] for pc in clouds)
            pts_pad = np.zeros((B, maxN, 4), dtype=np.float32)
            for i, pc in enumerate(clouds):
                pts_pad[i, :pc.shape[0]] = pc
            pts_b = torch.from_numpy(pts_pad)  # [B,N,4] float32  # [attached_file:1]
            yield imgs_b, pts_b  # [attached_file:1]
