import os, yaml, torch, cv2
import numpy as np
from src.data.nuscenes_loader import NuScenesMiniSample
from src.models.tokenizers import ImagePatchTokenizer, LidarPillarTokenizer
from src.models.omnivec2_core import OmniVec2Tiny
from src.utils.device import get_device

# Free MPS cache if available
if hasattr(torch.mps, "empty_cache"):
    torch.mps.empty_cache()

cfg = yaml.safe_load(open("config/dataset.yaml"))
root = cfg["nuscenes"]["dataroot"]
ver = cfg["nuscenes"]["version"]
cams = cfg["nuscenes"]["cameras"]
lidar = cfg["nuscenes"]["lidar"]
H, W = cfg["nuscenes"]["image_size"]

ds = NuScenesMiniSample(root, ver, cams, lidar)
tok, imgs, pts = ds.get_first_sample()

# Use 1–3 cameras to reduce tokens; start with 1 for MPS
sel = ["CAM_FRONT"]
ims = [cv2.resize(imgs[c], (W, H)) for c in sel]
im = np.concatenate(ims, axis=0)  # [H, W, 3] if one cam
if len(sel) > 1:
    im = np.concatenate(ims, axis=0)  # [kH, W, 3]
im = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).float() / 255.0  # [1,3,H*,W]

pc = torch.from_numpy(pts).unsqueeze(0).float()  # [1,N,4]

dev = get_device()
im, pc = im.to(dev), pc.to(dev)

img_tok = ImagePatchTokenizer(embed_dim=96, patch=32).to(dev)(im)
lid_tok = LidarPillarTokenizer(cell=1.0, embed_dim=96).to(dev)(pc)

model = OmniVec2Tiny(dim=96, heads=3, ff=192, depth=1).to(dev)
out = model(img_tok, lid_tok)
print("Fused embedding:", out.shape)
