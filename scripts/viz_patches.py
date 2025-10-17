import os, yaml, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
from src.data.nuscenes_loader import NuScenesMiniSample
from src.models.tokenizers import ImagePatchTokenizer

# Load config and sample
cfg = yaml.safe_load(open("config/dataset.yaml"))
root, ver = cfg["nuscenes"]["dataroot"], cfg["nuscenes"]["version"]
cams = cfg["nuscenes"]["cameras"]
H, W = cfg["nuscenes"]["image_size"]
patch = 32

ds = NuScenesMiniSample(root, ver, cams, cfg["nuscenes"]["lidar"])
_, imgs, _ = ds.get_first_sample()
img = cv2.resize(imgs["CAM_FRONT"], (W, H))

# Tokenize
tok = ImagePatchTokenizer(embed_dim=96, patch=patch).eval()
img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
with torch.no_grad():
    t = tok(img_t)             # [1, N, D]
B, N, D = t.shape
Hp, Wp = H//patch, W//patch
mags = t.norm(dim=-1).view(Hp, Wp).cpu().numpy()

# Draw patch grid on image
vis = img.copy()
for y in range(patch, H, patch):
    cv2.line(vis, (0,y), (W,y), (0,255,0), 1)
for x in range(patch, W, patch):
    cv2.line(vis, (x,0), (x,H), (0,255,0), 1)

# Heatmap overlay of token magnitudes
mags_norm = (mags - mags.min()) / (mags.max() - mags.min() + 1e-6)
mags_up = cv2.resize(mags_norm, (W, H), interpolation=cv2.INTER_NEAREST)
heat = (plt.cm.viridis(mags_up)[..., :3] * 255).astype(np.uint8)
overlay = (0.6*vis + 0.4*heat).astype(np.uint8)

os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/viz_image_grid.png", vis)
cv2.imwrite("outputs/viz_token_heat.png", overlay)
print("Saved outputs/viz_image_grid.png and outputs/viz_token_heat.png")
