import os, yaml
from src.data.nuscenes_loader import NuScenesMiniSample

cfg = yaml.safe_load(open("config/dataset.yaml"))
root = cfg["nuscenes"]["dataroot"]
ver = cfg["nuscenes"]["version"]
cams = cfg["nuscenes"]["cameras"]
lidar = cfg["nuscenes"]["lidar"]

assert os.path.isdir(root), f"nuScenes root not found: {root}"
ds = NuScenesMiniSample(root, ver, cams, lidar)
tok, imgs, pts = ds.get_first_sample()
print("Sample:", tok)
print("Images:", {k: (v.shape if v is not None else None) for k,v in imgs.items()})
print("LiDAR points:", pts.shape)
