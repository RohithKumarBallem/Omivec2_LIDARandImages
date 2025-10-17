# scripts/viz_lidar_tokens.py
import os, yaml, random, torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from src.utils.device import get_device
from src.models.tokenizers import PointTokenTokenizer

def load_lidar(nusc, lidar_channel, token):
    sample = nusc.get("sample", token)
    sd_l = nusc.get("sample_data", sample["data"][lidar_channel])
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_l["filename"]))
    pts = pc.points.T.astype(np.float32)  # [N,4] XYZI
    return pts

def kmeans(x, K, iters=15, seed=0):
    rng = np.random.RandomState(seed)
    N = x.shape[0]
    K = min(K, N)
    idx = rng.choice(N, K, replace=False)
    cent = x[idx].copy()
    for _ in range(iters):
        d = ((x[:, None, :] - cent[None, :, :])**2).sum(-1)  # [N,K]
        a = d.argmin(1)
        for k in range(K):
            sel = (a == k)
            if sel.any():
                cent[k] = x[sel].mean(0)
    d = ((x[:, None, :] - cent[None, :, :])**2).sum(-1)
    a = d.argmin(1)
    return a, cent

def safe_colors_from_clusters(assign, energy):
    # base categorical color in [0,1]
    cmap = plt.get_cmap("tab20")
    base = np.array([cmap(int(i) % 20)[:3] for i in assign], dtype=np.float32)  # [T,3]
    # brightness from energy in [0,1]
    e = (energy - float(energy.min())) / (float(energy.max() - energy.min()) + 1e-6)
    cols = base * (0.6 + 0.4 * e[:, None])  # [0.6,1.0]
    cols = np.clip(cols, 0.0, 1.0).astype(np.float32)
    assert cols.ndim == 2 and cols.shape[1] == 3, f"bad cols shape {cols.shape}"
    assert np.isfinite(cols).all(), "non-finite color values"
    return cols

def main():
    cfg = yaml.safe_load(open("config/dataset.yaml"))
    root = cfg["nuscenes"]["dataroot"]
    ver  = cfg["nuscenes"]["version"]
    lidar = cfg["nuscenes"]["lidar"]

    nusc = NuScenes(version=ver, dataroot=root, verbose=False)

    # collect sample tokens
    tokens = []
    for sc in nusc.scene:
        t = sc["first_sample_token"]
        while t:
            tokens.append(t)
            t = nusc.get("sample", t)["next"]
    token = random.choice(tokens)

    # load lidar
    pts_np = load_lidar(nusc, lidar, token)  # [N,4]
    if pts_np.shape[0] < 10:
        print("Too few points; choose another sample.")
        return
    xyz = pts_np[:, :3]
    inten = pts_np[:, 3:4]

    # tokenize T points
    dev = get_device()
    T = 1024
    tok = PointTokenTokenizer(in_ch=4, embed_dim=96, num_tokens=T).to(dev)
    pts_t = torch.from_numpy(pts_np).unsqueeze(0).float().to(dev)  # [1,N,4]
    with torch.no_grad():
        toks = tok(pts_t)  # [1,T,96]
    toks = toks.squeeze(0).cpu().numpy()  # [T,96]
    tok_energy = np.linalg.norm(toks, axis=1).astype(np.float32)  # [T]

    # pick T raw points (approximate tokenizer sampling)
    N = xyz.shape[0]
    idx = np.random.choice(N, size=T, replace=False if N >= T else True)
    xyz_T = xyz[idx]                 # [T,3]
    inten_T = inten[idx, 0]          # [T]

    # cluster sampled points into K pseudo "patches"
    K = min(32, xyz_T.shape[0])
    assign, centers = kmeans(xyz_T, K, iters=20, seed=0)  # [T], [K,3]

    # colors: categorical hue with energy brightness, guaranteed in [0,1]
    cols = safe_colors_from_clusters(assign, tok_energy)

    os.makedirs("viz", exist_ok=True)

    # 2D BEV scatter (x,y) with explicit facecolors
    assert np.isfinite(xyz_T).all(), "NaNs in coordinates"
    plt.figure(figsize=(6,6))
    plt.scatter(xyz_T[:,0], xyz_T[:,1], s=6, linewidths=0, facecolors=cols, edgecolors='none')
    plt.title("LiDAR tokens: clusters colored, brightness=token norm")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.axis("equal")
    plt.tight_layout()
    plt.savefig("viz/lidar_tokens_bev.png", dpi=200)
    plt.close()

    # Alternative BEV using continuous colormap on energy (fallback)
    fig = plt.figure(figsize=(6,6))
    norm = Normalize(vmin=float(tok_energy.min()), vmax=float(tok_energy.max()))
    sc = plt.scatter(xyz_T[:,0], xyz_T[:,1], s=6, linewidths=0, c=tok_energy, cmap='viridis', norm=norm)
    plt.colorbar(sc, label="token norm")
    plt.title("LiDAR tokens: energy colormap")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.axis("equal")
    plt.tight_layout()
    plt.savefig("viz/lidar_tokens_bev_energy.png", dpi=200)
    plt.close()

    # 3D scatter with facecolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz_T[:,0], xyz_T[:,1], xyz_T[:,2], s=5, linewidths=0, c=cols)
    ax.set_title("LiDAR tokens 3D: clusters colored, brightness=token norm")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig("viz/lidar_tokens_3d.png", dpi=200)
    plt.close()

    print("Saved viz/lidar_tokens_bev.png, viz/lidar_tokens_bev_energy.png, and viz/lidar_tokens_3d.png")

if __name__ == "__main__":
    main()
