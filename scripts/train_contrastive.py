# scripts/train_contrastive.py
import os, yaml, random, cv2, torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from src.utils.device import get_device
from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer
from src.models.omnivec2_core import OmniVec2Tiny
from src.utils.training import ProjectionHead, info_nce_loss

def load_sample(nusc, cams, lidar, H, W, token):
    sample = nusc.get("sample", token)
    # single camera to keep memory small
    sd = nusc.get("sample_data", sample["data"][cams[0]])
    img = cv2.imread(os.path.join(nusc.dataroot, sd["filename"]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H))
    # lidar
    sd_l = nusc.get("sample_data", sample["data"][lidar])
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_l["filename"]))
    pts = pc.points.T.astype(np.float32)  # [N,4] XYZI

    im = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0  # [1,3,H,W]
    pts = torch.from_numpy(pts).unsqueeze(0).float()  # [1,N,4]
    return im, pts

def collect_tokens(nusc):
    tokens = []
    for sc in nusc.scene:
        t = sc["first_sample_token"]
        while t:
            tokens.append(t)
            samp = nusc.get("sample", t)
            t = samp["next"]
    return tokens

def pad_points_batch(clouds, K=None, device="cpu"):
    """
    clouds: list of [N_i,4] float tensors (CPU)  -> returns [B, maxN or K, 4] float on device.
    If K is provided, randomly samples or repeats to K per sample. Otherwise pads to maxN.  # [attached_file:1]
    """
    B = len(clouds)
    if K is not None:
        out = []
        for pc in clouds:
            N = pc.shape[0]
            if N >= K:
                idx = torch.randint(0, N, (K,))
                out.append(pc[idx])
            else:
                reps = (K + N - 1) // N
                out.append(pc.repeat(reps, 1)[:K])
        return torch.stack(out, dim=0).to(device)
    else:
        maxN = max(pc.shape[0] for pc in clouds)
        pts = torch.zeros(B, maxN, 4, dtype=torch.float32)
        for i, pc in enumerate(clouds):
            pts[i, :pc.shape[0]] = pc
        return pts.to(device)

def main():
    # Config-driven dataset paths and sensors
    cfg = yaml.safe_load(open("config/dataset.yaml"))
    root = cfg["nuscenes"]["dataroot"]
    ver  = cfg["nuscenes"]["version"]
    cams = cfg["nuscenes"]["cameras"]
    lidar = cfg["nuscenes"]["lidar"]
    H, W = cfg["nuscenes"]["image_size"]

    nusc = NuScenes(version=ver, dataroot=root, verbose=False)
    tokens = collect_tokens(nusc)

    dev = get_device()
    # Tokenizers (LiDAR point tokens tuned for variance)
    img_tok = ImagePatchTokenizer(embed_dim=96, patch=32).to(dev)
    lid_tok = PointTokenTokenizer(in_ch=4, embed_dim=96, num_tokens=512, mlp_hidden=256).to(dev)
    # Tiny backbone for optional monitoring
    backbone = OmniVec2Tiny(dim=96, heads=3, ff=192, depth=1).to(dev)
    # Projection heads
    proj_img = ProjectionHead(96, 128).to(dev)
    proj_lid = ProjectionHead(96, 128).to(dev)

    opt = torch.optim.AdamW(
        list(img_tok.parameters())
        + list(lid_tok.parameters())
        + list(proj_img.parameters())
        + list(proj_lid.parameters()),
        lr=3e-4, weight_decay=1e-4
    )

    steps = min(50, len(tokens))
    B = 4  # ensure B >= 2
    assert B >= 2, "Batch size must be >= 2 for InfoNCE"

    for step in range(steps):
        # Build batch with padding (handles variable N for LiDAR)
        ims, clouds = [], []
        for _ in range(B):
            t = tokens[random.randrange(len(tokens))]
            im1, pts1 = load_sample(nusc, cams, lidar, H, W, t)  # im1: [1,3,H,W], pts1: [1,N_i,4]
            ims.append(im1)
            clouds.append(pts1.squeeze(0))  # [N_i,4] on CPU

        im = torch.cat(ims, dim=0).to(dev)              # [B,3,H,W]
        pts = pad_points_batch(clouds, K=None, device=dev)  # [B,maxN,4] or [B,K,4]

        ti = img_tok(im)                                # [B, N_img, 96]
        tl = lid_tok(pts)                               # [B, 512, 96]

        # Optional fused embedding for logging
        with torch.no_grad():
            z_fused = backbone(ti, tl)                  # [B, 96]

        # Per-modality pooling and InfoNCE
        zi_pool = ti.mean(dim=1)                        # [B, 96]
        zl_pool = tl.mean(dim=1)                        # [B, 96]
        li = proj_img(zi_pool)                          # [B, 128]
        ll = proj_lid(zl_pool)                          # [B, 128]

        loss = info_nce_loss(li, ll, tau=0.5)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % 5 == 0:
            with torch.no_grad():
                li_std = li.std(dim=-1).mean().item()
                ll_std = ll.std(dim=-1).mean().item()
            print(f"step {step+1}/{steps} loss {loss.item():.4f} fused_norm={z_fused.norm().mean().item():.2f} li_std={li_std:.4f} ll_std={ll_std:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "img_tok": img_tok.state_dict(),
        "lid_tok": lid_tok.state_dict(),
        "proj_img": proj_img.state_dict(),
        "proj_lid": proj_lid.state_dict(),
    }, "checkpoints/omnivec2_mini_contrastive.pt")
    print("Saved checkpoint to checkpoints/omnivec2_mini_contrastive.pt")

if __name__ == "__main__":
    main()
