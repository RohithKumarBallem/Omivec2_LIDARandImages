# scripts/train_contrastive.py
import os, yaml, random, cv2, torch
import numpy as np
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from src.utils.device import get_device
from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer
from src.models.omnivec2_core import OmniVec2Tiny
from src.utils.training import info_nce_loss  # signature: info_nce_loss(z_img, z_lid, tau)

def load_sample(nusc, cams, lidar, H, W, token):
    sample = nusc.get("sample", token)
    sd = nusc.get("sample_data", sample["data"][cams[0]])
    img = cv2.imread(os.path.join(nusc.dataroot, sd["filename"]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H))
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
    If K is provided, randomly samples or repeats to K per sample. Otherwise pads to maxN.
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
    cfg = yaml.safe_load(open("config/dataset.yaml"))
    root = cfg["nuscenes"]["dataroot"]
    ver  = cfg["nuscenes"]["version"]
    cams = cfg["nuscenes"]["cameras"]
    lidar = cfg["nuscenes"]["lidar"]
    H, W = cfg["nuscenes"]["image_size"]

    nusc = NuScenes(version=ver, dataroot=root, verbose=False)
    tokens = collect_tokens(nusc)

    dev = get_device()
    img_tok = ImagePatchTokenizer(embed_dim=96, patch=32).to(dev)
    lid_tok = PointTokenTokenizer(in_ch=4, embed_dim=96, num_tokens=512, mlp_hidden=256).to(dev)
    backbone = OmniVec2Tiny(dim=96, heads=3, ff=192, depth=1).to(dev)  # shared Transformer

    opt = torch.optim.AdamW(
        list(img_tok.parameters()) + list(lid_tok.parameters()) + list(backbone.parameters()),
        lr=3e-4, weight_decay=1e-4
    )

    steps = min(50, len(tokens))
    B = 4
    assert B >= 2, "Batch size must be >= 2 for InfoNCE"

    for step in range(steps):
        ims, clouds = [], []
        for _ in range(B):
            t = tokens[random.randrange(len(tokens))]
            im1, pts1 = load_sample(nusc, cams, lidar, H, W, t)
            ims.append(im1)
            clouds.append(pts1.squeeze(0))

        im = torch.cat(ims, dim=0).to(dev)                  # [B,3,H,W]
        pts = pad_points_batch(clouds, K=None, device=dev)  # [B,maxN,4] or [B,K,4]

        # Tokenize
        t_img = img_tok(im)                                  # [B, T_img, 96]
        t_lid = lid_tok(pts)                                 # [B, 512,    96]
        T_img = t_img.size(1)

        # Shared Transformer fusion: expect backbone to accept (img_tokens, lidar_tokens)
        fused = backbone(t_img, t_lid)                       # either [B,96] or [B, T_img+512, 96]
        if fused.ndim == 2:
            # If backbone outputs pooled embedding, also create per-modality fused emb via residual path
            # Fallback: use pre-fusion pools but this reduces fusion effect; better make backbone return sequence.
            zi_fused = t_img.mean(dim=1)                    # [B,96]
            zl_fused = t_lid.mean(dim=1)                    # [B,96]
            z_scene = fused                                 # [B,96] for logging
        else:
            # Slice fused tokens into image and LiDAR spans, then pool each span
            fused_img = fused[:, :T_img, :]                 # [B, T_img, 96]
            fused_lid = fused[:, T_img:, :]                 # [B, 512,   96]
            zi_fused = fused_img.mean(dim=1)                # [B,96]
            zl_fused = fused_lid.mean(dim=1)                # [B,96]
            z_scene = fused.mean(dim=1)                     # [B,96] optional scene summary

        # Normalize and compute InfoNCE on fused modality pools
        zi = F.normalize(zi_fused, dim=-1)
        zl = F.normalize(zl_fused, dim=-1)
        loss = info_nce_loss(zi, zl, tau=0.5)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(img_tok.parameters()) + list(lid_tok.parameters()) + list(backbone.parameters()), 1.0)
        opt.step()

        if (step + 1) % 5 == 0:
            with torch.no_grad():
                sims = zi @ zl.t()
                sim_pos = sims.diag().mean().item()
                zi_std = zi.std(dim=-1).mean().item()
                zl_std = zl.std(dim=-1).mean().item()
            print(f"step {step+1}/{steps} loss {loss.item():.4f} scene_norm={z_scene.norm(dim=-1).mean().item():.2f} sim+={sim_pos:.3f} zi_std={zi_std:.3f} zl_std={zl_std:.3f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "img_tok": img_tok.state_dict(),
        "lid_tok": lid_tok.state_dict(),
        "backbone": backbone.state_dict(),
    }, "checkpoints/omnivec2_fused_contrastive_96D.pt")
    print("Saved checkpoint to checkpoints/omnivec2_fused_contrastive_96D.pt")

if __name__ == "__main__":
    main()
