import os, yaml, random, cv2, torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from src.utils.device import get_device
from src.models.tokenizers import ImagePatchTokenizer, LidarPillarTokenizer
from src.models.omnivec2_core import OmniVec2Tiny
from src.utils.training import ProjectionHead, info_nce_loss

def load_sample(nusc, cams, lidar, H, W, token):
    sample = nusc.get("sample", token)
    imgs = []
    for cam in [cams[0]]:  # single cam for memory
        sd = nusc.get("sample_data", sample["data"][cam])
        img = cv2.imread(os.path.join(nusc.dataroot, sd["filename"]))
        img = cv2.resize(img, (W, H))
        imgs.append(img)
    from nuscenes.utils.data_classes import LidarPointCloud
    sd_l = nusc.get("sample_data", sample["data"][lidar])
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_l["filename"]))
    pts = pc.points.T.astype(np.float32)  # [N,4]
    im = np.concatenate(imgs, axis=0) if len(imgs) > 1 else imgs[0]
    im = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).float() / 255.0
    pts = torch.from_numpy(pts).unsqueeze(0).float()
    return im, pts

def main():
    cfg = yaml.safe_load(open("config/dataset.yaml"))
    root = cfg["nuscenes"]["dataroot"]
    ver  = cfg["nuscenes"]["version"]
    cams = cfg["nuscenes"]["cameras"]
    lidar = cfg["nuscenes"]["lidar"]
    H, W = cfg["nuscenes"]["image_size"]

    nusc = NuScenes(version=ver, dataroot=root, verbose=False)
    tokens = []
    for sc in nusc.scene:
        t = sc["first_sample_token"]
        while t != "":
            tokens.append(t)
            samp = nusc.get("sample", t)
            t = samp["next"]

    dev = get_device()
    img_tok = ImagePatchTokenizer(embed_dim=96, patch=32).to(dev)
    lid_tok = LidarPillarTokenizer(cell=1.0, embed_dim=96).to(dev)
    backbone = OmniVec2Tiny(dim=96, heads=3, ff=192, depth=1).to(dev)
    proj_img = ProjectionHead(96, 96).to(dev)
    proj_lid = ProjectionHead(96, 96).to(dev)

    opt = torch.optim.AdamW(list(img_tok.parameters())+
                            list(lid_tok.parameters())+
                            list(backbone.parameters())+
                            list(proj_img.parameters())+
                            list(proj_lid.parameters()), lr=2e-4, weight_decay=1e-4)

    steps = min(50, len(tokens))  # short run on mini
    backbone.train()
    for step in range(steps):
        # pick random token
        t = tokens[random.randrange(len(tokens))]
        im, pts = load_sample(nusc, cams, lidar, H, W, t)
        im, pts = im.to(dev), pts.to(dev)

        zi = img_tok(im)             # [B, Ni, D]
        zl = lid_tok(pts)            # [B, Nl, D]
        z  = backbone(zi, zl)        # [B, D]
        zi_proj = proj_img(z)
        zl_proj = proj_lid(z)

        loss = info_nce_loss(zi_proj, zl_proj, tau=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step+1) % 5 == 0:
            print(f"step {step+1}/{steps} loss {loss.item():.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "img_tok": img_tok.state_dict(),
        "lid_tok": lid_tok.state_dict(),
        "backbone": backbone.state_dict(),
        "proj_img": proj_img.state_dict(),
        "proj_lid": proj_lid.state_dict(),
    }, "checkpoints/omnivec2_mini_contrastive.pt")
    print("Saved checkpoint to checkpoints/omnivec2_mini_contrastive.pt")

if __name__ == "__main__":
    main()
