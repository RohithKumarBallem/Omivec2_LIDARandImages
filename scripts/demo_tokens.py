import os  # [attached_file:1]
import torch  # [attached_file:1]
from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer  # [attached_file:1]
from src.models.omnivec2_core import OmniVec2Tiny  # [attached_file:1]
from src.data.nuscenes_loader import NuScenesMiniSample  # assumes your existing loader [attached_file:1]

def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"  # [attached_file:1]
    device = torch.device(dev)  # [attached_file:1]

    # Minimal sample fetch (adapt to your loader API)
    ds = NuScenesMiniSample(...)  # fill with your existing params [attached_file:1]
    token, imgs_dict, points = ds.get_first_sample()  # [attached_file:1]

    # Preprocess to tensors
    imgs = torch.stack([torch.from_numpy(imgs_dict[k]).permute(2,0,1) for k in sorted(imgs_dict.keys())], dim=0).float() / 255.0  # [attached_file:1]
    # Just use the first camera for demo
    img = imgs[0:1].to(device)  # [1, 3, H, W] [attached_file:1]
    pts = torch.from_numpy(points).unsqueeze(0).to(device).float()  # [1, N, C] [attached_file:1]

    img_tok = ImagePatchTokenizer(embed_dim=96, patch=32).to(device)  # [attached_file:1]
    lid_tok = PointTokenTokenizer(in_ch=4, embed_dim=96, num_tokens=1024).to(device)  # [attached_file:1]
    backbone = OmniVec2Tiny(dim=96, heads=3, ff=192, depth=1).to(device)  # [attached_file:1]

    with torch.no_grad():
        ti = img_tok(img)      # [1, N_img, 96] [attached_file:1]
        tl = lid_tok(pts)      # [1, 1024, 96] [attached_file:1]
        z = backbone(ti, tl)   # [1, 96] [attached_file:1]

    print("Image tokens:", ti.shape, "LiDAR tokens:", tl.shape, "Fused:", z.shape)  # [attached_file:1]

if __name__ == "__main__":
    main()  # [attached_file:1]
