import torch
import torch.nn as nn
import torch.nn.functional as F

class ImagePatchTokenizer(nn.Module):
    def __init__(self, in_ch=3, embed_dim=96, patch=32):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        t = self.proj(x)                         # [B, D, H/P, W/P]
        B, D, Hp, Wp = t.shape
        t = t.permute(0,2,3,1).reshape(B, Hp*Wp, D)
        return self.norm(t)                      # [B, N, D]

class LidarPillarTokenizer(nn.Module):
    def __init__(self, x_range=(-50.0, 50.0), y_range=(-50.0, 50.0), cell=1.0, embed_dim=96):
        super().__init__()
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.cell = float(cell)

        H = int(round((self.y1 - self.y0) / self.cell))
        W = int(round((self.x1 - self.x0) / self.cell))
        self.H, self.W = H, W

        self.mlp = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, points):
        assert points.dim() == 3 and points.size(-1) >= 4
        B, N, _ = points.shape
        device = points.device
        H, W = self.H, self.W
        K = H * W

        xs = ((points[..., 0] - self.x0) / self.cell)
        ys = ((points[..., 1] - self.y0) / self.cell)
        xi = xs.floor().long()
        yi = ys.floor().long()
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)

        sum_feat = torch.zeros(B, H, W, 4, device=device)
        count = torch.zeros(B, H, W, 1, device=device)

        for b in range(B):
            v = valid[b]
            if v.sum() == 0:
                continue
            xb = xi[b][v]
            yb = yi[b][v]
            pts = points[b][v, :4]      # [M,4]
            flat = yb * W + xb          # [M]

            sum_flat = torch.zeros(K, 4, device=device)
            cnt_flat = torch.zeros(K, 1, device=device)

            sum_flat = sum_flat.index_add(0, flat, pts)
            ones = torch.ones(flat.numel(), 1, device=device)
            cnt_flat = cnt_flat.index_add(0, flat, ones)

            sum_feat[b] = sum_flat.view(H, W, 4)
            count[b] = cnt_flat.view(H, W, 1)

        denom = torch.clamp(count, min=1.0)
        mean_feat = sum_feat / denom
        feat = torch.cat([mean_feat, count], dim=-1)  # [B,H,W,5]
        tok = self.mlp(feat).view(B, H*W, -1)
        return self.norm(tok)

