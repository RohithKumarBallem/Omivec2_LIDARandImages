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

# src/models/tokenizers.py (append this next to ImagePatchTokenizer)

class PointTokenTokenizer(nn.Module):
    """
    - Inputs: points [B, N, C] with columns (x, y, z, intensity, ...).
    - Outputs: tokens [B, T, D], where T is a fixed token count via subsampling.
    """
    def __init__(self, in_ch=4, embed_dim=96, num_tokens=1024, mlp_hidden=128, dropout=0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(in_ch, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, points):
        """
        points: [B, N, C] tensor, typically C>=4 = (x,y,z,intensity)
        returns: [B, T, D] tokens
        """
        assert points.dim() == 3, "points must be [B, N, C]"
        B, N, C = points.shape
        T = min(self.num_tokens, N)
        device = points.device

        # Random per-batch subsample indices to get a fixed-length token sequence.
        # If N < T, we’ll duplicate some indices to pad up to T.
        if N >= T:
            idx = torch.randint(0, N, (B, T), device=device)
        else:
            # repeat and trim to length T
            base = torch.arange(N, device=device).unsqueeze(0).repeat(B, (T + N - 1) // N)
            idx = base[:, :T]

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
        pts = points[batch_idx, idx]                  # [B, T, C]

        # Optional simple normalization: center and scale coordinates per batch
        xyz = pts[..., :3]
        mean = xyz.mean(dim=1, keepdim=True)
        std = xyz.std(dim=1, keepdim=True).clamp_min(1e-6)
        xyz_norm = (xyz - mean) / std
        if C > 3:
            feats = torch.cat([xyz_norm, pts[..., 3:]], dim=-1)
        else:
            feats = xyz_norm

        tok = self.proj(feats)                        # [B, T, D]
        tok = self.drop(tok)
        return self.norm(tok)                         # [B, T, D]

