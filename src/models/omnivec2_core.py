import torch
import torch.nn as nn

class CrossModalBlock(nn.Module):
    def __init__(self, dim=96, heads=3, ff=192):
        super().__init__()
        self.attn_img_to_lidar = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_lidar_to_img = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_i = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)
        self.ffn_i = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
        self.ffn_l = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))

    def forward(self, img_tok, lidar_tok):
        # Cross attention both ways
        qi, _ = self.attn_img_to_lidar(img_tok, lidar_tok, lidar_tok)  # [B, Ni, D]
        ql, _ = self.attn_lidar_to_img(lidar_tok, img_tok, img_tok)    # [B, Nl, D]
        img_tok = img_tok + self.norm_i(qi)
        lidar_tok = lidar_tok + self.norm_l(ql)
        img_tok = img_tok + self.ffn_i(img_tok)
        lidar_tok = lidar_tok + self.ffn_l(lidar_tok)
        return img_tok, lidar_tok

class OmniVec2Tiny(nn.Module):
    def __init__(self, dim=96, heads=3, ff=192, depth=1):
        super().__init__()
        self.blocks = nn.ModuleList([CrossModalBlock(dim, heads, ff) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
    def forward(self, img_tok, lidar_tok):
        for blk in self.blocks:
            img_tok, lidar_tok = blk(img_tok, lidar_tok)
        fused = 0.5 * (img_tok.mean(dim=1) + lidar_tok.mean(dim=1))  # [B,D]
        return self.head(fused)
