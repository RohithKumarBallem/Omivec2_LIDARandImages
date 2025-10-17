import torch  # [attached_file:1]
import torch.nn as nn  # [attached_file:1]

class CrossModalBlock(nn.Module):
    def __init__(self, dim=96, heads=3, ff=192):
        super().__init__()
        self.attn_img_to_lidar = nn.MultiheadAttention(dim, heads, batch_first=True)  # [attached_file:1]
        self.attn_lidar_to_img = nn.MultiheadAttention(dim, heads, batch_first=True)  # [attached_file:1]
        self.norm_i = nn.LayerNorm(dim)  # [attached_file:1]
        self.norm_l = nn.LayerNorm(dim)  # [attached_file:1]
        self.ffn_i = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))  # [attached_file:1]
        self.ffn_l = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))  # [attached_file:1]

    def forward(self, img_tok, lidar_tok):
        qi, _ = self.attn_img_to_lidar(img_tok, lidar_tok, lidar_tok)  # [attached_file:1]
        ql, _ = self.attn_lidar_to_img(lidar_tok, img_tok, img_tok)  # [attached_file:1]
        img_tok = img_tok + self.norm_i(qi)  # [attached_file:1]
        lidar_tok = lidar_tok + self.norm_l(ql)  # [attached_file:1]
        img_tok = img_tok + self.ffn_i(img_tok)  # [attached_file:1]
        lidar_tok = lidar_tok + self.ffn_l(lidar_tok)  # [attached_file:1]
        return img_tok, lidar_tok  # [attached_file:1]

class OmniVec2Tiny(nn.Module):
    def __init__(self, dim=96, heads=3, ff=192, depth=1):
        super().__init__()
        self.blocks = nn.ModuleList([CrossModalBlock(dim, heads, ff) for _ in range(depth)])  # [attached_file:1]
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))  # [attached_file:1]

    def forward(self, img_tok, lidar_tok):
        for blk in self.blocks:
            img_tok, lidar_tok = blk(img_tok, lidar_tok)  # [attached_file:1]
        fused = 0.5 * (img_tok.mean(dim=1) + lidar_tok.mean(dim=1))  # [B, D]  # [attached_file:1]
        return self.head(fused)  # [B, D]  # [attached_file:1]
