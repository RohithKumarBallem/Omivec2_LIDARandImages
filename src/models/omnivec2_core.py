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
    """
    Shared multimodal Transformer encoder.
    - Inputs: image_tokens [B, T_img, D], lidar_tokens [B, T_lid, D]
    - Adds learned modality/type embeddings to each span
    - Runs TransformerEncoder over concatenated sequence
    - Returns fused token sequence [B, T_img+T_lid, D]
    """
    def __init__(self, dim=96, heads=3, ff=192, depth=1, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.type_embed = nn.Parameter(torch.randn(2, dim) * 0.02)  # [2,D] for (img, lidar)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(dim)

        # Optional positional encoding (sinusoidal)
        self.use_pos = True
        self.pos_cache = {}

    def _sinusoidal_pos(self, B, T, D, device):
        key = (T, D, device)
        if key in self.pos_cache:
            return self.pos_cache[key]
        pos = torch.arange(T, device=device).unsqueeze(1)  # [T,1]
        i = torch.arange(D, device=device).unsqueeze(0)    # [1,D]
        div = torch.exp(-torch.arange(0, D, 2, device=device) * (torch.log(torch.tensor(10000.0, device=device)) / D))
        pe = torch.zeros(T, D, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0).expand(B, T, D)               # [B,T,D]
        self.pos_cache[key] = pe
        return pe

    def forward(self, img_tok: torch.Tensor, lid_tok: torch.Tensor):
        """
        img_tok: [B, T_img, D]
        lid_tok: [B, T_lid, D]
        returns: fused [B, T_img+T_lid, D]
        """
        B, T_img, D = img_tok.shape
        _, T_lid, D2 = lid_tok.shape
        assert D == self.dim and D2 == self.dim, "Token dims must match model dim"

        # Add modality/type embeddings
        img_add = self.type_embed[0].unsqueeze(0).unsqueeze(1).expand(B, T_img, D)
        lid_add = self.type_embed[1].unsqueeze(0).unsqueeze(1).expand(B, T_lid, D)
        img = img_tok + img_add
        lid = lid_tok + lid_add

        # Concatenate
        seq = torch.cat([img, lid], dim=1)                 # [B, T_img+T_lid, D]

        # Positional encoding
        if self.use_pos:
            pos = self._sinusoidal_pos(B, seq.size(1), D, seq.device)
            seq = seq + pos

        # Transformer encoder
        fused = self.encoder(seq)                           # [B, T_img+T_lid, D]
        fused = self.out_norm(fused)
        return fused
