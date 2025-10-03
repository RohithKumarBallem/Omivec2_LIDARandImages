import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=96, proj_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
    def forward(self, x):
        return self.net(x)

def cosine_sim(a, b, eps=1e-8):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()

def info_nce_loss(z_img, z_lid, tau=0.1):
    # z_*: [B,D], positives are diagonal
    sim = cosine_sim(z_img, z_lid) / tau         # [B,B]
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_i = F.cross_entropy(sim, labels)
    loss_l = F.cross_entropy(sim.t(), labels)
    return 0.5 * (loss_i + loss_l)
