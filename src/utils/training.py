import torch  # [attached_file:1]
import torch.nn as nn  # [attached_file:1]
import torch.nn.functional as F  # [attached_file:1]

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=96, proj_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )  # [attached_file:1]
    def forward(self, x):
        return self.net(x)  # [attached_file:1]

def cosine_sim(a, b, eps=1e-8):
    a = F.normalize(a, dim=-1)  # [attached_file:1]
    b = F.normalize(b, dim=-1)  # [attached_file:1]
    return a @ b.t()  # [attached_file:1]

def info_nce_loss(z_img, z_lid, tau=0.1):
    sim = cosine_sim(z_img, z_lid) / tau  # [attached_file:1]
    labels = torch.arange(sim.size(0), device=sim.device)  # [attached_file:1]
    loss_i = F.cross_entropy(sim, labels)  # [attached_file:1]
    loss_l = F.cross_entropy(sim.t(), labels)  # [attached_file:1]
    return 0.5 * (loss_i + loss_l)  # [attached_file:1]
