import torch
import torch.nn as nn
import torch.nn.functional as F

# =================
# TIME EMBEDDING
# =================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(1, dim)

    def forward(self, t):
        t = t.float().unsqueeze(-1)
        return self.linear(t)


# =============
# GENERATOR 
# =============
class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()

        self.nz = nz
        self.time_embed = TimeEmbedding(32)

        self.conv1 = nn.Conv2d(1 + 32, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)

        self.z_fc = nn.Linear(nz, 128)

    def forward(self, xt, t, z):
        B, _, H, W = xt.shape

        # time embedding
        t_emb = self.time_embed(t)
        t_emb = t_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)

        x = torch.cat([xt, t_emb], dim=1)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))

        # inject z
        z_emb = self.z_fc(z).view(B, 128, 1, 1)
        h = h + z_emb

        h = F.relu(self.conv3(h))
        out = torch.tanh(self.conv4(h))

        return out


# ======================
# DISCRIMINATOR
# ======================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embed = TimeEmbedding(32)

        self.net = nn.Sequential(
            nn.Conv2d(2 + 32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 64 * 64, 1)
        )

    def forward(self, xt, t, xt_plus):
        B, _, H, W = xt.shape

        t_emb = self.time_embed(t)
        t_emb = t_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)

        x = torch.cat([xt, xt_plus, t_emb], dim=1)

        return self.net(x)