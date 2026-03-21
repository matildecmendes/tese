import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================
# TIME EMBEDDING
# ======================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        t = t.float().unsqueeze(-1)
        return self.linear(t)


# ======================
# BASIC BLOCK
# ======================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# ======================
# GENERATOR (U-NET)
# ======================
class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()

        self.nz = nz
        self.time_embed = TimeEmbedding(64)

        # Encoder
        self.enc1 = ConvBlock(1 + 64, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 256)

        # Decoder
        self.dec3 = ConvBlock(256 + 256, 128)
        self.dec2 = ConvBlock(128 + 128, 64)
        self.dec1 = ConvBlock(64 + 64, 64)

        self.final = nn.Conv2d(64, 1, 1)

        # z embedding
        self.z_fc = nn.Linear(nz, 256)

        self.pool = nn.MaxPool2d(2)

    def forward(self, xt, t, z):
        B, _, H, W = xt.shape

        # ===== time embedding =====
        t_emb = self.time_embed(t)
        t_emb = t_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)

        x = torch.cat([xt, t_emb], dim=1)

        # ===== encoder =====
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # ===== bottleneck + z =====
        b = self.bottleneck(self.pool(e3))

        z_emb = self.z_fc(z).view(B, 256, 1, 1)
        b = b + z_emb

        # ===== decoder =====
        d3 = F.interpolate(b, scale_factor=2)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = torch.tanh(self.final(d1))

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