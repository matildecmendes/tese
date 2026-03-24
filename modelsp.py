import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# RES BLOCK (GENERATOR)
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super().__init__()
        self.up = up

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        h = x

        if self.up:
            h = F.interpolate(h, scale_factor=2)
            x = F.interpolate(x, scale_factor=2)

        h = self.conv1(h)
        noise = torch.randn_like(h) * self.noise_strength
        h = F.leaky_relu(h + noise, 0.2)

        h = F.leaky_relu(self.conv2(h), 0.2)
        return h + self.skip(x)


# -------------------------
# RES BLOCK (DISCRIMINATOR)
# -------------------------
class DiscResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=False):
        super().__init__()
        self.down = down

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        h = x

        h = F.leaky_relu(self.conv1(h), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)

        if self.down:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)

        return h + self.skip(x)


# -------------------------
# GENERATOR
# -------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.fc = nn.Linear(z_dim, 512 * 4 * 4)

        self.blocks = nn.Sequential(
            ResBlock(512, 256, up=True),
            ResBlock(256, 128, up=True),
            ResBlock(128, 64, up=True),
            ResBlock(64, 32, up=True),
            ResBlock(32, 16, up=True),
            ResBlock(16, 8, up=True),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 512, 4, 4)
        h = self.blocks(h)
        return torch.tanh(self.to_img(h))


# -------------------------
# DISCRIMINATOR
# -------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList([
            DiscResBlock(1, 32, down=True),
            DiscResBlock(32, 64, down=True),
            DiscResBlock(64, 128, down=True),
            DiscResBlock(128, 256, down=True),
            DiscResBlock(256, 512, down=True),
        ])

        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc = nn.Linear(512*4*4, 1)

    def forward(self, x):
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)

        x = self.pool(x)
        out = self.fc(x.view(x.size(0), -1))
        return out, feats
