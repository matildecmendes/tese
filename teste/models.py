import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Generator ----------

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()

        self.fc = nn.Linear(z_dim, 512*4*4)

        self.blocks = nn.ModuleList([
            ConvBlock(512,512),
            ConvBlock(512,256),
            ConvBlock(256,128),
            ConvBlock(128,64),
            ConvBlock(64,32)
        ])

        self.to_img = nn.Conv2d(32,1,1)

    def forward(self, z):
        x = self.fc(z).view(-1,512,4,4)

        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = block(x)

        return torch.tanh(self.to_img(x))


# ---------- Discriminator ----------

class Discriminator(nn.Module):
    def __init__(self, T):
        super().__init__()

        self.embed = nn.Embedding(T, 64)

        self.blocks = nn.ModuleList([
            ConvBlock(1+64,64),
            ConvBlock(64,128),
            ConvBlock(128,256),
            ConvBlock(256,512)
        ])

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512,1)
        )

    def forward(self, x, t):
        t_emb = self.embed(t)[:,:,None,None]
        t_emb = t_emb.expand(-1,-1,x.size(2),x.size(3))

        x = torch.cat([x, t_emb], dim=1)

        for block in self.blocks:
            x = block(x)
            x = F.avg_pool2d(x,2)

        return self.final(x)
