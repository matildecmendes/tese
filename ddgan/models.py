import torch
import torch.nn as nn
import torch.nn.functional as F

# TIME EMBEDDING

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )


    def forward(self, t):
        t = t.float().unsqueeze(-1)
        return self.net(t)



# GENERATOR RESBLOCK

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super().__init__()
        self.up = up


        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)


        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        self.noise_strength = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        h = x


        if self.up:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")


        h = self.conv1(h)


        noise = torch.randn_like(h) * self.noise_strength
        h = F.leaky_relu(h + noise, 0.2)


        h = F.leaky_relu(self.conv2(h), 0.2)


        return h + self.skip(x)



# GENERATOR

class Generator(nn.Module):
    def __init__(self, nz=256):
        super().__init__()


        self.nz = nz
        self.time_embed = TimeEmbedding(64)


        # input: xt (1) + t (64)
        self.input_conv = nn.Conv2d(1 + 64, 64, 3, padding=1)


        self.blocks = nn.Sequential(
            ResBlock(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 64),
            ResBlock(64, 64),
        )


        self.to_img = nn.Conv2d(64, 1, 3, padding=1)


        self.z_fc = nn.Linear(nz, 64)


    def forward(self, xt, t, z):
        B, _, H, W = xt.shape


        # time embedding
        t_emb = self.time_embed(t)
        t_emb = t_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)


        x = torch.cat([xt, t_emb], dim=1)
        x = self.input_conv(x)


        h = self.blocks(x)


        # inject z
        z_emb = self.z_fc(z).view(B, 64, 1, 1)
        h = h + z_emb


        out = torch.tanh(self.to_img(h))


        return out



# DISCRIMINATOR BLOCK

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




# DISCRIMINATOR

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()


        self.time_embed = TimeEmbedding(64)


        # input: xt (1) + xt_plus (1) + t (64)
        self.input_conv = nn.Conv2d(2 + 64, 64, 3, padding=1)


        self.blocks = nn.ModuleList([
            DiscResBlock(64, 128, down=True),
            DiscResBlock(128, 256, down=True),
            DiscResBlock(256, 512, down=True),
        ])


        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(512 * 4 * 4, 1)


    def forward(self, xt, t, xt_plus):
        B, _, H, W = xt.shape


        # time embedding
        t_emb = self.time_embed(t)
        t_emb = t_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)


        x = torch.cat([xt * 1.0, xt_plus, t_emb], dim=1)
        x = self.input_conv(x)


        for block in self.blocks:
            x = block(x)


        x = self.pool(x)
        out = self.fc(x.view(B, -1))


        return out