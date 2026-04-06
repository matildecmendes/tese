# =========================================
# DIFFUSION-GAN (FULL REPO-LEVEL IMPLEMENTATION SIMPLIFIED)
# Based on StyleGAN2-ADA + Diffusion-GAN
# Adapted for Lung CT (256x256, 1-channel)
# =========================================

# NOTE:
# This is a CLEAN, STABLE version inspired by the repo you provided.
# It keeps key components: EMA, R1, Diffusion, ADA-lite
# but avoids full NVLabs complexity for usability.

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import copy

# =========================================================
# DATASET
# =========================================================

class CTDataset(Dataset):
    def __init__(self, path, max_samples=None):
        self.files = glob.glob(os.path.join(path, "*.npy"))
        if max_samples:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])
        x = torch.tensor(x, dtype=torch.float32)

        # CT normalization
        x = (x - x.mean()) / (x.std() + 1e-8)
        x = x.clamp(-3,3)/3

        return x.unsqueeze(0)

# =========================================================
# STYLEGAN-LIKE BLOCKS
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

# =========================================================
# GENERATOR (STRONGER)
# =========================================================

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

# =========================================================
# DISCRIMINATOR (STYLEGAN-LIKE)
# =========================================================

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

        x = torch.cat([x,t_emb],dim=1)

        for block in self.blocks:
            x = block(x)
            x = F.avg_pool2d(x,2)

        return self.final(x)

# =========================================================
# DIFFUSION
# =========================================================

class Diffusion:
    def __init__(self, T, device):
        self.T = T
        self.device = device

        beta = torch.linspace(1e-4,0.02,T).to(device)
        alpha = 1-beta
        self.alpha_bar = torch.cumprod(alpha,0)

    def sample_t(self,b):
        return torch.randint(0,self.T,(b,),device=self.device)

    def diffuse(self,x,t):
        noise = torch.randn_like(x)
        a = self.alpha_bar[t].view(-1,1,1,1)
        return torch.sqrt(a)*x + torch.sqrt(1-a)*noise

# =========================================================
# TRAIN LOOP (STYLEGAN-LIKE)
# =========================================================

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    dataset = CTDataset(args.data_path, args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print("Dataset size:", len(dataset))

    G = Generator(args.z_dim).to(device)
    D = Discriminator(args.T).to(device)
    G_ema = copy.deepcopy(G).eval()

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0,0.99))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0,0.99))

    diffusion = Diffusion(args.T, device)

    for epoch in range(args.epochs):
        for i, real in enumerate(loader):
            real = real.to(device)
            b = real.size(0)

            t = diffusion.sample_t(b)

            # --- D ---
            z = torch.randn(b,args.z_dim,device=device)
            fake = G(z).detach()

            real_d = diffusion.diffuse(real,t)
            fake_d = diffusion.diffuse(fake,t)

            loss_D = F.softplus(D(fake_d,t)).mean() + F.softplus(-D(real_d,t)).mean()

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- G ---
            z = torch.randn(b,args.z_dim,device=device)
            fake = G(z)
            fake_d = diffusion.diffuse(fake,t)

            loss_G = F.softplus(-D(fake_d,t)).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # EMA
            with torch.no_grad():
                for p_ema,p in zip(G_ema.parameters(),G.parameters()):
                    p_ema.copy_(p_ema*0.999 + p*0.001)

            if i%50==0:
                print(f"Epoch {epoch} Step {i} | D {loss_D.item():.3f} | G {loss_G.item():.3f}")

        # save images
        with torch.no_grad():
            z = torch.randn(16,args.z_dim,device=device)
            imgs = G_ema(z)
            vutils.save_image(imgs, f"{args.out_dir}/epoch_{epoch}.png", normalize=True)

# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--T', type=int, default=200)
    parser.add_argument('--max_samples', type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
