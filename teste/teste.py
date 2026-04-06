# =========================================
# DIFFUSION-GAN (STYLEGAN INSPIRED)
# Lung CT (256x256, 1-channel)
# =========================================

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import copy

# =========================================================
# DATASET
# =========================================================

class CTDataset(Dataset):
    def __init__(self, path, max_samples=None):
        self.files = glob.glob(os.path.join(path, "*.npy"))
        if max_samples is not None:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])
        x = torch.tensor(x, dtype=torch.float32)

        # Normalize (important for CT)
        x = (x - x.mean()) / (x.std() + 1e-8)
        x = x.clamp(-3, 3) / 3.0

        x = x.unsqueeze(0)
        return x

# =========================================================
# STYLE MAPPING NETWORK
# =========================================================

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=256, w_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, w_dim), nn.LeakyReLU(0.2),
            nn.Linear(w_dim, w_dim), nn.LeakyReLU(0.2),
            nn.Linear(w_dim, w_dim), nn.LeakyReLU(0.2)
        )

    def forward(self, z):
        return self.net(z)

# =========================================================
# MODULATED CONV
# =========================================================

class ModulatedConv(nn.Module):
    def __init__(self, in_c, out_c, w_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.style = nn.Linear(w_dim, in_c)

    def forward(self, x, w):
        s = self.style(w).unsqueeze(-1).unsqueeze(-1)
        x = x * (s + 1)
        return self.conv(x)

# =========================================================
# GENERATOR (STYLEGAN-LIKE)
# =========================================================

class Generator(nn.Module):
    def __init__(self, z_dim=256, w_dim=256):
        super().__init__()

        self.mapping = MappingNetwork(z_dim, w_dim)

        self.const = nn.Parameter(torch.randn(1, 512, 8, 8))

        def block(in_c, out_c):
            return nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ModulatedConv(in_c, out_c, w_dim),
                nn.LeakyReLU(0.2)
            ])

        self.blocks = nn.ModuleList([
            block(512, 256),
            block(256, 128),
            block(128, 64),
            block(64, 32)
        ])

        self.to_img = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, z):
        w = self.mapping(z)
        x = self.const.repeat(z.size(0), 1, 1, 1)

        for up, conv, act in self.blocks:
            x = up(x)
            x = conv(x, w)
            x = act(x)

        return torch.tanh(self.to_img(x))

# =========================================================
# DISCRIMINATOR (WITH SPECTRAL NORM + T)
# =========================================================

class Discriminator(nn.Module):
    def __init__(self, T, emb_dim=128):
        super().__init__()
        self.embed = nn.Embedding(T, emb_dim)

        def block(in_c, out_c):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1)),
                nn.LeakyReLU(0.2)
            )

        self.net = nn.Sequential(
            block(1 + emb_dim, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.AdaptiveAvgPool2d(1)  
        )

        self.fc = nn.Linear(512, 1)

    def forward(self, x, t):
        t_emb = self.embed(t)[:, :, None, None]
        t_emb = t_emb.expand(-1, -1, x.size(2), x.size(3))

        x = torch.cat([x, t_emb], dim=1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =========================================================
# DIFFUSION
# =========================================================

class Diffusion:
    def __init__(self, T=200, device='cuda'):
        self.T = T
        self.device = device

        beta = torch.linspace(1e-4, 0.02, T).to(device)
        alpha = 1. - beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)

    def sample_t(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device)

    def diffuse(self, x, t):
        noise = torch.randn_like(x)
        a_bar = self.alpha_bar[t].view(-1,1,1,1)
        return torch.sqrt(a_bar)*x + torch.sqrt(1-a_bar)*noise

# =========================================================
# SAVE
# =========================================================

def save_samples(G, epoch, z_dim, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    G.eval()
    with torch.no_grad():
        z = torch.randn(8, z_dim, device=device)
        fake = G(z).cpu().squeeze().numpy()

    fig, axes = plt.subplots(2,4, figsize=(8,4))
    for i in range(2):
        for j in range(4):
            axes[i,j].imshow(fake[i*4+j], cmap='gray')
            axes[i,j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch}.png"))
    plt.close()

# =========================================================
# TRAIN
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

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.99))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.99))

    diffusion = Diffusion(args.T, device)

    for epoch in range(args.epochs):
        for i, real in enumerate(loader):
            real = real.to(device)
            b = real.size(0)

            t = diffusion.sample_t(b)

            # D
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z).detach()

            real_d = diffusion.diffuse(real, t)
            fake_d = diffusion.diffuse(fake, t)

            loss_D = F.softplus(D(fake_d, t)).mean() + F.softplus(-D(real_d, t)).mean()

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # G
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z)
            fake_d = diffusion.diffuse(fake, t)

            loss_G = F.softplus(-D(fake_d, t)).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # EMA
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p_ema * 0.999 + p * 0.001)

            if i % 10 == 0:
                print(f"Epoch {epoch} | Step {i} | D {loss_D.item():.3f} | G {loss_G.item():.3f}")

        save_samples(G_ema, epoch, args.z_dim, device, args.out_dir)

# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--T', type=int, default=200)
    parser.add_argument('--max_samples', type=int, default=None)

    args = parser.parse_args()
    train(args)
