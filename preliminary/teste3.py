# =============================
# DIFFUSION-GAN
# Lung CT (1-channel, 256x256)
# =============================

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# =========================================================
# DATASET
# =========================================================

class CTDataset(Dataset):
    def __init__(self, path, max_samples=None):
        self.files = glob.glob(os.path.join(path, "*.npy"))
        
        if max_samples is not None:
            self.files = self.files[:max_samples]

        #print(f"[DATASET] Found {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])  # (256,256)
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(0)  # (1,256,256)
        return x

# =========================================================
# SIMPLE GENERATOR (StyleGAN2-like)
# =========================================================

class Generator(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.fc = nn.Linear(z_dim, 512*8*8)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 512, 8, 8)
        return self.net(x)

# =========================================================
# DISCRIMINATOR WITH T-EMBEDDING
# =========================================================

class Discriminator(nn.Module):
    def __init__(self, T, emb_dim=256):
        super().__init__()
        self.embed = nn.Embedding(T, emb_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(1 + emb_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Linear(512*16*16, 1)

    def forward(self, x, t):
        t_emb = self.embed(t)
        t_emb = t_emb[:, :, None, None]
        t_emb = t_emb.expand(-1, -1, x.size(2), x.size(3))

        x = torch.cat([x, t_emb], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =========================================================
# DIFFUSION PROCESS
# =========================================================

class Diffusion:
    def __init__(self, T=1000, device='cuda'):
        self.T = T
        self.device = device

        beta = torch.linspace(1e-4, 0.02, T).to(device)
        alpha = 1. - beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)

    def sample_t(self, batch_size):
        weights = torch.arange(1, self.T+1, device=self.device).float()
        weights = weights / weights.sum()
        return torch.multinomial(weights, batch_size, replacement=True)

    def diffuse(self, x, t):
        noise = torch.randn_like(x)
        a_bar = self.alpha_bar[t].view(-1,1,1,1)
        return torch.sqrt(a_bar)*x + torch.sqrt(1-a_bar)*noise
    
# =========================================================
# SAVE SAMPLES
# =========================================================
def save_samples(G, D, epoch, z_dim, device, out_dir):
    G.eval()

    with torch.no_grad():
        z = torch.randn(8, z_dim, device=device)
        fake = G(z).cpu()

    imgs = fake.squeeze().numpy()

    fig, axes = plt.subplots(2,4, figsize=(8,4))

    for i in range(2):
        for j in range(4):
            idx = i*4+j
            axes[i,j].imshow(imgs[idx], cmap="gray")
            axes[i,j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch}.png"))
    plt.close()

    # guardar modelos a cada epoch
    checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict()
    }, checkpoint_path)

    print(f"[SAVE] Epoch {epoch} saved")

# =========================================================
# TRAIN
# =========================================================

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    dataset = CTDataset(args.data_path, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print("Dataset size:", len(dataset))

    z_dim=args.z_dim

    G = Generator().to(device)
    D = Discriminator(T=args.T).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.99))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.99))

    diffusion = Diffusion(T=args.T, device=device)

    print("Starting training loop...")

    for epoch in range(args.epochs):
        print("Entered dataloader loop")

        for i, real in enumerate(loader):
            real = real.to(device)
            b = real.size(0)

            # =====================
            # SAMPLE TIMESTEPS
            # =====================
            t = diffusion.sample_t(b)

            # =====================
            # TRAIN D
            # =====================
            z = torch.randn(b, args.z_dim).to(device)
            fake = G(z).detach()

            real_diff = diffusion.diffuse(real, t)
            fake_diff = diffusion.diffuse(fake, t)

            D_real = D(real_diff, t)
            D_fake = D(fake_diff, t)

            loss_D = F.softplus(D_fake).mean() + F.softplus(-D_real).mean()

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # =====================
            # TRAIN G
            # =====================
            z = torch.randn(b, args.z_dim).to(device)
            fake = G(z)
            fake_diff = diffusion.diffuse(fake, t)

            D_fake = D(fake_diff, t)
            loss_G = F.softplus(-D_fake).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if i % 10 == 0:
                print(f"Epoch {epoch} | Step {i} | D {loss_D.item():.3f} | G {loss_G.item():.3f}")

        save_samples(G, D, epoch, z_dim, device, args.out_dir)

# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()
    train(args)
