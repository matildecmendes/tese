# =========================================================
# DIFFUSION-GAN FOR LUNG CT 
# =========================================================
import os
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import copy

from dataset import CTDataset
from models import Generator, Discriminator
from diffusion import Diffusion


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

            # Sample timestep
            t = diffusion.sample_t(b)

            # ------------------
            # Train Discriminator
            # ------------------
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z).detach()

            real_d = diffusion.diffuse(real, t)
            fake_d = diffusion.diffuse(fake, t)

            loss_D = F.softplus(D(fake_d,t)).mean() + F.softplus(-D(real_d,t)).mean()

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ------------------
            # Train Generator
            # ------------------
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z)
            fake_d = diffusion.diffuse(fake, t)

            loss_G = F.softplus(-D(fake_d,t)).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # EMA
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p_ema * 0.999 + p * 0.001)

            if i % 50 == 0:
                print(f"Epoch {epoch} | Step {i} | D {loss_D.item():.3f} | G {loss_G.item():.3f}")

        # Save images
        with torch.no_grad():
            z = torch.randn(16, args.z_dim, device=device)
            imgs = G_ema(z)
            vutils.save_image(imgs, f"{args.out_dir}/epoch_{epoch}.png", normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
