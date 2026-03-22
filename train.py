import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import os
import argparse

from dataset import CTSliceDataset
from models import Generator, Discriminator
from diffusion import *
from utils import update_ema


def train(data_path, output_path, batch_size, epochs, limit):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    os.makedirs(output_path, exist_ok=True)

    # ======================
    # DATA
    # ======================
    dataset = CTSliceDataset(data_path, limit=limit)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"[INFO] Dataset size: {len(dataset)}")

    # ======================
    # MODELS
    # ======================
    nz = 256
    T = 4  

    G = Generator(nz).to(device)
    D = Discriminator().to(device)

    ema_G = Generator(nz).to(device)
    ema_G.load_state_dict(G.state_dict())

    optG = torch.optim.Adam(G.parameters(), lr=2e-4)
    optD = torch.optim.Adam(D.parameters(), lr=5e-5)

    coeff = get_diffusion_coefficients(T, device)

    # ======================
    # TRAIN
    # ======================
    for epoch in range(epochs):

        print(f"\n===== EPOCH {epoch} =====")

        for i, x0 in enumerate(loader):

            x0 = x0.to(device)

            # ======================
            # TRAIN D
            # ======================
            D.zero_grad()

            t = torch.randint(0, T-1, (x0.size(0),), device=device)

            xt, xt_plus = q_sample_pairs(coeff, x0, t)

            xt = xt.detach().requires_grad_(True)

            # REAL
            D_real = D(xt, t, xt_plus)
            loss_real = F.softplus(-0.9 * D_real).mean()

            # FAKE
            z = torch.randn(x0.size(0), nz, device=device)

            x0_pred = G(xt_plus.detach(), t, z)

            xt_fake = sample_posterior(coeff, x0_pred, xt_plus, t)

            D_fake = D(xt_fake.detach(), t, xt_plus)
            loss_fake = F.softplus(D_fake).mean()

            # R1 reduzido
            grad = torch.autograd.grad(
                outputs=D_real.sum(),
                inputs=xt,
                create_graph=True
            )[0]

            r1 = grad.pow(2).mean()

            loss_D = loss_real + loss_fake + 0.1 * r1

            loss_D.backward()
            optD.step()

            # ======================
            # TRAIN G
            # ======================
            for _ in range(2):

                G.zero_grad()

                t = torch.randint(0, T-1, (x0.size(0),), device=device)

                xt, xt_plus = q_sample_pairs(coeff, x0, t)

                z = torch.randn(x0.size(0), nz, device=device)

                x0_pred = G(xt_plus, t, z)

                xt_fake = sample_posterior(coeff, x0_pred, xt_plus, t)

                D_fake = D(xt_fake, t, xt_plus)

                loss_G = F.softplus(-D_fake).mean()

                loss_G.backward()
                optG.step()

                update_ema(ema_G, G)

            # ======================
            # LOGS
            # ======================
            if i % 10 == 0:
                print(f"[Epoch {epoch} | Batch {i}] D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")

            # DEBUG
            if i == 0:
                print("x0 min/max:", x0.min().item(), x0.max().item())
                print("xt min/max:", xt.min().item(), xt.max().item())
                print("xt_plus min/max:", xt_plus.min().item(), xt_plus.max().item())
                print("x0_pred min/max:", x0_pred.min().item(), x0_pred.max().item())

            # ======================
            # SAMPLING
            # ======================
            if i % 200 == 0:
                samples = sample_ddgan(ema_G, coeff, 8, T, nz, device)

                torchvision.utils.save_image(
                    samples,
                    os.path.join(output_path, f"samples_epoch_{epoch}_batch_{i}.png"),
                    normalize=True
                )

        # ======================
        # SAVE
        # ======================
        torch.save(G.state_dict(), os.path.join(output_path, f"G_epoch_{epoch}.pth"))
        torch.save(D.state_dict(), os.path.join(output_path, f"D_epoch_{epoch}.pth"))

        print(f"[INFO] Epoch {epoch} done")

    print("[INFO] Training finished.")