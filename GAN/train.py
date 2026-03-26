import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CTDataset
from models import Generator, Discriminator
from utils import save_images


def train(data_path, output_path, batch_size=4, epochs=5, limit=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_path, exist_ok=True)

    # -------- Dataset --------
    dataset = CTDataset(data_path)

    if limit is not None:
        dataset.files = dataset.files[:limit]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Dataset size: {len(dataset)}")

    # -------- Models --------
    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # -------- Training --------
    for epoch in range(epochs):
        for i, real in enumerate(loader):

            real = real.to(device)
            batch_size_curr = real.size(0)

            # DEBUG REAL
            if i == 0:
                print("REAL min/max:", real.min().item(), real.max().item())

            real_labels = torch.ones(batch_size_curr, device=device)
            fake_labels = torch.zeros(batch_size_curr, device=device)

            # -------- Train D --------
            noise = torch.randn(batch_size_curr, 100, 1, 1, device=device)
            fake = G(noise)

            # DEBUG FAKE
            if i == 0:
                print("FAKE min/max:", fake.min().item(), fake.max().item())

            D_real = D(real)
            D_fake = D(fake.detach())

            loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # -------- Train G --------
            D_fake = D(fake)
            loss_G = criterion(D_fake, real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if i % 20 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] Batch {i} "
                    f"Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}"
                )

        # -------- Save images --------
        with torch.no_grad():
            noise = torch.randn(16, 100, 1, 1, device=device)
            fake = G(noise)
            save_images(fake, epoch, output_path)

        print(f"Saved samples for epoch {epoch}")

    print("Training finished.")


# ENTRY POINT

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple GAN Training")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset slices")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save outputs (NAS)")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        limit=args.limit
    )