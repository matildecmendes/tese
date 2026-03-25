import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CTDataset
from models import Generator, Discriminator
from utils import save_images
import config


def train():
    dataset = CTDataset(config.DATA_PATH, config.IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    G = Generator(config.LATENT_DIM).to(config.DEVICE)
    D = Discriminator().to(config.DEVICE)

    criterion = nn.BCELoss()

    opt_G = torch.optim.Adam(G.parameters(), lr=config.LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=config.LR, betas=(0.5, 0.999))

    for epoch in range(config.EPOCHS):
        for i, real in enumerate(loader):

            real = real.to(config.DEVICE)
            batch_size = real.size(0)

            # Labels
            real_labels = torch.ones(batch_size, device=config.DEVICE)
            fake_labels = torch.zeros(batch_size, device=config.DEVICE)

            # -------- Train D --------
            noise = torch.randn(batch_size, config.LATENT_DIM, 1, 1, device=config.DEVICE)
            fake = G(noise)

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

            if i % 50 == 0:
                print(f"[{epoch}/{config.EPOCHS}] Batch {i} "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

        # guardar imagens
        with torch.no_grad():
            noise = torch.randn(16, config.LATENT_DIM, 1, 1, device=config.DEVICE)
            fake = G(noise)
            save_images(fake, epoch, config.SAVE_PATH)


if __name__ == "__main__":
    train()