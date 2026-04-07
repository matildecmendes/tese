import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from data.dataset import NPYDataset
from models.generator import Generator
from models.discriminator import Discriminator
from diffusion.diffusion import Diffusion

cfg = Config()
device = torch.device(cfg.device)

dataset = NPYDataset("./data")
loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

G = Generator(cfg.z_dim).to(device)
D = Discriminator().to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5,0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.5,0.999))

diffusion = Diffusion(cfg.T_max, cfg.T_min, cfg.sigma, device)

def d_loss(real_logits, fake_logits):
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()

def g_loss(fake_logits):
    return F.softplus(-fake_logits).mean()

for epoch in range(cfg.epochs):

    for real in loader:
        real = real.to(device)
        b = real.size(0)

        # -----------------
        # DISCRIMINATOR
        # -----------------
        z = torch.randn(b, cfg.z_dim, device=device)
        fake = G(z).detach()

        t = diffusion.sample_t(b)

        real_t = diffusion.apply(real,t)
        fake_t = diffusion.apply(fake,t)

        real_logits = D(real_t,t)
        fake_logits = D(fake_t,t)

        loss_D = d_loss(real_logits, fake_logits)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        diffusion.update_T(real_logits, fake_logits)

        # -----------------
        # GENERATOR
        # -----------------
        z = torch.randn(b, cfg.z_dim, device=device)
        fake = G(z)

        fake_t = diffusion.apply(fake,t)
        fake_logits = D(fake_t,t)

        loss_G = g_loss(fake_logits)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"[Epoch {epoch}] D={loss_D.item():.3f} G={loss_G.item():.3f} T={diffusion.T}")