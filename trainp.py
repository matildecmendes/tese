import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasetp import CTDataset
from modelsp import Generator, Discriminator
from diffusionp import Diffusion
from utils import save_samples

import torch.nn.functional as F

# -------------------------
# LOSSES
# -------------------------
def adv_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets)

def feature_matching_loss(fake_feats, real_feats):
    loss = 0
    for f, r in zip(fake_feats, real_feats):
        loss += torch.mean(torch.abs(f - r.detach()))
    return loss


# -------------------------
# TRAIN
# -------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CTDataset(args.data_path)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True)

    G = Generator(args.z_dim).to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.999))

    diffusion = Diffusion(device=device)

    for epoch in range(args.epochs):
        for real in tqdm(loader, desc=f"Epoch {epoch}"):

            real = real.to(device)
            b = real.size(0)

            # ---------------- D ----------------
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z).detach()

            real_logits, real_feats = D(real)
            fake_logits, fake_feats = D(fake)

            d_loss = adv_loss(real_logits, torch.ones_like(real_logits)) + \
                     adv_loss(fake_logits, torch.zeros_like(fake_logits))

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ---------------- G ----------------
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z)

            fake_logits, fake_feats = D(fake)
            _, real_feats = D(real)

            g_loss = adv_loss(fake_logits, torch.ones_like(fake_logits))
            g_loss += args.fm_lambda * feature_matching_loss(fake_feats, real_feats)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        save_samples(G, args.z_dim, device, args.out_dir, epoch)

    torch.save(G.state_dict(), f"{args.out_dir}/G.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--fm_lambda", type=float, default=0.5)

    args = parser.parse_args()
    main(args)