import argparse
import torch
from torch.utils.data import DataLoader

from dataset import CTDataset
from models import Generator, Discriminator
from diffusion import Diffusion
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

    # DATASET

    dataset = CTDataset(args.data_path)

    if args.limit is not None:
        dataset.files = dataset.files[:args.limit]

    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    G = Generator(args.z_dim).to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.999))

    diffusion = Diffusion(device=device)

    # LOOP DE TREINO

    for epoch in range(args.epochs):
        for i, real in enumerate(loader):

            real = real.to(device)
            b = real.size(0)

            # ---------------- D ----------------
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z).detach()

            x_t_real, _ = diffusion.forward(real)
            x_t_fake, _ = diffusion.forward(fake)

            real_logits, real_feats = D(x_t_real)
            fake_logits, fake_feats = D(x_t_fake)

            d_loss = adv_loss(real_logits, torch.ones_like(real_logits)) + \
                    adv_loss(fake_logits, torch.zeros_like(fake_logits))

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ---------------- G ----------------
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z)

            x_t_fake, _ = diffusion.forward(fake)

            fake_logits, fake_feats = D(x_t_fake)
            _, real_feats = D(x_t_real)

            g_loss = adv_loss(fake_logits, torch.ones_like(fake_logits))
            g_loss += args.fm_lambda * feature_matching_loss(fake_feats, real_feats)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            # ---------------- PRINT DEBUG ----------------
            if i % 10 == 0:
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] "
                    f"[Batch {i}/{len(loader)}] "
                    f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f}"
                )

                print(
                    f"  REAL   min/max: {real.min().item():.3f} / {real.max().item():.3f}"
                )
                print(
                    f"  x_t    min/max: {x_t_real.min().item():.3f} / {x_t_real.max().item():.3f}"
                )
                print(
                    f"  FAKE   min/max: {fake.min().item():.3f} / {fake.max().item():.3f}"
                )

        # guardar imagens só no fim do epoch
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
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args)