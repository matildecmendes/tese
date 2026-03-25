import argparse
import torch
from torch.utils.data import DataLoader

from preliminary.dataset import CTDataset
from preliminary.models import Generator, Discriminator
from preliminary.diffusion import Diffusion
from preliminary.utils import save_samples

import torch.nn.functional as F

# -------------------------
# LOSSES
# -------------------------
def adv_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets)

def feature_matching_loss(fake_feats, real_feats):
    loss = 0
    for fake_f, real_f in zip(fake_feats, real_feats):
        loss += torch.mean(torch.abs(fake_f - real_f.detach()))
    return loss


# -------------------------
# TRAIN
# -------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    for epoch in range(args.epochs):
        for i, real in enumerate(loader):

            real = real.to(device)
            b = real.size(0)

            # --- D ---
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z).detach()

            # amostra t uma vez
            real_d, t = diffusion.forward(real)
            
            # usa o MESMO t para fake
            fake_d = diffusion.apply_at_t(fake, t)


            real_logits, real_feats = D(real_d)
            fake_logits, fake_feats = D(fake_d)

            real_labels = torch.ones_like(real_logits) * 0.9
            fake_labels = torch.zeros_like(fake_logits)

            loss_D = (
                adv_loss(real_logits, real_labels) +
                adv_loss(fake_logits, fake_labels)
            )

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- G ---
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z)

            fake_d = diffusion.apply_at_t(fake, t)

            fake_logits, fake_feats = D(fake_d)

            with torch.no_grad():
                _, real_feats = D(real_d)

            loss_G = (
                adv_loss(fake_logits, torch.ones_like(fake_logits) * 0.9) +
                args.fm_lambda * feature_matching_loss(fake_feats, real_feats)
            )

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{i}/{len(loader)}] "
                    f"D_loss: {loss_D.item():.3f} G_loss: {loss_G.item():.3f}",
                    flush=True)
                
            if i == 0:
                print("REAL SHAPE:", real.shape)
                print("FAKE SHAPE:", fake.shape)
                print("Real min | real max", real.min(), real.max())
                print("Fake min | fake max", fake.min(), fake.max())

        save_samples(G, args.z_dim, device, args.output_path, epoch)

    torch.save(G.state_dict(), f"{args.output_path}/G.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--fm_lambda", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args)