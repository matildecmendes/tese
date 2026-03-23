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


    torch.backends.cudnn.benchmark = True


    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "log.txt")


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
    T = 4  # continua baixo para facilitar aprendizagem


    G = Generator(nz).to(device)
    D = Discriminator().to(device)


    ema_G = Generator(nz).to(device)
    ema_G.load_state_dict(G.state_dict())


    # LR 
    optG = torch.optim.Adam(G.parameters(), lr=2e-4)
    optD = torch.optim.Adam(D.parameters(), lr=1e-5)  # ↓ mais lento


    coeff = get_diffusion_coefficients(T, device)


    # ======================
    # TRAIN
    # ======================
    for epoch in range(epochs):
        print(f"\n===== EPOCH {epoch} =====")


        for i, x0 in enumerate(loader):
            x0 = x0.to(device)


            # ======================
            # TRAIN D (1x)
            # ======================
            D.zero_grad()


            t = torch.randint(0, T-1, (x0.size(0),), device=device)
            xt, xt_plus = q_sample_pairs(coeff, x0, t)


            xt = xt.to(device)
            xt_plus = xt_plus.to(device)

            # versão limpa para R1
            xt_clean = xt.detach().clone().requires_grad_(True)

            # versões com noise para o discriminador
            xt_noisy = xt_clean + 0.01 * torch.randn_like(xt_clean)
            xt_plus_noisy = xt_plus + 0.01 * torch.randn_like(xt_plus)

            # REAL (com smoothing leve)
            D_real = D(xt_noisy, t, xt_plus_noisy)
            loss_real = F.softplus(-0.9 * D_real).mean()


            # FAKE
            z = torch.randn(x0.size(0), nz, device=device)
            x0_pred = G(xt_plus, t, z)
            xt_fake = sample_posterior(coeff, x0_pred, xt_plus, t)


            D_fake = D(xt_fake, t, xt_plus_noisy)
            loss_fake = F.softplus(D_fake).mean()


            # R1 (reduzido)
            grad = torch.autograd.grad(
                outputs=D_real.sum(),
                inputs=xt_clean,
                create_graph=True
            )[0]
            r1 = (grad_real.pow(2).view(grad_real.size(0), -1).sum(1)).mean()
            print("R1 real:", r1.item())


            loss_D = loss_real + loss_fake + 10 * r1
            loss_D.backward()
            optD.step()


            # ======================
            # TRAIN G 
            # ======================
            for _ in range(2):
                G.zero_grad()


                t = torch.randint(0, T-1, (x0.size(0),), device=device)
                xt, xt_plus = q_sample_pairs(coeff, x0, t)
                xt_plus = xt_plus.to(device)


                z = torch.randn(x0.size(0), nz, device=device)
                x0_pred = G(xt_plus, t, z)
                xt_fake = sample_posterior(coeff, x0_pred, xt_plus, t)


                D_fake = D(xt_fake, t, xt_plus)
                loss_G = F.softplus(-D_fake).mean()

                if torch.isnan(loss_G) or torch.isinf(loss_G):
                    print("NaN detected in G loss")
                    break
                if torch.isnan(x0_pred).any():
                    print("NaN in generator output")

                loss_G.backward()
                optG.step()


            update_ema(ema_G, G)


            # ======================
            # LOGS
            # ======================
            if i % 10 == 0:
                print(f"[Epoch {epoch} | Batch {i}] D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")


                with open(log_file, "a") as f:
                    f.write(
                        f"Epoch {epoch} Batch {i} "
                        f"D:{loss_D.item():.4f} "
                        f"G:{loss_G.item():.4f} "
                        f"D_real:{loss_real.item():.4f} "
                        f"D_fake:{loss_fake.item():.4f} "
                        f"R1:{r1.item():.4f}\n"
                    )


            # DEBUG visual do generator
            if i % 50 == 0:
                torchvision.utils.save_image(
                    x0_pred[:8],
                    os.path.join(output_path, f"x0_pred_epoch{epoch}_batch{i}.png"),
                    normalize=True
                )


            if i == 0:
                print("x0 min/max:", x0.min().item(), x0.max().item())
                print("xt min/max:", xt.min().item(), xt.max().item())
                print("xt_plus min/max:", xt_plus.min().item(), xt_plus.max().item())
                print("x0_pred min/max:", x0_pred.min().item(), x0_pred.max().item())


            if i == 0:
                torchvision.utils.save_image(x0[:4], f"{output_path}/debug_x0.png", normalize=True)
                torchvision.utils.save_image(xt[:4], f"{output_path}/debug_xt.png", normalize=True)
                torchvision.utils.save_image(xt_plus[:4], f"{output_path}/debug_xt_plus.png", normalize=True)
                torchvision.utils.save_image(x0_pred[:4], f"{output_path}/debug_x0_pred.png", normalize=True)
                torchvision.utils.save_image(xt_fake[:4], f"{output_path}/debug_xt_fake.png", normalize=True)


            if i == 0:
                print("D_real:", D_real.mean().item())
                print("D_fake:", D_fake.mean().item())

            print("grad mean:", grad.abs().mean().item())


        # ======================
        # SAMPLING
        # ======================
        samples = sample_ddgan(ema_G, coeff, 8, T, nz, device)


        torchvision.utils.save_image(
            samples,
            os.path.join(output_path, f"samples_epoch_{epoch}.png"),
            normalize=True
        )


        # ======================
        # SAVE MODELS
        # ======================
        torch.save(G.state_dict(), os.path.join(output_path, f"G_epoch_{epoch}.pth"))
        torch.save(D.state_dict(), os.path.join(output_path, f"D_epoch_{epoch}.pth"))


        print(f"[INFO] Epoch {epoch} done")


    print("[INFO] Training finished.")




# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="DDGAN Training")


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
