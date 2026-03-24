import os
import torch
import torchvision.utils as vutils

def save_samples(G, z_dim, device, out_dir, epoch, n=16):
    os.makedirs(out_dir, exist_ok=True)

    G.eval()
    with torch.no_grad():
        z = torch.randn(n, z_dim, device=device)
        fake = G(z)

        fake = (fake + 1) / 2  # [-1,1] → [0,1]

        grid = vutils.make_grid(fake, nrow=4)
        vutils.save_image(grid, f"{out_dir}/epoch_{epoch:04d}.png")