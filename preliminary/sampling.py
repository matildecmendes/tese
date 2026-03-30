import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# IMPORTA A MESMA ARQUITETURA
# (podes importar do teu train.py ou copiar)
# =========================================================
from train import Generator  # assume que está no mesmo folder

# =========================================================
# ARGUMENTS
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--num_samples", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=16)

args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# LOAD CHECKPOINT
# =========================================================
checkpoint = torch.load(args.model_path, map_location=device)

G = Generator().to(device)
G.load_state_dict(checkpoint["G"])
G.eval()

print(f"[INFO] Loaded model from epoch {checkpoint['epoch']}")

# =========================================================
# SAMPLING
# =========================================================
z_dim = 256

generated = 0
img_idx = 0

while generated < args.num_samples:

    b = min(args.batch_size, args.num_samples - generated)

    z = torch.randn(b, z_dim, device=device)

    with torch.no_grad():
        fake = G(z).cpu()

    for i in range(b):
        img = fake[i, 0].numpy()

        plt.imshow(img, cmap="gray")
        plt.axis("off")

        path = os.path.join(args.out_dir, f"sample_{img_idx}.png")
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

        img_idx += 1

    generated += b

    print(f"[PROGRESS] {generated}/{args.num_samples}")

print(f"[DONE] Generated {generated} images")