import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# MESMO GENERATOR
# =========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super().__init__()
        self.up = up
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        #self.noise_strength = nn.Parameter(torch.zeros(1))
        self.noise_strength = nn.Parameter(torch.zeros(1) - 2)

    def forward(self, x):
        h = x
        if self.up:
            h = F.interpolate(h, scale_factor=2)
            x = F.interpolate(x, scale_factor=2)

        h = self.conv1(h)
        h = h + torch.randn_like(h) * self.noise_strength
        h = F.leaky_relu(h, 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)

        return h + self.skip(x)

class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()

        self.z_dim = z_dim

        # noise → feature map (pequeno)
        self.fc = nn.Linear(z_dim, 256 * 4 * 4)

        # reduzir resolução da imagem real
        self.down = nn.AvgPool2d(8)  # 256 → 32

        # entrada: real (1 canal) + noise (256 canais)
        self.initial = nn.Conv2d(1 + 256, 256, 3, padding=1)

        # upsampling progressivo
        self.blocks = nn.Sequential(
            ResBlock(256,128,True),  # 32 → 64
            ResBlock(128,64,True),   # 64 → 128
            ResBlock(64,32,True),    # 128 → 256
            ResBlock(32,16,False),
            ResBlock(16,8,False),
        )

        self.to_img = nn.Conv2d(8,1,3,padding=1)

    def forward(self, x, z):
        """
        x: [B,1,256,256]
        z: [B,z_dim]
        """

        B = x.size(0)

        # ↓ imagem real para 32x32
        x_small = self.down(x)

        # z → feature map 32x32
        z_feat = self.fc(z).view(B,256,4,4)
        z_feat = F.interpolate(z_feat, size=(32,32))

        # concat
        h = torch.cat([x_small, z_feat], dim=1)

        h = self.initial(h)
        h = self.blocks(h)

        return torch.tanh(self.to_img(h))

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

G = ConditionalGenerator().to(device)
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
        real = torch.zeros(b, 1, 256, 256, device=device)
        fake = G(real, z)

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