import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# =========================
# GENERATOR 
# =========================
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


# =========================
# DATA
# =========================

def load_image(path, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("L")
    return transform(img)


def get_image_paths(folder):
    exts = ["png", "jpg", "jpeg"]
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.split('.')[-1] in exts
    ])


def save_image(tensor, path):
    img = tensor.squeeze().numpy()
    img = (img + 1) / 2  # [-1,1] → [0,1]
    img = (img * 255).clip(0,255).astype(np.uint8)

    Image.fromarray(img).save(path)


# =========================
# MAIN
# =========================

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    checkpoint = torch.load(args.model_path, map_location=device)

    G = ConditionalGenerator().to(device)
    G.load_state_dict(checkpoint["G"])
    G.eval()

    print(f"[INFO] Loaded model epoch {checkpoint['epoch']}")

    # paths
    real_paths = get_image_paths(args.real_dir)

    os.makedirs(os.path.join(args.out_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "fake"), exist_ok=True)

    z_dim = 256

    # escolher N imagens reais
    if args.num_samples <= len(real_paths):
        selected_real = np.random.choice(real_paths, args.num_samples, replace=False)
    else:
        selected_real = np.random.choice(real_paths, args.num_samples, replace=True)

    for idx, path in enumerate(tqdm(selected_real)):

        # save real
        img_real = Image.open(path).convert("L")
        real_save = os.path.join(args.out_dir, "real", f"{idx:05d}.png")
        img_real.save(real_save)

        # generate fake
        z = torch.randn(1, z_dim, device=device)

        with torch.no_grad():
            real_tensor = load_image(path).unsqueeze(0).to(device)
            real_tensor = real_tensor * 2 - 1  # normalizar para [-1,1]
            fake = G(real_tensor, z)

        fake_save = os.path.join(args.out_dir, "fake", f"{idx:05d}.png")
        save_image(fake.cpu(), fake_save)

    print(f"[DONE] Generated {args.num_samples} pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)

    args = parser.parse_args()

    main(args)