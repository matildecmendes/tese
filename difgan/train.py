import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# =========================================================
# ARGUMENTS
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr_g", type=float, default=2e-4)
parser.add_argument("--lr_d", type=float, default=1e-4)
parser.add_argument("--T", type=int, default=100)
parser.add_argument("--max_samples", type=int, default=None)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# DATASET
# =========================================================
class CTDataset(Dataset):
    def __init__(self, root, max_samples=None):
        self.files = sorted(glob.glob(os.path.join(root, "*.npy")))
        assert len(self.files) > 0, "Dataset vazio"

        if max_samples is not None:
            self.files = self.files[:max_samples]

        print(f"[DATASET] Found {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx]).astype(np.float32)

        if x.min() >= 0:
            x = x * 2 - 1

        x = torch.from_numpy(x).unsqueeze(0)
        return x

dataset = CTDataset(args.data_path, args.max_samples)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

# =========================================================
# DIFFUSION
# =========================================================
class Diffusion:
    def __init__(self, T_init=20, T_min=5, T_max=100):
        self.T = T_init
        self.T_min = T_min
        self.T_max = T_max

        self.update_schedule()

        self.target = 0.6   # alvo do discriminador
        self.adjust_step = 1

    def update_schedule(self):
        self.betas = torch.linspace(1e-4, 0.02, self.T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample_t(self, b):
        return torch.randint(0, self.T, (b,), device=device)

    def apply(self, x, t):
        alpha_bar = self.alpha_bars[t].view(-1,1,1,1)
        noise = torch.randn_like(x)
        return torch.sqrt(alpha_bar)*x + torch.sqrt(1-alpha_bar)*noise
    
    def update_T(self, real_logits, fake_logits):
        with torch.no_grad():
            real_prob = torch.sigmoid(real_logits)
            fake_prob = torch.sigmoid(fake_logits)

            acc = (real_prob.mean() + (1 - fake_prob.mean())) / 2

            if acc > self.target:
                self.T = min(self.T + self.adjust_step, self.T_max)
            else:
                self.T = max(self.T - self.adjust_step, self.T_min)

            self.update_schedule()

diffusion = Diffusion(T_init=20, T_min=5, T_max=100)

# =========================================================
# GENERATOR
# =========================================================
class Generator(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.fc = nn.Linear(z_dim, 512*4*4)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,4,2,1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(-1,512,4,4)
        return self.net(h)

# =========================================================
# TIMESTEP-DISCRIMINATOR 
# =========================================================
class Discriminator(nn.Module):
    def __init__(self, T):
        super().__init__()

        self.embed = nn.Embedding(T, 64)

        self.conv = nn.Sequential(
            nn.Conv2d(1+64, 64, 4,2,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,4,2,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,512,4,2,1),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Linear(512*16*16,1)

    def forward(self, x, t):
        emb = self.embed(t).view(t.size(0), -1, 1, 1)
        emb = emb.expand(-1, -1, x.size(2), x.size(3))

        x = torch.cat([x, emb], dim=1)
        h = self.conv(x)
        return self.fc(h.view(h.size(0), -1))
    
# =========================================================
# SAVE SAMPLES
# =========================================================

def save_samples(G, epoch, z_dim, device, out_dir):
    G.eval()

    with torch.no_grad():
        z = torch.randn(8, z_dim, device=device)
        fake = G(z).cpu()

    imgs = fake.squeeze().numpy()

    fig, axes = plt.subplots(2,4, figsize=(8,4))

    for i in range(2):
        for j in range(4):
            idx = i*4+j
            axes[i,j].imshow(imgs[idx], cmap="gray")
            axes[i,j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch}.png"))
    plt.close()

    # guardar modelos a cada epoch
    checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict()
    }, checkpoint_path)

    print(f"[SAVE] Epoch {epoch} saved")


# =========================================================
# DEBUG DIFFUSION
# =========================================================

def save_diffusion_debug(diffusion, real, fake, epoch, out_dir):
    with torch.no_grad():
        t_values = [0, 10, 30, 60, 90]

        imgs = []
        for t_val in t_values:
            t = torch.full((real.size(0),), t_val, device=real.device, dtype=torch.long)

            real_t = diffusion.apply(real, t)
            fake_t = diffusion.apply(fake, t)

            imgs.append((real_t.cpu(), fake_t.cpu()))

    fig, axes = plt.subplots(len(t_values), 4, figsize=(6,8))

    for i, (r, f) in enumerate(imgs):
        for j in range(2):
            axes[i, j*2].imshow(r[j,0], cmap="gray")
            axes[i, j*2+1].imshow(f[j,0], cmap="gray")

            axes[i, j*2].axis("off")
            axes[i, j*2+1].axis("off")

    plt.savefig(os.path.join(out_dir, f"diff_debug_{epoch}.png"))
    plt.close()

# =========================================================
# INIT
# =========================================================
z_dim = 256
G = Generator(z_dim).to(device)
D = Discriminator(args.T).to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5,0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5,0.999))

# =========================================================
# TRAIN
# =========================================================
for epoch in range(args.epochs):

    loss_D_avg = 0
    loss_G_avg = 0

    for i, real in enumerate(loader):

        real = real.to(device)
        b = real.size(0)

        # ======================
        # DISCRIMINATOR
        # ======================
        z = torch.randn(b, z_dim, device=device)
        fake = G(z).detach()

        t = diffusion.sample_t(b)

        real_t = diffusion.apply(real, t)
        fake_t = diffusion.apply(fake, t)

        real_logits = D(real_t, t)
        fake_logits = D(fake_t, t)

        loss_D = (
            F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)*0.9) +
            F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        )

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        if i % 10 == 0:
            diffusion.update_T(real_logits.detach(), fake_logits.detach())

        # ======================
        # GENERATOR
        # ======================
        z = torch.randn(b, z_dim, device=device)
        fake = G(z)
        fake_t = diffusion.apply(fake, t)

        fake_logits = D(fake_t, t)

        loss_G = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)*0.9
        )

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        loss_D_avg += loss_D.item()
        loss_G_avg += loss_G.item()

        if i % 50 == 0:
            print(f"[E{epoch}][{i}] D={loss_D.item():.3f} | G={loss_G.item():.3f}")
            print(f"   real_logit: {real_logits.mean().item():.3f}")
            print(f"   fake_logit: {fake_logits.mean().item():.3f}")
            print(f"   gap: {(real_logits.mean()-fake_logits.mean()).item():.3f}")

    print(f"\n[EPOCH {epoch}] D={loss_D_avg/len(loader):.3f} | G={loss_G_avg/len(loader):.3f}")

    # SAVE IMAGES
    save_samples(G, epoch, z_dim, device, args.out_dir)

    # DEBUG DIFFUSION (comentar se não quiser mais)
    #save_diffusion_debug(diffusion, real, fake, epoch, args.out_dir)