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
    def __init__(self, T):
        self.T = T
        self.betas = torch.linspace(1e-4, 0.01, T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample_t(self, b):
        t = torch.randint(0, self.T, (b,), device=device)

        # prioridade a timesteps maiores
        weights = (t.float() / self.T) ** 2
        t = (weights * self.T).long().clamp(max=self.T-1)

        return t

    def apply(self, x, t):
        alpha_bar = self.alpha_bars[t].view(-1,1,1,1)
        noise = torch.randn_like(x)
        return torch.sqrt(alpha_bar)*x + torch.sqrt(1-alpha_bar)*noise

diffusion = Diffusion(args.T)

# =========================================================
# BLOCO BASE
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# UP/DOWN BLOCKS
# ========================================================= 

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        h = self.conv(x)
        return h, self.pool(h)
    
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # garantir alinhamento
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    
# =========================================================
# RESIDUAL BLOCK
# =========================================================
class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=2)

        h = self.conv1(x_up)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)

        return F.relu(h + self.skip(x_up))
# =========================================================
# GENERATOR
# =========================================================
class Generator(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()

        self.fc = nn.Linear(z_dim, 512*8*8)

        # encoder
        self.down1 = Down(512, 256)   # 8 → 4
        self.down2 = Down(256, 128)   # 4 → 2
        self.down3 = Down(128, 64)    # 2 → 1

        # bottleneck
        self.middle = ConvBlock(64, 64)

        # decoder
        self.up1 = Up(64+64, 128)     # 1 → 2
        self.up2 = Up(128+128, 256)   # 2 → 4
        self.up3 = Up(256+256, 128)   # 4 → 8

        # upsampling final até 256
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 16
            nn.Conv2d(128,64,3,padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # 32
            nn.Conv2d(64,32,3,padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # 64
            nn.Conv2d(32,16,3,padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # 128
            nn.Conv2d(16,8,3,padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # 256
            nn.Conv2d(8,1,3,padding=1)
        )

    def forward(self, z):
        x = self.fc(z).view(-1,512,8,8)

        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)

        x = self.middle(x)

        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)

        x = self.final_up(x)

        return torch.tanh(x)
    
# =========================================================
# TIMESTEP-DISCRIMINATOR (SpectralNorm + features)
# =========================================================
def SN(layer):
    return nn.utils.spectral_norm(layer)

class Discriminator(nn.Module):
    def __init__(self, T):
        super().__init__()

        self.embed = nn.Embedding(T, 64)

        self.blocks = nn.ModuleList([
            SN(nn.Conv2d(1+64, 64, 4,2,1)),
            SN(nn.Conv2d(64,128,4,2,1)),
            SN(nn.Conv2d(128,256,4,2,1)),
            SN(nn.Conv2d(256,512,4,2,1)),
            SN(nn.Conv2d(512,512,4,2,1)),
        ])

        self.fc = SN(nn.Linear(512*8*8,1))

    def forward(self, x, t):
        emb = self.embed(t).view(t.size(0), -1, 1, 1)
        emb = emb.expand(-1, -1, x.size(2), x.size(3))

        x = torch.cat([x, emb], dim=1)

        feats = []
        for layer in self.blocks:
            x = F.leaky_relu(layer(x), 0.2)
            feats.append(x)

        return self.fc(x.view(x.size(0),-1)), feats

# =========================================================
# LOSSES
# =========================================================
def fm_loss(fake_feats, real_feats):
    return sum(torch.mean(torch.abs(f - r.detach())) for f, r in zip(fake_feats, real_feats))
    
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

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

G.apply(init_weights)

G.train()
D.train()

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
        z = torch.randn(b, z_dim, device=device)*1.2
        fake = G(z).detach()

        t = diffusion.sample_t(b)

        real_t = diffusion.apply(real, t)
        fake_t = diffusion.apply(fake, t)

        real_t = real_t + 0.05 * torch.randn_like(real_t)
        fake_t = fake_t + 0.05 * torch.randn_like(fake_t)

        real_logits, _ = D(real_t, t)
        fake_logits, _ = D(fake_t, t)

        #label smoothing (antes era 0.9)
        loss_D = (
            F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)*0.85) +
            F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        )

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ======================
        # GENERATOR
        # ======================
        z = torch.randn(b, z_dim, device=device)*1.2
        fake = G(z)
        fake_t = diffusion.apply(fake, t)

        fake_logits, fake_feats = D(fake_t, t)
        #_, real_feats = D(real_t, t)

        real_t_clean = diffusion.apply(real, t)
        _, real_feats = D(real_t_clean, t)

        # label smoothing (antes era 0.9)
        adv = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)*0.85
        )

        fm = fm_loss(fake_feats, real_feats)

        loss_G = adv + 0.02 * fm

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        loss_D_avg += loss_D.item()
        loss_G_avg += loss_G.item()

        if i % 50 == 0:
            print(f"[E{epoch}][{i}] D={loss_D.item():.3f} | G={loss_G.item():.3f}")
            print(f"   real_logit: {real_logits.mean().item():.3f}")
            print(f"   fake_logit: {fake_logits.mean().item():.3f}")

    print(f"\n[EPOCH {epoch}] D={loss_D_avg/len(loader):.3f} | G={loss_G_avg/len(loader):.3f}")

    save_samples(G, epoch, z_dim, device, args.out_dir)