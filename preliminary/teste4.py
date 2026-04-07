# =========================================
# DIFFUSION-GAN versão 4
# =========================================

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
parser.add_argument("--lr_d", type=float, default=5e-5)
parser.add_argument("--T", type=int, default=100)
parser.add_argument("--max_samples", type=int, default=None)

args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# DATASET
# =========================================================
class CTDataset(Dataset):
    def __init__(self, root, max_samples=None):
        self.files = sorted(glob.glob(os.path.join(root, "*.npy")))
        if max_samples is not None:
            self.files = self.files[:max_samples]
        #print(f"[DATASET] Found {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx]).astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0)
        return x

dataset = CTDataset(args.data_path, args.max_samples)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

# =========================================================
# DIFFUSION
# - adaptive difficulty
# - sigma=0.05 (noise scale)
# =========================================================
class Diffusion:
    def __init__(self, T_max=500, T_min=5, sigma=0.05):
        self.T_max = T_max
        self.T_min = T_min
        self.T = T_min
        self.sigma = sigma

        self.update_schedule()

        self.d_target = 0.6
        self.C = 1

        self.tepl = None

    def update_schedule(self):
        self.betas = torch.linspace(1e-4, 0.02, self.T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample_t(self, b):
        if self.tepl is None:
            return torch.randint(0, self.T, (b,), device=device)
        idx = torch.randint(0, len(self.tepl), (b,), device=device)
        return self.tepl[idx]

    def apply(self, x, t):
        # alpha_bar controla quantidade de sinal vs ruído
        alpha_bar = self.alpha_bars[t].view(-1,1,1,1)

        # ruído gaussiano
        noise = torch.randn_like(x)

        # forward diffusion
        return torch.sqrt(alpha_bar)*x + torch.sqrt(1-alpha_bar)*self.sigma*noise

    def update_T(self, real_logits):
        with torch.no_grad():
            # mede confiança do discriminator (overfitting proxy)
            r = torch.cat(real_logits)
            rd = torch.mean(torch.sign(torch.sigmoid(r) - 0.5))

            # ajustar dificuldade
            if rd > self.d_target:
                self.T = min(self.T + self.C, self.T_max)  # D muito forte → mais ruído
            else:
                self.T = max(self.T - self.C, self.T_min)  # D fraco → menos ruído

            # atualizar diffusion schedule
            self.update_schedule()

            # sampling com prioridade (mais peso para t grandes)
            weights = torch.arange(1, self.T+1, device=device).float()
            weights /= weights.sum()
            t_vals = torch.multinomial(weights, 32, replacement=True)

            # exploration list (mistura de fácil + difícil)
            self.tepl = torch.cat([
                torch.zeros(32, dtype=torch.long, device=device),
                t_vals
            ])

diffusion = Diffusion(T_max=500, T_min=5, sigma=0.05)

# =========================================================
# GENERATOR (UNET-LITE)
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.AvgPool2d(2)
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
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256*8*8)
        self.down1 = Down(256,128)
        self.down2 = Down(128,64)
        self.middle = ConvBlock(64,64)
        self.up1 = Up(64+64,128)
        self.up2 = Up(128+128,128)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,32,3,padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32,16,3,padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16,8,3,padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8,1,3,padding=1)
        )

    def forward(self, z):
        x = self.fc(z).view(-1,256,8,8)
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        x = self.middle(x)
        x = self.up1(x, s2)
        x = self.up2(x, s1)
        x = self.final(x)
        return torch.tanh(x)

# =========================================================
# DISCRIMINATOR
# =========================================================
def SN(layer): return nn.utils.spectral_norm(layer)

class Discriminator(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.embed = nn.Embedding(T, 64)
        self.blocks = nn.ModuleList([
            SN(nn.Conv2d(1+64,64,4,2,1)),
            SN(nn.Conv2d(64,128,4,2,1)),
            SN(nn.Conv2d(128,256,4,2,1)),
            SN(nn.Conv2d(256,512,4,2,1)),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = SN(nn.Linear(512,1))

    def forward(self, x, t):
        emb = self.embed(t).view(t.size(0), -1,1,1).expand(-1,-1,x.size(2),x.size(3))
        x = torch.cat([x, emb], dim=1)
        feats = []
        for l in self.blocks:
            x = F.leaky_relu(l(x),0.2)
            feats.append(x)
        x = self.pool(x)          # (B,512,1,1)
        x = x.view(x.size(0), -1) # (B,512)
        return self.fc(x), feats

# =========================================================
# MULTI-SCALE D
# =========================================================
class MultiScaleD(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.D1 = Discriminator(T)
        self.D2 = Discriminator(T)
        self.use_second = False 

    def forward(self, x, t):
        o1,f1 = self.D1(x,t)

        if self.use_second:
            x_small = F.avg_pool2d(x,2)
            o2,f2 = self.D2(x_small,t)
            return [o1,o2],[f1,f2]
        else:
            return [o1],[f1]
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
# INIT
# =========================================================
z_dim=256
G = Generator().to(device)
D = MultiScaleD(500).to(device)  # tem de cobrir T_max

opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5,0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5,0.999))

# =========================================================
# TRAIN
# =========================================================
print("Starting training...", flush=True)

for epoch in range(args.epochs):
    
    loss_D_avg = 0.0
    loss_G_avg = 0.0

    for i, real in enumerate(loader):
        real = real.to(device)
        b = real.size(0)

        z = torch.randn(b,z_dim,device=device)
        fake = G(z).detach()

        # sample timestep 
        t = diffusion.sample_t(b)

        # aplicar diffusion a real e fake
        real_t = diffusion.apply(real,t)
        fake_t = diffusion.apply(fake,t)

        # DISCRIMINATOR

        if epoch > 10:
            D.use_second = True

        real_logits, _ = D(real_t,t)
        fake_logits, _ = D(fake_t,t)

        loss_D = sum(
            F.binary_cross_entropy_with_logits(r, torch.ones_like(r)*0.9) +
            F.binary_cross_entropy_with_logits(f, torch.zeros_like(f))
            for r,f in zip(real_logits,fake_logits)
        )

        # atualizar D
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ADAPTIVE T (atualizar T adaptativamente (a cada 4 batches))
        if i % 4 == 0:
            diffusion.update_T(real_logits)

        # GENERATOR
        z = torch.randn(b,z_dim,device=device)

        # Generator forward (sem detach → permite gradiente passar pelo diffusion)
        fake = G(z)
        fake_t = diffusion.apply(fake,t)

        fake_logits, fake_feats = D(fake_t,t)
        # Feature matching
        _, real_feats = D(real_t,t)

        adv = sum(
            F.binary_cross_entropy_with_logits(f, torch.ones_like(f)*0.85)
            for f in fake_logits
        )

        fm = sum(
            sum(torch.mean(torch.abs(ff - rf.detach())) for ff,rf in zip(f1,f2))
            for f1,f2 in zip(fake_feats, real_feats)
        )

        loss_G = adv + 0.02 * fm

        loss_D_avg += loss_D.item()
        loss_G_avg += loss_G.item()

        opt_G.zero_grad()
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(),1.0)
        opt_G.step()

        # DEBUGS
        if i % 10 == 0:
            print(f"[E{epoch}][{i}/{len(loader)}] D={loss_D.item():.3f} | G={loss_G.item():.3f}", flush=True)
        
        if i % 50 == 0:
            print(f"T current: {diffusion.T}")

    print(f"\n[EPOCH {epoch}] D={loss_D_avg/len(loader):.3f} | G={loss_G_avg/len(loader):.3f}")

    save_samples(G, epoch, z_dim, device, args.out_dir)