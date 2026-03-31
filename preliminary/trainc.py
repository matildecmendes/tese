import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =========================================================
# ARGUMENTS (para .sh)
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr_g", type=float, default=2e-4)
parser.add_argument("--lr_d", type=float, default=5e-5)
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

        # =========================
        # DEBUG RAW 
        # =========================
        if idx == 0:
            print("\n[DATA DEBUG]")
            print("RAW min/max:", x.min(), x.max())
            print("RAW mean/std:", x.mean(), x.std())

        # =========================
        # VALIDAÇÃO AUTOMÁTICA
        # =========================
        assert x.shape == (256, 256), f"Shape errado: {x.shape}"

        # DETECTAR RANGE
        if x.min() >= -1.1 and x.max() <= 1.1:
            # em [-1,1]
            pass

        elif x.min() >= 0.0 and x.max() <= 1.0:
            # [0,1] → [-1,1]
            x = x * 2 - 1

        else:
            raise ValueError(f"Range inesperado: min={x.min()}, max={x.max()}")

        # =========================
        # DEBUG FINAL
        # =========================
        if idx == 0:
            print("FINAL min/max:", x.min(), x.max())
            print("FINAL mean/std:", x.mean(), x.std())
            print("")

        x = torch.from_numpy(x).unsqueeze(0)

        return x

dataset = CTDataset(args.data_path, args.max_samples)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

# =========================================================
# DIFFUSION
# =========================================================
class Diffusion:
    def __init__(self, T=25, beta_start=1e-4, beta_end=0.002):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, x):
        b = x.size(0)
        t = torch.randint(0, self.T, (b,), device=x.device)
        return self.apply_at_t(x, t), t

    def apply_at_t(self, x, t):
        alpha_bar_t = self.alpha_bars[t].view(-1,1,1,1)
        noise = torch.randn_like(x)
        return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

# =========================================================
# MODELS
# =========================================================
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

class DiscResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=False):
        super().__init__()
        self.down = down
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)

        if self.down:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)

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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiscResBlock(1,32,True),
            DiscResBlock(32,64,True),
            DiscResBlock(64,128,True),
            DiscResBlock(128,256,True),
            DiscResBlock(256,512,True),
        ])
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc = nn.Linear(512*4*4,1)

    def forward(self, x):
        feats = []
        for b in self.blocks:
            x = b(x)
            feats.append(x)
        x = self.pool(x)
        return self.fc(x.view(x.size(0),-1)), feats

# =========================================================
# LOSS
# =========================================================
def adv_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets)

def fm_loss(fake, real):
    return sum(torch.mean(torch.abs(f - r.detach())) for f,r in zip(fake,real))

# =========================================================
# INIT
# =========================================================
z_dim = 256
G = ConditionalGenerator(z_dim).to(device)
D = Discriminator().to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5,0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5,0.999))

diffusion = Diffusion()

z_fixed = torch.randn(8, z_dim, device=device)

# =========================================================
# SAVE FUNCTION
# =========================================================
def save_samples(G, epoch, loader):
    G.eval()

    real = next(iter(loader)).to(device)[:8]
    z = torch.randn(real.size(0), z_dim, device=device)*0.8

    with torch.no_grad():
        fake = G(real, z).cpu()

    grid = fake.squeeze().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(8,4))

    for i in range(2):
        for j in range(4):
            idx = i * 4 + j
            axes[i, j].imshow(grid[idx], cmap="gray")
            axes[i, j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)

    path = os.path.join(args.out_dir, f"epoch_{epoch}.png")
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

    checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict()
    }, checkpoint_path)

    print(f"[SAVE] Epoch {epoch} saved")

# =========================================================
# TRAIN
# =========================================================
for epoch in range(args.epochs):

    loss_D_avg = 0
    loss_G_avg = 0

    def save_debug_progress(G, diffusion, batch, epoch):
        G.eval()
        with torch.no_grad():

            z = torch.randn(batch.size(0), 256, device=device)
            fake = G(batch, z)

            # aplicar diffusion progressiva
            imgs = []

            for t_val in [0, 3, 6, 9]:
                t = torch.full((batch.size(0),), t_val, device=device, dtype=torch.long)
                img_t = diffusion.apply_at_t(fake, t)
                imgs.append(img_t.cpu())

        # guardar
        fig, axes = plt.subplots(len(imgs), 4, figsize=(6,6))

        for i in range(len(imgs)):
            for j in range(4):
                axes[i,j].imshow(imgs[i][j,0], cmap="gray")
                axes[i,j].axis("off")

        plt.savefig(os.path.join(args.out_dir, f"debug_epoch_{epoch}.png"), dpi=150)
        plt.close()

    for i, real in enumerate(loader):

        real = real.to(device)
        b = real.size(0)

        # ================= D =================
        z = torch.randn(b, z_dim, device=device)*0.6
        fake = G(real, z).detach()

        real_d, t = diffusion.forward(real)
        fake_d = diffusion.apply_at_t(fake, t)

        real_logits,_ = D(real_d)
        fake_logits,_ = D(fake_d)

        loss_D = adv_loss(real_logits, torch.ones_like(real_logits)*0.9) + \
                 adv_loss(fake_logits, torch.zeros_like(fake_logits))

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ================= G =================
        z = torch.randn(b, z_dim, device=device)*0.6
        fake = G(real, z)
        fake_d = diffusion.apply_at_t(fake, t)

        fake_logits, fake_feats = D(fake_d)
        _, real_feats = D(real_d)

        identity = torch.mean(torch.abs(fake - real)) #juntei para experimentar
        l1 = torch.mean(torch.abs(fake - real)) #juntei para experimentar

        loss_G = adv_loss(fake_logits, torch.ones_like(fake_logits)*0.9) + \
                 0.4 * fm_loss(fake_feats, real_feats) + 0.8 * identity + 0.5*l1  #antes era 0.5, esse parâmetro é o f_lambda

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        loss_D_avg += loss_D.item()
        loss_G_avg += loss_G.item()

        # DEBUG intermédio
        if i % 50 == 0:
            print(f"[STEP {i}] D={loss_D.item():.3f} | G={loss_G.item():.3f}")
            print(f"real_logits: {real_logits.mean().item():.3f}")
            print(f"fake_logits: {fake_logits.mean().item():.3f}")
            print(f"logit_gap: {(real_logits.mean()-fake_logits.mean()).item():.3f}")

    print(f"\n[EPOCH {epoch+1}] D={loss_D_avg/len(loader):.3f} | G={loss_G_avg/len(loader):.3f}")

    save_samples(G, epoch+1, loader)