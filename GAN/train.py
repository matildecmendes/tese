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

        if idx == 0:
            print("\n[DATA DEBUG]")
            print("RAW min/max:", x.min(), x.max())

        assert x.shape == (256, 256), f"Shape errado: {x.shape}"

        if x.min() >= -1.1 and x.max() <= 1.1:
            pass
        elif x.min() >= 0.0 and x.max() <= 1.0:
            x = x * 2 - 1
        else:
            raise ValueError(f"Range inesperado: {x.min()}, {x.max()}")

        if idx == 0:
            print("FINAL min/max:", x.min(), x.max())

        return torch.from_numpy(x).unsqueeze(0)

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
# MODELS
# =========================================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super().__init__()
        self.up = up
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        h = x
        if self.up:
            h = F.interpolate(h, scale_factor=2)
            x = F.interpolate(x, scale_factor=2)

        h = F.leaky_relu(self.conv1(h), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)

        return h + self.skip(x)

class DiscResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=False):
        super().__init__()
        self.down = down
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1))

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)

        if self.down:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)

        return h + self.skip(x)

class Generator(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.fc = nn.Linear(z_dim, 512 * 4 * 4)

        self.blocks = nn.Sequential(
            ResBlock(512,256,True),
            ResBlock(256,128,True),
            ResBlock(128,64,True),
            ResBlock(64,32,True),
            ResBlock(32,16,True),
            ResBlock(16,8,True),
        )

        self.to_img = nn.Conv2d(8,1,3,padding=1)

    def forward(self, z):
        h = self.fc(z).view(-1,512,4,4)
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
        self.fc = nn.utils.spectral_norm(nn.Linear(512*4*4,1))

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
G = Generator(z_dim).to(device)
D = Discriminator().to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5,0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5,0.999))

z_fixed = torch.randn(8, z_dim, device=device)

# =========================================================
# SAVE
# =========================================================
def save_samples(G, epoch):
    G.eval()
    with torch.no_grad():
        fake = G(z_fixed).cpu()

    grid = fake.squeeze().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(4,2))

    for i in range(2):
        for j in range(4):
            axes[i, j].imshow(grid[i*4+j], cmap="gray")
            axes[i, j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(os.path.join(args.out_dir, f"epoch_{epoch}.png"),
                bbox_inches='tight', pad_inches=0)
    plt.close()

    torch.save({
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict()
    }, os.path.join(args.out_dir, f"checkpoint_{epoch}.pt"))

    print(f"[SAVE] Epoch {epoch}")

# =========================================================
# TRAIN
# =========================================================
for epoch in range(args.epochs):

    loss_D_avg = 0
    loss_G_avg = 0

    for i, real in enumerate(loader):

        real = real.to(device)
        b = real.size(0)

        # ===== D =====
        z = torch.randn(b, z_dim, device=device)
        fake = G(z).detach()

        real_logits,_ = D(real)
        fake_logits,_ = D(fake)

        loss_D = adv_loss(real_logits, torch.ones_like(real_logits)*0.9) + \
                 adv_loss(fake_logits, torch.zeros_like(fake_logits))

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ===== G =====
        z = torch.randn(b, z_dim, device=device)
        fake = G(z)

        fake_logits, fake_feats = D(fake)
        _, real_feats = D(real)

        loss_G = adv_loss(fake_logits, torch.ones_like(fake_logits)*0.9) + \
                 0.5 * fm_loss(fake_feats, real_feats)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        loss_D_avg += loss_D.item()
        loss_G_avg += loss_G.item()

        if i % 50 == 0:
            print(f"[STEP {i}] D={loss_D.item():.3f} | G={loss_G.item():.3f}")

    print(f"\n[EPOCH {epoch+1}] D={loss_D_avg/len(loader):.3f} | G={loss_G_avg/len(loader):.3f}")

    save_samples(G, epoch+1)