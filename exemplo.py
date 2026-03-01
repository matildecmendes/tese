import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ---------------------------
# 1️⃣  Gerador (Generator)
# ---------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_dim=28*28):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh()  # valores entre -1 e 1
        )

    def forward(self, z):
        return self.model(z)


# ---------------------------
# 2️⃣  Discriminador (Discriminator)
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


# ---------------------------
# 3️⃣  Simples "diffusion step"
#     (adiciona e remove ruído)
# ---------------------------
def diffusion_step(x, t, noise_level=0.1):
    noise = torch.randn_like(x) * noise_level * t
    return x + noise


# ---------------------------
# 4️⃣  Treino simplificado
# ---------------------------
def train():
    epochs = 5
    batch_size = 32
    latent_dim = 100
    img_dim = 28 * 28

    G = Generator(latent_dim, img_dim)
    D = Discriminator(img_dim)

    optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for _ in range(100):  # 100 batches fake
            # ruído de entrada (vetor latente)
            z = torch.randn(batch_size, latent_dim)
            fake_imgs = G(z)

            # "passo de difusão" (adiciona ruído)
            fake_imgs_noisy = diffusion_step(fake_imgs, t=0.5)

            # treino do discriminador
            real_imgs = torch.randn(batch_size, img_dim)  # dados reais simulados
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            optimizer_D.zero_grad()
            real_loss = F.binary_cross_entropy(D(real_imgs), real_labels)
            fake_loss = F.binary_cross_entropy(D(fake_imgs_noisy.detach()), fake_labels)
            D_loss = real_loss + fake_loss
            D_loss.backward()
            optimizer_D.step()

            # treino do gerador
            optimizer_G.zero_grad()
            G_loss = F.binary_cross_entropy(D(fake_imgs_noisy), real_labels)
            G_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{epochs}] | D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f}")


if __name__ == "__main__":
    train()
