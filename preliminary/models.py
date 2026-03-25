import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Generator (Diffusion part)
# =========================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Discriminator
# =========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 64 * 64, 1)
        )

    def forward(self, x):
        return self.net(x)