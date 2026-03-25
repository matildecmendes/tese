import torch
import torch.nn as nn

# 1. Generator: Maps noise (z) to data space
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh() # Output scaled between -1 and 1
        )

    def forward(self, z):
        return self.model(z)

# 2. Discriminator: Classifies real/fake
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # Probability output
        )

    def forward(self, img):
        return self.model(img)
