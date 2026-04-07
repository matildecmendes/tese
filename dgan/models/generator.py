import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()

        self.fc = nn.Linear(z_dim, 512*8*8)

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
        )

    def forward(self, z):
        x = self.fc(z).view(-1,512,8,8)
        x = self.net(x)
        return torch.tanh(x)