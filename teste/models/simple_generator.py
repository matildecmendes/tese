import torch
import torch.nn as nn

class SimpleGenerator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=1, image_size=256):
        super().__init__()

        self.init_res = image_size // 16  # 16x downsample
        self.fc = nn.Linear(nz, ngf * 8 * self.init_res * self.init_res)

        self.net = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, 3, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 2, 3, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, nc, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), -1, self.init_res, self.init_res)
        return self.net(x)