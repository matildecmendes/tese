import torch
import torch.nn as nn
import torch.nn.functional as F

def SN(x): return nn.utils.spectral_norm(x)

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, t):
        half = 64
        emb = torch.exp(torch.arange(half, device=t.device) * -0.1)
        emb = t[:,None].float() * emb[None,:]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = F.relu(self.lin1(emb))
        return self.lin2(emb)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.t_embed = TimestepEmbedding(128)

        self.conv1 = SN(nn.Conv2d(1,64,4,2,1))
        self.conv2 = SN(nn.Conv2d(64,128,4,2,1))
        self.conv3 = SN(nn.Conv2d(128,256,4,2,1))
        self.conv4 = SN(nn.Conv2d(256,512,4,2,1))

        self.fc = SN(nn.Linear(512,1))

    def forward(self, x, t):
        t_emb = self.t_embed(t).unsqueeze(-1).unsqueeze(-1)

        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2(x),0.2)
        x = F.leaky_relu(self.conv3(x),0.2)
        x = F.leaky_relu(self.conv4(x),0.2)

        x = x + t_emb  # 🔥 conditioning simples mas eficaz

        x = x.mean([2,3])
        return self.fc(x)