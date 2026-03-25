import torch

class Diffusion:
    def __init__(self, T=30, beta_start=1e-4, beta_end=0.002, device="cuda"):
        self.T = T
        self.device = device

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