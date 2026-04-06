class Diffusion:
    def __init__(self, T, device):
        self.T = T
        self.device = device

        beta = torch.linspace(1e-4, 0.02, T).to(device)
        alpha = 1 - beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)

    def sample_t(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device)

    def diffuse(self, x, t):
        noise = torch.randn_like(x)
        a = self.alpha_bar[t].view(-1,1,1,1)
        return torch.sqrt(a)*x + torch.sqrt(1-a)*noise