import torch

class Diffusion:
    def __init__(self, T_max=200, T_min=5, sigma=1.0, device="cuda"):
        self.T_max = T_max
        self.T_min = T_min
        self.T = T_min
        self.device = device
        self.sigma = sigma

        self.d_target = 0.6
        self.C = 1

        self.update_schedule()

    def update_schedule(self):
        betas = torch.linspace(1e-4, 0.02, self.T, device=self.device)
        alphas = 1 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def sample_t(self, b):
        return torch.randint(0, self.T, (b,), device=self.device)

    def apply(self, x, t):
        alpha_bar = self.alpha_bars[t].view(-1,1,1,1)
        noise = torch.randn_like(x)
        return torch.sqrt(alpha_bar)*x + torch.sqrt(1-alpha_bar)*noise

    def update_T(self, real_logits, fake_logits):
        with torch.no_grad():
            real_prob = torch.sigmoid(real_logits).mean()
            fake_prob = torch.sigmoid(fake_logits).mean()

            d_acc = real_prob - fake_prob

            if d_acc > self.d_target:
                self.T = min(self.T + self.C, self.T_max)
            else:
                self.T = max(self.T - self.C, self.T_min)

            self.update_schedule()