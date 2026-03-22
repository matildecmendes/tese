import torch


def extract(a, t, shape):
    out = a.gather(0, t)
    return out.view(shape[0], 1, 1, 1)


def get_diffusion_coefficients(T, device):
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)


    coeff = {}
    coeff["betas"] = betas
    coeff["alphas"] = alphas
    coeff["alpha_bar"] = alpha_bar
    coeff["sigma"] = torch.sqrt(betas)
    coeff["a"] = torch.sqrt(alphas)
    coeff["sigma_bar"] = torch.sqrt(1 - alpha_bar)


    coeff["coef1"] = betas / (1 - alpha_bar)
    coeff["coef2"] = (1 - alphas) / (1 - alpha_bar)
    coeff["var"] = betas


    return coeff


def q_sample(coeff, x0, t, noise):
    return extract(coeff["alpha_bar"], t, x0.shape) * x0 + \
           extract(coeff["sigma_bar"], t, x0.shape) * noise


def q_sample_pairs(coeff, x0, t):
    noise = torch.randn_like(x0)
    xt = q_sample(coeff, x0, t, noise)


    t_plus = torch.clamp(t + 1, max=len(coeff["betas"]) - 1)


    xt_plus = extract(coeff["a"], t_plus, x0.shape) * xt + \
              extract(coeff["sigma"], t_plus, x0.shape) * noise


    return xt, xt_plus


def sample_posterior(coeff, x0, xt, t):
    mean = (
        extract(coeff["coef1"], t, xt.shape) * x0 +
        extract(coeff["coef2"], t, xt.shape) * xt
    )


    noise = torch.randn_like(xt)
    mask = (t != 0).float().view(-1,1,1,1)


    return mean + mask * torch.sqrt(coeff["var"][t]).view(-1,1,1,1) * noise


@torch.no_grad()
def sample_ddgan(G, coeff, n_samples, T, nz, device):
    x = torch.randn(n_samples, 1, 256, 256).to(device)


    for t in reversed(range(T)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        z = torch.randn(n_samples, nz, device=device)


        x0 = G(x, t_tensor, z)
        x = sample_posterior(coeff, x0, x, t_tensor)


    return x
