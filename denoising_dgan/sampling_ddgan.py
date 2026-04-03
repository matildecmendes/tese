import argparse
import torch
import os
import torchvision
import numpy as np

from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from pytorch_fid.fid_score import calculate_fid_given_paths

# =========================
# Diffusion utils (same as train)
# =========================

def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def get_time_schedule(args, device):
    t = np.linspace(0, 1, args.num_timesteps + 1)
    t = torch.from_numpy(t).float().to(device)
    return t

def get_sigma_schedule(args, device):
    t = torch.linspace(0, 1, args.num_timesteps + 1).to(device)

    if args.use_geometric:
        var = var_func_geometric(t, args.beta_min, args.beta_max)
    else:
        var = var_func_vp(t, args.beta_min, args.beta_max)

    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    betas = torch.cat([torch.tensor([1e-8], device=device), betas]).float()
    return betas

class Posterior_Coefficients():
    def __init__(self, args, device):
        betas = get_sigma_schedule(args, device)[1:]
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.], device=device), self.alphas_cumprod[:-1]
        ])

        self.posterior_variance = betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = (betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))


def sample_posterior(coefficients, x_0, x_t, t):
    mean = (
        extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
        + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    log_var = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)

    noise = torch.randn_like(x_t)
    nonzero_mask = (1 - (t == 0).float())

    return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise


def sample_from_model(coefficients, generator, args, device):
    x = torch.randn(args.batch_size, args.num_channels, args.image_size, args.image_size).to(device)

    with torch.no_grad():
        for i in reversed(range(args.num_timesteps)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(device)
            z = torch.randn(x.size(0), args.nz).to(device)

            x0 = generator(x, t, z)
            x = sample_posterior(coefficients, x0, x, t)

    return x

# =========================
# Main sampling
# =========================

def main(args):
    device = 'cuda:0'

    print("Loading model...")
    netG = NCSNpp(args).to(device)

    ckpt = torch.load(args.model_path, map_location=device)

    # remove DDP prefix if exists
    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[k.replace("module.", "")] = v

    netG.load_state_dict(new_ckpt)
    netG.eval()

    coeff = Posterior_Coefficients(args, device)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Start sampling...")

    total = 0
    n_batches = args.num_samples // args.batch_size

    for i in range(n_batches):
        fake = sample_from_model(coeff, netG, args, device)
        fake = (fake + 1) / 2  # [-1,1] -> [0,1]

        for j in range(fake.size(0)):
            idx = i * args.batch_size + j
            path = os.path.join(args.output_dir, f"{idx}.png")
            torchvision.utils.save_image(fake[j], path)

        total += fake.size(0)
        print(f"Generated {total}/{args.num_samples}")

    print("Sampling finished!")

    # =========================
    # FID
    # =========================
    if args.compute_fid:
        print("Computing FID...")

        paths = [args.output_dir, args.real_img_dir]
        fid = calculate_fid_given_paths(paths, batch_size=50, device=device, dims=2048)

        print(f"FID: {fid}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_path', type=str, required=True)

    # data
    parser.add_argument('--real_img_dir', type=str, default=None)

    # output
    parser.add_argument('--output_dir', type=str, required=True)

    # sampling
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)

    # diffusion
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
    parser.add_argument('--use_geometric', action='store_true', default=False)

    # model config (IMPORTANT: must match training)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_channels_dae', type=int, default=128)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1,2,2,4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attn_resolutions', default=(16,))
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--resamp_with_conv', action='store_false', default=True)
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--fir', action='store_false', default=True)
    parser.add_argument('--fir_kernel', default=[1,3,3,1])
    parser.add_argument('--skip_rescale', action='store_false', default=True)
    parser.add_argument('--resblock_type', default='biggan')
    parser.add_argument('--progressive', type=str, default='none')
    parser.add_argument('--progressive_input', type=str, default='residual')
    parser.add_argument('--progressive_combine', type=str, default='sum')
    parser.add_argument('--embedding_type', type=str, default='positional')
    parser.add_argument('--fourier_scale', type=float, default=16.)
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    parser.add_argument('--compute_fid', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
