import argparse
import torch
import os
import torchvision
import numpy as np

from pytorch_fid.fid_score import calculate_fid_given_paths


def sample_from_generator(generator, args, device):
    z = torch.randn(args.batch_size, args.nz).to(device)

    with torch.no_grad():
        fake = generator(z)

    return fake

# =========================
# Main sampling
# =========================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    from models.simple_generator import SimpleGenerator

    netG = SimpleGenerator(
        nz=args.nz,
        ngf=64,
        nc=args.num_channels,
        image_size=args.image_size
    ).to(device)

    ckpt = torch.load(args.model_path, map_location=device)

    # remove DDP prefix if exists
    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[k.replace("module.", "")] = v

    netG.load_state_dict(new_ckpt)
    netG.eval()


    os.makedirs(args.output_dir, exist_ok=True)

    print("Start sampling...")

    total = 0
    import math
    n_batches = math.ceil(args.num_samples / args.batch_size)

    for i in range(n_batches):
        fake = sample_from_generator(netG, args, device)
        fake = (fake + 1) / 2  # [-1,1] -> [0,1]

        for j in range(fake.size(0)):
            idx = i * args.batch_size + j

            if idx >= args.num_samples:
                break

            path = os.path.join(args.output_dir, f"{idx}.png")
            torchvision.utils.save_image(
                fake[j],
                path,
                normalize=True,
                value_range=(0, 1)
            )

        total += fake.size(0)

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

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--centered', action='store_false', default=True)

    args = parser.parse_args()

    main(args)
