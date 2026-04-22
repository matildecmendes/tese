import argparse
import os
import pickle
import torch
import torchvision
import numpy as np

from pytorch_fid.fid_score import calculate_fid_given_paths

# =========================
# Load model
# =========================

def load_model(model_path, device):
    print("Loading model...")

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    G = data['G_ema'].to(device)
    G.eval()

    return G

# =========================
# Sampling
# =========================

def generate_images(G, args, device):
    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    n_batches = args.num_samples // args.batch_size

    print("Start sampling...")

    for i in range(n_batches):
        z = torch.randn(args.batch_size, G.z_dim).to(device)

        # no conditioning
        c = None

        with torch.no_grad():
            img = G(z, c)

        # [-1,1] -> [0,1]
        img = (img + 1) / 2
        img = img.clamp(0, 1)

        for j in range(img.size(0)):
            idx = i * args.batch_size + j
            path = os.path.join(args.output_dir, f"{idx}.png")
            torchvision.utils.save_image(img[j], path)

        total += img.size(0)

        if i % 10 == 0:
            print(f"Generated {total}/{args.num_samples}")

    print("Sampling finished!")

# =========================
# Main
# =========================

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    G = load_model(args.model_path, device)

    generate_images(G, args, device)

    # =========================
    # FID
    # =========================
    if args.compute_fid:
        print("Computing FID...")

        paths = [args.output_dir, args.real_img_dir]
        fid = calculate_fid_given_paths(paths, batch_size=50, device=device, dims=2048)

        print(f"FID: {fid}")

# =========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--real_img_dir', type=str, default=None)
    parser.add_argument('--compute_fid', action='store_true')

    args = parser.parse_args()

    main(args)