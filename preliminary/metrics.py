import os
import argparse
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

# =========================
# Utils
# =========================

def load_image(path, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # [0,1]
    ])
    img = Image.open(path).convert("L")  # CT = grayscale
    img = transform(img)  # [1,H,W]

    return img


def to_rgb(img):
    return img.repeat(3, 1, 1)


def normalize_lpips(img):
    return img * 2 - 1


def get_image_paths(folder):
    exts = ["png", "jpg", "jpeg"]
    paths = []
    for ext in exts:
        paths.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)])
    return sorted(paths)


# =========================
# Sampling 
# =========================

def match_sample_sizes(real_paths, fake_paths, seed=42):
    random.seed(seed)

    n_fake = len(fake_paths)
    n_real = len(real_paths)

    if n_real > n_fake:
        real_paths = random.sample(real_paths, n_fake)
    elif n_fake > n_real:
        fake_paths = random.sample(fake_paths, n_real)

    return real_paths, fake_paths


# =========================
# SSIM
# =========================

def compute_ssim(real_paths, fake_paths):
    from skimage.metrics import structural_similarity as ssim

    scores = []

    for r, f in zip(real_paths, fake_paths):
        img_r = np.array(Image.open(r).convert("L"))
        img_f = np.array(Image.open(f).convert("L"))

        score = ssim(img_r, img_f, data_range=255)
        scores.append(score)

    return np.mean(scores)


# =========================
# LPIPS
# =========================

def compute_lpips(real_paths, fake_paths, device):
    import lpips

    loss_fn = lpips.LPIPS(net='alex').to(device)

    scores = []

    for r, f in tqdm(zip(real_paths, fake_paths), total=len(real_paths)):
        img_r = load_image(r).to(device)
        img_f = load_image(f).to(device)

        img_r = to_rgb(img_r)
        img_f = to_rgb(img_f)

        img_r = normalize_lpips(img_r)
        img_f = normalize_lpips(img_f)

        score = loss_fn(img_r.unsqueeze(0), img_f.unsqueeze(0))
        scores.append(score.item())

    return np.mean(scores)


# =========================
# FID 
# =========================

def compute_fid(real_paths, fake_paths, repeats=1):
    from pytorch_fid.fid_score import calculate_fid_given_paths
    import tempfile
    import shutil

    fid_scores = []

    for i in range(repeats):
        # sample subset (já equilibrado antes, mas randomizamos novamente)
        real_sample = random.sample(real_paths, len(fake_paths))

        # criar pastas temporárias
        with tempfile.TemporaryDirectory() as real_tmp, tempfile.TemporaryDirectory() as fake_tmp:
            
            for j, path in enumerate(real_sample):
                shutil.copy(path, os.path.join(real_tmp, f"{j}.png"))

            for j, path in enumerate(fake_paths):
                shutil.copy(path, os.path.join(fake_tmp, f"{j}.png"))

            fid = calculate_fid_given_paths(
                [real_tmp, fake_tmp],
                batch_size=32,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                dims=2048
            )

            fid_scores.append(fid)
            print(f"FID run {i+1}: {fid:.4f}")

    return np.mean(fid_scores), np.std(fid_scores)


# =========================
# MAIN
# =========================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    real_paths = get_image_paths(args.real)
    fake_paths = get_image_paths(args.fake)

    print(f"Original: {len(real_paths)} real vs {len(fake_paths)} fake")

    # balanceamento automático
    real_paths, fake_paths = match_sample_sizes(real_paths, fake_paths, seed=args.seed)

    print(f"Using: {len(real_paths)} real vs {len(fake_paths)} fake")

    if args.ssim:
        print("\nComputing SSIM...")
        ssim_score = compute_ssim(real_paths, fake_paths)
        print(f"SSIM: {ssim_score:.4f}")

    if args.lpips:
        print("\nComputing LPIPS...")
        lpips_score = compute_lpips(real_paths, fake_paths, device)
        print(f"LPIPS: {lpips_score:.4f}")

    if args.fid:
        print("\nComputing FID...")
        mean_fid, std_fid = compute_fid(real_paths, fake_paths, repeats=args.fid_repeats)
        print(f"\nFID: {mean_fid:.4f} ± {std_fid:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--real", type=str, required=True)
    parser.add_argument("--fake", type=str, required=True)

    parser.add_argument("--ssim", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--fid", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fid_repeats", type=int, default=1)

    args = parser.parse_args()

    main(args)