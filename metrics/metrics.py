import os
import argparse
import numpy as np
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
    # [1,H,W] -> [3,H,W]
    return img.repeat(3, 1, 1)


def normalize_lpips(img):
    # [0,1] -> [-1,1]
    return img * 2 - 1


def get_image_paths(folder):
    exts = ["png", "jpg", "jpeg"]
    paths = []
    for ext in exts:
        paths.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)])
    return sorted(paths)


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

def compute_fid(real_folder, fake_folder):
    import subprocess

    cmd = [
        "python", "-m", "pytorch_fid",
        real_folder,
        fake_folder
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)


# =========================
# MAIN
# =========================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    real_paths = get_image_paths(args.real)
    fake_paths = get_image_paths(args.fake)

    print(f"Found {len(real_paths)} real and {len(fake_paths)} fake images")

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
        compute_fid(args.real, args.fake)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--real", type=str, required=True)
    parser.add_argument("--fake", type=str, required=True)

    parser.add_argument("--ssim", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--fid", action="store_true")

    args = parser.parse_args()

    main(args)