import argparse
import os
import random
import subprocess
from glob import glob

import torch
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim

from pytorch_fid.fid_score import calculate_fid_given_paths

# =========================================================
# UTIL FUNCTIONS
# =========================================================

def get_image_paths(folder):
    return sorted(glob(os.path.join(folder, "*.png")))

def log_to_file(filepath, text):
    with open(filepath, "a") as f:
        f.write(text + "\n")

# =========================================================
# MS-SSIM
# =========================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])



def compute_msssim(paths, num_pairs=1000):
    print(f"[INFO] Computing MS-SSIM with {num_pairs} random pairs...")

    scores = []

    for i in range(num_pairs):
        idx1, idx2 = random.sample(range(len(paths)), 2)

        img1 = transform(Image.open(paths[idx1])).unsqueeze(0)
        img2 = transform(Image.open(paths[idx2])).unsqueeze(0)

        score = ms_ssim(img1, img2, data_range=1.0, size_average=True)
        scores.append(score.item())

        if i % 100 == 0:
            print(f"[MS-SSIM] Pair {i}/{num_pairs}")

    return sum(scores) / len(scores)

# =========================================================
# MAIN
# =========================================================


def main(args):
    random.seed(42)

    fake_paths = get_image_paths(args.fake_dir)
    real_paths = get_image_paths(args.real_dir)

    print(f"[INFO] Fake images: {len(fake_paths)}")
    print(f"[INFO] Real images: {len(real_paths)}")

    assert len(fake_paths) > 0, "No fake images found"
    assert len(real_paths) > 0, "No real images found"

    # =====================================================
    # SELECT RANDOM REAL IMAGES (NO SUBSET SAVED)
    # =====================================================
    n = len(fake_paths)
    print(f"[INFO] Randomly selecting {n} real images (no files saved)...")

    real_selected = random.sample(real_paths, n)

    # =====================================================
    # FID
    # =====================================================
    if args.compute_fid:
        print("\n===== COMPUTING FID =====")

        fid = calculate_fid_given_paths(
            [args.fake_dir, args.real_dir],
            batch_size=args.batch_size,
            device='cuda:0',
            dims=2048
        )

        result = f"FID: {fid}"
        print(f"[RESULT] {result}\n")
        log_to_file("fid.txt", result)
    # =====================================================
    # CMMD
    # =====================================================
    if args.compute_cmmd:
        print("\n===== COMPUTING CMMD =====")

        cmd = [
            "python",
            os.path.join(args.cmmd_path, "main.py"),
            args.real_dir,
            args.fake_dir,
            f"--batch_size={args.batch_size}",
            f"--max_count={len(fake_paths)}"
        ]

        print("[INFO] Running CMMD command:", " ".join(cmd))

        result = subprocess.run(cmd, check=True, text=True, capture_output=True)

        print(result.stdout)
        log_to_file("cmmd.txt", result.stdout)

    # =====================================================
    # MS-SSIM
    # =====================================================
    if args.compute_msssim:
        print("\n===== COMPUTING MS-SSIM =====")

        fake_score = compute_msssim(fake_paths)
        real_score = compute_msssim(real_selected)

        result_fake = f"MS-SSIM (fake): {fake_score}"
        result_real = f"MS-SSIM (real): {real_score}"

        print(f"[RESULT] {result_fake}")
        print(f"[RESULT] {result_real}\n")

        log_to_file("msssim.txt", result_fake)
        log_to_file("msssim.txt", result_real)

    # =====================================================
    # FRD
    # =====================================================
    if args.compute_frd:
        print("\n===== COMPUTING FRD =====")

        cmd = [
            "python", "-m", "frd_score",
            args.fake_dir,
            args.real_dir,
            "--match_sample_count",
            "--verbose",
            "--num_workers=16",
            "--frd_version", "v0"
        ]

        print("[INFO] Running FRD:", " ".join(cmd))

        result = subprocess.run(cmd, check=True, text=True, capture_output=True)

        print(result.stdout)
        log_to_file("frd.txt", result.stdout)


# =========================================================
# ARGUMENTS
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--fake_dir', type=str, required=True)
    parser.add_argument('--real_dir', type=str, required=True)

    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--compute_fid', action='store_true')
    parser.add_argument('--compute_cmmd', action='store_true')
    parser.add_argument('--compute_msssim', action='store_true')
    parser.add_argument('--compute_frd', action='store_true')

    parser.add_argument('--cmmd_path', type=str, default='cmmd-pytorch')

    args = parser.parse_args()

    main(args)
