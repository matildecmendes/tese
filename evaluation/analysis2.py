import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance

# =========================
# Utils
# =========================

def get_all_image_paths(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, f))
    return paths


def compute_mean_image(paths):
    mean = None
    count = 0

    for p in tqdm(paths):
        try:
            img = Image.open(p).convert("L")
            img = np.array(img, dtype=np.float32)

            if mean is None:
                mean = np.zeros_like(img)

            mean += img
            count += 1

        except Exception as e:
            print(f"Skipping {p}: {e}")

    mean /= count
    return mean


def collect_all_pixels(paths, max_images=None):
    pixels = []

    if max_images:
        paths = paths[:max_images]

    for p in tqdm(paths):
        try:
            img = Image.open(p).convert("L")
            img = np.array(img, dtype=np.float32)
            pixels.append(img.flatten())
        except:
            continue

    return np.concatenate(pixels)


def compute_histogram(pixels, bins=256):
    hist, _ = np.histogram(pixels, bins=bins, range=(0,255), density=True)
    return hist


def compute_kl(p, q):
    eps = 1e-8
    p = p + eps
    q = q + eps
    return np.sum(p * np.log(p / q))

def compute_emd(pixels_real, pixels_fake):
    return wasserstein_distance(pixels_real, pixels_fake)


def save_heatmap(img, out_path, title):
    plt.figure(figsize=(6,5))
    plt.title(title)
    plt.imshow(img, cmap="hot")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_histogram_plot(real_pixels, fake_pixels, out_path):
    plt.figure(figsize=(6,5))
    plt.hist(real_pixels, bins=100, alpha=0.5, label="Real", density=True)
    plt.hist(fake_pixels, bins=100, alpha=0.5, label="Fake", density=True)
    plt.legend()
    plt.title("Histogram Comparison")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# =========================
# Main
# =========================

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading paths...")
    real_paths = get_all_image_paths(args.real_dir)
    fake_paths = get_all_image_paths(args.fake_dir)

    print(f"Real images: {len(real_paths)}")
    print(f"Fake images: {len(fake_paths)}")

    # =========================
    # HEATMAPS
    # =========================
    if args.heatmaps:
        print("Computing mean images...")

        mean_real = compute_mean_image(real_paths)
        mean_fake = compute_mean_image(fake_paths)

        save_heatmap(mean_real, os.path.join(args.out_dir, "mean_real.png"), "Mean Real")
        save_heatmap(mean_fake, os.path.join(args.out_dir, "mean_fake.png"), "Mean Fake")

        diff = mean_real - mean_fake
        save_heatmap(diff, os.path.join(args.out_dir, "diff.png"), "Difference")

    # =========================
    # HISTOGRAMS
    # =========================
    if args.histograms or args.kl:
        print("Collecting pixels...")

        real_pixels = collect_all_pixels(real_paths, args.max_images)
        fake_pixels = collect_all_pixels(fake_paths, args.max_images)

    if args.histograms:
        save_histogram_plot(
            real_pixels,
            fake_pixels,
            os.path.join(args.out_dir, "hist_comparison.png")
        )

    # =========================
    # KL DIVERGENCE
    # =========================
    if args.kl:
        print("Computing KL divergence...")

        hist_real = compute_histogram(real_pixels)
        hist_fake = compute_histogram(fake_pixels)

        kl_rf = compute_kl(hist_real, hist_fake)
        kl_fr = compute_kl(hist_fake, hist_real)

        print(f"KL(real || fake): {kl_rf}")
        print(f"KL(fake || real): {kl_fr}")

        with open(os.path.join(args.out_dir, "kl.txt"), "w") as f:
            f.write(f"KL(real || fake): {kl_rf}\n")
            f.write(f"KL(fake || real): {kl_fr}\n")

    # =========================
    # EMD (Wasserstein)
    # =========================
    if args.emd:
        print("Computing Earth Mover Distance...")

        emd_value = compute_emd(real_pixels, fake_pixels)

        print(f"EMD: {emd_value}")

        with open(os.path.join(args.out_dir, "emd.txt"), "w") as f:
            f.write(f"EMD: {emd_value}\n")
            

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)


    parser.add_argument("--heatmaps", action="store_true")
    parser.add_argument("--histograms", action="store_true")
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--emd", action="store_true")

    # opcional (performance)
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()

    main(args)