import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

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


def save_heatmap(img, out_path, title):
    plt.figure(figsize=(6,5))
    plt.title(title)
    plt.imshow(img, cmap="hot")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_histogram(img, out_path, title):
    plt.figure(figsize=(6,5))
    plt.hist(img.flatten(), bins=100)
    plt.title(title)
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
    # Mean images
    # =========================
    print("Computing mean real...")
    mean_real = compute_mean_image(real_paths)

    print("Computing mean fake...")
    mean_fake = compute_mean_image(fake_paths)

    # =========================
    # Heatmaps
    # =========================
    save_heatmap(mean_real, os.path.join(args.out_dir, "mean_real.png"), "Mean Real")
    save_heatmap(mean_fake, os.path.join(args.out_dir, "mean_fake.png"), "Mean Fake")

    diff = mean_real - mean_fake
    save_heatmap(diff, os.path.join(args.out_dir, "diff.png"), "Difference")

    # =========================
    # Random histograms
    # =========================
    real_sample = np.array(Image.open(random.choice(real_paths)).convert("L"))
    fake_sample = np.array(Image.open(random.choice(fake_paths)).convert("L"))

    save_histogram(real_sample, os.path.join(args.out_dir, "hist_real.png"), "Real Histogram")
    save_histogram(fake_sample, os.path.join(args.out_dir, "hist_fake.png"), "Fake Histogram")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    main(args)