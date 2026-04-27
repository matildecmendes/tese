import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
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


def compute_mean_image(paths, max_images=None):
    mean = None
    count = 0

    if max_images:
        paths = paths[:max_images]

    for p in tqdm(paths):
        try:
            img = Image.open(p).convert("L")
            img = np.array(img, dtype=np.float32)

            if mean is None:
                mean = np.zeros_like(img)

            mean += img
            count += 1

        except:
            continue

    mean /= count
    return mean


def compute_histogram_streaming(paths, bins=256, max_images=None):
    hist = np.zeros(bins, dtype=np.float64)
    total_pixels = 0

    if max_images:
        paths = paths[:max_images]

    for p in tqdm(paths):
        try:
            img = Image.open(p).convert("L")
            img = np.array(img, dtype=np.float32)

            h, _ = np.histogram(img, bins=bins, range=(0,255))
            hist += h
            total_pixels += img.size

        except:
            continue

    hist /= total_pixels
    return hist


def compute_kl(p, q):
    eps = 1e-8
    p = p + eps
    q = q + eps
    return np.sum(p * np.log(p / q))


def compute_emd_from_hist(hist_real, hist_fake):
    bins = np.arange(len(hist_real))
    return wasserstein_distance(bins, bins, hist_real, hist_fake)


def save_heatmap(img, out_path, title):
    plt.figure(figsize=(6,5))
    plt.title(title)
    plt.imshow(img, cmap="hot")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_histogram_plot(hist_real, hist_fake, out_path):
    plt.figure(figsize=(6,5))
    plt.plot(hist_real, label="Real")
    plt.plot(hist_fake, label="Fake")
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

    # =====================================
    # MODO 1: COMPUTAR REAL (UMA VEZ)
    # =====================================
    if args.compute_real:
        if args.real_dir is None:
            raise ValueError("Precisas de --real_dir para calcular o real")

        print("Loading real dataset...")
        real_paths = get_all_image_paths(args.real_dir)
        print(f"Real images: {len(real_paths)}")

        print("Computing real mean...")
        mean_real = compute_mean_image(real_paths, args.max_images)

        print("Computing real histogram...")
        hist_real = compute_histogram_streaming(real_paths, args.max_images)

        np.save(os.path.join(args.out_dir, "real_mean.npy"), mean_real)
        np.save(os.path.join(args.out_dir, "real_hist.npy"), hist_real)

        print("Saved real_mean.npy and real_hist.npy")
        return

    # =====================================
    # MODO 2: ANALISAR FAKE
    # =====================================
    print("Loading fake dataset...")
    fake_paths = get_all_image_paths(args.fake_dir)
    print(f"Fake images: {len(fake_paths)}")

    # =========================
    # HEATMAP
    # =========================
    if args.heatmaps:
        print("Computing fake mean...")
        mean_fake = compute_mean_image(fake_paths, args.max_images)

        save_heatmap(mean_fake, os.path.join(args.out_dir, "mean_fake.png"), "Mean Fake")

        if args.real_mean_path:
            mean_real = np.load(args.real_mean_path)
            diff = mean_real - mean_fake
            save_heatmap(diff, os.path.join(args.out_dir, "diff.png"), "Difference")

    # =========================
    # HIST + KL + EMD
    # =========================
    if args.histograms or args.kl or args.emd:
        print("Computing fake histogram...")
        hist_fake = compute_histogram_streaming(fake_paths, args.max_images)

        if args.real_hist_path is None:
            raise ValueError("Precisas de --real_hist_path")

        hist_real = np.load(args.real_hist_path)

    if args.histograms:
        save_histogram_plot(hist_real, hist_fake,
                            os.path.join(args.out_dir, "hist_comparison.png"))

    if args.kl:
        print("Computing KL...")
        kl_rf = compute_kl(hist_real, hist_fake)
        kl_fr = compute_kl(hist_fake, hist_real)

        print(f"KL(real || fake): {kl_rf}")
        print(f"KL(fake || real): {kl_fr}")

        with open(os.path.join(args.out_dir, "kl.txt"), "w") as f:
            f.write(f"KL(real || fake): {kl_rf}\n")
            f.write(f"KL(fake || real): {kl_fr}\n")

    if args.emd:
        print("Computing EMD...")
        emd_value = compute_emd_from_hist(hist_real, hist_fake)

        print(f"EMD: {emd_value}")

        with open(os.path.join(args.out_dir, "emd.txt"), "w") as f:
            f.write(f"EMD: {emd_value}\n")

    print("Done!")


# =========================
# Args
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # MODOS
    parser.add_argument("--compute_real", action="store_true")

    # inputs
    parser.add_argument("--real_dir", type=str, default=None)
    parser.add_argument("--fake_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)

    # pré-computados
    parser.add_argument("--real_hist_path", type=str, default=None)
    parser.add_argument("--real_mean_path", type=str, default=None)

    # operações
    parser.add_argument("--heatmaps", action="store_true")
    parser.add_argument("--histograms", action="store_true")
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--emd", action="store_true")

    # performance
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()
    main(args)