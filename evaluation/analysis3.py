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
    return sorted(paths)


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

        except Exception as e:
            print(f"Erro a ler {p}: {e}")

    if count == 0:
        raise ValueError("Nenhuma imagem válida!")

    return mean / count


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

        except Exception as e:
            print(f"Erro a ler {p}: {e}")

    if total_pixels == 0:
        raise ValueError("Nenhum pixel válido encontrado!")

    hist /= total_pixels
    return hist


def compute_kl(p, q):
    eps = 1e-8
    p = p + eps
    q = q + eps
    return np.sum(p * np.log(p / q))


def compute_emd(hist_real, hist_fake):
    bins = np.arange(len(hist_real))
    return wasserstein_distance(bins, bins, hist_real, hist_fake)


def save_heatmap(img, path, title):
    plt.figure(figsize=(6,5))
    plt.imshow(img, cmap="hot")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_hist_plot(hist_real, hist_fake, path):
    plt.figure(figsize=(6,5))
    plt.plot(hist_real, label="Real")
    plt.plot(hist_fake, label="Fake")
    plt.legend()
    plt.title("Histogram Comparison")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =========================
# Main
# =========================

def main(args):

    os.makedirs(args.out_dir, exist_ok=True)

    # =========================
    # LOAD PATHS
    # =========================
    real_paths = get_all_image_paths(args.real_dir)
    fake_paths = get_all_image_paths(args.fake_dir)

    print("Real:", len(real_paths))
    print("Fake:", len(fake_paths))

    # =========================
    # HEATMAPS
    # =========================
    if args.heatmaps:
        mean_real = compute_mean_image(real_paths, args.max_images)
        mean_fake = compute_mean_image(fake_paths, args.max_images)

        save_heatmap(mean_real, os.path.join(args.out_dir, "mean_real.png"), "Mean Real")
        save_heatmap(mean_fake, os.path.join(args.out_dir, "mean_fake.png"), "Mean Fake")

        diff = mean_real - mean_fake
        save_heatmap(diff, os.path.join(args.out_dir, "diff.png"), "Difference")

    # =========================
    # HISTOGRAMS
    # =========================
    if args.histograms or args.kl or args.emd:
        print("Computing histograms...")

        hist_real = compute_histogram_streaming(real_paths, max_images=args.max_images)
        hist_fake = compute_histogram_streaming(fake_paths, max_images=args.max_images)

        print("Hist sums:", np.sum(hist_real), np.sum(hist_fake))

    if args.histograms:
        save_hist_plot(hist_real, hist_fake,
                       os.path.join(args.out_dir, "hist_comparison.png"))

    # =========================
    # KL
    # =========================
    if args.kl:
        kl_rf = compute_kl(hist_real, hist_fake)
        kl_fr = compute_kl(hist_fake, hist_real)

        print("KL(real||fake):", kl_rf)
        print("KL(fake||real):", kl_fr)

    # =========================
    # EMD
    # =========================
    if args.emd:
        emd = compute_emd(hist_real, hist_fake)
        print("EMD:", emd)

    print("Done!")


# =========================
# Args
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--heatmaps", action="store_true")
    parser.add_argument("--histograms", action="store_true")
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--emd", action="store_true")

    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()
    main(args)