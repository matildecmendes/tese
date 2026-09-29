import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# =========================
# PLOT STYLE
# =========================

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})


# =========================
# AUXILIARY FUNCTIONS
# =========================

def get_all_image_paths(folder):

    paths = []

    for root, _, files in os.walk(folder):
        for f in files:

            if f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp")
            ):
                paths.append(
                    os.path.join(root, f)
                )

    return sorted(paths)


def compute_mean_image(paths, max_images=None):

    mean = None
    count = 0

    if max_images is not None:
        paths = paths[:max_images]

    for p in tqdm(paths):

        try:

            img = Image.open(p).convert("L")
            img = np.array(
                img,
                dtype=np.float32
            )

            if mean is None:
                mean = np.zeros_like(
                    img,
                    dtype=np.float64
                )

            mean += img
            count += 1

        except Exception as e:

            print(
                f"Erro a ler {p}: {e}"
            )

    if count == 0:

        raise ValueError(
            "Nenhuma imagem válida encontrada!"
        )

    return mean / count


def compute_histogram(paths, bins=256, max_images=None):

    hist = np.zeros(
        bins,
        dtype=np.float64
    )

    total_pixels = 0

    if max_images is not None:
        paths = paths[:max_images]

    for p in tqdm(paths):

        try:

            img = Image.open(p).convert("L")

            img = np.array(
                img,
                dtype=np.float32
            )

            h, _ = np.histogram(
                img,
                bins=bins,
                range=(0, 255)
            )

            hist += h
            total_pixels += img.size

        except Exception as e:

            print(
                f"Erro a ler {p}: {e}"
            )

    if total_pixels == 0:

        raise ValueError(
            "Nenhum pixel válido encontrado!"
        )

    hist /= total_pixels

    # garantir que soma 1
    hist /= np.sum(hist)

    return hist


# =========================
# REAL MEAN HEATMAP
# =========================

def save_real_heatmap(mean_real, path):

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        mean_real,
        cmap="hot",
        vmin=0,
        vmax=255
    )

    plt.title(
        "Real",
        fontsize=18
    )

    cbar = plt.colorbar()

    cbar.ax.tick_params(
        labelsize=14
    )

    plt.xticks(
        fontsize=14
    )

    plt.yticks(
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# HISTOGRAM COMPARISON
# =========================

def save_hist_comparison(
    hist_real,
    hist_fake,
    path,
    title
):

    plt.figure(
        figsize=(6, 5)
    )

    x = np.arange(
        len(hist_real)
    )

    plt.bar(
        x,
        hist_real,
        color="tab:blue",
        edgecolor="tab:blue",
        alpha=0.5,
        label="Real"
    )

    plt.bar(
        x,
        hist_fake,
        color="tab:orange",
        edgecolor="tab:orange",
        alpha=0.6,
        label="Fake"
    )

    plt.ylim(
        0,
        0.02
    )

    plt.xlim(
        1,
        250
    )

    plt.legend(
        fontsize=14
    )

    plt.title(
        title,
        fontsize=18
    )

    plt.xticks(
        fontsize=14
    )

    plt.yticks(
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# MEAN HEATMAP DIFFERENCE
# =========================

def save_difference_heatmap(
    mean_real,
    mean_fake,
    path,
    title
):

    # Real - Fake
    diff = mean_real - mean_fake

    # escala simétrica:
    # por exemplo -30 e +30
    max_abs = np.max(
        np.abs(diff)
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        diff,
        cmap="bwr",
        vmin=-max_abs,
        vmax=max_abs
    )

    plt.title(
        title,
        fontsize=18
    )

    cbar = plt.colorbar()

    cbar.ax.tick_params(
        labelsize=14
    )

    plt.xticks(
        fontsize=14
    )

    plt.yticks(
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# MAIN
# =========================

def main(args):

    os.makedirs(
        args.out_dir,
        exist_ok=True
    )


    # =====================================
    # ONLY SAVE REAL HEATMAP
    # =====================================

    if args.save_real_heatmap:

        if args.real_mean_path is None:

            raise ValueError(
                "Precisas de indicar --real_mean_path"
            )

        mean_real = np.load(
            args.real_mean_path
        )

        save_real_heatmap(
            mean_real,
            os.path.join(
                args.out_dir,
                "mean_real.png"
            )
        )

        print(
            "✔ Real heatmap guardado"
        )

        return


    # =====================================
    # REAL DATA
    # =====================================

    if args.real_mean_path is None:

        raise ValueError(
            "Precisas de indicar --real_mean_path"
        )

    if args.real_hist_path is None:

        raise ValueError(
            "Precisas de indicar --real_hist_path"
        )

    mean_real = np.load(
        args.real_mean_path
    )

    hist_real = np.load(
        args.real_hist_path
    )


    # =====================================
    # FAKE DATA
    # =====================================

    if args.fake_dir is None:

        raise ValueError(
            "Precisas de indicar --fake_dir"
        )

    fake_paths = get_all_image_paths(
        args.fake_dir
    )

    print(
        "Fake images:",
        len(fake_paths)
    )

    if len(fake_paths) == 0:

        raise ValueError(
            "Fake dataset vazio!"
        )


    # =====================================
    # FAKE MEAN
    # =====================================

    print(
        "A calcular mean image..."
    )

    mean_fake = compute_mean_image(
        fake_paths,
        max_images=args.max_images
    )


    # =====================================
    # FAKE HISTOGRAM
    # =====================================

    print(
        "A calcular histogram..."
    )

    hist_fake = compute_histogram(
        fake_paths,
        max_images=args.max_images
    )


    # =====================================
    # HISTOGRAM COMPARISON
    # =====================================

    save_hist_comparison(
        hist_real,
        hist_fake,
        os.path.join(
            args.out_dir,
            "hist_comparison.png"
        ),
        args.title
    )


    # =====================================
    # MEAN HEATMAP DIFFERENCE
    # =====================================

    save_difference_heatmap(
        mean_real,
        mean_fake,
        os.path.join(
            args.out_dir,
            "mean_difference.png"
        ),
        args.title
    )


    print(
        "✔ hist_comparison.png guardado"
    )

    print(
        "✔ mean_difference.png guardado"
    )

    print(
        "✔ Done!"
    )


# =========================
# ARGUMENTS
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--fake_dir",
        type=str,
        default=None
    )


    parser.add_argument(
        "--real_mean_path",
        type=str,
        default=None
    )


    parser.add_argument(
        "--real_hist_path",
        type=str,
        default=None
    )


    parser.add_argument(
        "--out_dir",
        type=str,
        required=True
    )


    parser.add_argument(
        "--title",
        type=str,
        default="Model"
    )


    parser.add_argument(
        "--max_images",
        type=int,
        default=None
    )


    parser.add_argument(
        "--save_real_heatmap",
        action="store_true"
    )


    args = parser.parse_args()

    main(args)