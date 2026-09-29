import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


# =========================
# PLOT STYLE
# =========================

plt.rcParams.update({
    "font.size": 14,          # tamanho base do texto
    "axes.titlesize": 18,     # títulos dos gráficos
    "axes.labelsize": 16,     # labels dos eixos
    "xtick.labelsize": 14,    # números eixo X
    "ytick.labelsize": 14,    # números eixo Y
    "legend.fontsize": 14     # legenda Real / Fake
})


# =========================
# Utils
# =========================

def get_all_image_paths(folder):
    paths = []

    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
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

            h, _ = np.histogram(
                img,
                bins=bins,
                range=(0, 255)
            )

            hist += h
            total_pixels += img.size

        except Exception as e:
            print(f"Erro a ler {p}: {e}")

    if total_pixels == 0:
        raise ValueError("Nenhum pixel válido encontrado!")

    hist /= total_pixels

    # garantir normalização correta
    hist = hist / np.sum(hist)

    return hist


def compute_kl(p, q):

    eps = 1e-8

    p = p + eps
    q = q + eps

    # normalizar (evita NaN)
    p = p / np.sum(p)
    q = q / np.sum(q)

    return np.sum(p * np.log(p / q))


def compute_emd(hist_real, hist_fake):

    bins = np.arange(len(hist_real))

    return wasserstein_distance(
        bins,
        bins,
        hist_real,
        hist_fake
    )


# =========================
# HEATMAP
# =========================

def save_heatmap(img, path, title):

    plt.figure(figsize=(6, 5))

    plt.imshow(
        img,
        cmap="hot"
    )

    # título maior
    plt.title(
        title,
        fontsize=18
    )

    # colorbar com números maiores
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)

    # números dos eixos maiores
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# HISTOGRAMA INDIVIDUAL
# =========================

def save_hist_single(hist, path, title):

    plt.figure(figsize=(6, 5))

    x = np.arange(len(hist))

    plt.bar(
        x,
        hist,
        color="tab:blue",
        edgecolor="tab:blue"
    )

    # plt.yscale("log")

    plt.ylim(0, 0.02)
    plt.xlim(1, 250)

    plt.title(
        title,
        fontsize=18
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# HISTOGRAMA REAL VS FAKE
# =========================

def save_hist_comparison(hist_real, hist_fake, path):

    plt.figure(figsize=(6, 5))

    x = np.arange(len(hist_real))

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

    # plt.yscale("log")

    plt.ylim(0, 0.02)
    plt.xlim(1, 250)

    # legenda maior
    plt.legend(
        fontsize=14
    )

    # título maior AQUI
    plt.title(
        "Diffusion-StyleGAN2",
        fontsize=18
    )

    # escalas maiores
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# HISTOGRAMA OVERLAY
# =========================

def save_hist_overlay(hist_real, hist_fake, path):

    plt.figure(figsize=(6, 5))

    x = np.arange(len(hist_real))

    plt.plot(
        x,
        hist_real,
        label="Real",
        linewidth=2
    )

    plt.plot(
        x,
        hist_fake,
        label="Fake",
        linewidth=2
    )

    plt.legend(
        fontsize=14
    )

    plt.title(
        "Histogram Overlay",
        fontsize=18
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# DIFERENÇA DOS HISTOGRAMAS
# =========================

def save_hist_difference(hist_real, hist_fake, path):

    plt.figure(figsize=(6, 5))

    x = np.arange(len(hist_real))

    diff = hist_real - hist_fake

    plt.bar(
        x,
        diff
    )

    plt.axhline(
        0
    )

    plt.title(
        "Histogram Difference (Real - Fake)",
        fontsize=18
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# Main
# =========================

def main(args):

    os.makedirs(
        args.out_dir,
        exist_ok=True
    )

    # =========================
    # GERAR HEATMAP REAL DIRETAMENTE
    # =========================

    if args.real_heatmap:

        if args.real_dir is None:
            raise ValueError("Precisas de --real_dir")

        real_paths = get_all_image_paths(
            args.real_dir
        )

        print(
            "Real images:",
            len(real_paths)
        )

        mean_real = compute_mean_image(
            real_paths,
            max_images=args.max_images
        )

        save_heatmap(
            mean_real,
            os.path.join(
                args.out_dir,
                "mean_real.png"
            ),
            "Real Dataset Mean Heatmap"
        )

        print("✔ mean_real.png guardado")

        return


    # =========================
    # GUARDAR HISTOGRAMA REAL
    # =========================

    if args.save_real_hist:

        if args.real_hist_path is None:
            raise ValueError(
                "Precisas de --real_hist_path"
            )

        hist_real = np.load(
            args.real_hist_path
        )

        save_hist_single(
            hist_real,
            os.path.join(
                args.out_dir,
                "hist_real_nlst.png"
            ),
            "Real (NLST) Dataset Histogram"
        )

        print("✔ hist_real.png guardado")

        return


    # =========================
    # GUARDAR HEATMAP REAL
    # =========================

    if args.save_real_heatmap:

        if args.real_mean_path is None:
            raise ValueError(
                "Precisas de --real_mean_path"
            )

        mean_real = np.load(
            args.real_mean_path
        )

        save_heatmap(
            mean_real,
            os.path.join(
                args.out_dir,
                "mean_real.png"
            ),
            "Real Dataset Mean Heatmap"
        )

        print("✔ mean_real.png guardado")

        return


    # =====================================
    # MODO 1: GERAR REAL (.npy)
    # =====================================

    if args.compute_real:

        if args.real_dir is None:
            raise ValueError(
                "Precisas de --real_dir"
            )

        real_paths = get_all_image_paths(
            args.real_dir
        )

        print(
            "Real images:",
            len(real_paths)
        )

        mean_real = compute_mean_image(
            real_paths,
            args.max_images
        )

        hist_real = compute_histogram_streaming(
            real_paths,
            max_images=args.max_images
        )

        print(
            "Hist real sum:",
            np.sum(hist_real)
        )

        np.save(
            os.path.join(
                args.out_dir,
                "real_mean.npy"
            ),
            mean_real
        )

        np.save(
            os.path.join(
                args.out_dir,
                "real_hist.npy"
            ),
            hist_real
        )

        print(
            "✔ real_mean.npy e real_hist.npy guardados"
        )

        return


    # =====================================
    # MODO 2: ANALISAR FAKE
    # =====================================

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


    # =========================
    # HEATMAP
    # =========================

    if args.heatmaps:

        mean_fake = compute_mean_image(
            fake_paths,
            args.max_images
        )

        save_heatmap(
            mean_fake,
            os.path.join(
                args.out_dir,
                "mean_fake.png"
            ),
            "DDGAN (NLST) Generated Dataset Mean Heatmap"
        )  # MUDAR AQUI O TITULO


        if args.real_mean_path:

            mean_real = np.load(
                args.real_mean_path
            )

            diff = mean_real - mean_fake

            # Se quiseres diferença absoluta:
            # diff = np.abs(mean_real - mean_fake)

            save_heatmap(
                diff,
                os.path.join(
                    args.out_dir,
                    "diff.png"
                ),
                "Diffusion-StyleGAN2" #AQUI
            )


    # =========================
    # HISTOGRAMA FAKE
    # =========================

    if args.histograms or args.kl or args.emd:

        hist_fake = compute_histogram_streaming(
            fake_paths,
            max_images=args.max_images
        )

        print(
            "Hist fake sum:",
            np.sum(hist_fake)
        )

        np.save(
            os.path.join(
                args.out_dir,
                "hist_fake.npy"
            ),
            hist_fake
        )

        if args.histograms:

            save_hist_single(
                hist_fake,
                os.path.join(
                    args.out_dir,
                    "hist_fake.png"
                ),
                "DDPM Generated Dataset Histogram"
            )  # MUDAR AQUI O TITULO


    # =========================
    # COMPARAÇÃO
    # =========================

    if args.real_hist_path and (
        args.kl
        or args.emd
        or args.compare_hist
    ):

        hist_real = np.load(
            args.real_hist_path
        )

        print(
            "Hist real sum:",
            np.sum(hist_real)
        )


        if args.compare_hist:

            # barras
            save_hist_comparison(
                hist_real,
                hist_fake,
                os.path.join(
                    args.out_dir,
                    "hist_comparison.png"
                )
            )

            # overlay linhas
            save_hist_overlay(
                hist_real,
                hist_fake,
                os.path.join(
                    args.out_dir,
                    "hist_overlay.png"
                )
            )

            # diferença
            save_hist_difference(
                hist_real,
                hist_fake,
                os.path.join(
                    args.out_dir,
                    "hist_difference.png"
                )
            )


        if args.kl:

            kl_rf = compute_kl(
                hist_real,
                hist_fake
            )

            kl_fr = compute_kl(
                hist_fake,
                hist_real
            )

            print(
                "KL(real||fake):",
                kl_rf
            )

            print(
                "KL(fake||real):",
                kl_fr
            )


        if args.emd:

            emd = compute_emd(
                hist_real,
                hist_fake
            )

            print(
                "EMD:",
                emd
            )


    elif args.kl or args.emd:

        raise ValueError(
            "Precisas de --real_hist_path para KL/EMD"
        )


    print("✔ Done!")


# =========================
# Args
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    # modos
    parser.add_argument(
        "--compute_real",
        action="store_true"
    )

    parser.add_argument(
        "--save_real_hist",
        action="store_true"
    )

    parser.add_argument(
        "--save_real_heatmap",
        action="store_true"
    )


    # dirs
    parser.add_argument(
        "--real_dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--fake_dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True
    )


    # ficheiros
    parser.add_argument(
        "--real_hist_path",
        type=str,
        default=None
    )

    parser.add_argument(
        "--real_mean_path",
        type=str,
        default=None
    )


    # flags
    parser.add_argument(
        "--heatmaps",
        action="store_true"
    )

    parser.add_argument(
        "--histograms",
        action="store_true"
    )

    parser.add_argument(
        "--compare_hist",
        action="store_true"
    )

    parser.add_argument(
        "--kl",
        action="store_true"
    )

    parser.add_argument(
        "--emd",
        action="store_true"
    )

    parser.add_argument(
        "--real_heatmap",
        action="store_true"
    )


    parser.add_argument(
        "--max_images",
        type=int,
        default=None
    )


    args = parser.parse_args()

    main(args)