#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def normalize_hu(slice_ct, min_hu=-1000, max_hu=400):
    slice_ct = np.clip(slice_ct, min_hu, max_hu)
    slice_ct = (slice_ct - min_hu) / (max_hu - min_hu)
    slice_ct = 2 * slice_ct - 1
    return slice_ct.astype(np.float32)


def save_image(img, path, cmap="gray", vmin=None, vmax=None):
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def generate_preprocessing_examples(ct_path,
                                    mask_path,
                                    output_folder,
                                    slice_idx):

    os.makedirs(output_folder, exist_ok=True)

    ct = np.load(ct_path)
    mask = np.load(mask_path)

    if ct.shape != mask.shape:
        raise ValueError("CT and mask have different shapes.")

    original = ct[slice_idx]
    lung_mask = mask[slice_idx]

    # Aplicar máscara apenas para visualização
    masked = np.where(lung_mask > 0, original, -1000)

    # Clipping HU
    clipped = np.clip(original, -1000, 400)

    # Normalização [-1,1]
    normalized = normalize_hu(original)

    # Downsampling
    resized = normalized[::2, ::2]

    print("\n" + "=" * 60)
    print("PREPROCESSING INFORMATION")
    print("=" * 60)

    print_stats("01. Original CT", original)
    print_stats("02. Lung Mask", lung_mask)
    print_stats("03. Selected Slice (Masked)", masked)
    print_stats("04. HU Clipping", clipped)
    print_stats("05. Normalized", normalized)
    print_stats("06. Downsampled", resized)

    # -------------------------
    # Guardar imagens
    # -------------------------

    save_image(
        original,
        os.path.join(output_folder, "01_original_CT.png"),
        cmap="gray",
        vmin=-1000,
        vmax=400
    )

    save_image(
        lung_mask,
        os.path.join(output_folder, "02_lung_mask.png"),
        cmap="gray"
    )

    save_image(
        masked,
        os.path.join(output_folder, "03_selected_slice.png"),
        cmap="gray",
        vmin=-1000,
        vmax=400
    )

    save_image(
        clipped,
        os.path.join(output_folder, "04_HU_clipping.png"),
        cmap="gray",
        vmin=-1000,
        vmax=400
    )

    save_image(
        normalized,
        os.path.join(output_folder, "05_normalized.png"),
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    save_image(
        resized,
        os.path.join(output_folder, "06_downsampled_256.png"),
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    print(f"Images saved to: {output_folder}")

def print_stats(name, img):
    print(f"{name}")
    print(f"  Shape : {img.shape}")
    print(f"  Dtype : {img.dtype}")
    print(f"  Min   : {np.min(img):.4f}")
    print(f"  Max   : {np.max(img):.4f}")
    print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ct", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slice", type=int, required=True)

    args = parser.parse_args()

    generate_preprocessing_examples(
        args.ct,
        args.mask,
        args.output,
        args.slice

    )