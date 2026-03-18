#!/usr/bin/env python3

import os
import numpy as np
import argparse


# Função de normalização HU para pulmão


def normalize_hu(slice_ct, min_hu=-1000, max_hu=400):
    slice_ct = np.clip(slice_ct, min_hu, max_hu)
    slice_ct = (slice_ct - min_hu) / (max_hu - min_hu)  # [0,1]
    slice_ct = 2 * slice_ct - 1                        # [-1,1]
    return slice_ct.astype(np.float32)


# Função de processamento


def preprocess_dataset(ct_folder, mask_folder, output_folder):

    # Validar que os caminhos são pastas
    if not os.path.isdir(ct_folder):
        raise ValueError(f"CT folder does not exist or is not a directory: {ct_folder}")

    if not os.path.isdir(mask_folder):
        raise ValueError(f"Mask folder does not exist or is not a directory: {mask_folder}")

    os.makedirs(output_folder, exist_ok=True)

    # Apenas ficheiros .npy reais
    ct_files = sorted([
        f for f in os.listdir(ct_folder)
        if f.endswith(".npy") and os.path.isfile(os.path.join(ct_folder, f))
    ])  # DEBUG: [:1] para testar só um paciente

    if len(ct_files) == 0:
        raise ValueError("No .npy files found in CT folder.")

    total_patients = 0
    total_slices_kept = 0
    total_slices_discarded = 0

    for file_name in ct_files:

        total_patients += 1

        print("\n" + "=" * 60)
        print(f"Processing patient: {file_name}")
        print("=" * 60)

        ct_path = os.path.join(ct_folder, file_name)
        mask_path = os.path.join(mask_folder, file_name)

        if not os.path.isfile(mask_path):
            print("Mask file not found. Skipping.")
            continue

        try:
            ct = np.load(ct_path)
            mask = np.load(mask_path)
        except Exception as e:
            print(f"Error loading file: {e}")
            continue

        if ct.shape != mask.shape:
            print("Shape mismatch. Skipping.")
            continue

        print(f"Original shape: {ct.shape}")

        processed_slices = []
        kept = 0
        discarded = 0
        total_slices = mask.shape[0]

        # Loop pelas slices
        for i in range(total_slices):

            # -------- Barra de progresso --------
            progress = (i + 1) / total_slices
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "#" * filled_length + "-" * (bar_length - filled_length)

            print(
                f"\rProcessing slices: [{bar}] "
                f"{int(progress*100)}% ({i+1}/{total_slices})",
                end="",
                flush=True
            )
            # ------------------------------------

            if np.sum(mask[i]) > 0:

                slice_ct = ct[i]

                # Normalização HU
                slice_normalized = normalize_hu(slice_ct)

                # Downsampling 512 -> 256
                slice_resized = slice_normalized[::2, ::2]

                processed_slices.append(slice_resized)
                kept += 1
            else:
                discarded += 1

        print()  # nova linha depois da barra

        total_slices_kept += kept
        total_slices_discarded += discarded

        if kept > 0:
            processed_slices = np.array(processed_slices)
            output_path = os.path.join(output_folder, file_name)
            np.save(output_path, processed_slices)

            print(f"Saved shape: {processed_slices.shape}")
            print(f"Slices kept: {kept}")
            print(f"Slices discarded: {discarded}")
        else:
            print("No valid slices found.")

    print("\n" + "#" * 60)
    print("Processing complete")
    print(f"Total patients processed: {total_patients}")
    print(f"Total slices kept: {total_slices_kept}")
    print(f"Total slices discarded: {total_slices_discarded}")
    print("#" * 60)


# Entry point


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CT Preprocessing Script")

    parser.add_argument("--ct_folder", type=str, required=True,
                        help="Path to folder containing CT .npy files")

    parser.add_argument("--mask_folder", type=str, required=True,
                        help="Path to folder containing mask .npy files")

    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to save processed data")

    args = parser.parse_args()

    preprocess_dataset(
        ct_folder=args.ct_folder,
        mask_folder=args.mask_folder,
        output_folder=args.output_folder
    )