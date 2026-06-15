#!/usr/bin/env python3

import os
import argparse
import numpy as np
from PIL import Image


def npy_to_png(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    npy_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".npy")]
    )

    print(f"Found {len(npy_files)} .npy files")

    for idx, file_name in enumerate(npy_files):

        npy_path = os.path.join(input_folder, file_name)

        try:
            img = np.load(npy_path)

            # Assume imagens normalizadas em [-1,1]
            png = ((img + 1) / 2 * 255)
            png = np.clip(png, 0, 255).astype(np.uint8)

            png_name = os.path.splitext(file_name)[0] + ".png"
            png_path = os.path.join(output_folder, png_name)

            Image.fromarray(png).save(png_path)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(npy_files)}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert NPY images (-1,1) to PNG"
    )

    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing .npy files"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder where .png files will be saved"
    )

    args = parser.parse_args()

    npy_to_png(
        input_folder=args.input_folder,
        output_folder=args.output_folder
    )