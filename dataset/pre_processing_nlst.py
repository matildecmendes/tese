#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pydicom
import cv2
from PIL import Image


# --------------------------------------------------
# HU normalization
# --------------------------------------------------

def normalize_hu(slice_ct, min_hu=-1000, max_hu=400):

    slice_ct = np.clip(slice_ct, min_hu, max_hu)

    slice_ct = (slice_ct - min_hu) / (max_hu - min_hu)

    slice_ct = 2 * slice_ct - 1

    return slice_ct.astype(np.float32)


# --------------------------------------------------
# Load DICOM slice as HU
# --------------------------------------------------

def dicom_to_hu(dcm_path):

    ds = pydicom.dcmread(dcm_path)

    image = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    image = image * slope + intercept

    return image


# --------------------------------------------------
# Find all DICOM files recursively
# --------------------------------------------------

def find_dicom_files(root_folder):

    dicom_files = []

    for root, _, files in os.walk(root_folder):

        for f in files:

            if f.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, f))

    return sorted(dicom_files)


# --------------------------------------------------
# Main processing
# --------------------------------------------------

def process_dataset(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    dicom_files = find_dicom_files(input_folder)

    if len(dicom_files) == 0:
        raise ValueError("No DICOM files found.")

    print(f"Found {len(dicom_files)} DICOM files.")

    png_saved = False

    for idx, dcm_path in enumerate(dicom_files):

        try:

            hu = dicom_to_hu(dcm_path)

            normalized = normalize_hu(hu)

            resized = cv2.resize(
                normalized,
                (256, 256),
                interpolation=cv2.INTER_AREA
            )

            output_name = f"slice_{idx:07d}.npy"

            np.save(
                os.path.join(output_folder, output_name),
                resized.astype(np.float32)
            )

            # Save first PNG only
            if not png_saved:

                png_img = ((resized + 1) / 2 * 255)

                png_img = np.clip(png_img, 0, 255).astype(np.uint8)

                Image.fromarray(png_img).save(
                    os.path.join(output_folder, "first_slice_check.png")
                )

                png_saved = True

                print(
                    f"Saved validation PNG: "
                    f"{os.path.join(output_folder, 'first_slice_check.png')}"
                )

            if (idx + 1) % 100 == 0:

                print(
                    f"Processed {idx + 1}/{len(dicom_files)} slices"
                )

        except Exception as e:

            print(f"Error processing {dcm_path}")
            print(e)

    print("\nProcessing complete.")
    print(f"Saved {len(dicom_files)} slices.")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert NLST DICOM files to normalized 256x256 NPY slices"
    )

    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Root folder containing NLST DICOM files"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder where NPY slices will be saved"
    )

    args = parser.parse_args()

    process_dataset(
        input_folder=args.input_folder,
        output_folder=args.output_folder
    )