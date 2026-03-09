#!/usr/bin/env python3

import os
import json
import argparse


def create_dataset_split(ct_folder, output_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    # Verificar pasta
    if not os.path.isdir(ct_folder):
        raise ValueError(f"Folder not found: {ct_folder}")

    # Lista dos pacientes
    ct_files = sorted([
        f for f in os.listdir(ct_folder)
        if f.endswith(".npy") and os.path.isfile(os.path.join(ct_folder, f))
    ])

    if len(ct_files) == 0:
        raise ValueError("No .npy files found in the CT folder.")

    print(f"\nTotal patients found: {len(ct_files)}")

    # Calcular índices
    n = len(ct_files)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_files = ct_files[:train_end]
    val_files = ct_files[train_end:val_end]
    test_files = ct_files[val_end:]

    # Criar split
    split = {
        "train": train_files,
        "validation": val_files,
        "test": test_files
    }

    # Guardar JSON
    with open(output_file, "w") as f:
        json.dump(split, f, indent=4)

    print("\nDataset split created successfully\n")
    print(f"Train patients: {len(train_files)}")
    print(f"Validation patients: {len(val_files)}")
    print(f"Test patients: {len(test_files)}")

    print(f"\nSplit saved to: {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create dataset split for CT dataset")

    parser.add_argument("--ct_folder", type=str, required=True,
                        help="Path to folder containing CT .npy files")

    parser.add_argument("--output_file", type=str, default="dataset_split.json",
                        help="Path to save dataset split")

    args = parser.parse_args()

    create_dataset_split(
        ct_folder=args.ct_folder,
        output_file=args.output_file
    )