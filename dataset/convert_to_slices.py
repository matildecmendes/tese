#!/usr/bin/env python3

import os
import numpy as np
import argparse


def convert_volumes_to_slices(input_folder, output_folder):

    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    volume_files = sorted([
        f for f in os.listdir(input_folder)
        if f.endswith(".npy") and os.path.isfile(os.path.join(input_folder, f))
    ])

    #DEBUG volume_files = volume_files[:1] 

    if len(volume_files) == 0:
        raise ValueError("No .npy files found in input folder.")

    total_volumes = 0
    total_slices = 0

    for vol_file in volume_files:

        total_volumes += 1

        print("\n" + "=" * 60)
        print(f"Processing volume: {vol_file}")
        print("=" * 60)

        vol_path = os.path.join(input_folder, vol_file)

        try:
            volume = np.load(vol_path)
        except Exception as e:
            print(f"Error loading file: {e}")
            continue

        print(f"Volume shape: {volume.shape}")

        num_slices = volume.shape[0]

        for i in range(num_slices):

            slice_2d = volume[i]

            # nome do ficheiro com paciente + número da slice
            slice_name = f"{vol_file.replace('.npy','')}_slice_{i:04d}.npy"

            slice_path = os.path.join(output_folder, slice_name)

            try:
                np.save(slice_path, slice_2d)
            except OSError as e:
                print("\nERROR while saving slice!")
                print(e)
                print("Disk might be full. Stopping script safely.")
                return

            total_slices += 1
            slice_2d = volume[i]
            print(f"Slice {i} shape: {slice_2d.shape}")

        print(f"Slices saved from this volume: {num_slices}")

    print("\n" + "#" * 60)
    print("Conversion complete")
    print(f"Total volumes processed: {total_volumes}")
    print(f"Total slices saved: {total_slices}")
    print("#" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert CT volumes to individual slices")

    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder containing volume .npy files")

    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder where slices will be saved")

    args = parser.parse_args()

    convert_volumes_to_slices(
        input_folder=args.input_folder,
        output_folder=args.output_folder
    )