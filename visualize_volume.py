import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def visualize_volume(npy_path, slice_index=None):

    if not os.path.isfile(npy_path):
        raise ValueError(f"File not found: {npy_path}")

    volume = np.load(npy_path)

    print("=" * 50)
    print("Volume info")
    print("=" * 50)
    print("Shape:", volume.shape)
    print("Dtype:", volume.dtype)
    print("Min value:", volume.min())
    print("Max value:", volume.max())
    print("=" * 50)

    if len(volume.shape) != 3:
        raise ValueError("Expected a 3D volume (D, H, W)")

    total_slices = volume.shape[0]

    if slice_index is None:
        slice_index = total_slices // 2  # slice do meio

    if slice_index >= total_slices:
        raise ValueError("Slice index out of range")

    slice_img = volume[slice_index]

    print(f"Displaying slice {slice_index}/{total_slices-1}")

    plt.figure(figsize=(6, 6))
    plt.imshow(slice_img, cmap="gray")
    plt.title(f"Slice {slice_index}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize 3D CT volume")

    parser.add_argument("--file", type=str, required=True,
                        help="Path to .npy volume")

    parser.add_argument("--slice", type=int, default=None,
                        help="Slice index to visualize (default: middle slice)")

    args = parser.parse_args()

    visualize_volume(args.file, args.slice)