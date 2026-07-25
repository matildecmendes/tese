from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from tqdm import tqdm

from LAND.src.land2d.dataset_npy_2d import discover_npy_files, load_npy_image_2d, select_max_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a folder of preprocessed 2D lung CT .npy images.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--max_files",
        type=int,
        default=1000,
        help="Maximum number of files to inspect. Use 0 to inspect every file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_npy_files(args.dataset_path)
    selected = select_max_samples(files, args.max_files, args.seed)

    minima: list[float] = []
    maxima: list[float] = []
    means: list[float] = []

    print("\n========== LAND-2D DATASET CHECK ==========")
    print(f"Dataset folder: {Path(args.dataset_path).resolve()}")
    print(f"Total .npy files found: {len(files)}")
    print(f"Files inspected: {len(selected)}")
    print(f"Expected shape: ({args.resolution}, {args.resolution})")
    print("Expected range: [-1, 1]")
    print("===========================================\n")

    for path in tqdm(selected, desc="Checking .npy files"):
        image = load_npy_image_2d(path, expected_size=args.resolution)
        minima.append(float(image.min()))
        maxima.append(float(image.max()))
        means.append(float(image.mean()))

    print("\nDataset check completed successfully.")
    print(f"Global inspected minimum: {min(minima):.6f}")
    print(f"Global inspected maximum: {max(maxima):.6f}")
    print(f"Mean of image means: {statistics.fmean(means):.6f}")
    print("No invalid shapes, NaN/Inf values, or out-of-range values were found.")


if __name__ == "__main__":
    main()
