from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_SHAPES = {
    "HW": lambda x: x,
    "CHW": lambda x: x[0],
    "HWC": lambda x: x[..., 0],
}


def discover_npy_files(dataset_path: str | Path) -> list[Path]:
    """Find every .npy file recursively below ``dataset_path``."""
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a folder: {root}")

    files = sorted(path for path in root.rglob("*.npy") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .npy files were found below: {root}")
    return files


def split_file_list(
    files: Sequence[Path],
    train_portion: float = 0.9,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Create a deterministic shuffled train/validation split."""
    if not 0.0 < train_portion < 1.0:
        raise ValueError("train_portion must be between 0 and 1.")
    if len(files) < 2:
        raise ValueError("At least two .npy files are needed to create train/validation splits.")

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_portion)))
    return shuffled[:split_index], shuffled[split_index:]


def load_npy_image_2d(
    path: str | Path,
    expected_size: int = 256,
    range_tolerance: float = 1e-3,
) -> np.ndarray:
    """
    Load one grayscale 2D image and return shape ``(H, W)`` as float32.

    Accepted source shapes are ``(H, W)``, ``(1, H, W)`` and ``(H, W, 1)``.
    Values must already be normalized to [-1, 1]. No renormalization is applied.
    """
    path = Path(path)
    array = np.load(path, mmap_mode="r")

    if array.ndim == 2:
        image = np.asarray(array)
    elif array.ndim == 3 and array.shape[0] == 1:
        image = np.asarray(array[0])
    elif array.ndim == 3 and array.shape[-1] == 1:
        image = np.asarray(array[..., 0])
    else:
        raise ValueError(
            f"Unsupported shape {array.shape} in {path}. Expected (H,W), (1,H,W), or (H,W,1)."
        )

    if image.shape != (expected_size, expected_size):
        raise ValueError(
            f"Invalid spatial size {image.shape} in {path}. Expected ({expected_size}, {expected_size})."
        )

    image = image.astype(np.float32, copy=False)
    if not np.isfinite(image).all():
        raise ValueError(f"NaN or Inf detected in {path}.")

    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < -1.0 - range_tolerance or maximum > 1.0 + range_tolerance:
        raise ValueError(
            f"Values outside [-1,1] in {path}: min={minimum:.6f}, max={maximum:.6f}. "
            "The LAND-2D loader does not normalize inputs again."
        )

    return image


class NpyLungCT2DDataset(Dataset):
    """Dataset for preprocessed 2D grayscale lung CT slices stored as .npy files."""

    def __init__(
        self,
        files: Sequence[str | Path],
        expected_size: int = 256,
        horizontal_flip: bool = False,
        range_tolerance: float = 1e-3,
    ) -> None:
        if not files:
            raise ValueError("The dataset file list is empty.")
        self.files = [Path(path) for path in files]
        self.expected_size = expected_size
        self.horizontal_flip = horizontal_flip
        self.range_tolerance = range_tolerance

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        image = load_npy_image_2d(
            path,
            expected_size=self.expected_size,
            range_tolerance=self.range_tolerance,
        )

        # Copy before torch conversion because mmap arrays are read-only.
        tensor = torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).unsqueeze(0)

        if self.horizontal_flip and torch.rand(()) < 0.5:
            tensor = torch.flip(tensor, dims=(-1,))

        return {"image": tensor.contiguous(), "path": str(path)}


def select_max_samples(files: Sequence[Path], max_samples: int | None, seed: int = 42) -> list[Path]:
    """Optionally choose a deterministic random subset for smoke tests."""
    if max_samples is None or max_samples <= 0 or max_samples >= len(files):
        return list(files)
    rng = random.Random(seed)
    selected = list(files)
    rng.shuffle(selected)
    return sorted(selected[:max_samples])
