from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from LAND.src.land2d.autoencoder_kl_2d import AutoencoderKL2D
from LAND.src.land2d.dataset_npy_2d import NpyLungCT2DDataset, discover_npy_files, select_max_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct 2D .npy lung CT images with a trained LAND-2D VAE.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_comparison_png(original: np.ndarray, reconstruction: np.ndarray, path: Path) -> None:
    figure = plt.figure(figsize=(8, 4))
    axis1 = figure.add_subplot(1, 2, 1)
    axis1.imshow(original, cmap="gray", vmin=-1, vmax=1)
    axis1.set_title("Original")
    axis1.axis("off")
    axis2 = figure.add_subplot(1, 2, 2)
    axis2.imshow(reconstruction, cmap="gray", vmin=-1, vmax=1)
    axis2.set_title("Reconstruction")
    axis2.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = select_max_samples(discover_npy_files(args.dataset_path), args.num_samples, args.seed)
    dataset = NpyLungCT2DDataset(files, expected_size=args.resolution, horizontal_flip=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL2D.from_pretrained(args.vae_path).to(device).eval()
    vae.requires_grad_(False)

    sample_index = 0
    mse_values: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Reconstructing"):
            images = batch["image"].to(device)
            z_mu, _ = vae.encode(images)
            reconstructions = vae.decode(z_mu).clamp(-1, 1)

            for original_tensor, reconstruction_tensor in zip(images, reconstructions):
                original = original_tensor[0].float().cpu().numpy().astype(np.float32)
                reconstruction = reconstruction_tensor[0].float().cpu().numpy().astype(np.float32)
                mse_values.append(float(np.mean((original - reconstruction) ** 2)))

                np.save(output_dir / f"original_{sample_index:05d}.npy", original)
                np.save(output_dir / f"reconstruction_{sample_index:05d}.npy", reconstruction)
                save_comparison_png(
                    original,
                    reconstruction,
                    output_dir / f"comparison_{sample_index:05d}.png",
                )
                sample_index += 1

    print("\n========== VAE RECONSTRUCTION CHECK ==========")
    print(f"VAE: {Path(args.vae_path).resolve()}")
    print(f"Samples reconstructed: {sample_index}")
    print(f"Mean reconstruction MSE: {np.mean(mse_values):.8f}")
    print(f"Outputs: {output_dir}")
    print("Inspect the comparison PNG files before training the LDM.")
    print("==============================================")


if __name__ == "__main__":
    main()
