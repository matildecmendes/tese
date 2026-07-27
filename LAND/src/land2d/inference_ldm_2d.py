from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

from LAND.src.land2d.autoencoder_kl_2d import AutoencoderKL2D
from LAND.src.land2d.pipeline_ldm_2d import UnconditionalLDM2DPipeline
from LAND.src.land2d.unet_2d import UNetModel2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic 2D lung CT .npy images with LAND-2D.")
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--model_path", required=True, help="LDM output folder containing unet/ and scheduler/.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_png(image: np.ndarray, path: Path) -> None:
    figure = plt.figure(figsize=(4, 4))
    axis = figure.add_subplot(1, 1, 1)
    axis.imshow(image, cmap="gray", vmin=-1, vmax=1)
    axis.axis("off")
    figure.tight_layout(pad=0)
    figure.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.num_samples < 0:
        raise ValueError("num_samples cannot be negative.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.resolution <= 0:
        raise ValueError("resolution must be positive.")
    if args.steps <= 0:
        raise ValueError("steps must be positive.")

    model_path = Path(args.model_path).expanduser().resolve()
    vae_path = Path(args.vae_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    unet_dir = model_path / "unet"
    scheduler_dir = model_path / "scheduler"
    if not unet_dir.exists() or not scheduler_dir.exists():
        raise FileNotFoundError(
            f"Expected {unet_dir} and {scheduler_dir}. Pass the root LDM output folder as --model_path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNetModel2D.from_pretrained(unet_dir).to(device).eval()
    vae = AutoencoderKL2D.from_pretrained(vae_path).to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(scheduler_dir)
    if args.steps > scheduler.config.num_train_timesteps:
        raise ValueError(
            f"steps={args.steps} exceeds the scheduler training steps "
            f"({scheduler.config.num_train_timesteps})."
        )
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    pipeline = UnconditionalLDM2DPipeline(unet=unet, scheduler=scheduler, vae=vae, device=device)

    requested_indices = list(range(args.start_index, args.start_index + args.num_samples))
    indices_to_generate = []
    for index in requested_indices:
        output_path = output_dir / f"image_{index:05d}.npy"
        if args.overwrite or not output_path.exists():
            indices_to_generate.append(index)

    print("\n========== LAND-2D INFERENCE ==========")
    print(f"Model: {model_path}")
    print(f"VAE: {vae_path}")
    print(f"Output: {output_dir}")
    print(f"Requested samples: {args.num_samples}")
    print(f"Samples already present and skipped: {args.num_samples - len(indices_to_generate)}")
    print(f"Samples to generate: {len(indices_to_generate)}")
    print(f"Denoising steps: {args.steps}")
    print(f"Output format: float32 .npy, shape ({args.resolution}, {args.resolution}), range [-1,1]")
    print("=======================================\n")

    num_batches = ceil(len(indices_to_generate) / args.batch_size) if indices_to_generate else 0
    for batch_number in tqdm(range(num_batches), desc="Generating batches"):
        batch_indices = indices_to_generate[
            batch_number * args.batch_size : (batch_number + 1) * args.batch_size
        ]
        generators = [
            torch.Generator(device=device).manual_seed(args.seed + index)
            for index in batch_indices
        ]
        images = pipeline(
            batch_size=len(batch_indices),
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.steps,
            generators=generators,
            show_progress=False,
        )

        for index, image in zip(batch_indices, images):
            if image.shape != (args.resolution, args.resolution):
                raise RuntimeError(f"Unexpected generated shape {image.shape} for sample {index}.")
            if not np.isfinite(image).all():
                raise RuntimeError(f"NaN/Inf detected in generated sample {index}.")
            image = np.clip(image, -1.0, 1.0).astype(np.float32)
            #se quiser guardar .npy também!
            #np.save(output_dir / f"image_{index:05d}.npy", image)
            if args.save_png:
                save_png(image, output_dir / f"image_{index:05d}.png")

        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Finished. Synthetic images are in: {output_dir}")


if __name__ == "__main__":
    main()
