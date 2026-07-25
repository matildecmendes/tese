from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from LAND.src.land2d.autoencoder_kl_2d import AutoencoderKL2D
from LAND.src.land2d.dataset_npy_2d import NpyLungCT2DDataset, discover_npy_files, select_max_samples
from LAND.src.land2d.pipeline_ldm_2d import UnconditionalLDM2DPipeline
from LAND.src.land2d.unet_2d import UNetModel2D


def parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from error
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unconditional LAND latent diffusion on 2D lung CT slices.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=10)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--prediction_type", choices=["epsilon", "sample", "v_prediction"], default="v_prediction")
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", default="linear")
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--save_images_epochs", type=int, default=25)
    parser.add_argument("--sample_inference_steps", type=int, default=100)
    parser.add_argument("--num_sample_images", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=0, help="Optional smoke-test limit; 0 means unlimited.")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--model_channels", type=int, default=128)
    parser.add_argument("--channel_mult", type=parse_int_tuple, default=(1, 3, 4, 4, 4))
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument(
        "--horizontal_flip",
        action="store_true",
        help="Enable optional left-right flip augmentation (disabled by default).",
    )
    return parser.parse_args()


def save_png(image: np.ndarray, path: Path) -> None:
    figure = plt.figure(figsize=(4, 4))
    axis = figure.add_subplot(1, 1, 1)
    axis.imshow(image, cmap="gray", vmin=-1, vmax=1)
    axis.axis("off")
    figure.tight_layout(pad=0)
    figure.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def save_inference_components(
    accelerator: Accelerator,
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    output_dir: Path,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unet_dir = output_dir / "unet"
        scheduler_dir = output_dir / "scheduler"
        if unet_dir.exists():
            shutil.rmtree(unet_dir)
        if scheduler_dir.exists():
            shutil.rmtree(scheduler_dir)
        unwrapped_model.save_pretrained(unet_dir)
        scheduler.save_pretrained(scheduler_dir)


def save_resume_checkpoint(
    accelerator: Accelerator,
    checkpoint_dir: Path,
    epoch: int,
    global_step: int,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    accelerator.wait_for_everyone()
    accelerator.save_state(checkpoint_dir)
    if accelerator.is_main_process:
        with open(checkpoint_dir / "resume_metadata.json", "w", encoding="utf-8") as handle:
            json.dump({"epoch": epoch, "global_step": global_step}, handle, indent=2)


def main() -> None:
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("resolution must be positive.")
    if args.train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive.")
    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if args.ddpm_num_steps <= 0:
        raise ValueError("ddpm_num_steps must be positive.")
    if args.sample_inference_steps <= 0 or args.sample_inference_steps > args.ddpm_num_steps:
        raise ValueError("sample_inference_steps must be between 1 and ddpm_num_steps.")
    if any(multiplier <= 0 for multiplier in args.channel_mult):
        raise ValueError("Every channel_mult value must be positive.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    logging_dir = Path(args.logging_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    project_config = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        with open(output_dir / "args_2d.json", "w", encoding="utf-8") as handle:
            serializable_args = vars(args).copy()
            serializable_args["channel_mult"] = list(args.channel_mult)
            json.dump(serializable_args, handle, indent=2)

    files = select_max_samples(discover_npy_files(args.dataset_path), args.max_samples, args.seed)
    dataset = NpyLungCT2DDataset(
        files,
        expected_size=args.resolution,
        horizontal_flip=args.horizontal_flip,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(dataset) >= args.train_batch_size,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    vae = AutoencoderKL2D.from_pretrained(args.vae_path)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device)
    latent_channels = int(vae.config.latent_channels)
    if args.resolution % vae.spatial_scale_factor != 0:
        raise ValueError(
            f"resolution {args.resolution} is not divisible by the VAE scale factor "
            f"{vae.spatial_scale_factor}."
        )
    latent_resolution = args.resolution // vae.spatial_scale_factor
    unet_downsample_factor = 2 ** (len(args.channel_mult) - 1)
    if latent_resolution % unet_downsample_factor != 0:
        raise ValueError(
            f"Latent resolution {latent_resolution} must be divisible by the U-Net "
            f"downsampling factor {unet_downsample_factor}."
        )

    self_attention_blocks = ("input", "middle", "output") if args.attention else ()
    model = UNetModel2D(
        in_channels=latent_channels,
        model_channels=args.model_channels,
        out_channels=latent_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(16,),
        dropout=0.0,
        channel_mult=args.channel_mult,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        num_groups=32,
        additive_skips=True,
        self_attention_blocks=self_attention_blocks,
    )

    noise_scheduler = DDPMScheduler(
        clip_sample=False,
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_training_steps = args.num_epochs * updates_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=total_training_steps,
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        lr_scheduler,
    )

    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        checkpoint_dir = Path(args.resume_from_checkpoint).expanduser().resolve()
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_dir}")
        accelerator.load_state(checkpoint_dir)
        metadata_path = checkpoint_dir / "resume_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            start_epoch = int(metadata["epoch"]) + 1
            global_step = int(metadata["global_step"])

    accelerator.init_trackers(
        "LAND_2D_LDM",
        config={**vars(args), "channel_mult": list(args.channel_mult)},
    )

    accelerator.print("\n========== LAND-2D LDM TRAINING ==========")
    accelerator.print(f"Dataset: {Path(args.dataset_path).resolve()}")
    accelerator.print(f"Images: {len(dataset)}")
    accelerator.print(f"VAE: {Path(args.vae_path).resolve()}")
    accelerator.print(f"Latent channels: {latent_channels}")
    accelerator.print(f"VAE spatial scale factor: {vae.spatial_scale_factor}")
    accelerator.print(f"Expected latent size: {args.resolution // vae.spatial_scale_factor} x {args.resolution // vae.spatial_scale_factor}")
    accelerator.print(f"Attention enabled: {args.attention}")
    accelerator.print(f"Horizontal flip augmentation: {args.horizontal_flip}")
    accelerator.print("Input images are already in [-1,1]; no second normalization is applied.")
    accelerator.print("==========================================\n")

    stop_training = False
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        progress = tqdm(
            train_loader,
            desc=f"LDM epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process,
        )
        epoch_loss = 0.0

        for batch in progress:
            images = batch["image"].to(accelerator.device, non_blocking=True)

            with torch.no_grad():
                with accelerator.autocast():
                    z_mu, z_sigma = vae.encode(images)
                    latents = vae.sampling(z_mu, z_sigma)

            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=latents.device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with accelerator.accumulate(model):
                prediction = model(x=noisy_latents, timesteps=timesteps)
                if args.prediction_type == "epsilon":
                    target = noise
                elif args.prediction_type == "sample":
                    target = latents
                elif args.prediction_type == "v_prediction":
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(latents.device)
                    sqrt_alpha = alphas_cumprod[timesteps].sqrt()
                    sqrt_one_minus_alpha = (1.0 - alphas_cumprod[timesteps]).sqrt()
                    broadcast_shape = (batch_size,) + (1,) * (latents.ndim - 1)
                    target = sqrt_alpha.view(broadcast_shape) * noise - sqrt_one_minus_alpha.view(broadcast_shape) * latents
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                if prediction.shape != target.shape:
                    raise RuntimeError(
                        f"U-Net output shape {tuple(prediction.shape)} does not match "
                        f"target shape {tuple(target.shape)}."
                    )
                if not torch.isfinite(prediction).all():
                    raise RuntimeError("NaN or Inf detected in the U-Net output.")

                elementwise_loss = F.mse_loss(prediction.float(), target.float(), reduction="none")
                per_sample_loss = elementwise_loss.mean(dim=tuple(range(1, elementwise_loss.ndim)))

                if args.snr_gamma is not None:
                    snr = compute_snr(noise_scheduler, timesteps)
                    gamma = torch.full_like(snr, args.snr_gamma)
                    weights = torch.minimum(snr, gamma)
                    if args.prediction_type == "epsilon":
                        weights = weights / snr.clamp_min(1e-8)
                    elif args.prediction_type == "v_prediction":
                        weights = weights / (snr + 1.0)
                    per_sample_loss = per_sample_loss * weights

                loss = per_sample_loss.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.detach())
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log(
                    {"train/loss": float(loss.detach()), "train/lr": lr_scheduler.get_last_lr()[0], "epoch": epoch + 1},
                    step=global_step,
                )
            progress.set_postfix(loss=float(loss.detach()), step=global_step)

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                stop_training = True
                break

        mean_epoch_loss = epoch_loss / max(1, len(train_loader))
        accelerator.log({"train/epoch_loss": mean_epoch_loss}, step=global_step)

        should_save_model = (
            (epoch + 1) % args.save_model_epochs == 0
            or epoch == args.num_epochs - 1
            or stop_training
        )
        if should_save_model:
            save_inference_components(accelerator, model, noise_scheduler, output_dir)
            save_resume_checkpoint(
                accelerator,
                output_dir / "resume_checkpoint",
                epoch,
                global_step,
            )

            if accelerator.is_main_process and (epoch + 1) % 50 == 0:
                snapshot_dir = output_dir / "epoch_ckpts" / f"epoch-{epoch + 1:04d}"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(snapshot_dir / "unet")
                noise_scheduler.save_pretrained(snapshot_dir / "scheduler")

        should_sample = (
            args.num_sample_images > 0
            and ((epoch + 1) % args.save_images_epochs == 0 or stop_training)
        )
        if should_sample:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.eval()
                pipeline = UnconditionalLDM2DPipeline(
                    unet=unwrapped_model,
                    scheduler=noise_scheduler,
                    vae=vae,
                    device=accelerator.device,
                )
                generators = [
                    torch.Generator(device=accelerator.device).manual_seed(args.seed + index)
                    for index in range(args.num_sample_images)
                ]
                samples = pipeline(
                    batch_size=args.num_sample_images,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.sample_inference_steps,
                    generators=generators,
                    show_progress=True,
                )
                sample_dir = output_dir / "training_samples" / f"epoch-{epoch + 1:04d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                for index, image in enumerate(samples):
                    np.save(sample_dir / f"sample_{index:03d}.npy", image)
                    save_png(image, sample_dir / f"sample_{index:03d}.png")
                model.train()

        if stop_training:
            break

    save_inference_components(accelerator, model, noise_scheduler, output_dir)
    accelerator.end_training()
    accelerator.print(f"Training complete. Inference model saved in: {output_dir}")


if __name__ == "__main__":
    main()
