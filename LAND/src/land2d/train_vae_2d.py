from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from LAND.src.land2d.autoencoder_kl_2d import AutoencoderKL2D
from LAND.src.land2d.dataset_npy_2d import (
    NpyLungCT2DDataset,
    discover_npy_files,
    select_max_samples,
    split_file_list,
)
from LAND2.src.vae.utils.losses import KL_loss


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LAND image VAE on 2D lung CT .npy slices.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_config_file", required=True)
    parser.add_argument("--train_config_file", required=True)
    parser.add_argument("--model_dir", default="checkpoints/vae_2d")
    parser.add_argument("--log_dir", default="logs/vae_2d")
    parser.add_argument("--run_name", default="vaeLAND2D")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--max_samples", type=int, default=0, help="Use a subset for smoke tests; 0 means all files.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--horizontal_flip",
        action="store_true",
        help="Enable optional left-right flip augmentation (disabled by default).",
    )
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def configure_logging(run_dir: Path, log_root: Path) -> logging.Logger:
    log_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("land2d.vae")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(run_dir / "train_vae_2d.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_training_state(
    path: Path,
    epoch: int,
    best_val_loss: float,
    autoencoder: AutoencoderKL2D,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler_g: torch.optim.lr_scheduler.LRScheduler,
    scheduler_d: torch.optim.lr_scheduler.LRScheduler,
    scaler_g: torch.amp.GradScaler,
    scaler_d: torch.amp.GradScaler,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "autoencoder": autoencoder.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "scheduler_g": scheduler_g.state_dict(),
            "scheduler_d": scheduler_d.state_dict(),
            "scaler_g": scaler_g.state_dict(),
            "scaler_d": scaler_d.state_dict(),
        },
        path,
    )


def add_reconstruction_images(
    writer: SummaryWriter,
    images: torch.Tensor,
    reconstructions: torch.Tensor,
    epoch: int,
    tag: str,
) -> None:
    originals = ((images[:4].detach().float().cpu() + 1.0) / 2.0).clamp(0, 1)
    recon = ((reconstructions[:4].detach().float().cpu() + 1.0) / 2.0).clamp(0, 1)
    comparison = torch.cat([originals, recon], dim=-1)
    writer.add_images(tag, comparison, global_step=epoch)


def main() -> None:
    args = parse_args()
    model_config = load_json(args.model_config_file)
    train_config = load_json(args.train_config_file)

    for key in ("epochs", "batch_size", "val_batch_size", "num_workers", "seed"):
        value = getattr(args, key)
        if value is not None:
            train_config[key] = value
    if args.horizontal_flip:
        train_config["horizontal_flip"] = True
    if args.no_amp:
        train_config["amp"] = False

    torch.manual_seed(int(train_config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(train_config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(train_config["amp"] and device.type == "cuda")

    if args.resume_from_checkpoint:
        state_path = Path(args.resume_from_checkpoint).expanduser().resolve()
        if not state_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {state_path}")
        run_dir = state_path.parent
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.model_dir).expanduser().resolve() / f"{args.run_name}_{timestamp}"
        state_path = run_dir / "training_state_last.pt"

    logger = configure_logging(run_dir, Path(args.log_dir).expanduser().resolve())
    tensorboard_dir = run_dir / "tensorboard"
    writer = SummaryWriter(tensorboard_dir)

    with open(run_dir / "model_config.json", "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as handle:
        json.dump(train_config, handle, indent=2)

    files = discover_npy_files(args.dataset_path)
    files = select_max_samples(files, args.max_samples, int(train_config["seed"]))
    train_files, val_files = split_file_list(
        files,
        train_portion=float(train_config["train_portion"]),
        seed=int(train_config["seed"]),
    )

    train_dataset = NpyLungCT2DDataset(
        train_files,
        expected_size=int(train_config["resolution"]),
        horizontal_flip=bool(train_config["horizontal_flip"]),
    )
    val_dataset = NpyLungCT2DDataset(
        val_files,
        expected_size=int(train_config["resolution"]),
        horizontal_flip=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(train_config["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= int(train_config["batch_size"]),
        persistent_workers=int(train_config["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_config["val_batch_size"]),
        shuffle=False,
        num_workers=int(train_config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(train_config["num_workers"]) > 0,
    )

    autoencoder = AutoencoderKL2D(**model_config).to(device)
    discriminator = PatchDiscriminator(
        spatial_dims=2,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    ).to(device)

    reconstruction_loss = (
        MSELoss() if str(train_config["reconstruction_loss"]).lower() == "l2" else L1Loss()
    )
    adversarial_loss = PatchAdversarialLoss(criterion="least_squares")
    perceptual_loss = PerceptualLoss(
        spatial_dims=2,
        network_type="squeeze",
        is_fake_3d=False,
    ).eval().to(device)
    perceptual_loss.requires_grad_(False)

    learning_rate = float(train_config["learning_rate"])
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, eps=1e-6 if amp_enabled else 1e-8)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, eps=1e-6 if amp_enabled else 1e-8)

    # def warmup_rule(epoch: int) -> float:
    #     if epoch < 10:
    #         return 0.01
    #     if epoch < 20:
    #         return 0.1
    #     return 1.0
    def warmup_rule(epoch: int) -> float:
        if epoch < 1:
            return 0.1
        return 1.0

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)
    scaler_g = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scaler_d = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_val_loss = math.inf
    if args.resume_from_checkpoint:
        checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
        autoencoder.load_state_dict(checkpoint["autoencoder"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])
        scheduler_d.load_state_dict(checkpoint["scheduler_d"])
        scaler_g.load_state_dict(checkpoint.get("scaler_g", {}))
        scaler_d.load_state_dict(checkpoint.get("scaler_d", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        logger.info("Resuming from %s at epoch %d", state_path, start_epoch)

    logger.info("Device: %s", device)
    logger.info("AMP enabled: %s", amp_enabled)
    logger.info("Training images: %d", len(train_dataset))
    logger.info("Validation images: %d", len(val_dataset))
    logger.info("Input shape: (B, 1, %d, %d)", train_config["resolution"], train_config["resolution"])
    logger.info("No input renormalization is applied; values remain in [-1,1].")
    logger.info("Autoencoder parameters: %d", sum(p.numel() for p in autoencoder.parameters()))

    global_step = 0
    epochs = int(train_config["epochs"])
    validation_interval = int(train_config["validation_interval"])

    for epoch in range(start_epoch, epochs):
        autoencoder.train()
        discriminator.train()
        epoch_totals = {"reconstruction": 0.0, "kl": 0.0, "perceptual": 0.0, "generator_adv": 0.0, "discriminator": 0.0}

        progress = tqdm(train_loader, desc=f"VAE train epoch {epoch + 1}/{epochs}")
        last_images = None
        last_reconstructions = None
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            optimizer_g.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                reconstructions, z_mu, z_sigma = autoencoder(images)
                if reconstructions.shape != images.shape:
                    raise RuntimeError(
                        f"VAE reconstruction shape {tuple(reconstructions.shape)} does not match "
                        f"input shape {tuple(images.shape)}."
                    )
                if not torch.isfinite(reconstructions).all():
                    raise RuntimeError("NaN or Inf detected in VAE reconstructions.")
                loss_reconstruction = reconstruction_loss(reconstructions, images)
                loss_kl = KL_loss(z_mu, z_sigma)

                # SqueezeNet/LPIPS expects RGB-like input, so repeat grayscale channels only for this loss.
                perceptual_recon = reconstructions.repeat(1, 3, 1, 1)
                perceptual_real = images.repeat(1, 3, 1, 1)
                loss_perceptual = perceptual_loss(perceptual_recon.float(), perceptual_real.float())

                logits_fake_for_g = discriminator(reconstructions.float())[-1]
                loss_generator_adv = adversarial_loss(
                    logits_fake_for_g,
                    target_is_real=True,
                    for_discriminator=False,
                )
                loss_g = (
                    loss_reconstruction
                    + float(train_config["kl_weight"]) * loss_kl
                    + float(train_config["perceptual_weight"]) * loss_perceptual
                    + float(train_config["adversarial_weight"]) * loss_generator_adv
                )

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            optimizer_d.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits_fake = discriminator(reconstructions.detach().float())[-1]
                logits_real = discriminator(images.detach().float())[-1]
                loss_d_fake = adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d_real = adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = 0.5 * (loss_d_fake + loss_d_real)

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            global_step += 1
            values = {
                "reconstruction": float(loss_reconstruction.detach()),
                "kl": float(loss_kl.detach()),
                "perceptual": float(loss_perceptual.detach()),
                "generator_adv": float(loss_generator_adv.detach()),
                "discriminator": float(loss_d.detach()),
            }
            for name, value in values.items():
                epoch_totals[name] += value
                writer.add_scalar(f"train/{name}", value, global_step)

            scale_factor = float((1.0 / z_mu.detach().flatten().std().clamp_min(1e-8)).cpu())
            writer.add_scalar("train/latent_scale_factor_diagnostic", scale_factor, global_step)
            progress.set_postfix(recon=values["reconstruction"], d=values["discriminator"])
            last_images = images
            last_reconstructions = reconstructions

        scheduler_g.step()
        scheduler_d.step()
        for name, total in epoch_totals.items():
            writer.add_scalar(f"epoch_train/{name}", total / max(1, len(train_loader)), epoch + 1)

        if last_images is not None and last_reconstructions is not None:
            add_reconstruction_images(writer, last_images, last_reconstructions, epoch + 1, "train/original_left_reconstruction_right")

        val_total = 0.0
        if (epoch + 1) % validation_interval == 0:
            autoencoder.eval()
            val_components = {"reconstruction": 0.0, "kl": 0.0, "perceptual": 0.0}
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"VAE validation epoch {epoch + 1}"):
                    images = batch["image"].to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                        reconstructions, z_mu, z_sigma = autoencoder(images)
                        loss_reconstruction = reconstruction_loss(reconstructions, images)
                        loss_kl = KL_loss(z_mu, z_sigma)
                        loss_perceptual = perceptual_loss(
                            reconstructions.repeat(1, 3, 1, 1).float(),
                            images.repeat(1, 3, 1, 1).float(),
                        )
                    val_components["reconstruction"] += float(loss_reconstruction)
                    val_components["kl"] += float(loss_kl)
                    val_components["perceptual"] += float(loss_perceptual)

                for name in val_components:
                    val_components[name] /= max(1, len(val_loader))
                    writer.add_scalar(f"validation/{name}", val_components[name], epoch + 1)

                val_total = (
                    val_components["reconstruction"]
                    + float(train_config["kl_weight"]) * val_components["kl"]
                    + float(train_config["perceptual_weight"]) * val_components["perceptual"]
                )
                writer.add_scalar("validation/weighted_total", val_total, epoch + 1)
                add_reconstruction_images(writer, images, reconstructions, epoch + 1, "validation/original_left_reconstruction_right")

            logger.info("Epoch %d validation loss: %.8f", epoch + 1, val_total)
            if val_total < best_val_loss:
                best_val_loss = val_total
                best_dir = run_dir / "vae_best_epoch"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                autoencoder.save_pretrained(best_dir)
                torch.save(discriminator.state_dict(), run_dir / "discriminator_best_epoch.pt")
                logger.info("Saved new best VAE to %s", best_dir)

        last_dir = run_dir / "vae_last_epoch"
        if last_dir.exists():
            shutil.rmtree(last_dir)
        autoencoder.save_pretrained(last_dir)
        torch.save(discriminator.state_dict(), run_dir / "discriminator_last_epoch.pt")
        save_training_state(
            state_path,
            epoch,
            best_val_loss,
            autoencoder,
            discriminator,
            optimizer_g,
            optimizer_d,
            scheduler_g,
            scheduler_d,
            scaler_g,
            scaler_d,
        )
        logger.info("Completed epoch %d/%d", epoch + 1, epochs)

    writer.close()
    logger.info("VAE training completed. Best VAE: %s", run_dir / "vae_best_epoch")


if __name__ == "__main__":
    main()
