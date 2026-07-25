from __future__ import annotations

import inspect
from typing import Sequence

import numpy as np
import torch
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

from LAND.src.land2d.autoencoder_kl_2d import AutoencoderKL2D
from LAND.src.land2d.unet_2d import UNetModel2D


class UnconditionalLDM2DPipeline:
    """Minimal unconditional 2D latent diffusion sampling pipeline for LAND."""

    def __init__(
        self,
        unet: UNetModel2D,
        scheduler: DDPMScheduler,
        vae: AutoencoderKL2D,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.unet = unet.to(self.device).eval()
        self.vae = vae.to(self.device).eval()
        self.scheduler = scheduler
        self.latent_channels = int(self.vae.config.latent_channels)
        self.vae_scale_factor = int(self.vae.spatial_scale_factor)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 1000,
        latents: torch.Tensor | None = None,
        generators: Sequence[torch.Generator] | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"Image size ({height}, {width}) must be divisible by VAE scale factor {self.vae_scale_factor}."
            )

        latent_shape = (
            batch_size,
            self.latent_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        dtype = next(self.unet.parameters()).dtype

        if latents is None:
            if generators is None:
                latents = torch.randn(latent_shape, device=self.device, dtype=dtype)
            else:
                if len(generators) != batch_size:
                    raise ValueError("Number of generators must equal batch_size.")
                latent_list = [
                    torch.randn(latent_shape[1:], generator=generator, device=self.device, dtype=dtype)
                    for generator in generators
                ]
                latents = torch.stack(latent_list, dim=0)
        else:
            if tuple(latents.shape) != latent_shape:
                raise ValueError(f"Received latent shape {tuple(latents.shape)}, expected {latent_shape}.")
            latents = latents.to(device=self.device, dtype=dtype)

        latents = latents * self.scheduler.init_noise_sigma
        set_timesteps_parameters = inspect.signature(self.scheduler.set_timesteps).parameters
        if "device" in set_timesteps_parameters:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        else:
            self.scheduler.set_timesteps(num_inference_steps)

        timesteps = self.scheduler.timesteps
        iterator = tqdm(timesteps, desc="Denoising", leave=False) if show_progress else timesteps
        for timestep in iterator:
            model_input = self.scheduler.scale_model_input(latents, timestep)
            timestep_batch = timestep.expand(batch_size).to(self.device).long()
            prediction = self.unet(x=model_input, timesteps=timestep_batch)

            step_kwargs = {}
            if "generator" in inspect.signature(self.scheduler.step).parameters and generators:
                # Diffusers' randn_tensor supports one generator per batch element.
                # Passing the complete list keeps every sample tied to seed + sample index,
                # including the random variance added at each DDPM step.
                step_kwargs["generator"] = list(generators)
            latents = self.scheduler.step(prediction, timestep, latents, **step_kwargs).prev_sample

        images = self.vae.decode(latents).clamp(-1.0, 1.0)
        return images[:, 0].float().cpu().numpy().astype(np.float32)
