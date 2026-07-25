from __future__ import annotations

from collections.abc import Sequence

from diffusers import ConfigMixin, ModelMixin
from monai.networks.nets import AutoencoderKL


class AutoencoderKL2D(AutoencoderKL, ModelMixin, ConfigMixin):
    """
    MONAI AutoencoderKL configured for 2D and wrapped with Diffusers-style
    ``save_pretrained`` / ``from_pretrained`` support.

    This replaces LAND's memory-optimized MAISI 3D VAE only in the new 2D path.
    The original 3D files remain unchanged.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] = (1, 1, 1),
        num_channels: Sequence[int] = (32, 64, 128),
        attention_levels: Sequence[bool] = (False, False, False),
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = False,
        with_decoder_nonlocal_attn: bool = False,
        include_fc: bool = False,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        use_checkpointing: bool = False,
        use_convtranspose: bool = False,
    ) -> None:
        self.register_to_config(
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=list(num_res_blocks),
            num_channels=list(num_channels),
            attention_levels=list(attention_levels),
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_encoder_nonlocal_attn=with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=with_decoder_nonlocal_attn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            use_checkpointing=use_checkpointing,
            use_convtranspose=use_convtranspose,
        )

        super().__init__(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=tuple(num_res_blocks),
            channels=tuple(num_channels),
            attention_levels=tuple(attention_levels),
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_encoder_nonlocal_attn=with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=with_decoder_nonlocal_attn,
            use_checkpoint=use_checkpointing,
            use_convtranspose=use_convtranspose,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

    @property
    def spatial_scale_factor(self) -> int:
        return 2 ** (len(self.config.num_channels) - 1)
