from __future__ import annotations

from collections.abc import Sequence

from diffusers.configuration_utils import register_to_config

from LAND.src.unet.unet import UNetModel


class UNetModel2D(UNetModel):
    """LAND's custom diffusion U-Net fixed to two spatial dimensions."""

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 128,
        out_channels: int = 4,
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (16,),
        dropout: float = 0.0,
        channel_mult: Sequence[int] = (1, 3, 4, 4, 4),
        conv_resample: bool = True,
        use_checkpoint: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        num_groups: int = 32,
        additive_skips: bool = True,
        verbose: bool = False,
        self_attention_blocks: Sequence[str] = (),
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_resolutions),
            dropout=dropout,
            channel_mult=tuple(channel_mult),
            conv_resample=conv_resample,
            dims=2,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            num_groups=num_groups,
            resample_2d=True,
            additive_skips=additive_skips,
            verbose=verbose,
            self_attention_blocks=tuple(self_attention_blocks),
            cross_attention_blocks=(),
            cross_attention_dim=None,
        )
