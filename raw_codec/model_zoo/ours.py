from __future__ import annotations

from typing import Any

from raw_codec.models.elic import ELICRawCodec


_COMMON: dict[str, Any] = {
    "architecture": "elic_raw_mobile_conv_hyperprior",
    "raw_bit_depth": 14,
    "in_channels": 4,
    "out_channels": 3,
    "hidden_channels": 256,
    "latent_channels": 320,
    "hyper_channels": 192,
    "channel_groups": (16, 16, 32, 64, 192),
    "hq_synthesis_blocks": (4, 3, 2),
    "output_activation": "clamp_ste",
    "min_likelihood": 1.0e-9,
    "scale_table_min": 0.11,
    "scale_table_max": 256.0,
    "scale_table_levels": 64,
    "fixed_qstep": 0.15,
}


class OursM(ELICRawCodec):
    """Paper Ours-M: medium mobile sender and receiver-heavy decoder."""

    def __init__(self, **overrides: Any) -> None:
        kwargs = {
            **_COMMON,
            "mobile_analysis_channels": (224, 448, 640),
            "mobile_analysis_blocks": (1, 3, 5),
            **overrides,
        }
        super().__init__(**kwargs)


class OursS(ELICRawCodec):
    """Paper Ours-S: narrower sender with the same latent/decoder topology."""

    def __init__(self, **overrides: Any) -> None:
        kwargs = {
            **_COMMON,
            "mobile_analysis_channels": (160, 256, 320),
            "mobile_analysis_blocks": (1, 3, 5),
            **overrides,
        }
        super().__init__(**kwargs)
