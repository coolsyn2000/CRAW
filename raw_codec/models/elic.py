from __future__ import annotations

import math
import struct
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import conv, deconv, update_registered_buffers

from .naf_blocks import LayerNorm2d, NAFBlock


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def clamp_ste(x: torch.Tensor, minimum: float = 0.0, maximum: float = 1.0) -> torch.Tensor:
    clamped = x.clamp(minimum, maximum)
    return x + (clamped - x).detach()


def conv1x1(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1)


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1)


class DepthResidualBlock(nn.Module):
    """Depth-wise nonlinear embedding used by MLICv2's token mixer."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.net = nn.Sequential(
            conv1x1(channels, channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.GELU(),
            conv1x1(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GateBlock(nn.Module):
    """Lightweight point-wise interaction from LLIC/MLICv2."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.project_in = conv1x1(channels, 2 * channels)
        self.project_out = conv1x1(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.project_in(x).chunk(2, dim=1)
        return self.project_out(first * second)


class SimpleTokenMixingBlock(nn.Module):
    """MLICv2 STMT block for efficient high-rate transform capacity."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.norm1 = LayerNorm2d(channels)
        self.embedding = DepthResidualBlock(channels)
        self.token_mixer = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
        )
        self.token_projection = conv1x1(channels, channels)
        self.norm2 = LayerNorm2d(channels)
        self.gate = GateBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_projection(
            self.token_mixer(self.embedding(self.norm1(x)))
        )
        return x + self.gate(self.norm2(x))


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        hidden = channels // 2
        self.net = nn.Sequential(
            conv1x1(channels, hidden),
            nn.ReLU(inplace=True),
            conv3x3(hidden, hidden),
            nn.ReLU(inplace=True),
            conv1x1(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AttentionBlock(nn.Module):
    class ResidualUnit(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            hidden = int(channels) // 2
            self.net = nn.Sequential(
                conv1x1(channels, hidden),
                nn.ReLU(inplace=True),
                conv3x3(hidden, hidden),
                nn.ReLU(inplace=True),
                conv1x1(hidden, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.relu(x + self.net(x), inplace=True)

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.branch_a = nn.Sequential(*[self.ResidualUnit(channels) for _ in range(3)])
        self.branch_b = nn.Sequential(
            *[self.ResidualUnit(channels) for _ in range(3)],
            conv1x1(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch_a(x) * torch.sigmoid(self.branch_b(x))


class CheckerboardMaskedConv2d(nn.Conv2d):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        mask = torch.zeros_like(self.weight)
        mask[:, :, 0::2, 1::2] = 1
        mask[:, :, 1::2, 0::2] = 1
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ELICAnalysisTransform(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_channels: int) -> None:
        super().__init__()
        n = int(hidden_channels)
        m = int(latent_channels)
        self.net = nn.Sequential(
            conv(int(in_channels), n),
            *[ResidualBottleneckBlock(n) for _ in range(3)],
            conv(n, n),
            *[ResidualBottleneckBlock(n) for _ in range(3)],
            AttentionBlock(n),
            conv(n, n),
            *[ResidualBottleneckBlock(n) for _ in range(3)],
            conv(n, m),
            AttentionBlock(m),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ELICSynthesisTransform(nn.Module):
    """ELIC synthesis transform with a RAW-task-specific 2x RGB output head."""

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        latent_channels: int,
        output_activation: str,
    ) -> None:
        super().__init__()
        n = int(hidden_channels)
        m = int(latent_channels)
        self.output_activation = str(output_activation).lower()
        self.net = nn.Sequential(
            AttentionBlock(m),
            deconv(m, n),
            *[ResidualBottleneckBlock(n) for _ in range(3)],
            deconv(n, n),
            AttentionBlock(n),
            *[ResidualBottleneckBlock(n) for _ in range(3)],
            deconv(n, n),
            *[ResidualBottleneckBlock(n) for _ in range(3)],
            deconv(n, int(out_channels) * 4),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.output_activation in {"", "none", "identity"}:
            return x
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "clamp":
            return x.clamp(0.0, 1.0)
        if self.output_activation == "clamp_ste":
            return clamp_ste(x)
        raise ValueError(f"Unsupported output_activation={self.output_activation!r}.")


class ELICHQAnalysisTransform(nn.Module):
    """Higher-resolution ELIC analysis transform for packed RAW restoration."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_channels: int,
        block_counts: list[int] | tuple[int, int, int],
        block_factory: Callable[[int], nn.Module] = NAFBlock,
    ) -> None:
        super().__init__()
        n = int(hidden_channels)
        m = int(latent_channels)
        counts = tuple(int(value) for value in block_counts)
        if len(counts) != 3 or any(value < 0 for value in counts):
            raise ValueError("HQ analysis block_counts must contain three non-negative integers.")
        self.net = nn.Sequential(
            conv(int(in_channels), n),
            *(block_factory(n) for _ in range(counts[0])),
            conv(n, n),
            *(block_factory(n) for _ in range(counts[1])),
            AttentionBlock(n),
            conv(n, m),
            *(block_factory(m) for _ in range(counts[2])),
            AttentionBlock(m),
        )
        self.detail_split_index = counts[0] + counts[1] + 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_with_detail_feature(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detail_feature = self.net[: self.detail_split_index](x)
        latent = self.net[self.detail_split_index :](detail_feature)
        return latent, detail_feature


class ELICMobileConvAnalysisTransform(nn.Module):
    """NPU-friendly packed-RAW analysis using only dense convs, ReLU, and adds."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        stage_channels: list[int] | tuple[int, int, int],
        block_counts: list[int] | tuple[int, int, int],
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in stage_channels)
        counts = tuple(int(value) for value in block_counts)
        if len(channels) != 3 or any(value <= 0 for value in channels):
            raise ValueError(
                "Mobile analysis stage_channels must contain three positive integers."
            )
        if len(counts) != 3 or any(value < 0 for value in counts):
            raise ValueError(
                "Mobile analysis block_counts must contain three non-negative integers."
            )

        self.channels = channels
        self.block_counts = counts

        modules: list[nn.Module] = []
        previous_channels = int(in_channels)
        for stage_channels_value, stage_blocks in zip(channels, counts):
            modules.extend(
                (
                    nn.Conv2d(
                        previous_channels,
                        stage_channels_value,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
            modules.extend(
                ResidualBottleneckBlock(stage_channels_value)
                for _ in range(stage_blocks)
            )
            previous_channels = stage_channels_value
        modules.append(conv1x1(previous_channels, int(latent_channels)))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ELICHQSynthesisTransform(nn.Module):
    """Three-stage restoration decoder mirroring the HQ packed-RAW encoder."""

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        latent_channels: int,
        output_activation: str,
        block_counts: list[int] | tuple[int, int, int],
        block_factory: Callable[[int], nn.Module] = NAFBlock,
    ) -> None:
        super().__init__()
        n = int(hidden_channels)
        m = int(latent_channels)
        counts = tuple(int(value) for value in block_counts)
        if len(counts) != 3 or any(value < 0 for value in counts):
            raise ValueError("HQ synthesis block_counts must contain three non-negative integers.")
        self.output_activation = str(output_activation).lower()
        self.net = nn.Sequential(
            *(block_factory(m) for _ in range(counts[0])),
            AttentionBlock(m),
            deconv(m, n),
            *(block_factory(n) for _ in range(counts[1])),
            AttentionBlock(n),
            deconv(n, n),
            *(block_factory(n) for _ in range(counts[2])),
            deconv(n, int(out_channels) * 4),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.output_activation in {"", "none", "identity"}:
            return x
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "clamp":
            return x.clamp(0.0, 1.0)
        if self.output_activation == "clamp_ste":
            return clamp_ste(x)
        raise ValueError(f"Unsupported output_activation={self.output_activation!r}.")


class ELICHQDetailSynthesisTransform(ELICHQSynthesisTransform):
    """ELIC-HQ decoder with a coded 32x32 skip and a receiver-only detail head."""

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        latent_channels: int,
        output_activation: str,
        block_counts: list[int] | tuple[int, int, int],
        detail_decoder_channels: int,
        detail_decoder_blocks: int,
        block_factory: Callable[[int], nn.Module] = NAFBlock,
    ) -> None:
        super().__init__(
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            output_activation=output_activation,
            block_counts=block_counts,
            block_factory=block_factory,
        )
        counts = tuple(int(value) for value in block_counts)
        n = int(hidden_channels)
        detail_n = int(detail_decoder_channels)
        detail_blocks = int(detail_decoder_blocks)
        if detail_n <= 0:
            raise ValueError("detail_decoder_channels must be positive.")
        if detail_blocks < 0:
            raise ValueError("detail_decoder_blocks must be non-negative.")
        self.detail_injection_index = counts[0] + 1
        self.refinement_source_index = sum(counts) + 3
        self.refinement = nn.Sequential(
            deconv(n, detail_n),
            *(block_factory(detail_n) for _ in range(detail_blocks)),
            conv3x3(detail_n, int(out_channels) * 4),
            nn.PixelShuffle(2),
        )
        final_projection = self.refinement[-2]
        nn.init.zeros_(final_projection.weight)
        if final_projection.bias is not None:
            nn.init.zeros_(final_projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        detail_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        refinement_source = None
        for index, module in enumerate(self.net):
            x = module(x)
            if index == self.detail_injection_index and detail_feature is not None:
                if x.shape != detail_feature.shape:
                    raise ValueError(
                        "Detail feature shape does not match the 32x32 synthesis feature: "
                        f"{tuple(detail_feature.shape)} vs {tuple(x.shape)}."
                    )
                x = x + detail_feature
            if index == self.refinement_source_index:
                refinement_source = x
        if refinement_source is None:
            raise RuntimeError("Failed to capture the ELIC-HQ refinement source feature.")
        x = x + self.refinement(refinement_source)
        if self.output_activation in {"", "none", "identity"}:
            return x
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "clamp":
            return x.clamp(0.0, 1.0)
        if self.output_activation == "clamp_ste":
            return clamp_ste(x)
        raise ValueError(f"Unsupported output_activation={self.output_activation!r}.")


class ELICRawCodec(nn.Module):
    """ELIC adapted to packed noisy Bayer RAW -> clean linear camera RGB.

    The transform and SCCTX entropy model follow the public CVPR 2022
    reimplementation. Hyperprior-only variants retain the same transforms but
    remove the sequential context model. The task-specific changes are a
    four-channel input and a sub-pixel full-resolution RGB head.
    """

    _CONTAINER_HEADER = struct.Struct(">4sB3x6I")

    def __init__(
        self,
        architecture: str = "elic_raw",
        in_channels: int = 4,
        out_channels: int = 3,
        hidden_channels: int = 192,
        latent_channels: int = 320,
        hyper_channels: int | None = None,
        analysis_hidden_channels: int | None = None,
        synthesis_hidden_channels: int | None = None,
        channel_groups: list[int] | tuple[int, ...] = (16, 16, 32, 64, 192),
        channel_context_hidden: list[int] | tuple[int, int] = (224, 128),
        parameter_aggregation_hidden: list[int] | tuple[int, int] = (640, 512),
        output_activation: str = "clamp_ste",
        raw_bit_depth: int = 14,
        min_likelihood: float = 1.0e-9,
        scale_table_min: float = 0.11,
        scale_table_max: float = 256.0,
        scale_table_levels: int = 64,
        fixed_qstep: float = 1.0,
        hq_analysis_blocks: list[int] | tuple[int, int, int] = (2, 2, 4),
        hq_synthesis_blocks: list[int] | tuple[int, int, int] = (4, 3, 2),
        hq_block_type: str = "naf",
        mobile_analysis_channels: list[int] | tuple[int, int, int] = (128, 192, 256),
        mobile_analysis_blocks: list[int] | tuple[int, int, int] = (0, 2, 3),
        detail_channels: int = 48,
        detail_qstep: float = 0.15,
        detail_decoder_channels: int = 128,
        detail_decoder_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.architecture = str(architecture).lower()
        self.use_scctx = self.architecture not in {
            "elic_hq_hyperprior",
            "elic_raw_hq_hyperprior",
            "elic_raw_restoration_hq_hyperprior",
            "elic_hq_hyperprior_lite",
            "elic_raw_hq_hyperprior_lite",
            "elic_raw_restoration_hq_hyperprior_lite",
            "elic_mobile_conv_hyperprior",
            "elic_raw_mobile_conv_hyperprior",
            "elic_raw_restoration_mobile_conv_hyperprior",
        }
        # Autoregressive context parameters must match the full-precision
        # entropy-coding path during evaluation.
        self.requires_full_precision_evaluation = True
        self.encoder_in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.analysis_hidden_channels = int(analysis_hidden_channels or hidden_channels)
        self.synthesis_hidden_channels = int(synthesis_hidden_channels or hidden_channels)
        self.latent_channels = int(latent_channels)
        self.hyper_channels = int(hyper_channels or hidden_channels)
        self.raw_bit_depth = int(raw_bit_depth)
        self.min_likelihood = float(min_likelihood)
        self.channel_groups = tuple(int(value) for value in channel_groups)
        if sum(self.channel_groups) != self.latent_channels:
            raise ValueError(
                f"ELIC channel_groups sum to {sum(self.channel_groups)}, expected {self.latent_channels}."
            )
        channel_context_hidden = tuple(int(value) for value in channel_context_hidden)
        parameter_aggregation_hidden = tuple(
            int(value) for value in parameter_aggregation_hidden
        )
        if len(channel_context_hidden) != 2 or any(
            value <= 0 for value in channel_context_hidden
        ):
            raise ValueError("channel_context_hidden must contain two positive integers.")
        if len(parameter_aggregation_hidden) != 2 or any(
            value <= 0 for value in parameter_aggregation_hidden
        ):
            raise ValueError(
                "parameter_aggregation_hidden must contain two positive integers."
            )
        self.channel_context_hidden = channel_context_hidden
        self.parameter_aggregation_hidden = parameter_aggregation_hidden
        self.scale_table_min = float(scale_table_min)
        self.scale_table_max = float(scale_table_max)
        self.scale_table_levels = int(scale_table_levels)
        self.fixed_qstep = float(fixed_qstep)
        if self.fixed_qstep <= 0.0:
            raise ValueError("fixed_qstep must be positive.")
        self.use_hq_transform = self.architecture in {
            "elic_hq",
            "elic_raw_hq",
            "elic_raw_restoration_hq",
            "elic_hq_hyperprior",
            "elic_raw_hq_hyperprior",
            "elic_raw_restoration_hq_hyperprior",
            "elic_hq_hyperprior_lite",
            "elic_raw_hq_hyperprior_lite",
            "elic_raw_restoration_hq_hyperprior_lite",
            "elic_hq_detail",
            "elic_raw_hq_detail",
            "elic_raw_restoration_hq_detail",
        }
        self.use_mobile_analysis = self.architecture in {
            "elic_mobile_conv_hyperprior",
            "elic_raw_mobile_conv_hyperprior",
            "elic_raw_restoration_mobile_conv_hyperprior",
        }
        self.use_detail_stream = self.architecture in {
            "elic_hq_detail",
            "elic_raw_hq_detail",
            "elic_raw_restoration_hq_detail",
        }
        self.hq_block_type = str(hq_block_type).lower().replace("-", "_")
        hq_block_factories: dict[str, Callable[[int], nn.Module]] = {
            "naf": NAFBlock,
            "nafblock": NAFBlock,
            "stmt": SimpleTokenMixingBlock,
            "mlicv2_stmt": SimpleTokenMixingBlock,
        }
        if self.hq_block_type not in hq_block_factories:
            raise ValueError(
                f"Unsupported hq_block_type={hq_block_type!r}; "
                f"expected one of {sorted(hq_block_factories)}."
            )
        hq_block_factory = hq_block_factories[self.hq_block_type]
        self.transform_downsample_stages = (
            3 if self.use_hq_transform or self.use_mobile_analysis else 4
        )

        analysis_n = self.analysis_hidden_channels
        synthesis_n = self.synthesis_hidden_channels
        hyper_n = self.hyper_channels
        m = self.latent_channels
        if self.use_hq_transform or self.use_mobile_analysis:
            if self.use_mobile_analysis:
                self.g_a = ELICMobileConvAnalysisTransform(
                    self.encoder_in_channels,
                    m,
                    mobile_analysis_channels,
                    mobile_analysis_blocks,
                )
            else:
                self.g_a = ELICHQAnalysisTransform(
                    self.encoder_in_channels,
                    analysis_n,
                    m,
                    hq_analysis_blocks,
                    hq_block_factory,
                )
            if self.use_detail_stream:
                self.g_s = ELICHQDetailSynthesisTransform(
                    self.out_channels,
                    synthesis_n,
                    m,
                    output_activation,
                    hq_synthesis_blocks,
                    detail_decoder_channels,
                    detail_decoder_blocks,
                    hq_block_factory,
                )
            else:
                self.g_s = ELICHQSynthesisTransform(
                    self.out_channels,
                    synthesis_n,
                    m,
                    output_activation,
                    hq_synthesis_blocks,
                    hq_block_factory,
                )
        else:
            self.g_a = ELICAnalysisTransform(self.encoder_in_channels, analysis_n, m)
            self.g_s = ELICSynthesisTransform(
                self.out_channels,
                synthesis_n,
                m,
                output_activation,
            )
        self.h_a = nn.Sequential(
            conv3x3(m, hyper_n),
            nn.ReLU(inplace=True),
            conv(hyper_n, hyper_n),
            nn.ReLU(inplace=True),
            conv(hyper_n, hyper_n),
        )
        self.h_s = nn.Sequential(
            deconv(hyper_n, hyper_n),
            nn.ReLU(inplace=True),
            deconv(hyper_n, hyper_n * 3 // 2),
            nn.ReLU(inplace=True),
            conv3x3(hyper_n * 3 // 2, 2 * m),
        )

        channel_context_first, channel_context_second = self.channel_context_hidden
        support_channels = []
        for index in range(1, len(self.channel_groups)):
            if index == 1:
                support_channels.append(self.channel_groups[0])
            else:
                support_channels.append(self.channel_groups[0] + self.channel_groups[index - 1])
        self.channel_context = nn.ModuleList(
            nn.Sequential(
                conv(
                    in_count,
                    channel_context_first,
                    kernel_size=5,
                    stride=1,
                ),
                nn.ReLU(inplace=True),
                conv(
                    channel_context_first,
                    channel_context_second,
                    kernel_size=5,
                    stride=1,
                ),
                nn.ReLU(inplace=True),
                conv(
                    channel_context_second,
                    2 * self.channel_groups[index],
                    kernel_size=5,
                    stride=1,
                ),
            )
            for index, in_count in enumerate(support_channels, start=1)
        )
        self.spatial_context = nn.ModuleList(
            CheckerboardMaskedConv2d(
                channels,
                2 * channels,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for channels in self.channel_groups
        )
        aggregation_first, aggregation_second = self.parameter_aggregation_hidden
        self.parameter_aggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(
                    2 * m + 2 * channels + (0 if index == 0 else 2 * channels),
                    aggregation_first,
                ),
                nn.ReLU(inplace=True),
                conv1x1(aggregation_first, aggregation_second),
                nn.ReLU(inplace=True),
                conv1x1(aggregation_second, 2 * channels),
            )
            for index, channels in enumerate(self.channel_groups)
        )
        if not self.use_scctx:
            # Preserve the ELIC-HQ transforms while removing every sequential
            # channel/checkerboard context parameter from the deployable model.
            self.channel_context = nn.ModuleList()
            self.spatial_context = nn.ModuleList()
            self.parameter_aggregation = nn.ModuleList()
        self.entropy_bottleneck = EntropyBottleneck(hyper_n)
        self.gaussian_conditional = GaussianConditional(None)
        if self.use_detail_stream:
            self.detail_channels = int(detail_channels)
            self.detail_qstep = float(detail_qstep)
            if self.detail_channels <= 0:
                raise ValueError("detail_channels must be positive.")
            if self.detail_qstep <= 0.0:
                raise ValueError("detail_qstep must be positive.")
            self.detail_analysis_projection = conv1x1(analysis_n, self.detail_channels)
            self.detail_prediction = deconv(m, self.detail_channels)
            self.detail_to_synthesis = conv1x1(self.detail_channels, synthesis_n)
            nn.init.zeros_(self.detail_to_synthesis.weight)
            if self.detail_to_synthesis.bias is not None:
                nn.init.zeros_(self.detail_to_synthesis.bias)
            self.detail_entropy_bottleneck = EntropyBottleneck(self.detail_channels)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ELICRawCodec":
        model_cfg = dict(config.get("model", {}))
        codec_cfg = dict(config.get("codec", {}))
        codec_cfg.update(model_cfg)
        return cls(**codec_cfg)

    def checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        return self.state_dict()

    def checkpoint_is_partial(self) -> bool:
        return False

    def _build_input(self, packed_raw: torch.Tensor, noise_map: torch.Tensor | None) -> torch.Tensor:
        if int(packed_raw.shape[1]) != self.encoder_in_channels:
            raise ValueError(
                f"ELIC expects {self.encoder_in_channels} packed RAW channels, got {packed_raw.shape[1]}."
            )
        if noise_map is not None:
            raise ValueError("The ELIC baseline intentionally does not accept a noise map.")
        return packed_raw

    def analysis_latent(
        self,
        packed_raw: torch.Tensor,
        noise_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.g_a(self._build_input(packed_raw, noise_map))

    def analysis_latent_with_detail(
        self,
        packed_raw: torch.Tensor,
        noise_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_detail_stream:
            raise RuntimeError("The selected ELIC architecture has no coded detail stream.")
        model_input = self._build_input(packed_raw, noise_map)
        return self.g_a.forward_with_detail_feature(model_input)

    @staticmethod
    def _anchor_mask(reference: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(reference, dtype=torch.bool)
        mask[:, :, 0::2, 0::2] = True
        mask[:, :, 1::2, 1::2] = True
        return mask

    @staticmethod
    def _pack_checkerboard(tensor: torch.Tensor, anchor: bool) -> torch.Tensor:
        batch, channels, height, width = tensor.shape
        if height % 2 or width % 2:
            raise ValueError(f"ELIC checkerboard coding requires even H/W, got {(height, width)}.")
        packed = tensor.new_empty((batch, channels, height, width // 2))
        if anchor:
            packed[:, :, 0::2, :] = tensor[:, :, 0::2, 0::2]
            packed[:, :, 1::2, :] = tensor[:, :, 1::2, 1::2]
        else:
            packed[:, :, 0::2, :] = tensor[:, :, 0::2, 1::2]
            packed[:, :, 1::2, :] = tensor[:, :, 1::2, 0::2]
        return packed

    @staticmethod
    def _unpack_checkerboard(packed: torch.Tensor, width: int, anchor: bool) -> torch.Tensor:
        batch, channels, height, half_width = packed.shape
        if half_width * 2 != int(width):
            raise ValueError("Packed checkerboard width does not match the requested output width.")
        output = packed.new_zeros((batch, channels, height, int(width)))
        if anchor:
            output[:, :, 0::2, 0::2] = packed[:, :, 0::2, :]
            output[:, :, 1::2, 1::2] = packed[:, :, 1::2, :]
        else:
            output[:, :, 0::2, 1::2] = packed[:, :, 0::2, :]
            output[:, :, 1::2, 0::2] = packed[:, :, 1::2, :]
        return output

    def _hyper_parameters(
        self,
        z_hat: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, dim=1)
        if latent_means.shape[-2:] != output_size:
            latent_means = F.interpolate(latent_means, size=output_size, mode="nearest")
            latent_scales = F.interpolate(latent_scales, size=output_size, mode="nearest")
        return latent_means, latent_scales

    def _support(
        self,
        slice_index: int,
        decoded_slices: list[torch.Tensor],
        latent_means: torch.Tensor,
        latent_scales: torch.Tensor,
    ) -> torch.Tensor:
        hyper_support = torch.cat([latent_means, latent_scales], dim=1)
        if slice_index == 0:
            return hyper_support
        channel_input = (
            decoded_slices[0]
            if slice_index == 1
            else torch.cat([decoded_slices[0], decoded_slices[slice_index - 1]], dim=1)
        )
        channel_support = self.channel_context[slice_index - 1](channel_input)
        return torch.cat([channel_support, hyper_support], dim=1)

    def _anchor_parameters(
        self,
        slice_index: int,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channels = self.channel_groups[slice_index]
        spatial_zeros = support.new_zeros(
            (support.shape[0], 2 * channels, support.shape[-2], support.shape[-1])
        )
        means, scales = self.parameter_aggregation[slice_index](
            torch.cat([spatial_zeros, support], dim=1)
        ).chunk(2, dim=1)
        return means, scales

    def _nonanchor_parameters(
        self,
        slice_index: int,
        anchor_hat: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_support = self.spatial_context[slice_index](anchor_hat)
        means, scales = self.parameter_aggregation[slice_index](
            torch.cat([spatial_support, support], dim=1)
        ).chunk(2, dim=1)
        return means, scales

    @staticmethod
    def _centered_quantize(
        values: torch.Tensor,
        means: torch.Tensor,
        training: bool,
        qstep: torch.Tensor,
    ) -> torch.Tensor:
        centered = (values - means) / qstep
        quantized = ste_round(centered) if training else torch.round(centered)
        return quantized * qstep + means

    def _rate_from_likelihoods(
        self,
        y_likelihoods: torch.Tensor,
        z_likelihoods: torch.Tensor,
        raw_shape: torch.Size | tuple[int, ...],
    ) -> dict[str, torch.Tensor]:
        if len(raw_shape) != 4:
            raise ValueError(f"Expected bitrate_shape [B,C,H,W], got {tuple(raw_shape)}.")
        denominator = max(int(raw_shape[1]) * int(raw_shape[2]) * int(raw_shape[3]), 1)
        y_bits = -torch.log2(y_likelihoods.float().flatten(1).clamp_min(self.min_likelihood)).sum(1)
        z_bits = -torch.log2(z_likelihoods.float().flatten(1).clamp_min(self.min_likelihood)).sum(1)
        sample_bits = y_bits + z_bits
        sample_bpp = sample_bits / float(denominator)
        total_bits = sample_bits.sum()
        symbols = y_likelihoods[0].numel() + z_likelihoods[0].numel()
        bits_per_symbol_per_sample = sample_bits / float(max(symbols, 1))
        original_bits = float(y_likelihoods.shape[0] * denominator * self.raw_bit_depth)
        return {
            "bits_per_symbol": bits_per_symbol_per_sample.mean(),
            "bits_per_symbol_per_sample": bits_per_symbol_per_sample,
            "total_bits": total_bits,
            "bpp": sample_bpp.mean(),
            "sample_bpp": sample_bpp,
            "compression_ratio": total_bits.new_tensor(original_bits) / total_bits.clamp_min(1.0e-9),
            "estimated_entropy": bits_per_symbol_per_sample.mean().detach(),
            "latent_bpp": (y_bits / float(denominator)).mean(),
            "hyper_bpp": (z_bits / float(denominator)).mean(),
            "quantized_latent_unique_count": total_bits.new_zeros(()),
        }

    def _add_detail_rate(
        self,
        bitrate: dict[str, torch.Tensor],
        detail_likelihoods: torch.Tensor,
        y_likelihoods: torch.Tensor,
        z_likelihoods: torch.Tensor,
        raw_shape: torch.Size | tuple[int, ...],
    ) -> dict[str, torch.Tensor]:
        denominator = max(int(raw_shape[1]) * int(raw_shape[2]) * int(raw_shape[3]), 1)
        detail_bits = -torch.log2(
            detail_likelihoods.float().flatten(1).clamp_min(self.min_likelihood)
        ).sum(1)
        main_bits = bitrate["sample_bpp"].float() * float(denominator)
        sample_bits = main_bits + detail_bits
        total_bits = sample_bits.sum()
        symbols = (
            y_likelihoods[0].numel()
            + z_likelihoods[0].numel()
            + detail_likelihoods[0].numel()
        )
        bits_per_symbol_per_sample = sample_bits / float(max(symbols, 1))
        original_bits = float(
            detail_likelihoods.shape[0] * denominator * self.raw_bit_depth
        )
        combined = dict(bitrate)
        combined.update(
            {
                "bits_per_symbol": bits_per_symbol_per_sample.mean(),
                "bits_per_symbol_per_sample": bits_per_symbol_per_sample,
                "total_bits": total_bits,
                "bpp": (sample_bits / float(denominator)).mean(),
                "sample_bpp": sample_bits / float(denominator),
                "compression_ratio": total_bits.new_tensor(original_bits)
                / total_bits.clamp_min(1.0e-9),
                "estimated_entropy": bits_per_symbol_per_sample.mean().detach(),
                "main_bpp": bitrate["bpp"],
                "detail_bpp": (detail_bits / float(denominator)).mean(),
                "latent_bpp": bitrate["latent_bpp"]
                + (detail_bits / float(denominator)).mean(),
            }
        )
        return combined

    def forward(
        self,
        noisy_raw: torch.Tensor,
        noise_map: torch.Tensor | None = None,
        bitrate_shape: torch.Size | tuple[int, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        detail_source = None
        if self.use_detail_stream:
            y, detail_source = self.analysis_latent_with_detail(noisy_raw, noise_map)
        else:
            y = self.analysis_latent(noisy_raw, noise_map)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        latent_means, latent_scales = self._hyper_parameters(z_hat, y.shape[-2:])

        qstep = y.new_tensor(self.fixed_qstep)
        if not self.use_scctx:
            means_hat = latent_means
            scales_hat = latent_scales
            y_hat = self._centered_quantize(
                y,
                means_hat,
                self.training,
                qstep,
            )
            _, likelihood = self.gaussian_conditional(
                y / qstep,
                scales_hat / qstep,
                means=means_hat / qstep,
            )
            y_likelihoods = likelihood
            hard_symbols = torch.round((y - means_hat) / qstep).detach()
            hard_y_hat = (hard_symbols * qstep + means_hat.detach()).detach()
        else:
            decoded_slices: list[torch.Tensor] = []
            likelihood_slices: list[torch.Tensor] = []
            mean_slices: list[torch.Tensor] = []
            scale_slices: list[torch.Tensor] = []
            hard_symbol_slices: list[torch.Tensor] = []
            y_slices = torch.split(y, self.channel_groups, dim=1)
            for slice_index, y_slice in enumerate(y_slices):
                support = self._support(
                    slice_index,
                    decoded_slices,
                    latent_means,
                    latent_scales,
                )
                means_anchor, scales_anchor = self._anchor_parameters(
                    slice_index,
                    support,
                )
                anchor_mask = self._anchor_mask(y_slice)
                anchor_values = torch.where(
                    anchor_mask,
                    y_slice,
                    torch.zeros_like(y_slice),
                )
                anchor_decoder = torch.where(
                    anchor_mask,
                    self._centered_quantize(
                        y_slice,
                        means_anchor,
                        self.training,
                        qstep,
                    ),
                    torch.zeros_like(y_slice),
                )
                if self.training:
                    noisy_anchor = anchor_values + (
                        torch.empty_like(anchor_values).uniform_(-0.5, 0.5)
                        * qstep
                    )
                    anchor_context = torch.where(
                        anchor_mask,
                        noisy_anchor,
                        torch.zeros_like(noisy_anchor),
                    )
                else:
                    anchor_context = anchor_decoder

                means_nonanchor, scales_nonanchor = self._nonanchor_parameters(
                    slice_index,
                    anchor_context,
                    support,
                )
                means = torch.where(anchor_mask, means_anchor, means_nonanchor)
                scales = torch.where(anchor_mask, scales_anchor, scales_nonanchor)
                nonanchor_decoder = torch.where(
                    ~anchor_mask,
                    self._centered_quantize(
                        y_slice,
                        means_nonanchor,
                        self.training,
                        qstep,
                    ),
                    torch.zeros_like(y_slice),
                )
                y_hat_slice = anchor_decoder + nonanchor_decoder
                _, likelihood = self.gaussian_conditional(
                    y_slice / qstep,
                    scales / qstep,
                    means=means / qstep,
                )
                decoded_slices.append(y_hat_slice)
                likelihood_slices.append(likelihood)
                mean_slices.append(means)
                scale_slices.append(scales)
                hard_symbol_slices.append(
                    torch.round((y_slice - means) / qstep).detach()
                )

            y_hat = torch.cat(decoded_slices, dim=1)
            means_hat = torch.cat(mean_slices, dim=1)
            scales_hat = torch.cat(scale_slices, dim=1)
            hard_symbols = torch.cat(hard_symbol_slices, dim=1)
            hard_y_hat = (hard_symbols * qstep + means_hat.detach()).detach()
            y_likelihoods = torch.cat(likelihood_slices, dim=1)
        detail_outputs: dict[str, torch.Tensor] = {}
        detail_likelihoods = None
        if self.use_detail_stream:
            if detail_source is None:
                raise RuntimeError("Missing ELIC-HQ detail source feature.")
            detail_target = self.detail_analysis_projection(detail_source)
            detail_prediction = self.detail_prediction(y_hat)
            if detail_prediction.shape[-2:] != detail_target.shape[-2:]:
                detail_prediction = F.interpolate(
                    detail_prediction,
                    size=detail_target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            detail_residual_scaled = (
                detail_target - detail_prediction
            ) / self.detail_qstep
            detail_hat_scaled, detail_likelihoods = self.detail_entropy_bottleneck(
                detail_residual_scaled
            )
            detail_hat = detail_prediction + detail_hat_scaled * self.detail_qstep
            detail_feature = self.detail_to_synthesis(detail_hat)
            reconstruction = self.g_s(y_hat, detail_feature=detail_feature)
            detail_outputs = {
                "detail_source": detail_source,
                "detail_target": detail_target,
                "detail_prediction": detail_prediction,
                "detail_latent": detail_residual_scaled,
                "quantized_detail_latent": detail_hat_scaled,
                "detail_qstep": y.new_full(
                    (y.shape[0], 1, 1, 1),
                    self.detail_qstep,
                ),
            }
        else:
            reconstruction = self.g_s(y_hat)
        if bitrate_shape is not None and reconstruction.shape[-2:] != tuple(bitrate_shape[-2:]):
            reconstruction = F.interpolate(
                reconstruction,
                size=tuple(bitrate_shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
        rate_shape = bitrate_shape if bitrate_shape is not None else reconstruction.shape
        bitrate = self._rate_from_likelihoods(y_likelihoods, z_likelihoods, rate_shape)
        if detail_likelihoods is not None:
            bitrate = self._add_detail_rate(
                bitrate,
                detail_likelihoods,
                y_likelihoods,
                z_likelihoods,
                rate_shape,
            )
        qstep_output = y.new_full((y.shape[0], 1, 1, 1), self.fixed_qstep)
        auxiliary_loss = self.entropy_bottleneck.loss()
        if self.use_detail_stream:
            auxiliary_loss = auxiliary_loss + self.detail_entropy_bottleneck.loss()
        return {
            "recon_linear_rgb": reconstruction,
            "latent": y,
            "hyper_latent": z,
            "quantized_hyper_latent": z_hat,
            "quantized_latent": y_hat,
            "quantized_latent_symbols": y_hat - means_hat,
            "hard_quantized_latent": hard_y_hat,
            "hard_quantized_symbols": hard_symbols,
            "qstep": qstep_output,
            "bitrate": bitrate,
            "latent_scales": scales_hat,
            "latent_means": means_hat,
            "aux_loss": auxiliary_loss,
            **detail_outputs,
        }

    def scale_table(self) -> torch.Tensor:
        return torch.exp(
            torch.linspace(
                math.log(self.scale_table_min),
                math.log(self.scale_table_max / min(self.fixed_qstep, 1.0)),
                self.scale_table_levels,
            )
        )

    def update(self, force: bool = False) -> bool:
        updated = self.gaussian_conditional.update_scale_table(self.scale_table(), force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        if self.use_detail_stream:
            updated |= self.detail_entropy_bottleneck.update(force=force)
        return updated

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        if self.use_detail_stream:
            update_registered_buffers(
                self.detail_entropy_bottleneck,
                "detail_entropy_bottleneck",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )
        return super().load_state_dict(state_dict, strict=strict)

    def _compress_latents(
        self,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[
        list[bytes],
        list[bytes] | list[tuple[list[bytes], list[bytes]]],
        torch.Tensor,
    ]:
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        latent_means, latent_scales = self._hyper_parameters(z_hat, y.shape[-2:])
        qstep = float(self.fixed_qstep)
        if not self.use_scctx:
            normalized_y = y / qstep
            normalized_means = latent_means / qstep
            indexes = self.gaussian_conditional.build_indexes(
                latent_scales / qstep
            )
            y_strings = self.gaussian_conditional.compress(
                normalized_y,
                indexes,
                means=normalized_means,
            )
            y_hat = self.gaussian_conditional.quantize(
                normalized_y,
                "dequantize",
                means=normalized_means,
            ) * qstep
            return z_strings, y_strings, y_hat

        decoded_slices: list[torch.Tensor] = []
        y_strings: list[tuple[list[bytes], list[bytes]]] = []
        for slice_index, y_slice in enumerate(torch.split(y, self.channel_groups, dim=1)):
            support = self._support(slice_index, decoded_slices, latent_means, latent_scales)
            means_anchor, scales_anchor = self._anchor_parameters(slice_index, support)
            y_anchor = self._pack_checkerboard(y_slice, anchor=True)
            mean_anchor = self._pack_checkerboard(means_anchor, anchor=True)
            scale_anchor = self._pack_checkerboard(scales_anchor, anchor=True)
            normalized_y_anchor = y_anchor / qstep
            normalized_mean_anchor = mean_anchor / qstep
            indexes_anchor = self.gaussian_conditional.build_indexes(scale_anchor / qstep)
            anchor_strings = self.gaussian_conditional.compress(
                normalized_y_anchor,
                indexes_anchor,
                means=normalized_mean_anchor,
            )
            anchor_quantized = self.gaussian_conditional.quantize(
                normalized_y_anchor,
                "dequantize",
                means=normalized_mean_anchor,
            ) * qstep
            anchor_hat = self._unpack_checkerboard(
                anchor_quantized, width=y.shape[-1], anchor=True
            )

            means_nonanchor, scales_nonanchor = self._nonanchor_parameters(
                slice_index, anchor_hat, support
            )
            y_nonanchor = self._pack_checkerboard(y_slice, anchor=False)
            mean_nonanchor = self._pack_checkerboard(means_nonanchor, anchor=False)
            scale_nonanchor = self._pack_checkerboard(scales_nonanchor, anchor=False)
            normalized_y_nonanchor = y_nonanchor / qstep
            normalized_mean_nonanchor = mean_nonanchor / qstep
            indexes_nonanchor = self.gaussian_conditional.build_indexes(
                scale_nonanchor / qstep
            )
            nonanchor_strings = self.gaussian_conditional.compress(
                normalized_y_nonanchor,
                indexes_nonanchor,
                means=normalized_mean_nonanchor,
            )
            nonanchor_quantized = self.gaussian_conditional.quantize(
                normalized_y_nonanchor,
                "dequantize",
                means=normalized_mean_nonanchor,
            ) * qstep
            nonanchor_hat = self._unpack_checkerboard(
                nonanchor_quantized, width=y.shape[-1], anchor=False
            )
            decoded_slices.append(anchor_hat + nonanchor_hat)
            y_strings.append((anchor_strings, nonanchor_strings))
        return z_strings, y_strings, torch.cat(decoded_slices, dim=1)

    def _decompress_latents(
        self,
        z_strings: list[bytes],
        y_strings: list[bytes] | list[tuple[list[bytes], list[bytes]]],
        z_shape: tuple[int, int],
        y_shape: tuple[int, int],
    ) -> torch.Tensor:
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        latent_means, latent_scales = self._hyper_parameters(z_hat, y_shape)
        qstep = float(self.fixed_qstep)
        if not self.use_scctx:
            normalized_means = latent_means / qstep
            indexes = self.gaussian_conditional.build_indexes(
                latent_scales / qstep
            )
            return self.gaussian_conditional.decompress(
                y_strings,
                indexes,
                means=normalized_means,
            ) * qstep

        decoded_slices: list[torch.Tensor] = []
        for slice_index, channels in enumerate(self.channel_groups):
            support = self._support(slice_index, decoded_slices, latent_means, latent_scales)
            means_anchor, scales_anchor = self._anchor_parameters(slice_index, support)
            mean_anchor = self._pack_checkerboard(means_anchor, anchor=True)
            scale_anchor = self._pack_checkerboard(scales_anchor, anchor=True)
            normalized_mean_anchor = mean_anchor / qstep
            indexes_anchor = self.gaussian_conditional.build_indexes(scale_anchor / qstep)
            anchor_quantized = self.gaussian_conditional.decompress(
                y_strings[slice_index][0],
                indexes_anchor,
                means=normalized_mean_anchor,
            ) * qstep
            anchor_hat = self._unpack_checkerboard(
                anchor_quantized, width=y_shape[1], anchor=True
            )

            means_nonanchor, scales_nonanchor = self._nonanchor_parameters(
                slice_index, anchor_hat, support
            )
            mean_nonanchor = self._pack_checkerboard(means_nonanchor, anchor=False)
            scale_nonanchor = self._pack_checkerboard(scales_nonanchor, anchor=False)
            normalized_mean_nonanchor = mean_nonanchor / qstep
            indexes_nonanchor = self.gaussian_conditional.build_indexes(
                scale_nonanchor / qstep
            )
            nonanchor_quantized = self.gaussian_conditional.decompress(
                y_strings[slice_index][1],
                indexes_nonanchor,
                means=normalized_mean_nonanchor,
            ) * qstep
            nonanchor_hat = self._unpack_checkerboard(
                nonanchor_quantized, width=y_shape[1], anchor=False
            )
            decoded = anchor_hat + nonanchor_hat
            if int(decoded.shape[1]) != channels:
                raise RuntimeError("Decoded ELIC slice has an unexpected channel count.")
            decoded_slices.append(decoded)
        return torch.cat(decoded_slices, dim=1)

    @torch.no_grad()
    def actual_entropy_encode_decode(
        self,
        model_output: dict[str, torch.Tensor],
        bitrate_shape: torch.Size | tuple[int, ...],
    ) -> dict[str, float | int]:
        if self.use_detail_stream:
            raise NotImplementedError(
                "Actual container coding for the new detail stream is not implemented yet; "
                "use entropy-estimated bpp while training ELIC-HQ-Detail."
            )
        self.update(force=False)
        y = model_output["latent"].detach().float()
        z = model_output["hyper_latent"].detach().float()
        total_start = time.perf_counter()
        encode_start = time.perf_counter()
        z_strings, y_strings, encoder_y_hat = self._compress_latents(y, z)
        encode_seconds = time.perf_counter() - encode_start
        decode_start = time.perf_counter()
        decoded_y = self._decompress_latents(
            z_strings,
            y_strings,
            z_shape=(int(z.shape[-2]), int(z.shape[-1])),
            y_shape=(int(y.shape[-2]), int(y.shape[-1])),
        )
        decode_seconds = time.perf_counter() - decode_start
        expected = model_output["hard_quantized_latent"].detach().float()
        roundtrip_error = max(
            float((decoded_y - encoder_y_hat).abs().max().item()),
            float((decoded_y - expected).abs().max().item()),
        )

        payload_bytes = 0
        file_bytes = 0
        header_bytes = 0
        batch_size = int(y.shape[0])
        for sample_index in range(batch_size):
            streams = [z_strings[sample_index]]
            if self.use_scctx:
                for anchor_strings, nonanchor_strings in y_strings:
                    streams.extend(
                        [
                            anchor_strings[sample_index],
                            nonanchor_strings[sample_index],
                        ]
                    )
            else:
                streams.append(y_strings[sample_index])
            sample_payload = sum(len(stream) for stream in streams)
            sample_header = self._CONTAINER_HEADER.size + 4 * len(streams)
            payload_bytes += sample_payload
            header_bytes += sample_header
            file_bytes += sample_payload + sample_header

        denominator = max(
            int(bitrate_shape[1]) * int(bitrate_shape[2]) * int(bitrate_shape[3]) * batch_size,
            1,
        )
        return {
            "payload_bpp": 8.0 * payload_bytes / denominator,
            "file_bpp": 8.0 * file_bytes / denominator,
            "payload_bytes": payload_bytes,
            "file_bytes": file_bytes,
            "header_bytes": header_bytes,
            "encode_seconds": encode_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": time.perf_counter() - total_start,
            "roundtrip_max_error": roundtrip_error,
        }
