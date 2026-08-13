from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from raw_codec.data.unprocess import linear_camera_rgb_to_srgb


Reduction = Literal["mean", "sum", "none"]

_LPIPS_CACHE: dict[tuple[str, str], nn.Module] = {}

LINEAR_CAMERA_DOMAINS = {"camera", "linear_camera", "linear_camera_rgb", "camera_linear", "linear"}
SRGB_DOMAINS = {"srgb", "s_rgb", "display", "rendered", "rgb"}


def _reduce_loss(loss: torch.Tensor, reduction: Reduction = "mean") -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


def _as_nchw(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"{name} expects [B,C,H,W] tensors, got {tuple(x.shape)}.")
    return x.float()


def _loss_domain_name(domain: str | None) -> str:
    domain = str(domain or "camera").lower()
    if domain in LINEAR_CAMERA_DOMAINS:
        return "camera"
    if domain in SRGB_DOMAINS:
        return "srgb"
    raise ValueError(f"Unsupported loss domain: {domain}")


def convert_loss_domain(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    domain: str | None = "camera",
    gamma: float = 2.2,
    ccm: list[list[float]] | tuple[tuple[float, ...], ...] | torch.Tensor | None = None,
    awb_gains: dict[str, float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return prediction and target in the requested RGB loss domain.

    ``camera`` leaves tensors in linear camera RGB. ``srgb`` applies the same
    approximate forward AWB, CCM, and gamma path used by rawsim previews.
    """
    domain_name = _loss_domain_name(domain)
    if domain_name == "camera":
        return pred, target
    return (
        linear_camera_rgb_to_srgb(pred, gamma=gamma, ccm=ccm, awb_gains=awb_gains),
        linear_camera_rgb_to_srgb(target, gamma=gamma, ccm=ccm, awb_gains=awb_gains),
    )


def _gaussian_kernel(
    channels: int,
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.view(1, 1, window_size, window_size).expand(channels, 1, -1, -1).contiguous()


def _filter2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    x = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(x, kernel, groups=x.shape[1])


def _ssim_and_contrast(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"SSIM expects matching shapes, got {pred.shape} and {target.shape}.")
    with torch.amp.autocast(device_type=pred.device.type, enabled=False):
        pred = _as_nchw(pred, "SSIM")
        target = _as_nchw(target, "SSIM")
        _, channels, height, width = pred.shape
        window_size = min(int(window_size), int(height), int(width))
        if window_size % 2 == 0:
            window_size -= 1
        window_size = max(window_size, 3)
        kernel = _gaussian_kernel(channels, window_size, sigma, pred.device, pred.dtype)

        mu_x = _filter2d(pred, kernel)
        mu_y = _filter2d(target, kernel)
        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()
        mu_xy = mu_x * mu_y

        sigma_x_sq = (_filter2d(pred * pred, kernel) - mu_x_sq).clamp_min(0.0)
        sigma_y_sq = (_filter2d(target * target, kernel) - mu_y_sq).clamp_min(0.0)
        sigma_xy = _filter2d(pred * target, kernel) - mu_xy

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        contrast = (2.0 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2).clamp_min(1e-12)
        ssim = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ).clamp_min(1e-12)
        ssim = ssim.clamp(0.0, 1.0)
        contrast = contrast.clamp(0.0, 1.0)
        return ssim.flatten(1).mean(dim=1), contrast.flatten(1).mean(dim=1)


def _ms_ssim_score(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    pred = _as_nchw(pred, "MS-SSIM").clamp(0.0, data_range)
    target = _as_nchw(target, "MS-SSIM").clamp(0.0, data_range)
    if pred.shape != target.shape:
        raise ValueError(f"MS-SSIM expects matching shapes, got {pred.shape} and {target.shape}.")

    weights = pred.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    min_hw = int(min(pred.shape[-2:]))
    levels = min(len(weights), max(1, int(torch.floor(torch.log2(pred.new_tensor(max(min_hw, 1)))).item()) - 2))
    weights = weights[:levels]
    weights = weights / weights.sum().clamp_min(1e-12)

    contrast_values: list[torch.Tensor] = []
    ssim_value = pred.new_ones(pred.shape[0])
    x = pred
    y = target
    for level in range(levels):
        ssim_value, contrast = _ssim_and_contrast(x, y, data_range=data_range)
        if level < levels - 1:
            contrast_values.append(contrast.clamp(1e-6, 1.0))
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)

    scale_values = contrast_values + [ssim_value.clamp(1e-6, 1.0)]
    score = pred.new_ones(pred.shape[0])
    for scale_value, weight in zip(scale_values, weights):
        score = score * scale_value.pow(weight)
    return score.clamp(0.0, 1.0)


def _perceptual_rgb(x: torch.Tensor, data_range: float, *, to_minus_one_one: bool) -> torch.Tensor:
    x = _as_nchw(x, "perceptual loss").clamp(0.0, data_range) / float(data_range)
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    elif x.shape[1] != 3:
        raise ValueError(
            "LPIPS is an RGB perceptual metric. Expected 1 or 3 channels, "
            f"got {x.shape[1]}."
        )
    if to_minus_one_one:
        x = x * 2.0 - 1.0
    return x.contiguous()


def _lpips_model(net: str, device: torch.device) -> nn.Module:
    key = (str(net).lower(), str(device))
    if key not in _LPIPS_CACHE:
        try:
            import lpips  # type: ignore
        except ImportError as exc:
            raise ImportError("LPIPS loss requires `lpips`. Install it with `pip install lpips`.") from exc
        model = lpips.LPIPS(net=key[0]).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _LPIPS_CACHE[key] = model
    return _LPIPS_CACHE[key]


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
    reduction: Reduction = "mean",
    domain: str | None = "camera",
    gamma: float = 2.2,
    ccm: list[list[float]] | tuple[tuple[float, ...], ...] | torch.Tensor | None = None,
    awb_gains: dict[str, float] | torch.Tensor | None = None,
) -> torch.Tensor:
    pred, target = convert_loss_domain(pred, target, domain=domain, gamma=gamma, ccm=ccm, awb_gains=awb_gains)
    loss = torch.sqrt((pred - target).square() + float(eps) ** 2)
    return _reduce_loss(loss, reduction)


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: Reduction = "mean",
    domain: str | None = "camera",
    gamma: float = 2.2,
    ccm: list[list[float]] | tuple[tuple[float, ...], ...] | torch.Tensor | None = None,
    awb_gains: dict[str, float] | torch.Tensor | None = None,
) -> torch.Tensor:
    pred, target = convert_loss_domain(pred, target, domain=domain, gamma=gamma, ccm=ccm, awb_gains=awb_gains)
    score, _ = _ssim_and_contrast(pred.clamp(0.0, data_range), target.clamp(0.0, data_range), data_range=data_range)
    return _reduce_loss(1.0 - score, reduction)


def ms_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: Reduction = "mean",
    domain: str | None = "camera",
    gamma: float = 2.2,
    ccm: list[list[float]] | tuple[tuple[float, ...], ...] | torch.Tensor | None = None,
    awb_gains: dict[str, float] | torch.Tensor | None = None,
) -> torch.Tensor:
    pred, target = convert_loss_domain(pred, target, domain=domain, gamma=gamma, ccm=ccm, awb_gains=awb_gains)
    return _reduce_loss(1.0 - _ms_ssim_score(pred, target, data_range=data_range), reduction)


def lpips_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    net: str = "alex",
    reduction: Reduction = "mean",
    domain: str | None = "srgb",
    gamma: float = 2.2,
    ccm: list[list[float]] | tuple[tuple[float, ...], ...] | torch.Tensor | None = None,
    awb_gains: dict[str, float] | torch.Tensor | None = None,
) -> torch.Tensor:
    pred, target = convert_loss_domain(pred, target, domain=domain, gamma=gamma, ccm=ccm, awb_gains=awb_gains)
    pred_rgb = _perceptual_rgb(pred, data_range, to_minus_one_one=True)
    target_rgb = _perceptual_rgb(target, data_range, to_minus_one_one=True)
    score = _lpips_model(net, pred_rgb.device)(pred_rgb, target_rgb)
    if score.ndim == 0:
        score = score.view(1)
    return _reduce_loss(score.flatten(1).mean(dim=1), reduction)


charboineer_loss = charbonnier_loss
msssim_loss = ms_ssim_loss


__all__ = [
    "charboineer_loss",
    "charbonnier_loss",
    "convert_loss_domain",
    "lpips_loss",
    "msssim_loss",
    "ms_ssim_loss",
    "ssim_loss",
]
