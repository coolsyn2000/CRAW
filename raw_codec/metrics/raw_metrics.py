from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def image_psnr(recon: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(recon, target)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=recon.device) / mse.clamp_min(1e-12))


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
    recon: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
    window_size: int = 11,
    sigma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.amp.autocast(device_type=recon.device.type, enabled=False):
        recon = recon.float()
        target = target.float()
        _, channels, height, width = recon.shape
        window_size = min(int(window_size), int(height), int(width))
        if window_size % 2 == 0:
            window_size -= 1
        window_size = max(window_size, 3)
        kernel = _gaussian_kernel(channels, window_size, sigma, recon.device, recon.dtype)

        mu_x = _filter2d(recon, kernel)
        mu_y = _filter2d(target, kernel)
        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()
        mu_xy = mu_x * mu_y

        sigma_x_sq = (_filter2d(recon * recon, kernel) - mu_x_sq).clamp_min(0.0)
        sigma_y_sq = (_filter2d(target * target, kernel) - mu_y_sq).clamp_min(0.0)
        sigma_xy = _filter2d(recon * target, kernel) - mu_xy

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        contrast = (2.0 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2).clamp_min(1e-12)
        ssim = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ).clamp_min(1e-12)
        ssim = ssim.clamp(0.0, 1.0)
        contrast = contrast.clamp(0.0, 1.0)
        return ssim.flatten(1).mean(dim=1), contrast.flatten(1).mean(dim=1)


@torch.no_grad()
def image_ssim(recon: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Single-scale SSIM for image tensors in [0, data_range]."""
    if recon.shape != target.shape:
        raise ValueError(f"SSIM expects matching shapes, got {recon.shape} and {target.shape}.")
    if recon.ndim != 4:
        raise ValueError(f"SSIM expects [B,C,H,W] tensors, got {recon.shape}.")
    score, _ = _ssim_and_contrast(
        recon.float().clamp(0.0, data_range),
        target.float().clamp(0.0, data_range),
        data_range=data_range,
    )
    return score.mean()


@torch.no_grad()
def image_ms_ssim(recon: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Multi-scale SSIM for image tensors in [0, 1]."""
    if recon.shape != target.shape:
        raise ValueError(f"MS-SSIM expects matching shapes, got {recon.shape} and {target.shape}.")
    if recon.ndim != 4:
        raise ValueError(f"MS-SSIM expects [B,C,H,W] tensors, got {recon.shape}.")

    weights = recon.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    min_hw = int(min(recon.shape[-2:]))
    levels = min(len(weights), max(1, int(math.floor(math.log2(max(min_hw, 1)))) - 2))
    weights = weights[:levels]
    weights = weights / weights.sum().clamp_min(1e-12)

    x = recon.clamp(0.0, data_range)
    y = target.clamp(0.0, data_range)
    contrast_values: list[torch.Tensor] = []
    ssim_value = x.new_ones(x.shape[0])
    for level in range(levels):
        ssim_value, contrast = _ssim_and_contrast(x, y, data_range=data_range)
        if level < levels - 1:
            contrast_values.append(contrast.clamp(1e-6, 1.0))
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)

    scale_values = contrast_values + [ssim_value.clamp(1e-6, 1.0)]
    score = x.new_ones(recon.shape[0])
    for scale_value, weight in zip(scale_values, weights):
        score = score * scale_value.pow(weight)
    return score.clamp(0.0, 1.0).mean()


@torch.no_grad()
def image_lpips(
    recon: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    net: str = "alex",
    domain: str | None = "srgb",
) -> torch.Tensor:
    from raw_codec.losses.loss_zoo import lpips_loss

    return lpips_loss(
        recon,
        target,
        data_range=data_range,
        net=net,
        domain=domain,
    )


@torch.no_grad()
def tensor_stats(tensor: torch.Tensor, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_min": float(tensor.min().item()),
        f"{prefix}_max": float(tensor.max().item()),
        f"{prefix}_mean": float(tensor.mean().item()),
        f"{prefix}_std": float(tensor.std(unbiased=False).item()),
    }


@torch.no_grad()
def calculate_quality_metrics(
    recon: torch.Tensor,
    target: torch.Tensor,
    latent: torch.Tensor | None = None,
) -> dict[str, float]:
    mse = F.mse_loss(recon, target)
    l1 = F.l1_loss(recon, target)
    psnr = 10.0 * math.log10(1.0 / max(float(mse.item()), 1e-12))
    metrics = {
        "l1": float(l1.item()),
        "mse": float(mse.item()),
        "psnr": psnr,
        "ssim": float(image_ssim(recon, target).item()),
        "ms_ssim": float(image_ms_ssim(recon, target).item()),
    }
    metrics.update(tensor_stats(recon, "recon"))
    if latent is not None:
        metrics.update(tensor_stats(latent, "latent"))
    return metrics


psnr_metric = image_psnr
ssim_metric = image_ssim
ms_ssim_metric = image_ms_ssim
lpips_metric = image_lpips
