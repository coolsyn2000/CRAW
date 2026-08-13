from __future__ import annotations

import torch


BAYER_TILES = {
    "RGGB": (("R", "Gr"), ("Gb", "B")),
    "BGGR": (("B", "Gb"), ("Gr", "R")),
    "GRBG": (("Gr", "R"), ("B", "Gb")),
    "GBRG": (("Gb", "B"), ("R", "Gr")),
}
COLOR_INDEX = {"R": 0, "Gr": 1, "Gb": 1, "G": 1, "B": 2}


def inverse_gamma(srgb: torch.Tensor, gamma: float | str = 2.2) -> torch.Tensor:
    if isinstance(gamma, str) and gamma.lower() in {"srgb", "iec61966-2-1"}:
        srgb = srgb.clamp(0.0, 1.0)
        return torch.where(
            srgb <= 0.04045,
            srgb / 12.92,
            ((srgb + 0.055) / 1.055).pow(2.4),
        ).clamp(0.0, 1.0)
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    return srgb.clamp(0.0, 1.0).pow(gamma).clamp(0.0, 1.0)


def gamma_encode(
    linear_rgb: torch.Tensor,
    gamma: float | str = 2.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    if isinstance(gamma, str) and gamma.lower() in {"srgb", "iec61966-2-1"}:
        linear_rgb = linear_rgb.clamp(0.0, 1.0)
        return torch.where(
            linear_rgb <= 0.0031308,
            12.92 * linear_rgb,
            1.055 * linear_rgb.pow(1.0 / 2.4) - 0.055,
        ).clamp(0.0, 1.0)
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    linear_rgb = linear_rgb.clamp(0.0, 1.0).clamp_min(float(eps))
    return linear_rgb.pow(1.0 / gamma).clamp(0.0, 1.0)


def _as_nchw(rgb: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if rgb.ndim == 3:
        return rgb.unsqueeze(0), True
    if rgb.ndim == 4:
        return rgb, False
    raise ValueError(f"Expected RGB tensor [3,H,W] or [B,3,H,W], got {tuple(rgb.shape)}.")


def _awb_gains_tensor(
    awb_gains: dict[str, float] | torch.Tensor | None,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if awb_gains is None:
        gains = torch.ones(batch_size, 3, dtype=dtype, device=device)
    elif isinstance(awb_gains, dict):
        gains = torch.tensor(
            [
                float(awb_gains.get("red_gain", 1.0)),
                float(awb_gains.get("green_gain", 1.0)),
                float(awb_gains.get("blue_gain", 1.0)),
            ],
            dtype=dtype,
            device=device,
        ).view(1, 3).expand(batch_size, 3)
    else:
        gains = awb_gains.to(device=device, dtype=dtype)
        if gains.ndim == 1:
            gains = gains.view(1, 3).expand(batch_size, 3)
        elif gains.ndim != 2 or gains.shape[-1] != 3:
            raise ValueError(f"awb_gains must have shape [3] or [B,3], got {tuple(gains.shape)}.")
        if gains.shape[0] == 1 and batch_size != 1:
            gains = gains.expand(batch_size, 3)
        if gains.shape[0] != batch_size:
            raise ValueError(f"awb_gains batch size {gains.shape[0]} does not match RGB batch size {batch_size}.")
    return gains.view(batch_size, 3, 1, 1)


def linear_camera_rgb_to_srgb(
    linear_camera_rgb: torch.Tensor,
    gamma: float | str = 2.2,
    ccm: list[list[float]] | tuple[tuple[float, ...], ...] | torch.Tensor | None = None,
    awb_gains: dict[str, float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate camera-linear RGB display rendering.

    The raw-sim unprocess path uses inverse gamma, inverse CCM, and inverse AWB.
    This applies the matching forward AWB, CCM, and gamma encoding for losses
    and debug previews.
    """
    rgb, squeezed = _as_nchw(linear_camera_rgb)
    if rgb.shape[-3] != 3:
        raise ValueError(f"Expected RGB tensor with 3 channels, got {rgb.shape[-3]}.")

    rgb = rgb.clamp(0.0, 1.0)
    gains = _awb_gains_tensor(awb_gains, rgb.shape[0], rgb.dtype, rgb.device)
    rgb = rgb * gains

    b, _, h, w = rgb.shape
    if ccm is None:
        ccm_t = torch.eye(3, dtype=rgb.dtype, device=rgb.device).expand(b, 3, 3)
    else:
        ccm_t = torch.as_tensor(ccm, dtype=rgb.dtype, device=rgb.device)
        if ccm_t.ndim == 2:
            ccm_t = ccm_t.unsqueeze(0).expand(b, 3, 3)
        elif ccm_t.ndim == 3 and ccm_t.shape[0] == 1 and b != 1:
            ccm_t = ccm_t.expand(b, 3, 3)
    if ccm_t.shape != (b, 3, 3):
        raise ValueError(
            f"ccm must have shape [3,3] or [B,3,3], got {tuple(ccm_t.shape)}."
        )

    flat = rgb.permute(0, 2, 3, 1).reshape(b, h * w, 3)
    srgb_linear = torch.bmm(flat, ccm_t.transpose(1, 2))
    srgb_linear = srgb_linear.reshape(b, h, w, 3).permute(0, 3, 1, 2)
    srgb = gamma_encode(srgb_linear, gamma=gamma)
    return srgb.squeeze(0) if squeezed else srgb


def srgb_to_linear_camera_rgb(
    srgb: torch.Tensor,
    gamma: float | str = 2.2,
    ccm: list[list[float]] | tuple[tuple[float, ...], ...] | torch.Tensor | None = None,
    awb_gains: dict[str, float] | None = None,
    safe_invert_awb: bool = False,
) -> torch.Tensor:
    """Approximate sRGB unprocessing into camera linear RGB.

    This mirrors the compact raw-sim path used by Open-AISP: inverse gamma,
    inverse color correction, then inverse AWB gains.
    """
    if srgb.shape[-3] != 3:
        raise ValueError(f"Expected RGB tensor with 3 channels, got {srgb.shape[-3]}.")

    linear_rgb = inverse_gamma(srgb, gamma=gamma)
    if ccm is None:
        ccm_t = torch.eye(3, dtype=linear_rgb.dtype, device=linear_rgb.device)
    else:
        ccm_t = torch.as_tensor(ccm, dtype=linear_rgb.dtype, device=linear_rgb.device)
    if ccm_t.shape != (3, 3):
        raise ValueError(f"ccm must be 3x3, got {tuple(ccm_t.shape)}.")

    inv_ccm = torch.linalg.inv(ccm_t)
    flat = linear_rgb.permute(1, 2, 0).reshape(-1, 3)
    camera_rgb = (flat @ inv_ccm.T).reshape(linear_rgb.shape[1], linear_rgb.shape[2], 3).permute(2, 0, 1)

    awb_gains = awb_gains or {}
    gains = torch.tensor(
        [
            float(awb_gains.get("red_gain", 1.0)),
            float(awb_gains.get("green_gain", 1.0)),
            float(awb_gains.get("blue_gain", 1.0)),
        ],
        dtype=camera_rgb.dtype,
        device=camera_rgb.device,
    ).view(3, 1, 1)
    inverse_gains = torch.rsqrt(gains.square().clamp_min(1e-16))
    if safe_invert_awb:
        gray = camera_rgb.mean(dim=0, keepdim=True)
        mask = ((gray - 0.9) / 0.1).clamp(0.0, 1.0).square()
        inverse_gains = torch.maximum(mask + (1.0 - mask) * inverse_gains, inverse_gains)
    return (camera_rgb * inverse_gains).clamp(0.0, 1.0)


def bayer_channel_names(pattern: str = "RGGB") -> list[str]:
    tile = BAYER_TILES[pattern.upper()]
    return [tile[0][0], tile[0][1], tile[1][0], tile[1][1]]


def remosaic_linear_rgb_to_packed_raw(linear_camera_rgb: torch.Tensor, pattern: str = "RGGB") -> torch.Tensor:
    """Pack linear camera RGB into Bayer RAW planes [R, G1, G2, B]-style order."""
    pattern = pattern.upper()
    if pattern not in BAYER_TILES:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    if linear_camera_rgb.shape[-3] != 3:
        raise ValueError(f"Expected linear RGB tensor with 3 channels, got {linear_camera_rgb.shape[-3]}.")

    _, h, w = linear_camera_rgb.shape
    h = h - h % 2
    w = w - w % 2
    rgb = linear_camera_rgb[:, :h, :w]
    tile = BAYER_TILES[pattern]
    packed = []
    for row in range(2):
        for col in range(2):
            color = tile[row][col]
            packed.append(rgb[COLOR_INDEX[color], row::2, col::2])
    return torch.stack(packed, dim=0).clamp(0.0, 1.0)


def unprocess_srgb_to_packed_raw(
    srgb: torch.Tensor,
    gamma: float | str = 2.2,
    ccm: list[list[float]] | tuple[tuple[float, ...], ...] | torch.Tensor | None = None,
    awb_gains: dict[str, float] | None = None,
    pattern: str = "RGGB",
) -> torch.Tensor:
    linear_camera_rgb = srgb_to_linear_camera_rgb(
        srgb,
        gamma=gamma,
        ccm=ccm,
        awb_gains=awb_gains,
        safe_invert_awb=False,
    )
    return remosaic_linear_rgb_to_packed_raw(linear_camera_rgb, pattern=pattern)
