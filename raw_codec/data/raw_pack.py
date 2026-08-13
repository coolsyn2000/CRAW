from __future__ import annotations

import torch


def pack_bayer(raw_bayer: torch.Tensor, pattern: str = "RGGB") -> torch.Tensor:
    """Pack a single-channel Bayer tensor into four RGGB planes.

    Args:
        raw_bayer: Tensor with shape ``[..., H, W]`` or ``[..., 1, H, W]``.
        pattern: CFA pattern. v1 implements RGGB only.
    """
    if pattern.upper() != "RGGB":
        raise NotImplementedError("Only RGGB packing is implemented in v1.")

    if raw_bayer.dim() >= 3 and raw_bayer.shape[-3] == 1:
        raw_bayer = raw_bayer.squeeze(-3)

    h, w = raw_bayer.shape[-2:]
    if h % 2 != 0 or w % 2 != 0:
        raw_bayer = raw_bayer[..., : h - (h % 2), : w - (w % 2)]

    r = raw_bayer[..., 0::2, 0::2]
    g1 = raw_bayer[..., 0::2, 1::2]
    g2 = raw_bayer[..., 1::2, 0::2]
    b = raw_bayer[..., 1::2, 1::2]
    return torch.stack([r, g1, g2, b], dim=-3)


def unpack_packed_raw(packed_raw: torch.Tensor, pattern: str = "RGGB") -> torch.Tensor:
    """Unpack four-plane RGGB RAW into a single Bayer mosaic tensor."""
    if pattern.upper() != "RGGB":
        raise NotImplementedError("Only RGGB unpacking is implemented in v1.")
    if packed_raw.shape[-3] != 4:
        raise ValueError(f"Expected 4 packed RAW channels, got {packed_raw.shape[-3]}.")

    *prefix, _, h, w = packed_raw.shape
    raw = torch.empty(*prefix, h * 2, w * 2, device=packed_raw.device, dtype=packed_raw.dtype)
    raw[..., 0::2, 0::2] = packed_raw[..., 0, :, :]
    raw[..., 0::2, 1::2] = packed_raw[..., 1, :, :]
    raw[..., 1::2, 0::2] = packed_raw[..., 2, :, :]
    raw[..., 1::2, 1::2] = packed_raw[..., 3, :, :]
    return raw


def approximate_packed_raw_from_rgb(rgb: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """Convert display RGB in [0, 1] to an approximate linear packed RAW tensor."""
    if rgb.shape[-3] != 3:
        raise ValueError(f"Expected RGB tensor with 3 channels, got {rgb.shape[-3]}.")
    linear = rgb.clamp(0.0, 1.0).pow(gamma)
    r = linear[..., 0, :, :]
    g = linear[..., 1, :, :]
    b = linear[..., 2, :, :]
    return torch.stack([r, g, g, b], dim=-3).clamp(0.0, 1.0)


def packed_raw_to_rgb(packed_raw: torch.Tensor) -> torch.Tensor:
    """Approximate packed RAW as RGB for debugging visualization."""
    if packed_raw.shape[-3] != 4:
        raise ValueError(f"Expected 4 packed RAW channels, got {packed_raw.shape[-3]}.")
    r = packed_raw[..., 0, :, :]
    g = 0.5 * (packed_raw[..., 1, :, :] + packed_raw[..., 2, :, :])
    b = packed_raw[..., 3, :, :]
    return torch.stack([r, g, b], dim=-3).clamp(0.0, 1.0)
