from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


_CFA_FLIPS = {
    "RGGB": (),
    "GRBG": (-1,),
    "GBRG": (-2,),
    "BGGR": (-2, -1),
}


@dataclass(frozen=True)
class RawUint16Metadata:
    width: int
    height: int
    cfa_pattern: str
    black_level: tuple[float, float, float, float]
    white_level: float
    bit_depth: int
    model_input_include_black_level: bool
    document: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "RawUint16Metadata":
        source = Path(path)
        document = json.loads(source.read_text(encoding="utf-8"))
        if str(document.get("dtype", "")).lower() not in {"uint16", "<u2"}:
            raise ValueError("RAW sidecar dtype must be uint16.")
        if str(document.get("endianness", "")).lower() not in {"little", "little-endian"}:
            raise ValueError("ERIC quick inference requires little-endian uint16 RAW.")
        pattern = str(document["cfa_pattern"]).upper()
        if pattern not in _CFA_FLIPS:
            raise ValueError(f"Unsupported CFA pattern {pattern!r}.")
        black = document["black_level"]
        if isinstance(black, (int, float)):
            black_values = (float(black),) * 4
        else:
            black_values = tuple(float(value) for value in black)
            if len(black_values) != 4:
                raise ValueError("black_level must be a scalar or four row-major CFA values.")
        metadata = cls(
            width=int(document["width"]),
            height=int(document["height"]),
            cfa_pattern=pattern,
            black_level=black_values,
            white_level=float(document["white_level"]),
            bit_depth=int(document["bit_depth"]),
            model_input_include_black_level=bool(
                document.get("model_input_include_black_level", True)
            ),
            document=document,
        )
        metadata.validate()
        return metadata

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("RAW width and height must be positive even integers.")
        if self.bit_depth <= 0 or self.bit_depth > 16:
            raise ValueError("bit_depth must be in [1,16].")
        if any(value < 0.0 for value in self.black_level):
            raise ValueError("black_level values must be non-negative.")
        if any(value >= self.white_level for value in self.black_level):
            raise ValueError("Every black_level value must be below white_level.")


def load_raw_uint16(
    raw_path: str | Path,
    metadata_path: str | Path,
) -> tuple[np.ndarray, RawUint16Metadata]:
    """Load a headerless, unpacked, little-endian uint16 Bayer frame."""

    metadata = RawUint16Metadata.from_json(metadata_path)
    source = Path(raw_path)
    expected_bytes = metadata.width * metadata.height * 2
    actual_bytes = source.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"RAW size mismatch: expected {expected_bytes} bytes for "
            f"{metadata.width}x{metadata.height}, got {actual_bytes}."
        )
    raw = np.fromfile(source, dtype="<u2").reshape(metadata.height, metadata.width)
    return raw, metadata


def _normalize_native_mosaic(
    raw: np.ndarray,
    metadata: RawUint16Metadata,
) -> torch.Tensor:
    values = torch.from_numpy(raw.astype(np.float32, copy=False))
    black = values.new_empty(values.shape)
    black[0::2, 0::2] = metadata.black_level[0]
    black[0::2, 1::2] = metadata.black_level[1]
    black[1::2, 0::2] = metadata.black_level[2]
    black[1::2, 1::2] = metadata.black_level[3]
    white = values.new_tensor(metadata.white_level)
    if metadata.model_input_include_black_level:
        return (values / white).clamp(0.0, 1.0)
    return ((values - black) / (white - black)).clamp(0.0, 1.0)


def normalize_and_pack_raw(
    raw: np.ndarray,
    metadata: RawUint16Metadata,
) -> torch.Tensor:
    """Normalize native sensor codes and spatially canonicalize to packed RGGB."""

    mosaic = _normalize_native_mosaic(raw, metadata)
    flips = _CFA_FLIPS[metadata.cfa_pattern]
    if flips:
        mosaic = torch.flip(mosaic, dims=flips)
    return torch.stack(
        (
            mosaic[0::2, 0::2],
            mosaic[0::2, 1::2],
            mosaic[1::2, 0::2],
            mosaic[1::2, 1::2],
        ),
        dim=0,
    ).contiguous()


def native_rgb_from_canonical(rgb: torch.Tensor, cfa_pattern: str) -> torch.Tensor:
    """Undo the spatial CFA canonicalization on a decoded RGB image."""

    pattern = str(cfa_pattern).upper()
    if pattern not in _CFA_FLIPS:
        raise ValueError(f"Unsupported CFA pattern {pattern!r}.")
    flips = _CFA_FLIPS[pattern]
    return torch.flip(rgb, dims=flips) if flips else rgb
