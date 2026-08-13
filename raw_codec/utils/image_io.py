from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in [0, 1] to an HWC uint8 image."""
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}.")
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels for image save, got {tensor.shape[0]}.")
    arr = tensor.permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def save_rgb_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(tensor)).save(path)

