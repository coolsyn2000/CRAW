from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


_DYNAMIC_ENTROPY_BUFFER_SUFFIXES = (
    "._quantized_cdf",
    "._offset",
    "._cdf_length",
    ".scale_table",
)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    step: int = 0,
    config: dict | None = None,
    extra_state: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.checkpoint_state_dict() if hasattr(model, "checkpoint_state_dict") else model.state_dict()
    payload: dict[str, Any] = {
        "model": model_state,
        "epoch": epoch,
        "step": step,
        "config": config or {},
    }
    if hasattr(model, "checkpoint_is_partial") and model.checkpoint_is_partial():
        payload["partial_model_state"] = True
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    strict = bool(strict and not checkpoint.get("partial_model_state", False))
    state_dict = checkpoint["model"]
    if not strict:
        current_state = model.state_dict()
        compatible_state = {}
        skipped = []
        for key, value in state_dict.items():
            if key in current_state and current_state[key].shape != value.shape:
                # CompressAI entropy models resize these registered buffers in
                # their load_state_dict override before copying checkpoint data.
                if key.endswith(_DYNAMIC_ENTROPY_BUFFER_SUFFIXES):
                    compatible_state[key] = value
                    continue
                skipped.append((key, tuple(value.shape), tuple(current_state[key].shape)))
                continue
            compatible_state[key] = value
        if skipped:
            preview = ", ".join(
                f"{key}:{old_shape}->{new_shape}" for key, old_shape, new_shape in skipped[:8]
            )
            suffix = "" if len(skipped) <= 8 else f", ... total={len(skipped)}"
            print(f"warning=skipped_incompatible_checkpoint_tensors {preview}{suffix}", flush=True)
        model.load_state_dict(compatible_state, strict=False)
    else:
        model.load_state_dict(state_dict, strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            print(f"warning=skipped_optimizer_state reason={exc}", flush=True)
    return checkpoint
