from __future__ import annotations

from typing import Any

from torch import nn

from .ours import OursM, OursS


_KINDS = {
    "ours_m": "joint_codec",
    "ours_s": "joint_codec",
}


def _model_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    model_cfg = dict(config.get("model", {}))
    name = str(model_cfg.pop("name", "")).lower().replace("-", "_")
    model_cfg.pop("family", None)
    if not name:
        raise KeyError("Config is missing model.name.")
    if name not in _KINDS:
        raise ValueError(f"Unknown paper model {name!r}; choose from {sorted(_KINDS)}.")
    return name, model_cfg


def model_kind(config: dict[str, Any]) -> str:
    name, _ = _model_config(config)
    return _KINDS[name]


def build_model(config: dict[str, Any]) -> nn.Module:
    name, kwargs = _model_config(config)
    if name == "ours_m":
        return OursM(**kwargs)
    if name == "ours_s":
        return OursS(**kwargs)
    raise AssertionError(name)
