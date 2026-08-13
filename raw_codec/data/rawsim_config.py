from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_rawsim_document(document: dict[str, Any]) -> dict[str, Any]:
    if "rawsim" in document and isinstance(document["rawsim"], dict):
        return deepcopy(document["rawsim"])
    return deepcopy(document)


def attach_rawsim_config(config: dict[str, Any], rawsim_config_path: str | Path | None = None) -> dict[str, Any]:
    merged = deepcopy(config)
    path = rawsim_config_path or merged.get("rawsim_config")
    if path:
        rawsim = normalize_rawsim_document(load_yaml_config(path))
        rawsim["_config_dir"] = str(Path(path).expanduser().resolve().parent)
        merged["rawsim"] = rawsim
    return merged


def apply_evaluation_data_config(
    config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    """Overlay evaluation data semantics without replacing the trained model config."""
    merged = deepcopy(config)
    for key in ("rawsim", "target", "unprocess", "noise"):
        if key in evaluation_config:
            merged[key] = deepcopy(evaluation_config[key])
    evaluation_val = evaluation_config.get("data", {}).get("val")
    if evaluation_val is not None:
        merged.setdefault("data", {})["val"] = deepcopy(evaluation_val)
    return merged


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def resolve_rawsim_config(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config.get("rawsim", {}))


def resolve_cfa_pattern(config: dict[str, Any]) -> str:
    rawsim = resolve_rawsim_config(config)
    return str(
        rawsim.get(
            "bayer_pattern",
            rawsim.get("cfa_pattern", config.get("data", {}).get("cfa_pattern", "RGGB")),
        )
    ).upper()


def resolve_noise_config(config: dict[str, Any]) -> dict[str, Any]:
    rawsim = resolve_rawsim_config(config)
    noise_cfg: dict[str, Any] = {}
    _deep_update(noise_cfg, rawsim.get("noise", {}))
    if "channels" in rawsim and "channels" not in noise_cfg:
        noise_cfg["channels"] = deepcopy(rawsim["channels"])
    _deep_update(noise_cfg, config.get("noise", {}))
    config_dir = rawsim.get("_config_dir")
    if config_dir:
        for key in ("profile_path", "noise_profile_path", "data_noise_path"):
            value = noise_cfg.get(key)
            if value and not Path(value).expanduser().is_absolute():
                candidate = Path(config_dir) / value
                if candidate.exists():
                    noise_cfg[key] = str(candidate)
    return noise_cfg


def resolve_unprocess_config(config: dict[str, Any]) -> dict[str, Any]:
    rawsim = resolve_rawsim_config(config)
    unprocess_cfg: dict[str, Any] = {}
    _deep_update(unprocess_cfg, rawsim.get("unprocess", {}))
    _deep_update(unprocess_cfg, config.get("unprocess", {}))
    return unprocess_cfg


def rawsim_dataset_defaults(config: dict[str, Any], split: str) -> dict[str, Any]:
    rawsim = resolve_rawsim_config(config)
    dataset_cfg: dict[str, Any] = {}
    _deep_update(dataset_cfg, rawsim.get("dataset", {}))
    _deep_update(dataset_cfg, rawsim.get(split, {}))
    return dataset_cfg
