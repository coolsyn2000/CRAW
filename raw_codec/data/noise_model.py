from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


DEFAULT_KB_CHANNELS = {
    "R": {"K": 0.00009, "B": 0.00045**2},
    "Gr": {"K": 0.00008, "B": 0.00040**2},
    "Gb": {"K": 0.00008, "B": 0.00041**2},
    "B": {"K": 0.00010, "B": 0.00048**2},
}

PROFILE_NOISE_MODELS = {
    "poisson_gaussian_iso_profile",
    "iso_profile",
    "darktable_profile",
    "poisson_gaussian_darktable",
}

PROFILE_CHANNEL_INDEX = {"R": 0, "GR": 1, "GB": 1, "G": 1, "B": 2}
DISABLED_NOISE_MODELS = {"", "none", "off", "disable", "disabled", "clean", "identity"}


def _channel_variance_multipliers(
    noise_cfg: dict | None,
    channel_names: list[str] | tuple[str, ...],
    *,
    ndim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return scalar and per-Bayer-channel variance calibration factors."""
    noise_cfg = noise_cfg or {}
    scalar = float(noise_cfg.get("strength_multiplier", 1.0))
    configured = noise_cfg.get(
        "channel_strength_multipliers",
        noise_cfg.get("per_channel_strength_multipliers", noise_cfg.get("variance_channel_multipliers")),
    )
    if configured is None:
        values = [1.0] * len(channel_names)
    elif isinstance(configured, dict):
        normalized = {str(key).upper(): float(value) for key, value in configured.items()}
        values = []
        for name in channel_names:
            key = str(name).upper()
            fallback = "G" if key in {"GR", "GB"} else key
            if key in normalized:
                values.append(normalized[key])
            elif fallback in normalized:
                values.append(normalized[fallback])
            else:
                raise KeyError(f"Missing channel variance multiplier for {name!r}.")
    else:
        values = [float(value) for value in configured]
        if len(values) != len(channel_names):
            raise ValueError(
                "channel_strength_multipliers must have one value per channel: "
                f"expected {len(channel_names)}, got {len(values)}."
            )
    if scalar <= 0.0 or any(value <= 0.0 for value in values):
        raise ValueError("Noise variance strength multipliers must be positive.")
    shape = (len(channel_names),) + (1,) * (ndim - 1)
    if ndim == 4:
        shape = (1, len(channel_names), 1, 1)
    return torch.tensor(values, dtype=dtype, device=device).reshape(shape) * scalar


def noise_simulation_enabled(noise_cfg: dict | None = None) -> bool:
    noise_cfg = noise_cfg or {}
    enabled = noise_cfg.get("enabled", noise_cfg.get("simulate_noise", True))
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}
    if not bool(enabled):
        return False
    model = str(noise_cfg.get("model", "poisson_gaussian")).lower()
    return model not in DISABLED_NOISE_MODELS


def _noise_map_options(noise_cfg: dict | None = None) -> dict:
    noise_cfg = noise_cfg or {}
    options = dict(noise_cfg.get("noise_map", {}))
    for key in ("channels", "domain", "scale", "clamp_max"):
        legacy_key = f"noise_map_{key}"
        if legacy_key in noise_cfg and key not in options:
            options[key] = noise_cfg[legacy_key]
    return options


def _noise_map_uses_per_channel(noise_cfg: dict | None = None) -> bool:
    channels = str(_noise_map_options(noise_cfg).get("channels", "mean")).lower()
    return channels in {"per_channel", "channel", "channels", "packed", "raw", "4", "rggb"}


def _format_noise_map_from_variance(variance: torch.Tensor, noise_cfg: dict | None = None) -> torch.Tensor:
    options = _noise_map_options(noise_cfg)
    variance = variance.clamp_min(0.0)
    if not _noise_map_uses_per_channel(noise_cfg):
        channel_dim = 0 if variance.ndim == 3 else 1
        variance = variance.mean(dim=channel_dim, keepdim=True)

    domain = str(options.get("domain", "std")).lower()
    if domain in {"variance", "var", "variance_map"}:
        noise_map = variance
    elif domain in {"std", "sigma", "stdev", "standard_deviation"}:
        noise_map = torch.sqrt(variance.clamp_min(1e-12))
    else:
        raise ValueError(f"Unsupported noise_map.domain: {domain}")

    scale = float(options.get("scale", 1.0))
    if scale != 1.0:
        noise_map = noise_map * scale
    if "clamp_max" in options and options["clamp_max"] is not None:
        noise_map = noise_map.clamp_max(float(options["clamp_max"]))
    return noise_map.clamp_min(0.0)


def black_level_normalized(noise_cfg: dict | None = None) -> float:
    noise_cfg = noise_cfg or {}
    black_level = noise_cfg.get("black_level", noise_cfg.get("black_level_code"))
    if black_level is None:
        return 0.0
    black_level = float(black_level)
    white_level = float(noise_cfg.get("white_level", 255.0 if black_level > 1.0 else 1.0))
    if white_level <= 0:
        raise ValueError("white_level must be positive.")
    black = black_level / white_level if white_level > 1.0 or black_level > 1.0 else black_level
    return min(max(black, 0.0), 1.0 - 1e-6)


def signal_scale_from_config(noise_cfg: dict | None = None) -> float:
    return 1.0 - black_level_normalized(noise_cfg)


def apply_black_level_to_signal(
    signal_raw: torch.Tensor,
    noise_cfg: dict | None = None,
    *,
    clamp_signal: bool = True,
    clamp_raw: bool = True,
) -> torch.Tensor:
    """Map black-subtracted signal RAW in [0, 1] to normalized stored RAW."""
    black = black_level_normalized(noise_cfg)
    signal = signal_raw.clamp(0.0, 1.0) if clamp_signal else signal_raw
    stored = black + signal * (1.0 - black)
    return stored.clamp(0.0, 1.0) if clamp_raw else stored


def subtract_black_level_from_raw(stored_raw: torch.Tensor, noise_cfg: dict | None = None) -> torch.Tensor:
    """Map normalized stored RAW back to black-subtracted signal RAW."""
    black = black_level_normalized(noise_cfg)
    if black <= 0.0:
        return stored_raw.clamp(0.0, 1.0)
    return ((stored_raw - black) / max(1.0 - black, 1e-8)).clamp(0.0, 1.0)


def add_poisson_gaussian_noise(
    clean_raw: torch.Tensor,
    shot_noise: float = 0.01,
    read_noise: float = 0.0001,
    noise_cfg: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a simple Poisson-Gaussian RAW noise approximation.

    The variance model is ``var = shot_noise * clean + read_noise``.
    """
    clean_signal = clean_raw.clamp(0.0, 1.0)
    variance_signal = (shot_noise * clean_signal + read_noise).clamp_min(0.0)
    noise = torch.randn(clean_raw.shape, device=clean_raw.device, dtype=clean_raw.dtype)
    noisy_signal = clean_signal + torch.sqrt(variance_signal.clamp_min(1e-12)) * noise
    noisy = apply_black_level_to_signal(noisy_signal, noise_cfg, clamp_signal=False)
    variance = variance_signal * signal_scale_from_config(noise_cfg) ** 2
    return noisy, variance


def sample_analog_gain(noise_cfg: dict | None = None) -> float:
    noise_cfg = noise_cfg or {}
    if "analog_gain_range" in noise_cfg and noise_cfg["analog_gain_range"] is not None:
        low, high = noise_cfg["analog_gain_range"]
        low = max(float(low), 1e-8)
        high = max(float(high), low)
        if noise_cfg.get("analog_gain_sampling", "log_uniform") == "uniform":
            return float(torch.empty(1).uniform_(low, high).item())
        log_gain = torch.empty(1).uniform_(math.log(low), math.log(high)).item()
        return float(math.exp(log_gain))
    return float(noise_cfg.get("analog_gain", 1.0))


def _profile_path_from_config(noise_cfg: dict | None) -> str:
    noise_cfg = noise_cfg or {}
    path = (
        noise_cfg.get("profile_path")
        or noise_cfg.get("noise_profile_path")
        or noise_cfg.get("data_noise_path")
    )
    if not path:
        raise ValueError("ISO-profile noise model requires noise.profile_path.")
    return str(Path(path).expanduser().resolve())


def _profile_model_from_config(noise_cfg: dict | None) -> str | None:
    noise_cfg = noise_cfg or {}
    model = (
        noise_cfg.get("profile_model")
        or noise_cfg.get("camera_model")
        or noise_cfg.get("noise_profile_model")
    )
    return str(model) if model else None


def _sample_profile_model_from_config(noise_cfg: dict | None) -> str | None:
    """Select one camera profile while keeping camera and ISO sampling coupled."""
    noise_cfg = noise_cfg or {}
    configured = noise_cfg.get("profile_models", noise_cfg.get("camera_models"))
    if not configured:
        return _profile_model_from_config(noise_cfg)
    models = [str(value) for value in configured]
    if not models:
        raise ValueError("noise.profile_models must contain at least one camera model.")
    weights = noise_cfg.get("profile_model_weights", noise_cfg.get("camera_model_weights"))
    if weights is None:
        index = int(torch.randint(0, len(models), (1,)).item())
    else:
        weights_t = torch.tensor([float(value) for value in weights], dtype=torch.float32)
        if weights_t.numel() != len(models):
            raise ValueError("noise.profile_model_weights must match noise.profile_models.")
        if bool(torch.any(weights_t < 0)) or float(weights_t.sum().item()) <= 0.0:
            raise ValueError("noise.profile_model_weights must be non-negative and sum to a positive value.")
        index = int(torch.multinomial(weights_t, 1).item())
    return models[index]


def _select_profile_config(noise_cfg: dict | None) -> dict:
    selected = dict(noise_cfg or {})
    model = _sample_profile_model_from_config(selected)
    if model is not None:
        selected["profile_model"] = model
    selected.pop("profile_models", None)
    selected.pop("camera_models", None)
    selected.pop("profile_model_weights", None)
    selected.pop("camera_model_weights", None)
    return selected


def _model_key(value: Any) -> str:
    return str(value).replace("-", "").replace("_", "").replace(" ", "").casefold()


def _profile_entry_matches_model(model_key: str, entry: Any, requested_model: str) -> bool:
    requested = _model_key(requested_model)
    candidates: list[Any] = [model_key]
    if isinstance(entry, dict):
        candidates.extend([entry.get("model"), entry.get("name")])
        candidates.extend(entry.get("aliases", []))
    return any(_model_key(candidate) == requested for candidate in candidates if candidate)


def _available_profile_models(document: dict) -> list[str]:
    available: list[str] = []
    for key in ("camera_profiles", "profiles_by_model", "models", "cameras"):
        grouped = document.get(key)
        if isinstance(grouped, dict):
            available.extend(str(model) for model in grouped)
        elif isinstance(grouped, list):
            for entry in grouped:
                if isinstance(entry, dict) and entry.get("model"):
                    available.append(str(entry["model"]))
    if document.get("model"):
        available.append(str(document["model"]))
    return sorted(set(available))


def _profiles_from_grouped_document(
    document: dict,
    requested_model: str | None,
) -> tuple[list[dict], str]:
    default_model = str(document.get("default_model") or document.get("model") or "")
    selected_model = requested_model or default_model

    if not requested_model and document.get("profiles"):
        return list(document["profiles"]), str(document.get("model", default_model or "unknown"))
    if requested_model and document.get("profiles") and _model_key(document.get("model")) == _model_key(requested_model):
        return list(document["profiles"]), str(document.get("model", requested_model))

    for key in ("camera_profiles", "profiles_by_model", "models", "cameras"):
        grouped = document.get(key)
        if isinstance(grouped, dict):
            for model, entry in grouped.items():
                if selected_model and not _profile_entry_matches_model(str(model), entry, selected_model):
                    continue
                profiles = entry.get("profiles", []) if isinstance(entry, dict) else entry
                return list(profiles), str(model)
        elif isinstance(grouped, list):
            for entry in grouped:
                if not isinstance(entry, dict):
                    continue
                model = str(entry.get("model") or entry.get("name") or "")
                if selected_model and not _profile_entry_matches_model(model, entry, selected_model):
                    continue
                return list(entry.get("profiles", [])), model or selected_model or "unknown"

    available = ", ".join(_available_profile_models(document)) or "none"
    if requested_model:
        raise ValueError(f"Noise profile model {requested_model!r} not found. Available models: {available}.")
    raise ValueError(f"No noise profiles found. Available models: {available}.")


@lru_cache(maxsize=32)
def _load_noise_profile(
    profile_path: str,
    requested_model: str | None = None,
) -> tuple[tuple[float, ...], tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...], str]:
    with Path(profile_path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    profiles, camera_model = _profiles_from_grouped_document(document, requested_model)
    profiles = sorted(profiles, key=lambda item: float(item["iso"]))
    if not profiles:
        raise ValueError(f"No noise profiles found in {profile_path}.")

    iso_values: list[float] = []
    a_values: list[tuple[float, float, float]] = []
    b_values: list[tuple[float, float, float]] = []
    for profile in profiles:
        iso_values.append(float(profile["iso"]))
        a = tuple(float(value) for value in profile["a"])
        b = tuple(float(value) for value in profile["b"])
        if len(a) != 3 or len(b) != 3:
            raise ValueError(f"Expected 3-channel a/b values in {profile_path}.")
        a_values.append(a)
        b_values.append(b)

    return tuple(iso_values), tuple(a_values), tuple(b_values), camera_model or "unknown"


def _profile_point_is_positive(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(value > 0.0 for value in a) and all(value > 0.0 for value in b)


def _positive_profile_indices(
    iso_values: tuple[float, ...],
    a_values: tuple[tuple[float, float, float], ...],
    b_values: tuple[tuple[float, float, float], ...],
    low: float,
    high: float,
) -> list[int]:
    return [
        index
        for index, iso in enumerate(iso_values)
        if low <= iso <= high and _profile_point_is_positive(a_values[index], b_values[index])
    ]


def _profile_iso_range(noise_cfg: dict | None, iso_values: tuple[float, ...]) -> tuple[float, float]:
    noise_cfg = noise_cfg or {}
    iso_range = noise_cfg.get("profile_iso_range", noise_cfg.get("iso_range"))
    if iso_range is None:
        return float(min(iso_values)), float(max(iso_values))
    if len(iso_range) != 2:
        raise ValueError("noise.iso_range must contain [min_iso, max_iso].")
    low = float(iso_range[0])
    high = float(iso_range[1])
    if high < low:
        low, high = high, low
    return low, high


def _sample_log_uniform(low: float, high: float) -> float:
    log_iso = torch.empty(1).uniform_(math.log(low), math.log(high)).item()
    return float(math.exp(log_iso))


def _sample_iso_from_weighted_ranges(noise_cfg: dict, low: float, high: float) -> float:
    ranges = noise_cfg.get("iso_sampling_ranges", noise_cfg.get("iso_ranges"))
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("noise.iso_sampling_ranges must be a non-empty list for weighted ISO sampling.")

    candidates: list[tuple[float, float]] = []
    weights: list[float] = []
    for item in ranges:
        if isinstance(item, dict):
            range_low = item.get("min", item.get("low", item.get("start")))
            range_high = item.get("max", item.get("high", item.get("end")))
            weight = item.get("weight", 1.0)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            range_low = item[0]
            range_high = item[1]
            weight = item[2] if len(item) >= 3 else 1.0
        else:
            raise ValueError(
                "Each noise.iso_sampling_ranges item must be [low, high, weight] "
                "or a mapping with low/high/weight."
            )

        range_low = float(range_low)
        range_high = float(range_high)
        if range_high < range_low:
            range_low, range_high = range_high, range_low
        if range_high < low or range_low > high:
            continue
        range_low = max(range_low, float(low))
        range_high = min(range_high, float(high))
        weight = float(weight)
        if range_low <= 0.0 or range_high <= 0.0 or weight <= 0.0:
            continue
        candidates.append((range_low, range_high))
        weights.append(weight)

    if not candidates:
        raise ValueError(f"No valid weighted ISO ranges overlap [{low}, {high}].")
    weights_t = torch.tensor(weights, dtype=torch.float32)
    index = int(torch.multinomial(weights_t, 1).item())
    range_low, range_high = candidates[index]
    return _sample_log_uniform(max(range_low, 1e-8), max(range_high, range_low))


def _sample_iso_from_profile(noise_cfg: dict | None = None) -> float:
    noise_cfg = noise_cfg or {}
    if "iso" in noise_cfg and noise_cfg["iso"] is not None:
        return float(noise_cfg["iso"])

    profile_path = _profile_path_from_config(noise_cfg)
    profile_model = _profile_model_from_config(noise_cfg)
    iso_values, a_values, b_values, _ = _load_noise_profile(profile_path, profile_model)
    low, high = _profile_iso_range(noise_cfg, iso_values)
    low = max(low, min(iso_values), 1e-8)
    high = min(max(high, low), max(iso_values))
    positive_indices = _positive_profile_indices(iso_values, a_values, b_values, low, high)
    if not positive_indices:
        raise ValueError(f"No positive profile ISO values inside [{low}, {high}].")
    positive_iso_values = [iso_values[index] for index in positive_indices]
    low = max(low, min(positive_iso_values))
    high = min(high, max(positive_iso_values))
    sampling = str(noise_cfg.get("iso_sampling", "log_uniform")).lower()

    if sampling in {"discrete", "choice", "profile"}:
        candidates = [iso for iso in positive_iso_values if low <= iso <= high]
        if not candidates:
            raise ValueError(f"No profile ISO values inside [{low}, {high}].")
        index = int(torch.randint(0, len(candidates), (1,)).item())
        return float(candidates[index])
    if sampling == "uniform":
        return float(torch.empty(1).uniform_(low, high).item())
    if sampling in {"log_weighted_ranges", "weighted_log_uniform", "log_ranges", "log_buckets"}:
        return _sample_iso_from_weighted_ranges(noise_cfg, low, high)
    if sampling not in {"log_uniform", "log"}:
        raise ValueError(f"Unsupported noise.iso_sampling: {sampling}")
    return _sample_log_uniform(low, high)


def _noise_profile_tensors(
    noise_cfg: dict | None,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    profile_path = _profile_path_from_config(noise_cfg)
    profile_model = _profile_model_from_config(noise_cfg)
    iso_values, a_values, b_values, camera_model = _load_noise_profile(profile_path, profile_model)
    low, high = _profile_iso_range(noise_cfg, iso_values)

    selected = _positive_profile_indices(iso_values, a_values, b_values, low, high)
    if len(selected) < 2:
        raise ValueError(f"Need at least two positive ISO profile points inside [{low}, {high}].")

    iso = torch.tensor([iso_values[index] for index in selected], dtype=dtype, device=device)
    a = torch.tensor([a_values[index] for index in selected], dtype=dtype, device=device)
    b = torch.tensor([b_values[index] for index in selected], dtype=dtype, device=device)
    return iso, a, b, camera_model


def _interpolate_profile_kb(
    noise_cfg: dict | None,
    iso: float | torch.Tensor,
    channel_names: list[str] | tuple[str, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    table_iso, table_a, table_b, camera_model = _noise_profile_tensors(noise_cfg, dtype=dtype, device=device)
    iso_t = torch.as_tensor(iso, dtype=dtype, device=device)
    scalar = iso_t.ndim == 0
    iso_t = iso_t.clamp(float(table_iso[0].item()), float(table_iso[-1].item()))

    log_table_iso = torch.log(table_iso)
    log_iso = torch.log(iso_t)
    right = torch.searchsorted(log_table_iso, log_iso).clamp(1, table_iso.numel() - 1)
    left = right - 1
    denom = (log_table_iso[right] - log_table_iso[left]).clamp_min(torch.finfo(dtype).eps)
    weight = (log_iso - log_table_iso[left]) / denom

    log_a_left = torch.log(table_a[left])
    log_a_right = torch.log(table_a[right])
    log_b_left = torch.log(table_b[left])
    log_b_right = torch.log(table_b[right])
    if weight.ndim > 0:
        weight = weight.unsqueeze(-1)
    a_rgb = torch.exp(log_a_left + weight * (log_a_right - log_a_left))
    b_rgb = torch.exp(log_b_left + weight * (log_b_right - log_b_left))

    channel_indices = [PROFILE_CHANNEL_INDEX[name.upper()] for name in channel_names]
    if scalar:
        k = a_rgb[channel_indices].view(-1, 1, 1)
        b = b_rgb[channel_indices].view(-1, 1, 1)
        iso_out = iso_t
    else:
        k = a_rgb[:, channel_indices].view(iso_t.numel(), len(channel_indices), 1, 1)
        b = b_rgb[:, channel_indices].view(iso_t.numel(), len(channel_indices), 1, 1)
        iso_out = iso_t.reshape(-1)
    return k, b, iso_out, camera_model


def _channel_params_from_config(noise_cfg: dict | None) -> dict[str, dict[str, float]]:
    noise_cfg = noise_cfg or {}
    channels = noise_cfg.get("channels") or noise_cfg.get("calibration", {}).get("channels")
    if not channels:
        return DEFAULT_KB_CHANNELS
    return channels


def _lookup_channel(channels: dict[str, dict[str, float]], name: str) -> dict[str, float]:
    if name in channels:
        return channels[name]
    if name in {"Gr", "Gb"} and "G" in channels:
        return channels["G"]
    raise KeyError(f"Missing K/B noise parameters for channel '{name}'.")


def add_kb_poisson_gaussian_noise(
    clean_raw: torch.Tensor,
    noise_cfg: dict | None = None,
    channel_names: list[str] | tuple[str, ...] = ("R", "Gr", "Gb", "B"),
    analog_gain: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
    """Apply calibrated Poisson-Gaussian noise with K/B and analog gain.

    ``variance = K_eff * signal + B_eff`` where K scales linearly with analog
    gain by default. ``B`` is read-noise variance; to mirror Open-AISP's
    sigma-read scaling, it uses ``ratio ** (2 * read_noise_gain_exponent)``.
    """
    noise_cfg = noise_cfg or {}
    analog_gain = sample_analog_gain(noise_cfg) if analog_gain is None else float(analog_gain)
    base_gain = float(noise_cfg.get("base_analog_gain", noise_cfg.get("calibration", {}).get("base_analog_gain", 1.0)))
    ratio = analog_gain / max(base_gain, 1e-8)
    shot_exp = float(noise_cfg.get("shot_noise_gain_exponent", 1.0))
    read_exp = float(noise_cfg.get("read_noise_gain_exponent", 1.0))
    channels = _channel_params_from_config(noise_cfg)

    clean_signal = clean_raw.clamp(0.0, 1.0)
    k_values = []
    b_values = []
    for name in channel_names:
        params = _lookup_channel(channels, name)
        k_values.append(float(params["K"]) * ratio**shot_exp)
        if "B" in params:
            base_b = float(params["B"])
        else:
            base_b = float(params["sigma_read"]) ** 2
        b_values.append(base_b * ratio ** (2.0 * read_exp))

    k = torch.tensor(k_values, dtype=clean_raw.dtype, device=clean_raw.device).view(-1, 1, 1)
    b = torch.tensor(b_values, dtype=clean_raw.dtype, device=clean_raw.device).view(-1, 1, 1)
    variance_strength = _channel_variance_multipliers(
        noise_cfg,
        channel_names,
        ndim=clean_raw.ndim,
        dtype=clean_raw.dtype,
        device=clean_raw.device,
    )
    variance_signal = (k * clean_signal + b).clamp_min(1e-12) * variance_strength
    noise = torch.randn(clean_raw.shape, device=clean_raw.device, dtype=clean_raw.dtype)
    noisy_signal = clean_signal + torch.sqrt(variance_signal) * noise
    noisy = apply_black_level_to_signal(noisy_signal, noise_cfg, clamp_signal=False)
    variance = variance_signal * signal_scale_from_config(noise_cfg) ** 2
    metadata = {
        "analog_gain": analog_gain,
        "iso": analog_gain * 100.0,
        "K": k.detach().flatten().cpu(),
        "B": b.detach().flatten().cpu(),
        "variance_strength": variance_strength.detach().flatten().cpu(),
        "black_level": black_level_normalized(noise_cfg),
    }
    return noisy, variance, metadata


def add_profile_poisson_gaussian_noise(
    clean_raw: torch.Tensor,
    noise_cfg: dict | None = None,
    channel_names: list[str] | tuple[str, ...] = ("R", "Gr", "Gb", "B"),
    analog_gain: float | None = None,
    iso: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float | str]]:
    """Apply darktable-style ISO-profile Poisson-Gaussian noise.

    The profile stores per-ISO ``a`` and ``b`` values. We map them to
    ``variance = K(ISO) * signal + B(ISO)`` and interpolate in log-ISO/log-KB
    space, which preserves positivity and tracks calibrated camera ISO stops.
    """
    noise_cfg = _select_profile_config(noise_cfg)
    if iso is None:
        iso = float(analog_gain) * 100.0 if analog_gain is not None else _sample_iso_from_profile(noise_cfg)
    iso = float(iso)
    analog_gain = iso / 100.0
    clean_signal = clean_raw.clamp(0.0, 1.0)
    k, b, iso_t, camera_model = _interpolate_profile_kb(
        noise_cfg,
        iso,
        channel_names,
        dtype=clean_raw.dtype,
        device=clean_raw.device,
    )
    variance_strength = _channel_variance_multipliers(
        noise_cfg,
        channel_names,
        ndim=clean_raw.ndim,
        dtype=clean_raw.dtype,
        device=clean_raw.device,
    )
    k_eff = (k * variance_strength).clamp_min(1e-12)
    b_eff = (b * variance_strength).clamp_min(0.0)
    variance_signal = (k_eff * clean_signal + b_eff).clamp_min(1e-12)
    distribution = str(
        noise_cfg.get(
            "shot_noise_distribution",
            noise_cfg.get("profile_distribution", noise_cfg.get("distribution", "gaussian")),
        )
    ).lower()
    if distribution in {"poisson", "poisson_gaussian", "poisson-gaussian", "exact_poisson"}:
        shot = torch.poisson((clean_signal / k_eff).clamp_min(0.0)) * k_eff
        read = torch.randn_like(clean_raw) * torch.sqrt(b_eff)
        noisy_signal = shot + read
        distribution = "poisson_gaussian"
    elif distribution in {"gaussian", "heteroscedastic_gaussian", "normal"}:
        noise = torch.randn_like(clean_raw)
        noisy_signal = clean_signal + torch.sqrt(variance_signal) * noise
        distribution = "heteroscedastic_gaussian"
    else:
        raise ValueError(f"Unsupported profile noise distribution: {distribution}")
    noisy = apply_black_level_to_signal(noisy_signal, noise_cfg, clamp_signal=False)
    variance = variance_signal * signal_scale_from_config(noise_cfg) ** 2
    metadata = {
        "analog_gain": analog_gain,
        "iso": float(iso_t.item()),
        "K": k.detach().flatten().cpu(),
        "B": b.detach().flatten().cpu(),
        "variance_strength": variance_strength.detach().flatten().cpu(),
        "black_level": black_level_normalized(noise_cfg),
        "noise_profile_model": camera_model,
        "noise_distribution": distribution,
    }
    return noisy, variance, metadata


def estimate_noise_std_map_from_config(
    noisy_raw: torch.Tensor,
    noise_cfg: dict | None = None,
    channel_names: list[str] | tuple[str, ...] = ("R", "Gr", "Gb", "B"),
    analog_gain: float | torch.Tensor | None = None,
    iso: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate an encoder noise prior from RAW signal and noise calibration.

    AISP-style noise calibration fits ``variance = K * signal + B``. This
    function builds the same signal-dependent map for the encoder. By default it
    returns the legacy 1-channel std map; set ``noise.noise_map.channels`` to
    ``per_channel`` and ``noise.noise_map.domain`` to ``variance`` for 4-channel
    RGGB variance maps.
    """
    noise_cfg = noise_cfg or {}
    if not noise_simulation_enabled(noise_cfg):
        if noisy_raw.ndim == 3:
            variance = torch.zeros_like(noisy_raw)
        elif noisy_raw.ndim == 4:
            variance = torch.zeros_like(noisy_raw)
        else:
            raise ValueError(f"Expected noisy_raw with shape [C,H,W] or [B,C,H,W], got {noisy_raw.shape}.")
        if not _noise_map_uses_per_channel(noise_cfg):
            channel_dim = 0 if variance.ndim == 3 else 1
            variance = variance.mean(dim=channel_dim, keepdim=True)
        return variance

    model = noise_cfg.get("model", "poisson_gaussian").lower()
    signal_raw = subtract_black_level_from_raw(noisy_raw, noise_cfg)
    if noisy_raw.ndim == 3:
        gain_shape = (1, 1, 1)
    elif noisy_raw.ndim == 4:
        gain_shape = (-1, 1, 1, 1)
    else:
        raise ValueError(f"Expected noisy_raw with shape [C,H,W] or [B,C,H,W], got {noisy_raw.shape}.")

    if iso is None and analog_gain is not None and model in PROFILE_NOISE_MODELS:
        iso = torch.as_tensor(analog_gain, dtype=noisy_raw.dtype, device=noisy_raw.device) * 100.0

    if analog_gain is None:
        analog_gain = float(noise_cfg.get("analog_gain", noise_cfg.get("base_analog_gain", 1.0)))
    gain = torch.as_tensor(analog_gain, dtype=noisy_raw.dtype, device=noisy_raw.device)
    if gain.ndim == 0:
        gain = gain.reshape(1)
    if noisy_raw.ndim == 3 and gain.numel() != 1:
        raise ValueError("Per-sample analog_gain requires batched noisy_raw.")
    gain = gain.reshape(gain_shape if gain.numel() > 1 else (1,) * noisy_raw.ndim)

    if model in PROFILE_NOISE_MODELS:
        if iso is None:
            iso = float(noise_cfg.get("iso", 100.0 * float(noise_cfg.get("analog_gain", 1.0))))
        k, b, _, _ = _interpolate_profile_kb(
            noise_cfg,
            iso,
            channel_names,
            dtype=noisy_raw.dtype,
            device=noisy_raw.device,
        )
        variance_strength = _channel_variance_multipliers(
            noise_cfg,
            channel_names,
            ndim=noisy_raw.ndim,
            dtype=noisy_raw.dtype,
            device=noisy_raw.device,
        )
        variance_signal = (k * signal_raw + b) * variance_strength
        variance = variance_signal * signal_scale_from_config(noise_cfg) ** 2
    elif model in {"poisson_gaussian_kb", "kb", "open_aisp"}:
        base_gain = float(
            noise_cfg.get("base_analog_gain", noise_cfg.get("calibration", {}).get("base_analog_gain", 1.0))
        )
        ratio = gain / max(base_gain, 1e-8)
        shot_exp = float(noise_cfg.get("shot_noise_gain_exponent", 1.0))
        read_exp = float(noise_cfg.get("read_noise_gain_exponent", 1.0))
        variance_strength = _channel_variance_multipliers(
            noise_cfg,
            channel_names,
            ndim=noisy_raw.ndim,
            dtype=noisy_raw.dtype,
            device=noisy_raw.device,
        )
        channels = _channel_params_from_config(noise_cfg)

        k_values = []
        b_values = []
        for name in channel_names:
            params = _lookup_channel(channels, name)
            k_values.append(float(params["K"]))
            if "B" in params:
                b_values.append(float(params["B"]))
            else:
                b_values.append(float(params["sigma_read"]) ** 2)

        channel_shape = (-1, 1, 1) if noisy_raw.ndim == 3 else (1, -1, 1, 1)
        k = torch.tensor(k_values, dtype=noisy_raw.dtype, device=noisy_raw.device).view(channel_shape)
        b = torch.tensor(b_values, dtype=noisy_raw.dtype, device=noisy_raw.device).view(channel_shape)
        variance_signal = (
            k * ratio**shot_exp * signal_raw + b * ratio ** (2.0 * read_exp)
        ) * variance_strength
        variance = variance_signal * signal_scale_from_config(noise_cfg) ** 2
    else:
        shot_noise = float(noise_cfg.get("shot_noise", 0.01))
        read_noise = float(noise_cfg.get("read_noise", 0.0001))
        variance_signal = shot_noise * signal_raw + read_noise
        variance = variance_signal * signal_scale_from_config(noise_cfg) ** 2

    return _format_noise_map_from_variance(variance, noise_cfg)


def add_noise_from_config(
    clean_raw: torch.Tensor,
    noise_cfg: dict | None = None,
    channel_names: list[str] | tuple[str, ...] = ("R", "Gr", "Gb", "B"),
    analog_gain: float | None = None,
    iso: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float | str]]:
    noise_cfg = noise_cfg or {}
    if not noise_simulation_enabled(noise_cfg):
        noisy = apply_black_level_to_signal(clean_raw, noise_cfg)
        variance = torch.zeros_like(noisy)
        return noisy, variance, {
            "analog_gain": float(noise_cfg.get("analog_gain", 1.0)),
            "iso": float(noise_cfg.get("iso", 0.0)),
            "black_level": black_level_normalized(noise_cfg),
            "noise_model": "disabled",
        }

    model = noise_cfg.get("model", "poisson_gaussian").lower()
    if model in PROFILE_NOISE_MODELS:
        return add_profile_poisson_gaussian_noise(
            clean_raw,
            noise_cfg=noise_cfg,
            channel_names=channel_names,
            analog_gain=analog_gain,
            iso=iso,
        )
    if model in {"poisson_gaussian_kb", "kb", "open_aisp"}:
        return add_kb_poisson_gaussian_noise(
            clean_raw,
            noise_cfg=noise_cfg,
            channel_names=channel_names,
            analog_gain=analog_gain,
        )
    noisy, variance = add_poisson_gaussian_noise(
        clean_raw,
        shot_noise=float(noise_cfg.get("shot_noise", 0.01)),
        read_noise=float(noise_cfg.get("read_noise", 0.0001)),
        noise_cfg=noise_cfg,
    )
    return noisy, variance, {
        "analog_gain": float(noise_cfg.get("analog_gain", 1.0)),
        "black_level": black_level_normalized(noise_cfg),
    }
