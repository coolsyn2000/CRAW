from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .noise_model import add_noise_from_config, apply_black_level_to_signal, estimate_noise_std_map_from_config
from .raw_pack import approximate_packed_raw_from_rgb, packed_raw_to_rgb
from .sidd_raw_dataset import demosaic_canonical_rggb
from .unprocess import (
    bayer_channel_names,
    remosaic_linear_rgb_to_packed_raw,
    srgb_to_linear_camera_rgb,
)


class SyntheticRawDataset(Dataset):
    """Synthetic packed RAW dataset for fast prototype training and validation."""

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        mode: str = "random",
        image_folder: str | list[str] | tuple[str, ...] | None = None,
        patch_size: int = 256,
        length: int = 512,
        shot_noise: float = 0.01,
        read_noise: float = 0.0001,
        noise_cfg: dict | None = None,
        unprocess_cfg: dict | None = None,
        cfa_pattern: str = "RGGB",
        image_patch_size: int | None = None,
        augment: bool = False,
        deterministic: bool = False,
        base_seed: int = 0,
        fixed_analog_gains: list[float] | tuple[float, ...] | None = None,
        fixed_isos: list[float] | tuple[float, ...] | None = None,
        same_scene_gain_sweep: bool = False,
        simulate_noise: bool = True,
        linear_target_from_clean_raw: bool = False,
        defer_linear_rgb: bool = False,
        image_split: str = "all",
        validation_fraction: float = 0.01,
        image_split_seed: int = 0,
    ) -> None:
        self.mode = mode.lower()
        if image_folder is None:
            self.image_folders: list[Path] = []
        elif isinstance(image_folder, (list, tuple)):
            self.image_folders = [Path(folder).expanduser() for folder in image_folder]
        else:
            self.image_folders = [Path(image_folder).expanduser()]
        self.patch_size = int(patch_size)
        self.length = int(length)
        self.noise_cfg = dict(noise_cfg or {"shot_noise": shot_noise, "read_noise": read_noise})
        self.unprocess_cfg = dict(unprocess_cfg or {})
        sensor_profile_path = self.unprocess_cfg.get("sensor_profile_path")
        self.sensor_profiles: dict = {}
        if sensor_profile_path:
            profile_path = Path(str(sensor_profile_path)).expanduser()
            with profile_path.open("r", encoding="utf-8") as handle:
                profile_document = json.load(handle)
            self.sensor_profiles = dict(profile_document.get("cameras", profile_document))
        self.cfa_pattern = cfa_pattern.upper()
        self.channel_names = bayer_channel_names(self.cfa_pattern)
        self.image_patch_size = int(image_patch_size) if image_patch_size else self.patch_size * 2
        self.augment = bool(augment)
        self.deterministic = bool(deterministic)
        self.base_seed = int(base_seed)
        self.fixed_analog_gains = [float(gain) for gain in fixed_analog_gains or []]
        self.fixed_isos = [float(iso) for iso in fixed_isos or []]
        self.same_scene_gain_sweep = bool(same_scene_gain_sweep)
        self.simulate_noise = bool(simulate_noise)
        self.linear_target_from_clean_raw = bool(linear_target_from_clean_raw)
        if self.mode == "image" and self.linear_target_from_clean_raw:
            raise ValueError(
                "Image-mode clean linear RGB must remain the camera-linear output of "
                "sRGB unprocessing. Remosaicing and demosaicing it creates a degraded "
                "pseudo-target with false-color edges."
            )
        self.defer_linear_rgb = bool(defer_linear_rgb)
        self.image_split = str(image_split).lower()
        self.validation_fraction = float(validation_fraction)
        self.image_split_seed = int(image_split_seed)
        self.use_unprocess = bool(self.unprocess_cfg.get("enabled", self.mode == "image"))
        self.exposure_cfg = dict(
            self.unprocess_cfg.get(
                "exposure",
                self.unprocess_cfg.get("exposure_augmentation", {}),
            )
            or {}
        )

        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.image_patch_size % 2 != 0:
            self.image_patch_size -= 1

        self.image_paths: list[Path] = []
        if self.mode == "image":
            if not self.image_folders:
                raise ValueError("image_folder or image_folders is required when data mode is 'image'.")
            missing_folders = [str(folder) for folder in self.image_folders if not folder.exists()]
            if missing_folders:
                raise FileNotFoundError(f"Image folder does not exist: {missing_folders[0]}")
            image_paths: list[Path] = []
            for folder in self.image_folders:
                image_paths.extend(
                    p for p in folder.rglob("*") if p.suffix.lower() in self.IMAGE_SUFFIXES
                )
            self.image_paths = self._select_image_split(sorted(image_paths))
            if not self.image_paths:
                raise FileNotFoundError(
                    f"No images remain in image_split={self.image_split!r} under: "
                    + ", ".join(str(folder) for folder in self.image_folders)
                )
            if length <= 0:
                self.length = len(self.image_paths)
        elif self.mode != "random":
            raise ValueError(f"Unknown dataset mode: {mode}")

    def _select_image_split(self, image_paths: list[Path]) -> list[Path]:
        split = self.image_split.replace("-", "_")
        if split in {"", "all", "none", "full"}:
            return image_paths
        if split == "validation":
            split = "val"
        if split not in {"train", "val"}:
            raise ValueError("image_split must be 'all', 'train', or 'val'.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between zero and one.")

        selected: list[Path] = []
        threshold = int(self.validation_fraction * (1 << 64))
        for path in image_paths:
            key = f"{self.image_split_seed}:{path.as_posix()}".encode("utf-8")
            bucket = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
            is_validation = bucket < threshold
            if (split == "val" and is_validation) or (split == "train" and not is_validation):
                selected.append(path)
        return selected

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.deterministic:
            if self.same_scene_gain_sweep and self._fixed_noise_setting_count():
                clean_index = self._clean_index(index)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(self.base_seed + int(clean_index))
                    clean_item = self._make_clean_item(clean_index)
                with torch.random.fork_rng(devices=[]):
                    # Keep exposure and the standard-normal noise realization fixed
                    # across gains so only the configured noise scale changes.
                    torch.manual_seed(self.base_seed + 1_000_003 + int(clean_index))
                    return self._make_sample_from_clean(index, clean_item, clean_index)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.base_seed + int(index))
                return self._make_sample(index)
        return self._make_sample(index)

    def _linear_rgb_include_black_level(self) -> bool:
        return bool(
            self.unprocess_cfg.get(
                "linear_rgb_include_black_level",
                self.unprocess_cfg.get("target_include_black_level", False),
            )
        )

    def _prepare_linear_rgb_target(self, linear_rgb: torch.Tensor) -> torch.Tensor:
        linear_rgb = linear_rgb.clamp(0.0, 1.0)
        if self._linear_rgb_include_black_level():
            return apply_black_level_to_signal(linear_rgb, self.noise_cfg)
        return linear_rgb

    def _sample_synthetic_exposure_scale(self) -> float:
        if not bool(self.exposure_cfg.get("enabled", False)):
            return 1.0
        identity_probability = float(
            self.exposure_cfg.get(
                "identity_probability",
                self.exposure_cfg.get("keep_probability", 0.0),
            )
        )
        if identity_probability > 0.0 and torch.rand(()) < identity_probability:
            return 1.0

        mode = str(self.exposure_cfg.get("sampling", self.exposure_cfg.get("mode", "log_uniform"))).lower()
        if mode in {"log_weighted_ranges", "weighted_log_ranges", "log_mixture"}:
            ranges = self.exposure_cfg.get("ranges", self.exposure_cfg.get("weighted_ranges"))
            if not ranges:
                raise ValueError("unprocess.exposure.ranges must be non-empty for weighted exposure sampling.")
            weights = torch.tensor([max(float(item[2]), 0.0) for item in ranges], dtype=torch.float32)
            if float(weights.sum().item()) <= 0.0:
                raise ValueError("unprocess.exposure.ranges weights must sum to a positive value.")
            selected = int(torch.multinomial(weights / weights.sum(), 1).item())
            low, high = float(ranges[selected][0]), float(ranges[selected][1])
            low, high = max(min(low, high), 1e-6), max(max(low, high), 1e-6)
            return float(torch.empty(1).uniform_(math.log(low), math.log(high)).exp().item())

        scale_range = self.exposure_cfg.get("scale_range", self.exposure_cfg.get("range", [1.0, 1.0]))
        low = float(scale_range[0])
        high = float(scale_range[1] if len(scale_range) > 1 else scale_range[0])
        low, high = min(low, high), max(low, high)
        if mode in {"uniform", "linear_uniform"}:
            return float(torch.empty(1).uniform_(low, high).item())
        if mode in {"log_uniform", "log"}:
            low, high = max(low, 1e-6), max(high, 1e-6)
            return float(torch.empty(1).uniform_(math.log(low), math.log(high)).exp().item())
        raise ValueError(f"Unsupported unprocess.exposure sampling: {mode}")

    def _apply_synthetic_exposure(
        self,
        clean_raw: torch.Tensor,
        clean_extras: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
        exposure_scale = self._sample_synthetic_exposure_scale()
        clean_raw = clean_raw.clamp(0.0, 1.0)
        if exposure_scale == 1.0:
            return clean_raw, clean_extras, exposure_scale

        scaled_extras: dict[str, torch.Tensor] = {}
        for key, value in clean_extras.items():
            if torch.is_tensor(value) and key in {"linear_rgb", "camera_rgb"}:
                scaled_extras[key] = (value * exposure_scale).clamp(0.0, 1.0)
            else:
                scaled_extras[key] = value
        return (clean_raw * exposure_scale).clamp(0.0, 1.0), scaled_extras, exposure_scale

    def _clean_index(self, index: int) -> int:
        fixed_count = self._fixed_noise_setting_count()
        if self.same_scene_gain_sweep and fixed_count:
            return int(index) // fixed_count
        return int(index)

    def _fixed_noise_setting_count(self) -> int:
        if self.fixed_isos:
            return len(self.fixed_isos)
        return len(self.fixed_analog_gains)

    def _make_sample(self, index: int) -> dict[str, torch.Tensor]:
        clean_index = self._clean_index(index)
        clean_item = self._make_clean_item(clean_index)
        return self._make_sample_from_clean(index, clean_item, clean_index)

    def _make_clean_item(self, index: int) -> dict[str, torch.Tensor]:
        if self.mode == "image":
            return self._load_image_sample(index)
        clean_raw = self._make_smooth_random_raw()
        linear_rgb = F.interpolate(
            packed_raw_to_rgb(clean_raw).unsqueeze(0),
            size=(self.patch_size * 2, self.patch_size * 2),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return {"clean_raw": clean_raw, "linear_rgb": linear_rgb}

    def _make_sample_from_clean(
        self,
        index: int,
        clean_item: dict[str, torch.Tensor] | torch.Tensor,
        clean_index: int,
    ) -> dict[str, torch.Tensor]:
        if isinstance(clean_item, torch.Tensor):
            clean_raw = clean_item
            clean_extras: dict[str, torch.Tensor] = {}
        else:
            clean_raw = clean_item["clean_raw"]
            clean_extras = {key: value for key, value in clean_item.items() if key != "clean_raw"}
        selected_sensor_model = clean_extras.pop("_noise_profile_model", None)

        clean_raw, clean_extras, exposure_scale = self._apply_synthetic_exposure(clean_raw, clean_extras)
        if "linear_rgb" in clean_extras and torch.is_tensor(clean_extras["linear_rgb"]):
            clean_extras["linear_rgb"] = self._prepare_linear_rgb_target(clean_extras["linear_rgb"])
        if "camera_rgb" in clean_extras and torch.is_tensor(clean_extras["camera_rgb"]):
            clean_extras["camera_rgb"] = clean_extras["camera_rgb"].clamp(0.0, 1.0)

        analog_gain = None
        iso = None
        gain_index = -1
        if self.fixed_isos:
            gain_index = int(index) % len(self.fixed_isos)
            iso = self.fixed_isos[gain_index]
        elif self.fixed_analog_gains:
            gain_index = int(index) % len(self.fixed_analog_gains)
            analog_gain = self.fixed_analog_gains[gain_index]

        clean_raw_with_black = apply_black_level_to_signal(clean_raw, self.noise_cfg)
        if self.simulate_noise:
            noise_cfg = dict(self.noise_cfg)
            if selected_sensor_model is not None:
                noise_cfg["profile_model"] = str(selected_sensor_model)
                noise_cfg.pop("profile_models", None)
                noise_cfg.pop("camera_models", None)
            noisy_raw, noise_variance, noise_meta = add_noise_from_config(
                clean_raw,
                noise_cfg=noise_cfg,
                channel_names=self.channel_names,
                analog_gain=analog_gain,
                iso=iso,
            )
            analog_gain = noise_meta.get("analog_gain")
            iso = noise_meta.get("iso")
            noise_map_cfg = dict(noise_cfg)
            selected_profile = noise_meta.get("noise_profile_model")
            if selected_profile:
                noise_map_cfg["profile_model"] = str(selected_profile)
                noise_map_cfg.pop("profile_models", None)
                noise_map_cfg.pop("camera_models", None)
            noise_map = estimate_noise_std_map_from_config(
                noisy_raw,
                noise_cfg=noise_map_cfg,
                channel_names=self.channel_names,
                analog_gain=analog_gain,
                iso=iso,
            )
        else:
            noisy_raw = clean_raw_with_black
            noise_variance = torch.zeros_like(clean_raw_with_black)
            noise_map = torch.zeros((1, *clean_raw_with_black.shape[-2:]), dtype=clean_raw_with_black.dtype)
            analog_gain = 1.0 if analog_gain is None else analog_gain
            iso = 0.0 if iso is None else iso
        if self.linear_target_from_clean_raw:
            if self.cfa_pattern != "RGGB":
                raise ValueError("linear_target_from_clean_raw currently requires canonical RGGB.")
            linear_from_raw = demosaic_canonical_rggb(clean_raw_with_black)
            clean_extras["linear_rgb"] = linear_from_raw
            clean_extras["camera_rgb"] = linear_from_raw
        sample = {
            "clean_raw": clean_raw_with_black,
            "noisy_raw": noisy_raw,
            "noise_map": noise_map,
            "noise_variance": noise_variance,
            "noise_sigma": torch.sqrt(noise_variance.clamp_min(1e-12)),
            "image_id": torch.tensor(int(clean_index), dtype=torch.long),
            "exposure_scale": torch.tensor(float(exposure_scale), dtype=torch.float32),
        }
        if analog_gain is not None:
            sample["analog_gain"] = torch.tensor(float(analog_gain), dtype=torch.float32)
        if iso is not None:
            sample["iso"] = torch.tensor(float(iso), dtype=torch.float32)
        if gain_index >= 0:
            sample["gain_index"] = torch.tensor(int(gain_index), dtype=torch.long)
        sample.update(clean_extras)
        if self.defer_linear_rgb:
            empty_target = torch.empty(0, dtype=clean_raw_with_black.dtype)
            sample["linear_rgb"] = empty_target
            sample["camera_rgb"] = empty_target
        return sample

    def _make_smooth_random_raw(self) -> torch.Tensor:
        p = self.patch_size
        low_size = max(4, math.ceil(p / 32))
        low_freq = torch.rand(4, low_size, low_size)
        smooth = F.interpolate(
            low_freq.unsqueeze(0),
            size=(p, p),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        yy = torch.linspace(0.0, 1.0, p).view(1, p, 1)
        xx = torch.linspace(0.0, 1.0, p).view(1, 1, p)
        gradient = (0.5 * xx + 0.5 * yy).expand(4, p, p)
        channel_gain = 0.75 + 0.5 * torch.rand(4, 1, 1)
        clean = (0.65 * smooth + 0.35 * gradient) * channel_gain
        return clean.clamp(0.0, 1.0)

    def _load_image_sample(self, index: int) -> dict[str, torch.Tensor]:
        path = self.image_paths[index % len(self.image_paths)]
        image = Image.open(path).convert("RGB")
        image = self._resize_if_needed(image)
        rgb = self._random_crop_rgb(image)
        rgb = self._augment_rgb(rgb)
        if self.use_unprocess:
            sensor_model, sensor_profile = self._sample_sensor_profile()
            awb_gains = (
                {
                    "red_gain": float(sensor_profile["awb_gains"][0]),
                    "green_gain": float(sensor_profile["awb_gains"][1]),
                    "blue_gain": float(sensor_profile["awb_gains"][2]),
                }
                if sensor_profile is not None
                else self._sample_awb_gains()
            )
            ccm = (
                sensor_profile["camera_to_srgb"]
                if sensor_profile is not None
                else self.unprocess_cfg.get("ccm")
            )
            gamma = self.unprocess_cfg.get(
                "transfer_function",
                self.unprocess_cfg.get("gamma", 2.2),
            )
            camera_rgb = srgb_to_linear_camera_rgb(
                rgb,
                gamma=gamma,
                ccm=ccm,
                awb_gains=awb_gains,
                safe_invert_awb=bool(self.unprocess_cfg.get("safe_invert_awb", False)),
            )
            target_mode = str(self.unprocess_cfg.get("target_linear_rgb", "camera")).lower()
            if target_mode in {"camera", "linear_camera", "linear_camera_rgb"}:
                linear_rgb = camera_rgb
            else:
                raise ValueError(f"Unsupported unprocess.target_linear_rgb: {target_mode}")

            clean_raw = remosaic_linear_rgb_to_packed_raw(camera_rgb, pattern=self.cfa_pattern)
            if clean_raw.shape[-1] != self.patch_size or clean_raw.shape[-2] != self.patch_size:
                clean_raw = F.interpolate(
                    clean_raw.unsqueeze(0),
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            target_size = self.patch_size * 2
            if linear_rgb.shape[-1] != target_size or linear_rgb.shape[-2] != target_size:
                linear_rgb = F.interpolate(
                    linear_rgb.unsqueeze(0),
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            if awb_gains is None:
                awb_tensor = torch.ones(3, dtype=torch.float32)
            else:
                awb_tensor = torch.tensor(
                    [
                        float(awb_gains.get("red_gain", 1.0)),
                        float(awb_gains.get("green_gain", 1.0)),
                        float(awb_gains.get("blue_gain", 1.0)),
                    ],
                    dtype=torch.float32,
                )
            sample = {
                "clean_raw": clean_raw.clamp(0.0, 1.0),
                "linear_rgb": linear_rgb.clamp(0.0, 1.0),
                "camera_rgb": camera_rgb.clamp(0.0, 1.0),
                "awb_gains": awb_tensor,
            }
            if sensor_model is not None:
                sample["_noise_profile_model"] = sensor_model
            return sample
        return {
            "clean_raw": approximate_packed_raw_from_rgb(rgb),
            "linear_rgb": rgb.clamp(0.0, 1.0),
            "awb_gains": torch.ones(3, dtype=torch.float32),
        }

    def _sample_range(self, values: list[float] | tuple[float, ...], default: float) -> float:
        if not values:
            return float(default)
        if len(values) == 1:
            return float(values[0])
        low = float(values[0])
        high = float(values[1])
        if high < low:
            low, high = high, low
        return float(torch.empty(1).uniform_(low, high).item())

    def _sample_awb_gains(self) -> dict[str, float] | None:
        fixed_gains = self.unprocess_cfg.get("awb_gains")
        if fixed_gains:
            return {key: float(value) for key, value in fixed_gains.items()}

        awb_cfg = dict(self.unprocess_cfg.get("awb", {}))
        if not bool(awb_cfg.get("enabled", False)):
            return None
        red_gain = self._sample_range(awb_cfg.get("red_gain_range", [1.9, 2.4]), default=2.1)
        blue_gain = self._sample_range(awb_cfg.get("blue_gain_range", [1.5, 1.9]), default=1.7)
        rgb_gain = self._sample_range(awb_cfg.get("rgb_gain_range", [1.0, 1.4]), default=1.2)
        return {
            "red_gain": red_gain * rgb_gain,
            "green_gain": rgb_gain,
            "blue_gain": blue_gain * rgb_gain,
        }

    def _sample_sensor_profile(self) -> tuple[str | None, dict | None]:
        if not self.sensor_profiles:
            return None, None
        configured_models = self.noise_cfg.get(
            "profile_models",
            self.noise_cfg.get("camera_models", list(self.sensor_profiles)),
        )
        models = [str(model) for model in configured_models if str(model) in self.sensor_profiles]
        if not models:
            raise ValueError("No configured noise profile model has a matching SIDD color profile.")
        configured_weights = self.noise_cfg.get(
            "profile_model_weights",
            self.noise_cfg.get("camera_model_weights"),
        )
        if configured_weights is None:
            weights = torch.ones(len(models), dtype=torch.float32)
        else:
            all_models = [str(model) for model in configured_models]
            weight_by_model = {
                model: float(weight)
                for model, weight in zip(all_models, configured_weights)
            }
            weights = torch.tensor(
                [max(weight_by_model.get(model, 0.0), 0.0) for model in models],
                dtype=torch.float32,
            )
        if float(weights.sum()) <= 0.0:
            raise ValueError("SIDD sensor profile weights must sum to a positive value.")
        model = models[int(torch.multinomial(weights / weights.sum(), 1).item())]
        profiles = list(self.sensor_profiles[model].get("profiles", []))
        if not profiles:
            raise ValueError(f"SIDD sensor model {model!r} has no color profiles.")
        profile = profiles[int(torch.randint(len(profiles), (1,)).item())]
        return model, profile

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        p = self.image_patch_size
        if width >= p and height >= p:
            return image

        scale = max(p / max(width, 1), p / max(height, 1))
        new_size = (max(p, math.ceil(width * scale)), max(p, math.ceil(height * scale)))
        return image.resize(new_size, Image.BICUBIC)

    def _random_crop_rgb(self, image: Image.Image) -> torch.Tensor:
        width, height = image.size
        p = self.image_patch_size
        max_left = max(width - p, 0)
        max_top = max(height - p, 0)
        left = int(torch.randint(0, max_left + 1, (1,)).item()) if max_left else 0
        top = int(torch.randint(0, max_top + 1, (1,)).item()) if max_top else 0
        crop = image.crop((left, top, left + p, top + p))
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _augment_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return rgb
        if torch.rand(()) < 0.5:
            rgb = torch.flip(rgb, dims=(-1,))
        if torch.rand(()) < 0.5:
            rgb = torch.flip(rgb, dims=(-2,))
        if torch.rand(()) < 0.5:
            rgb = rgb.transpose(-1, -2).contiguous()
        return rgb
