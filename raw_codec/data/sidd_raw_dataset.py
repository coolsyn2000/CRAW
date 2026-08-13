from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .unprocess import BAYER_TILES


_SCENE_RE = re.compile(
    r"^(?P<instance>\d{4})_(?P<scene>\d{3})_(?P<camera>[A-Z0-9]+)_"
    r"(?P<iso>\d{5})_(?P<shutter>\d{5})_(?P<cct>\d{4})_(?P<brightness>[LNH])$"
)
_DEFAULT_BAYER_PATTERNS = {
    "GP": "BGGR",
    "IP": "RGGB",
    "S6": "GRBG",
    "N6": "BGGR",
    "G4": "BGGR",
}
_CANONICAL_CHANNELS = ("R", "Gr", "Gb", "B")


@dataclass(frozen=True)
class SiddRawPair:
    scene_id: str
    camera: str
    iso: float
    bayer_pattern: str
    noisy_path: Path
    gt_path: Path


def _load_mat_image(path: Path) -> torch.Tensor:
    try:
        from scipy.io import loadmat

        document = loadmat(path)
        arrays = [
            value
            for key, value in document.items()
            if not key.startswith("__")
            and isinstance(value, np.ndarray)
            and value.ndim in {2, 3}
            and np.issubdtype(value.dtype, np.number)
        ]
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as handle:
            arrays = [
                np.asarray(value).transpose(tuple(reversed(range(len(value.shape)))))
                for value in handle.values()
                if hasattr(value, "shape") and len(value.shape) in {2, 3}
            ]
    if not arrays:
        raise ValueError(f"No 2-D or 3-D numeric RAW array found in {path}.")

    array = max(arrays, key=lambda value: int(np.prod(value.shape)))
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 4 and array.shape[-1] != 4:
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] != 4:
        raise ValueError(f"Expected SIDD RAW channels last with four planes, got {array.shape} in {path}.")

    source_dtype = array.dtype
    tensor = torch.from_numpy(array.astype(np.float32, copy=True))
    if np.issubdtype(source_dtype, np.integer):
        tensor = tensor / float(np.iinfo(source_dtype).max)
    min_value = float(tensor.min())
    max_value = float(tensor.max())
    if min_value < -1e-5 or max_value > 1.0001:
        raise ValueError(
            f"SIDD RAW is expected to be black-level-subtracted and normalized to [0,1], "
            f"but {path} has range [{min_value:.6g}, {max_value:.6g}]."
        )
    return tensor.clamp(0.0, 1.0)


def _row_major_channel_names(pattern: str) -> tuple[str, str, str, str]:
    tile = BAYER_TILES[pattern.upper()]
    return tile[0][0], tile[0][1], tile[1][0], tile[1][1]


def _canonicalize_packed_raw(raw: torch.Tensor, pattern: str) -> torch.Tensor:
    """Map a native Bayer image to RGGB with a whole-image spatial transform.

    This preserves the relative sampling geometry of the four CFA planes. A
    channel-only permutation changes CFA labels without transforming the scene.
    """
    if raw.ndim == 2:
        raw = torch.stack(
            [
                raw[0::2, 0::2],
                raw[0::2, 1::2],
                raw[1::2, 0::2],
                raw[1::2, 1::2],
            ],
            dim=-1,
        )
    if raw.ndim != 3 or raw.shape[-1] != 4:
        raise ValueError(f"Expected SIDD RAW as [H,W] or [H,W,4], got {tuple(raw.shape)}.")

    pattern = pattern.upper()
    if pattern not in BAYER_TILES:
        raise ValueError(f"Unsupported Bayer pattern {pattern!r}.")

    height, width, _ = raw.shape
    mosaic = torch.empty((height * 2, width * 2), dtype=raw.dtype, device=raw.device)
    mosaic[0::2, 0::2] = raw[..., 0]
    mosaic[0::2, 1::2] = raw[..., 1]
    mosaic[1::2, 0::2] = raw[..., 2]
    mosaic[1::2, 1::2] = raw[..., 3]

    spatial_dims = {
        "RGGB": (),
        "GRBG": (1,),
        "GBRG": (0,),
        "BGGR": (0, 1),
    }[pattern]
    if spatial_dims:
        mosaic = torch.flip(mosaic, dims=spatial_dims)

    return torch.stack(
        [
            mosaic[0::2, 0::2],
            mosaic[0::2, 1::2],
            mosaic[1::2, 0::2],
            mosaic[1::2, 1::2],
        ],
        dim=0,
    ).contiguous()


def _unpack_canonical_rggb(packed_raw: torch.Tensor) -> torch.Tensor:
    if packed_raw.ndim != 3 or packed_raw.shape[0] != 4:
        raise ValueError(f"Expected canonical packed RAW [4,H,W], got {tuple(packed_raw.shape)}.")

    _, h, w = packed_raw.shape
    mosaic = torch.empty((h * 2, w * 2), dtype=packed_raw.dtype, device=packed_raw.device)
    mosaic[0::2, 0::2] = packed_raw[0]
    mosaic[0::2, 1::2] = packed_raw[1]
    mosaic[1::2, 0::2] = packed_raw[2]
    mosaic[1::2, 1::2] = packed_raw[3]
    return mosaic


def _demosaic_canonical_rggb_bilinear(packed_raw: torch.Tensor) -> torch.Tensor:
    mosaic_2d = _unpack_canonical_rggb(packed_raw)
    _, h, w = packed_raw.shape
    mosaic = mosaic_2d.view(1, 1, h * 2, w * 2)
    masks = torch.zeros(
        (3, 1, h * 2, w * 2),
        dtype=packed_raw.dtype,
        device=packed_raw.device,
    )
    samples = torch.zeros_like(masks)

    masks[0, 0, 0::2, 0::2] = 1.0
    masks[1, 0, 0::2, 1::2] = 1.0
    masks[1, 0, 1::2, 0::2] = 1.0
    masks[2, 0, 1::2, 1::2] = 1.0
    samples.copy_(mosaic.expand(3, -1, -1, -1) * masks)

    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=packed_raw.dtype,
        device=packed_raw.device,
    ).view(1, 1, 3, 3)
    values = F.conv2d(F.pad(samples, (1, 1, 1, 1), mode="reflect"), kernel)
    weights = F.conv2d(F.pad(masks, (1, 1, 1, 1), mode="reflect"), kernel)
    rgb = values / weights.clamp_min(1e-8)
    rgb = torch.where(masks > 0, samples, rgb)
    return rgb[:, 0].clamp(0.0, 1.0)


def _demosaic_canonical_rggb_opencv_ea(packed_raw: torch.Tensor) -> torch.Tensor:
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "OpenCV edge-aware demosaicing requires opencv-python-headless."
        ) from error

    mosaic = _unpack_canonical_rggb(packed_raw).detach().float().cpu().numpy()
    max_code = 16383.0
    mosaic_u16 = np.rint(np.clip(mosaic, 0.0, 1.0) * max_code).astype(np.uint16)
    # OpenCV's Bayer conversion aliases use BG2RGB for an RGGB input mosaic.
    rgb = cv2.cvtColor(mosaic_u16, cv2.COLOR_BayerBG2RGB_EA)
    rgb_tensor = torch.from_numpy(rgb.astype(np.float32) / max_code).permute(2, 0, 1)
    return rgb_tensor.to(device=packed_raw.device, dtype=packed_raw.dtype).clamp(0.0, 1.0)


def demosaic_canonical_rggb(
    packed_raw: torch.Tensor,
    method: str = "bilinear",
) -> torch.Tensor:
    """Demosaic a canonical RGGB tensor without applying the rest of an ISP."""
    normalized_method = str(method).lower().replace("-", "_")
    if normalized_method in {"bilinear", "torch_bilinear"}:
        return _demosaic_canonical_rggb_bilinear(packed_raw)
    if normalized_method in {"opencv_ea", "ea", "edge_aware"}:
        return _demosaic_canonical_rggb_opencv_ea(packed_raw)
    raise ValueError(
        f"Unsupported demosaic method {method!r}; expected 'bilinear' or 'opencv_ea'."
    )


def _load_bayer_patterns(root: Path, csv_path: str | Path | None) -> dict[str, str]:
    patterns = dict(_DEFAULT_BAYER_PATTERNS)
    path = Path(csv_path).expanduser() if csv_path else root / "bayer_patterns.csv"
    if not path.exists():
        return patterns
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            camera = str(row.get("camera_id", "")).upper()
            pattern = str(row.get("bayer_pattern", "")).upper()
            if camera and pattern in BAYER_TILES:
                patterns[camera] = pattern
    return patterns


def _load_nlf(root: Path, csv_path: str | Path | None) -> dict[str, tuple[float, ...]]:
    path = Path(csv_path).expanduser() if csv_path else root / "noise_level_functions.csv"
    if not path.exists():
        return {}
    result: dict[str, tuple[float, ...]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scene_id = str(row.get("scene_instance_id", ""))
            if not scene_id:
                continue
            result[scene_id] = (
                float(row["beta1_r"]),
                float(row["beta2_r"]),
                float(row["beta1_g"]),
                float(row["beta2_g"]),
                float(row["beta1_b"]),
                float(row["beta2_b"]),
            )
    return result


def _resolve_data_root(root: str | Path) -> Path:
    root = Path(root).expanduser()
    candidates = (
        root,
        root / "Data",
        root / "SIDD_Medium_Raw",
        root / "SIDD_Medium_Raw" / "Data",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(_SCENE_RE.match(path.name) for path in candidate.iterdir() if path.is_dir()):
            return candidate
    raise FileNotFoundError(f"Could not find SIDD Medium scene directories under {root}.")


class SiddRawDataset(Dataset):
    """SIDD Medium paired RAW loader with canonical RGGB input and camera-linear RGB targets."""

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        patch_size: int = 512,
        length: int = 0,
        deterministic: bool = True,
        base_seed: int = 0,
        augment: bool = False,
        max_pairs: int = 0,
        pair_indices: list[int] | tuple[int, ...] | None = None,
        bayer_patterns_csv: str | Path | None = None,
        noise_level_functions_csv: str | Path | None = None,
        canonical_black_level: float | None = None,
        canonical_white_level: float | None = None,
        linear_rgb_include_black_level: bool = False,
        target_include_black_level: bool = False,
        demosaic_method: str = "opencv_ea",
        **_: Any,
    ) -> None:
        self.root = Path(root).expanduser()
        self.data_root = _resolve_data_root(self.root)
        self.split = str(split).lower()
        self.patch_size = int(patch_size)
        self.length = int(length)
        self.deterministic = bool(deterministic)
        self.base_seed = int(base_seed)
        self.augment = bool(augment)
        self.canonical_black_level = (
            float(canonical_black_level) if canonical_black_level is not None else None
        )
        self.canonical_white_level = (
            float(canonical_white_level) if canonical_white_level is not None else None
        )
        self.include_black_level = bool(
            linear_rgb_include_black_level or target_include_black_level
        )
        self.demosaic_method = str(demosaic_method)
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if (self.canonical_black_level is None) != (self.canonical_white_level is None):
            raise ValueError("canonical_black_level and canonical_white_level must be provided together.")

        pattern_root = self.root if self.root.is_dir() else self.data_root
        self.bayer_patterns = _load_bayer_patterns(pattern_root, bayer_patterns_csv)
        self.nlf = _load_nlf(pattern_root, noise_level_functions_csv)
        self.pairs = self._load_pairs(max_pairs=max_pairs)
        if pair_indices is not None:
            requested = [int(index) for index in pair_indices]
            invalid = [index for index in requested if index < 0 or index >= len(self.pairs)]
            if invalid:
                raise IndexError(
                    f"SIDD pair_indices contains out-of-range indices {invalid}; "
                    f"dataset has {len(self.pairs)} pairs."
                )
            self.pairs = [self.pairs[index] for index in requested]
        if not self.pairs:
            raise FileNotFoundError(f"No paired SIDD Medium RAW files found under {self.data_root}.")
        if self.length <= 0:
            self.length = len(self.pairs)

    def _load_pairs(self, max_pairs: int) -> list[SiddRawPair]:
        pairs: list[SiddRawPair] = []
        for scene_dir in sorted(path for path in self.data_root.iterdir() if path.is_dir()):
            match = _SCENE_RE.match(scene_dir.name)
            if match is None:
                continue
            camera = match.group("camera")
            pattern = self.bayer_patterns.get(camera)
            if pattern is None:
                raise ValueError(f"No Bayer pattern registered for SIDD camera {camera}.")
            for noisy_path in sorted(scene_dir.glob("*_NOISY_RAW_*.MAT")):
                gt_path = noisy_path.with_name(noisy_path.name.replace("_NOISY_RAW_", "_GT_RAW_"))
                if not gt_path.exists():
                    continue
                pairs.append(
                    SiddRawPair(
                        scene_id=scene_dir.name,
                        camera=camera,
                        iso=float(match.group("iso")),
                        bayer_pattern=pattern,
                        noisy_path=noisy_path,
                        gt_path=gt_path,
                    )
                )
                if max_pairs > 0 and len(pairs) >= int(max_pairs):
                    return pairs
        return pairs

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + int(index) if self.deterministic else torch.seed())
        pair = self.pairs[int(index) % len(self.pairs)]
        noisy_signal = _canonicalize_packed_raw(_load_mat_image(pair.noisy_path), pair.bayer_pattern)
        clean_signal = _canonicalize_packed_raw(_load_mat_image(pair.gt_path), pair.bayer_pattern)
        if noisy_signal.shape != clean_signal.shape:
            raise ValueError(
                f"SIDD pair shape mismatch: {tuple(noisy_signal.shape)} vs {tuple(clean_signal.shape)}."
            )

        _, height, width = noisy_signal.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"SIDD image {pair.noisy_path} has packed shape {height}x{width}, "
                f"smaller than patch_size={self.patch_size}."
            )
        top = int(torch.randint(0, height - self.patch_size + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, width - self.patch_size + 1, (1,), generator=generator).item())
        crop = np.s_[..., top : top + self.patch_size, left : left + self.patch_size]
        noisy_signal = noisy_signal[crop]
        clean_signal = clean_signal[crop]

        if self.augment:
            if bool(torch.randint(0, 2, (1,), generator=generator).item()):
                noisy_signal = torch.flip(noisy_signal, dims=(-1,))
                clean_signal = torch.flip(clean_signal, dims=(-1,))
            if bool(torch.randint(0, 2, (1,), generator=generator).item()):
                noisy_signal = torch.flip(noisy_signal, dims=(-2,))
                clean_signal = torch.flip(clean_signal, dims=(-2,))

        noise_map = self._noise_map(pair.scene_id, noisy_signal)
        noisy_raw = self._include_canonical_black(noisy_signal)
        clean_raw = self._include_canonical_black(clean_signal)
        linear_rgb = demosaic_canonical_rggb(
            clean_signal,
            method=self.demosaic_method,
        )
        if self.include_black_level:
            linear_rgb = self._include_canonical_black(linear_rgb)

        black = float(self.canonical_black_level or 0.0)
        white = float(self.canonical_white_level or 1.0)
        return {
            "clean_raw": clean_raw.contiguous(),
            "noisy_raw": noisy_raw.contiguous(),
            "noise_map": noise_map.contiguous(),
            "noise_variance": noise_map.square().contiguous(),
            "noise_sigma": noise_map.contiguous(),
            "linear_rgb": linear_rgb.contiguous(),
            "camera_rgb": linear_rgb.contiguous(),
            "awb_gains": torch.ones(3, dtype=torch.float32),
            "exposure_scale": torch.tensor(1.0, dtype=torch.float32),
            "analog_gain": torch.tensor(max(pair.iso / 100.0, 1.0), dtype=torch.float32),
            "iso": torch.tensor(pair.iso, dtype=torch.float32),
            "native_black_level": torch.tensor(0.0, dtype=torch.float32),
            "black_level": torch.tensor(black, dtype=torch.float32),
            "white_level": torch.tensor(white, dtype=torch.float32),
            "image_id": torch.tensor(int(index) % len(self.pairs), dtype=torch.long),
            "scene": pair.scene_id,
            "camera": pair.camera,
            "bayer_pattern": pair.bayer_pattern,
            "noisy_path": str(pair.noisy_path),
            "clean_path": str(pair.gt_path),
        }

    def _include_canonical_black(self, signal: torch.Tensor) -> torch.Tensor:
        if not self.include_black_level:
            return signal
        black = float(self.canonical_black_level or 0.0)
        white = float(self.canonical_white_level or 1.0)
        normalized_black = black / max(white, 1.0)
        return (normalized_black + signal * (1.0 - normalized_black)).clamp(0.0, 1.0)

    def _noise_map(self, scene_id: str, noisy_signal: torch.Tensor) -> torch.Tensor:
        params = self.nlf.get(scene_id)
        if params is None:
            return torch.zeros((1, *noisy_signal.shape[-2:]), dtype=noisy_signal.dtype)
        beta1_r, beta2_r, beta1_g, beta2_g, beta1_b, beta2_b = params
        beta1 = noisy_signal.new_tensor([beta1_r, beta1_g, beta1_g, beta1_b]).view(4, 1, 1)
        beta2 = noisy_signal.new_tensor([beta2_r, beta2_g, beta2_g, beta2_b]).view(4, 1, 1)
        std = torch.sqrt((beta1 * noisy_signal + beta2).clamp_min(0.0))
        return std.mean(dim=0, keepdim=True)
