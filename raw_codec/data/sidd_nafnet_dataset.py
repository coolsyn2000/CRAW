from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from .sidd_color import load_sidd_render_profile_for_scene
from .sidd_raw_dataset import demosaic_canonical_rggb


class SiddNafnetRawDataset(Dataset):
    """Pre-tiled SIDD packed RAW pairs for NAFNet-style training."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        packed_crop_size: int = 128,
        length: int = 0,
        include_black_level: bool = True,
        canonical_black_level: float = 512.0,
        canonical_white_level: float = 16383.0,
        augment: bool = False,
        defer_linear_rgb: bool = False,
        demosaic_method: str = "opencv_ea",
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = str(split).lower()
        self.lmdb_path = self.root / f"{self.split}.lmdb"
        if not self.lmdb_path.is_dir():
            raise FileNotFoundError(
                f"Missing prepared SIDD NAFNet dataset at {self.lmdb_path}. "
                "Run tools/data/prepare_sidd_lmdb.py first."
            )
        with (self.lmdb_path / "meta_info.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.cfa_canonicalization = str(
            self.metadata.get("cfa_canonicalization", "legacy_channel_reorder")
        )
        self.entries = list(self.metadata["entries"])
        self.render_profiles = dict(self.metadata.get("render_profiles", {}))
        self.stored_packed_size = int(self.metadata["packed_size"])
        self.packed_crop_size = int(packed_crop_size)
        self.length = int(length) if int(length) > 0 else len(self.entries)
        self.include_black_level = bool(include_black_level)
        self.canonical_black_level = float(canonical_black_level)
        self.canonical_white_level = float(canonical_white_level)
        self.augment = bool(augment)
        self.defer_linear_rgb = bool(defer_linear_rgb)
        self.demosaic_method = str(demosaic_method)
        self._environment: lmdb.Environment | None = None
        self._render_profile_cache: dict[str, dict[str, Any]] = {}

        if not self.entries:
            raise ValueError(f"No entries found in {self.lmdb_path}.")
        if not 0 < self.packed_crop_size <= self.stored_packed_size:
            raise ValueError(
                f"packed_crop_size={self.packed_crop_size} must be in "
                f"[1,{self.stored_packed_size}]."
            )
        if self.augment:
            raise ValueError(
                "The canonical NAFNet SIDD protocol disables flips and rotations; "
                "packed Bayer augmentation requires channel-aware permutations."
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_environment"] = None
        return state

    def __del__(self) -> None:
        if self._environment is not None:
            self._environment.close()

    def _open_environment(self) -> lmdb.Environment:
        if self._environment is None:
            self._environment = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
                subdir=True,
            )
        return self._environment

    def __len__(self) -> int:
        return self.length

    def _read_array(self, index: int, kind: str) -> torch.Tensor:
        key = f"{index:08d}/{kind}".encode("ascii")
        with self._open_environment().begin(write=False) as transaction:
            value = transaction.get(key)
        if value is None:
            raise KeyError(f"Missing LMDB key {key!r} in {self.lmdb_path}.")
        shape = (4, self.stored_packed_size, self.stored_packed_size)
        array = np.frombuffer(value, dtype=np.uint16).reshape(shape).copy()
        return torch.from_numpy(array).float().div_(65535.0)

    def _add_black_level(self, signal: torch.Tensor) -> torch.Tensor:
        if not self.include_black_level:
            return signal
        black = self.canonical_black_level / max(self.canonical_white_level, 1.0)
        return black + signal * (1.0 - black)

    def _render_profile(self, entry: dict[str, Any]) -> dict[str, Any]:
        scene = str(entry["scene"])
        profile = self.render_profiles.get(scene)
        if profile is None:
            profile = self._render_profile_cache.get(scene)
        if profile is None:
            profile = load_sidd_render_profile_for_scene(
                entry["noisy_path"],
                scene,
            )
            self._render_profile_cache[scene] = profile
        return profile

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        stored_index = int(index) % len(self.entries)
        entry = self.entries[stored_index]
        noisy = self._read_array(stored_index, "noisy")
        clean = self._read_array(stored_index, "clean")

        margin = self.stored_packed_size - self.packed_crop_size
        if margin > 0:
            top = int(torch.randint(0, margin + 1, ()).item())
            left = int(torch.randint(0, margin + 1, ()).item())
            crop = np.s_[
                ...,
                top : top + self.packed_crop_size,
                left : left + self.packed_crop_size,
            ]
            noisy = noisy[crop]
            clean = clean[crop]

        noisy_raw = self._add_black_level(noisy).contiguous()
        clean_raw = self._add_black_level(clean).contiguous()
        linear_rgb = (
            torch.empty(0, dtype=clean_raw.dtype)
            if self.defer_linear_rgb
            else demosaic_canonical_rggb(
                clean_raw,
                method=self.demosaic_method,
            ).contiguous()
        )
        black = (
            self.canonical_black_level / max(self.canonical_white_level, 1.0)
            if self.include_black_level
            else 0.0
        )
        iso = float(entry["iso"])
        render_profile = self._render_profile(entry)
        return {
            "clean_raw": clean_raw,
            "noisy_raw": noisy_raw,
            "noise_map": torch.zeros(
                (1, self.packed_crop_size, self.packed_crop_size),
                dtype=noisy_raw.dtype,
            ),
            "linear_rgb": linear_rgb,
            "camera_rgb": linear_rgb,
            "awb_gains": torch.tensor(
                render_profile["awb_gains"], dtype=torch.float32
            ),
            "camera_to_srgb": torch.tensor(
                render_profile["camera_to_srgb"], dtype=torch.float32
            ),
            "exposure_scale": torch.tensor(1.0, dtype=torch.float32),
            "analog_gain": torch.tensor(max(iso / 100.0, 1.0), dtype=torch.float32),
            "iso": torch.tensor(iso, dtype=torch.float32),
            "native_black_level": torch.tensor(0.0, dtype=torch.float32),
            "black_level": torch.tensor(black, dtype=torch.float32),
            "white_level": torch.tensor(1.0, dtype=torch.float32),
            "image_id": torch.tensor(stored_index, dtype=torch.long),
            "scene": str(entry["scene"]),
            "camera": str(entry["camera"]),
            "bayer_pattern": "RGGB",
            "noisy_path": str(entry["noisy_path"]),
            "clean_path": str(entry["clean_path"]),
        }
