from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


RGB_TO_XYZ = np.asarray(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
ILLUMINANT_KELVIN = {
    17: 2856.0,
    18: 4874.0,
    19: 6774.0,
    20: 5503.0,
    21: 6504.0,
    22: 7504.0,
}


def sidd_metadata_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    name = path.name.replace("_NOISY_RAW_", "_METADATA_RAW_")
    name = name.replace("_GT_RAW_", "_METADATA_RAW_")
    metadata_path = path.with_name(name)
    if metadata_path == path:
        raise ValueError(f"Cannot derive SIDD metadata path from {path}.")
    return metadata_path


def _interpolate_xyz_to_camera(metadata: Any, cct: float) -> np.ndarray:
    matrix1 = np.asarray(metadata.ColorMatrix1, dtype=np.float64).reshape(3, 3)
    matrix2 = np.asarray(metadata.ColorMatrix2, dtype=np.float64).reshape(3, 3)
    temperature1 = ILLUMINANT_KELVIN.get(
        int(metadata.CalibrationIlluminant1), 2856.0
    )
    temperature2 = ILLUMINANT_KELVIN.get(
        int(metadata.CalibrationIlluminant2), 6504.0
    )
    reciprocal = 1.0 / max(float(cct), 1.0)
    reciprocal1 = 1.0 / temperature1
    reciprocal2 = 1.0 / temperature2
    weight1 = np.clip(
        (reciprocal - reciprocal2) / max(reciprocal1 - reciprocal2, 1.0e-12),
        0.0,
        1.0,
    )
    return weight1 * matrix1 + (1.0 - weight1) * matrix2


def sidd_render_profile_from_metadata(metadata: Any, cct: float) -> dict[str, Any]:
    neutral = np.asarray(metadata.AsShotNeutral, dtype=np.float64).reshape(3)
    gains = 1.0 / np.clip(neutral, 1.0e-8, None)
    gains = gains / gains[1]

    xyz_to_camera = _interpolate_xyz_to_camera(metadata, cct)
    srgb_to_camera = xyz_to_camera @ RGB_TO_XYZ
    srgb_to_camera = srgb_to_camera / srgb_to_camera.sum(
        axis=1, keepdims=True
    )
    return {
        "awb_gains": gains.tolist(),
        "camera_to_srgb": np.linalg.inv(srgb_to_camera).tolist(),
    }


def _mat_struct_to_namespace(value: Any) -> Any:
    if hasattr(value, "ColorMatrix1"):
        return value
    if isinstance(value, dict):
        return SimpleNamespace(**value)
    return value


@lru_cache(maxsize=256)
def load_sidd_render_profile(
    metadata_path: str | Path,
    cct: float,
) -> dict[str, Any]:
    from scipy.io import loadmat

    path = Path(metadata_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            "SIDD display rendering requires the matching metadata file; "
            f"missing {path}. Rebuild the LMDB to embed AWB/CCM profiles."
        )
    document = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "metadata" not in document:
        raise KeyError(f"SIDD metadata field is missing from {path}.")
    return sidd_render_profile_from_metadata(
        _mat_struct_to_namespace(document["metadata"]),
        cct,
    )


def load_sidd_render_profile_for_scene(
    raw_path: str | Path,
    scene: str,
) -> dict[str, Any]:
    try:
        cct = float(str(scene).split("_")[-2])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot parse CCT from SIDD scene {scene!r}.") from error
    metadata_path = sidd_metadata_path(raw_path)
    profile = dict(load_sidd_render_profile(metadata_path, cct))
    profile["render_metadata_path"] = str(metadata_path)
    return profile
