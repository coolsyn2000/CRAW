from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from raw_codec.data.sidd_color import sidd_render_profile_from_metadata
from raw_codec.data.sidd_raw_dataset import demosaic_canonical_rggb
from raw_codec.data.unprocess import linear_camera_rgb_to_srgb
from raw_codec.io import RawUint16Metadata, native_rgb_from_canonical, normalize_and_pack_raw
from raw_codec.utils.image_io import save_rgb_image


def _load_raw(path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat

        document = loadmat(path)
        arrays = [
            value
            for key, value in document.items()
            if not key.startswith("__")
            and isinstance(value, np.ndarray)
            and value.ndim == 2
            and np.issubdtype(value.dtype, np.number)
        ]
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as handle:
            arrays = [
                np.asarray(value).T
                for value in handle.values()
                if hasattr(value, "shape") and len(value.shape) == 2
            ]
    if not arrays:
        raise ValueError(f"No two-dimensional RAW array found in {path}.")
    raw = np.asarray(max(arrays, key=lambda value: int(value.size)), dtype=np.float32)
    if not np.isfinite(raw).all() or float(raw.min()) < -1.0e-6 or float(raw.max()) > 1.0001:
        raise ValueError(f"Expected normalized SIDD RAW in [0,1], got {raw.min()}..{raw.max()}.")
    return np.clip(raw, 0.0, 1.0)


def _load_metadata(path: Path):
    from scipy.io import loadmat

    document = loadmat(path, squeeze_me=True, struct_as_record=False)
    metadata = document.get("metadata")
    if metadata is None:
        raise ValueError(f"No metadata structure found in {path}.")
    return metadata


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _field(metadata: Any, name: str, default: Any = None) -> Any:
    return _json_value(getattr(metadata, name, default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export normalized SIDD MATLAB RAW as an unpacked uint16 Bayer frame."
    )
    parser.add_argument("--noisy-mat", required=True)
    parser.add_argument("--metadata-mat", required=True)
    parser.add_argument("--output-raw", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-noisy-png")
    parser.add_argument("--gt-mat")
    parser.add_argument("--output-gt-png")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--cfa-pattern", required=True, choices=("RGGB", "GRBG", "GBRG", "BGGR"))
    parser.add_argument("--bit-depth", type=int, default=10)
    parser.add_argument("--black-level", type=int, default=32)
    parser.add_argument("--white-level", type=int, default=1023)
    return parser.parse_args()


def _sensor_codes(raw: np.ndarray, black: int, white: int) -> np.ndarray:
    return np.rint(raw * float(white - black) + float(black)).clip(0, 65535).astype("<u2")


@torch.inference_mode()
def _render_sidd_png(
    codes: np.ndarray,
    document: dict[str, Any],
    output_path: str | Path,
) -> None:
    metadata = RawUint16Metadata(
        width=int(document["width"]),
        height=int(document["height"]),
        cfa_pattern=str(document["cfa_pattern"]),
        black_level=tuple(float(value) for value in document["black_level"]),
        white_level=float(document["white_level"]),
        bit_depth=int(document["bit_depth"]),
        model_input_include_black_level=True,
        document=document,
    )
    metadata.validate()
    packed = normalize_and_pack_raw(codes, metadata)
    camera_rgb = demosaic_canonical_rggb(packed, method="opencv_ea")
    camera_rgb = native_rgb_from_canonical(camera_rgb, metadata.cfa_pattern)
    black = sum(metadata.black_level) / len(metadata.black_level)
    normalized_black = black / metadata.white_level
    camera_signal = (
        (camera_rgb - normalized_black) / max(1.0 - normalized_black, 1.0e-8)
    ).clamp(0.0, 1.0)
    render = document["render"]
    display = linear_camera_rgb_to_srgb(
        camera_signal,
        gamma=2.2,
        ccm=render["camera_to_srgb"],
        awb_gains=torch.tensor(render["awb_gains"]),
    )
    save_rgb_image(display, output_path)
    print(f"rendered={Path(output_path).resolve()} shape={tuple(display.shape)}")


def main() -> None:
    args = parse_args()
    noisy_path = Path(args.noisy_mat).expanduser().resolve()
    metadata_path = Path(args.metadata_mat).expanduser().resolve()
    raw = _load_raw(noisy_path)
    metadata = _load_metadata(metadata_path)
    black = int(args.black_level)
    white = int(args.white_level)
    if black < 0 or white <= black or white > 65535:
        raise ValueError("Require 0 <= black_level < white_level <= 65535.")
    codes = _sensor_codes(raw, black, white)

    output_raw = Path(args.output_raw).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    codes.tofile(output_raw)

    source_black = _field(metadata, "BlackLevel", 0)
    source_white = _field(metadata, "WhiteLevel", None)
    try:
        cct = float(str(args.scene_id).split("_")[-2])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot parse CCT from scene id {args.scene_id!r}.") from error
    render_profile = sidd_render_profile_from_metadata(metadata, cct=cct)
    document = {
        "format": "headerless_unpacked_bayer",
        "version": 1,
        "file": output_raw.name,
        "dtype": "uint16",
        "endianness": "little",
        "width": int(codes.shape[1]),
        "height": int(codes.shape[0]),
        "row_stride_bytes": int(codes.shape[1] * 2),
        "cfa_pattern": str(args.cfa_pattern),
        "bit_depth": int(args.bit_depth),
        "black_level": [black, black, black, black],
        "white_level": white,
        "model_input_include_black_level": True,
        "model_input_normalization": "code/white_level",
        "black_level_subtracted_normalization": (
            "(code-black_level)/(white_level-black_level)"
        ),
        "source": {
            "dataset": "SIDD Medium Raw",
            "scene_id": str(args.scene_id),
            "frame_id": str(args.frame_id),
            "camera_make": _field(metadata, "Make"),
            "camera_model": _field(metadata, "Model"),
            "unique_camera_model": _field(metadata, "UniqueCameraModel"),
            "iso": _field(metadata, "ISOSpeedRatings"),
            "exposure_time_seconds": _field(metadata, "ExposureTime"),
            "f_number": _field(metadata, "FNumber"),
            "focal_length_mm": _field(metadata, "FocalLength"),
            "orientation": _field(metadata, "Orientation", 1),
            "source_black_level": source_black,
            "source_white_level": source_white,
            "source_noisy_mat": noisy_path.name,
            "source_metadata_mat": metadata_path.name,
        },
        "color": {
            "cct_kelvin": cct,
            "as_shot_neutral": _field(metadata, "AsShotNeutral"),
            "color_matrix_1": _field(metadata, "ColorMatrix1"),
            "color_matrix_2": _field(metadata, "ColorMatrix2"),
            "calibration_illuminant_1": _field(metadata, "CalibrationIlluminant1"),
            "calibration_illuminant_2": _field(metadata, "CalibrationIlluminant2"),
        },
        "render": {
            "transfer": "gamma_2.2",
            "awb_gains": render_profile["awb_gains"],
            "camera_to_srgb": render_profile["camera_to_srgb"],
        },
        "release_note": (
            "SIDD stores a black-level-subtracted normalized RAW signal. This asset "
            f"maps it to a {int(args.bit_depth)}-bit sensor-code domain using "
            f"code = round(signal * ({white} - {black}) + {black})."
        ),
    }
    output_json.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_noisy_png:
        _render_sidd_png(codes, document, args.output_noisy_png)
    if bool(args.gt_mat) != bool(args.output_gt_png):
        raise ValueError("--gt-mat and --output-gt-png must be provided together.")
    if args.gt_mat:
        gt = _load_raw(Path(args.gt_mat).expanduser().resolve())
        if gt.shape != raw.shape:
            raise ValueError(f"GT shape {gt.shape} does not match noisy RAW shape {raw.shape}.")
        _render_sidd_png(_sensor_codes(gt, black, white), document, args.output_gt_png)
    print(f"raw={output_raw} bytes={output_raw.stat().st_size}")
    print(f"metadata={output_json}")
    print(f"shape={codes.shape} range={int(codes.min())}..{int(codes.max())}")


if __name__ == "__main__":
    main()
