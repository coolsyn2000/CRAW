from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from raw_codec.data.unprocess import linear_camera_rgb_to_srgb
from raw_codec.io import (
    load_raw_uint16,
    native_rgb_from_canonical,
    normalize_and_pack_raw,
    read_craw,
    write_craw,
)
from raw_codec.model_zoo import OursM, OursS
from raw_codec.utils import load_checkpoint
from raw_codec.utils.image_io import save_rgb_image


REPO_ROOT = Path(__file__).resolve().parent
EXAMPLE_ROOT = REPO_ROOT / "assets" / "sidd_s6_iso3200"
DEFAULT_RAW = EXAMPLE_ROOT / "sidd_0104_s6_iso3200_noisy.raw"
DEFAULT_METADATA = EXAMPLE_ROOT / "sidd_0104_s6_iso3200_noisy.json"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "ours_s_lambda6e4_176k" / "inference_fp16.pth"
)
DEFAULT_CRAW = REPO_ROOT / "outputs" / "sidd_0104_ours_s.craw"
DEFAULT_PNG = REPO_ROOT / "outputs" / "sidd_0104_ours_s.png"


def _canonical_decoder_id(value: str) -> str:
    value = str(value).lower().replace("-", "_")
    if value not in {"ours_m", "ours_s"}:
        raise ValueError("decoder id must be ours_m or ours_s.")
    return value


def _checkpoint_method(checkpoint: dict[str, Any]) -> str | None:
    method = checkpoint.get("method")
    if not method:
        return None
    return str(method).lower().replace("-", "_")


def load_model(decoder_id: str, checkpoint_path: str | Path, device: torch.device):
    decoder_id = _canonical_decoder_id(decoder_id)
    model = OursM() if decoder_id == "ours_m" else OursS()
    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
        map_location="cpu",
        strict=True,
    )
    method = _checkpoint_method(checkpoint)
    if method is not None and method != decoder_id:
        raise ValueError(
            f"Checkpoint declares method={method!r}, but the bitstream requires "
            f"decoder_id={decoder_id!r}."
        )
    model = model.to(device=device, dtype=torch.float32).eval()
    model.update(force=True)
    if model.use_scctx:
        raise ValueError("ERIC .craw v1 supports the paper hyperprior-only models.")
    return model


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pad_whole_frame(packed: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int]:
    height, width = packed.shape[-2:]
    pad_bottom = (-height) % factor
    pad_right = (-width) % factor
    if not pad_bottom and not pad_right:
        return packed, 0, 0
    return (
        F.pad(packed, (0, pad_right, 0, pad_bottom), mode="reflect"),
        pad_bottom,
        pad_right,
    )


@torch.inference_mode()
def encode(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    decoder_id = _canonical_decoder_id(args.decoder_id)
    raw, metadata = load_raw_uint16(args.input, args.metadata)
    packed = normalize_and_pack_raw(raw, metadata).unsqueeze(0)
    model = load_model(decoder_id, args.checkpoint, device)
    factor = 2 ** int(model.transform_downsample_stages)
    packed, pad_bottom, pad_right = _pad_whole_frame(packed, factor)
    packed = packed.to(device=device, dtype=torch.float32)

    _synchronize(device)
    start = time.perf_counter()
    y = model.analysis_latent(packed)
    z = model.h_a(y)
    z_strings, y_strings, _ = model._compress_latents(y.float(), z.float())
    _synchronize(device)
    elapsed = time.perf_counter() - start
    if len(z_strings) != 1 or len(y_strings) != 1:
        raise RuntimeError("Whole-image quick inference requires batch size one.")
    if not isinstance(y_strings[0], bytes):
        raise RuntimeError("Unexpected contextual y stream for a hyperprior-only model.")

    geometry = {
        "image_height": metadata.height,
        "image_width": metadata.width,
        "cfa_pattern": metadata.cfa_pattern,
        "canonical_cfa_pattern": "RGGB",
        "packed_height": metadata.height // 2,
        "packed_width": metadata.width // 2,
        "packed_pad_bottom": pad_bottom,
        "packed_pad_right": pad_right,
        "y_height": int(y.shape[-2]),
        "y_width": int(y.shape[-1]),
        "z_height": int(z.shape[-2]),
        "z_width": int(z.shape[-1]),
    }
    write_craw(
        args.output,
        decoder_id=decoder_id,
        geometry=geometry,
        capture_metadata=metadata.document,
        streams=(("z", z_strings[0]), ("y", y_strings[0])),
    )
    output = Path(args.output)
    pixels = metadata.height * metadata.width
    file_bytes = output.stat().st_size
    payload_bytes = len(z_strings[0]) + len(y_strings[0])
    print(f"input={args.input}")
    print("inference=whole_image_single_pass tiling=false")
    print(f"decoder_id={decoder_id}")
    print(f"craw={output}")
    print(f"craw_bytes={file_bytes} payload_bytes={payload_bytes}")
    print(f"file_bpp={8.0 * file_bytes / pixels:.6f}")
    print(f"payload_bpp={8.0 * payload_bytes / pixels:.6f}")
    print(f"raw_to_craw_ratio={Path(args.input).stat().st_size / file_bytes:.3f}")
    print(f"encode_seconds={elapsed:.4f}")
    if args.reconstruction:
        decode_craw(
            input_path=output,
            checkpoint_path=args.checkpoint,
            reconstruction_path=args.reconstruction,
            device=device,
        )


@torch.inference_mode()
def decode_craw(
    *,
    input_path: str | Path,
    checkpoint_path: str | Path,
    reconstruction_path: str | Path,
    device: torch.device,
) -> None:
    container = read_craw(input_path)
    geometry = container.header["geometry"]
    model = load_model(container.decoder_id, checkpoint_path, device)
    _synchronize(device)
    start = time.perf_counter()
    decoded_y = model._decompress_latents(
        [container.streams["z"]],
        [container.streams["y"]],
        z_shape=(int(geometry["z_height"]), int(geometry["z_width"])),
        y_shape=(int(geometry["y_height"]), int(geometry["y_width"])),
    )
    reconstruction = model.g_s(decoded_y).float()
    reconstruction = reconstruction[
        ...,
        : int(geometry["image_height"]),
        : int(geometry["image_width"]),
    ]
    reconstruction = native_rgb_from_canonical(
        reconstruction,
        str(geometry["cfa_pattern"]),
    )
    _synchronize(device)
    elapsed = time.perf_counter() - start
    capture_metadata = container.header.get("capture_metadata", {})
    render = capture_metadata.get("render", {})
    if "awb_gains" not in render or "camera_to_srgb" not in render:
        raise ValueError(".craw capture metadata is missing the AWB/CCM render profile.")
    black_values = capture_metadata.get("black_level", [0.0])
    if isinstance(black_values, (int, float)):
        black = float(black_values)
    else:
        black = sum(float(value) for value in black_values) / len(black_values)
    white = float(capture_metadata.get("white_level", 1.0))
    normalized_black = black / max(white, 1.0)
    display_signal = (
        (reconstruction[0] - normalized_black)
        / max(1.0 - normalized_black, 1.0e-8)
    ).clamp(0.0, 1.0)
    display = linear_camera_rgb_to_srgb(
        display_signal,
        gamma=2.2,
        ccm=render["camera_to_srgb"],
        awb_gains=torch.tensor(render["awb_gains"]),
    )
    output = Path(reconstruction_path)
    if output.suffix.lower() != ".png":
        raise ValueError("Rendered reconstruction output must use the .png extension.")
    save_rgb_image(display, output)
    print(
        f"reconstruction={output} shape={(int(display.shape[1]), int(display.shape[2]), 3)} "
        "domain=rendered_srgb"
    )
    print(f"decode_seconds={elapsed:.4f}")


def decode(args: argparse.Namespace) -> None:
    decode_craw(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        reconstruction_path=args.reconstruction,
        device=torch.device(args.device),
    )


def inspect(args: argparse.Namespace) -> None:
    container = read_craw(args.input)
    path = Path(args.input)
    geometry = container.header["geometry"]
    pixels = int(geometry["image_height"]) * int(geometry["image_width"])
    print(f"craw={path}")
    print(f"decoder_id={container.decoder_id}")
    print(f"image={geometry['image_width']}x{geometry['image_height']}")
    print(f"file_bytes={path.stat().st_size} payload_bytes={container.payload_bytes}")
    print(f"file_bpp={8.0 * path.stat().st_size / pixels:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whole-image uint16 Bayer RAW inference with real ERIC rANS bitstreams."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Encode one complete RAW frame.")
    encode_parser.add_argument("--input", default=str(DEFAULT_RAW))
    encode_parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    encode_parser.add_argument("--decoder-id", choices=("ours_m", "ours_s"), default="ours_s")
    encode_parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    encode_parser.add_argument("--output", default=str(DEFAULT_CRAW))
    encode_parser.add_argument("--reconstruction", default=None)
    encode_parser.add_argument("--device", default="cuda")
    encode_parser.set_defaults(function=encode)

    decode_parser = subparsers.add_parser("decode", help="Decode one complete .craw frame.")
    decode_parser.add_argument("--input", default=str(DEFAULT_CRAW))
    decode_parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    decode_parser.add_argument(
        "--reconstruction",
        default=str(DEFAULT_PNG),
    )
    decode_parser.add_argument("--device", default="cuda")
    decode_parser.set_defaults(function=decode)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect and verify a .craw file.")
    inspect_parser.add_argument("--input", default=str(DEFAULT_CRAW))
    inspect_parser.set_defaults(function=inspect)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
