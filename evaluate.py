from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from raw_codec import build_model, load_config, model_kind
from raw_codec.engine import build_validation_loaders
from raw_codec.losses import RateDistortionLoss
from raw_codec.metrics import image_lpips, image_ms_ssim, image_psnr
from raw_codec.utils import load_checkpoint, seed_everything
from train import bitrate_shape, move_batch, target_from_batch


def forward(model, kind: str, batch: dict[str, Any], target: torch.Tensor, config: dict[str, Any]):
    input_key = str(config.get("input", {}).get("key", "noisy_raw"))
    if kind == "restoration":
        return model(batch[input_key]), None
    output = model(
        batch[input_key],
        noise_map=batch.get("noise_map") if bool(config.get("input", {}).get("use_noise_map", False)) else None,
        bitrate_shape=bitrate_shape(batch, target, config),
    )
    return output["recon_linear_rgb"], output


@torch.no_grad()
def evaluate_loader(model, kind, loader, device, config, compute_lpips: bool) -> dict[str, float]:
    renderer = RateDistortionLoss.from_config(config).to(device)
    sums = {"psnr": 0.0, "ms_ssim": 0.0, "lpips": 0.0, "bpp": 0.0}
    counts = {key: 0 for key in sums}
    for batch in loader:
        batch = move_batch(batch, device)
        target = target_from_batch(batch)
        recon, output = forward(model, kind, batch, target, config)
        batch_size = int(target.shape[0])
        sums["psnr"] += float(image_psnr(recon.float(), target.float()).item()) * batch_size
        sums["ms_ssim"] += float(image_ms_ssim(recon.float(), target.float()).item()) * batch_size
        counts["psnr"] += batch_size
        counts["ms_ssim"] += batch_size
        if compute_lpips:
            recon_srgb, target_srgb = renderer.to_srgb(recon.float(), target.float(), batch=batch)
            sums["lpips"] += float(
                image_lpips(
                    recon_srgb,
                    target_srgb,
                    net="alex",
                    domain="camera",
                ).item()
            ) * batch_size
            counts["lpips"] += batch_size
        if output is not None:
            sums["bpp"] += float(output["bitrate"]["bpp"].item()) * batch_size
            counts["bpp"] += batch_size
        elif torch.is_tensor(batch.get("codec_bpp")):
            sums["bpp"] += float(batch["codec_bpp"].float().mean().item()) * batch_size
            counts["bpp"] += batch_size
    return {
        key: sums[key] / counts[key] if counts[key] else float("nan")
        for key in sums
    } | {"samples": float(counts["psnr"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed SIDD/LSDIR paper splits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("train", {}).get("seed", 10)))
    device = torch.device(args.device)
    kind = model_kind(config)
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=True)
    if hasattr(model, "update"):
        model.update(force=True)
    model.eval()

    loaders = build_validation_loaders(config, device_type=device.type)
    selected = set(args.dataset)
    if selected:
        unknown = selected - set(loaders)
        if unknown:
            raise KeyError(f"Unknown validation set(s): {sorted(unknown)}")
        loaders = {name: loader for name, loader in loaders.items() if name in selected}

    result = {
        name: evaluate_loader(model, kind, loader, device, config, bool(args.lpips))
        for name, loader in loaders.items()
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=True)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
