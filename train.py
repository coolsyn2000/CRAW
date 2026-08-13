from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from raw_codec import build_model, load_config, model_kind
from raw_codec.engine import build_loader, build_validation_loaders
from raw_codec.losses import RateDistortionLoss
from raw_codec.metrics import calculate_quality_metrics
from raw_codec.utils import load_checkpoint, save_checkpoint, seed_everything


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def target_from_batch(batch: dict[str, Any]) -> torch.Tensor:
    if "linear_rgb" not in batch:
        raise KeyError("Dataset sample has no linear_rgb target.")
    return batch["linear_rgb"]


def bitrate_shape(batch: dict[str, Any], target: torch.Tensor, config: dict[str, Any]) -> torch.Size:
    denominator = str(config.get("bitrate", {}).get("denominator", "image_pixels")).lower()
    if denominator in {"image", "image_pixels", "hw"}:
        return torch.Size((target.shape[0], 1, target.shape[-2], target.shape[-1]))
    if denominator in {"target", "rgb"}:
        return target.shape
    if denominator in {"input", "raw", "packed_raw"}:
        return batch[str(config.get("input", {}).get("key", "noisy_raw"))].shape
    raise ValueError(f"Unsupported bitrate denominator: {denominator}")


def model_forward(
    model: torch.nn.Module,
    kind: str,
    batch: dict[str, Any],
    target: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    input_cfg = dict(config.get("input", {}))
    input_key = str(input_cfg.get("key", "noisy_raw"))
    model_input = batch[input_key]
    if kind == "restoration":
        return model(model_input), None

    kwargs: dict[str, Any] = {
        "noise_map": batch.get("noise_map") if bool(input_cfg.get("use_noise_map", False)) else None,
        "bitrate_shape": bitrate_shape(batch, target, config),
    }
    if bool(getattr(model, "uses_iso_condition", False)):
        kwargs["iso"] = batch["iso"]
    output = model(model_input, **kwargs)
    return output["recon_linear_rgb"], output


def split_parameters(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    auxiliary_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad and name.endswith(".quantiles")
    }
    main: list[torch.nn.Parameter] = []
    auxiliary: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (auxiliary if name in auxiliary_names else main).append(parameter)
    return main, auxiliary


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes)


def initialize_model(
    model: torch.nn.Module,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[int, int]:
    """Initialize a model from compatible tensors or leading channel slices."""

    init_cfg = dict(train_cfg.get("initialization", {}) or {})
    checkpoint_path = init_cfg.get("checkpoint")
    if not checkpoint_path:
        return 0, 0
    path = Path(str(checkpoint_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {path}")
    mode = str(init_cfg.get("mode", "compatible")).lower().replace("-", "_")
    if mode not in {"compatible", "channel_slice"}:
        raise ValueError("train.initialization.mode must be compatible or channel_slice.")
    prefixes = tuple(str(value) for value in init_cfg.get("include_prefixes", []) if str(value))
    source_checkpoint = torch.load(path, map_location=device, weights_only=False)
    source_state = source_checkpoint.get("model", source_checkpoint)
    target_state = model.state_dict()
    dynamic_suffixes = ("._quantized_cdf", "._offset", "._cdf_length", ".scale_table")
    compatible: dict[str, torch.Tensor] = {}
    loaded = 0
    sliced = 0
    considered = 0
    for name, source in source_state.items():
        if prefixes and not _matches_prefix(name, prefixes):
            continue
        considered += 1
        target = target_state.get(name)
        if target is None:
            continue
        if target.shape == source.shape:
            compatible[name] = source
            loaded += 1
            continue
        can_slice = (
            mode == "channel_slice"
            and source.ndim > 0
            and source.ndim == target.ndim
            and not name.endswith(dynamic_suffixes)
        )
        if can_slice:
            initialized = target.clone()
            overlap = tuple(
                slice(0, min(int(source_size), int(target_size)))
                for source_size, target_size in zip(source.shape, target.shape)
            )
            initialized[overlap] = source[overlap].to(
                device=initialized.device,
                dtype=initialized.dtype,
            )
            compatible[name] = initialized
            loaded += 1
            sliced += 1
    for name, value in target_state.items():
        if name.endswith(dynamic_suffixes) and name not in compatible:
            compatible[name] = value
    if not compatible:
        raise RuntimeError(f"No compatible tensors found in initialization checkpoint {path}.")
    model.load_state_dict(compatible, strict=False)
    print(
        f"initialization={path} loaded={loaded} sliced={sliced} skipped={considered - loaded}",
        flush=True,
    )
    return loaded, considered - loaded


def freeze_parameters(model: torch.nn.Module, train_cfg: dict[str, Any]) -> list[str]:
    prefixes = tuple(
        str(value)
        for value in train_cfg.get("freeze_parameter_prefixes", [])
        if str(value)
    )
    if not prefixes:
        return []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if _matches_prefix(name, prefixes):
            parameter.requires_grad_(False)
            frozen.append(name)
    unmatched = [
        prefix
        for prefix in prefixes
        if not any(_matches_prefix(name, (prefix,)) for name in frozen)
    ]
    if unmatched:
        raise ValueError(f"freeze_parameter_prefixes matched no parameters: {unmatched}")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Parameter freezing left no trainable parameters.")
    print(f"frozen_parameters={len(frozen)} prefixes={list(prefixes)}", flush=True)
    return frozen


def build_teacher(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module | None, str | None, dict[str, Any] | None]:
    distill_cfg = dict(config.get("distillation", {}) or {})
    if not bool(distill_cfg.get("enabled", False)):
        return None, None, None
    teacher_config_path = distill_cfg.get("teacher_config")
    teacher_checkpoint_path = distill_cfg.get("teacher_checkpoint")
    if not teacher_config_path or not teacher_checkpoint_path:
        raise ValueError("Enabled distillation requires teacher_config and teacher_checkpoint.")
    teacher_config = load_config(teacher_config_path)
    teacher_kind = model_kind(teacher_config)
    teacher = build_model(teacher_config).to(device)
    load_checkpoint(teacher_checkpoint_path, teacher, map_location=device, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    print(f"distillation_teacher={teacher_checkpoint_path}", flush=True)
    return teacher, teacher_kind, teacher_config


def distillation_losses(
    student_output: dict[str, Any],
    teacher_output: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    cfg = dict(config.get("distillation", {}) or {})
    latent_key = str(cfg.get("latent_key", "quantized_latent"))
    if latent_key not in student_output or latent_key not in teacher_output:
        raise KeyError(f"Distillation latent {latent_key!r} is missing from model output.")
    student_latent = student_output[latent_key]
    teacher_latent = teacher_output[latent_key].detach().to(student_latent)
    if student_latent.shape != teacher_latent.shape:
        raise ValueError(
            "Student and teacher latent shapes differ: "
            f"{tuple(student_latent.shape)} vs {tuple(teacher_latent.shape)}."
        )
    normalization = str(cfg.get("latent_normalization", "teacher_rms")).lower()
    if normalization in {"teacher_rms", "target_rms", "relative_rms"}:
        scale = teacher_latent.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        scale = scale.clamp_min(float(cfg.get("latent_scale_floor", 0.1))).to(student_latent)
        latent_loss = F.smooth_l1_loss(student_latent / scale, teacher_latent / scale)
    elif normalization in {"none", "off", "identity"}:
        latent_loss = F.smooth_l1_loss(student_latent, teacher_latent)
    else:
        raise ValueError(f"Unsupported latent normalization: {normalization}")
    student_recon = student_output["recon_linear_rgb"]
    teacher_recon = teacher_output["recon_linear_rgb"].detach().to(student_recon)
    reconstruction_loss = F.l1_loss(student_recon, teacher_recon)
    total = (
        float(cfg.get("latent_weight", 0.0)) * latent_loss
        + float(cfg.get("reconstruction_weight", 0.0)) * reconstruction_loss
    )
    return {
        "loss": total,
        "latent": latent_loss.detach(),
        "reconstruction": reconstruction_loss.detach(),
    }


def amp_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    max_steps: int,
    base_lr: float,
    warmup_steps: int,
    start_ratio: float,
    min_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        progress = (step + 1) / float(warmup_steps)
        ratio = start_ratio + (1.0 - start_ratio) * progress
    else:
        span = max(max_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / float(span), 0.0), 1.0)
        ratio = min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
    lr = base_lr * ratio
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    kind: str,
    loader,
    device: torch.device,
    config: dict[str, Any],
    criterion: RateDistortionLoss | None,
    amp_enabled: bool,
    dtype: torch.dtype,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    sample_count = 0
    evaluation_amp = amp_enabled and not bool(getattr(model, "requires_full_precision_evaluation", False))
    started = time.perf_counter()
    for batch in loader:
        batch = move_batch(batch, device)
        target = target_from_batch(batch)
        with autocast_context(device, evaluation_amp, dtype):
            recon, output = model_forward(model, kind, batch, target, config)
            if kind == "joint_codec":
                assert output is not None and criterion is not None
                loss_items = criterion(output, target, batch=batch)
                loss_value = float(loss_items["loss"].item())
                bpp_value = float(loss_items["bpp"].item())
            else:
                loss_value = float(F.l1_loss(recon, target).item())
                codec_bpp = batch.get("codec_bpp")
                bpp_value = float(codec_bpp.float().mean().item()) if torch.is_tensor(codec_bpp) else float("nan")
        quality = calculate_quality_metrics(recon.float(), target.float())
        count = int(target.shape[0])
        values = {
            "loss": loss_value,
            "bpp": bpp_value,
            "psnr": float(quality["psnr"]),
            "ms_ssim": float(quality["ms_ssim"]),
        }
        for key, value in values.items():
            if math.isfinite(value):
                sums[key] = sums.get(key, 0.0) + value * count
        sample_count += count
    if was_training:
        model.train()
    metrics = {key: value / max(sample_count, 1) for key, value in sums.items()}
    metrics["samples"] = float(sample_count)
    metrics["seconds"] = time.perf_counter() - started
    return metrics


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run(config: dict[str, Any], *, resume: str | None, model_only: bool) -> None:
    train_cfg = dict(config.get("train", {}))
    seed_everything(int(train_cfg.get("seed", 10)))
    device_name = str(train_cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(device_name)
    torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get("allow_tf32", False))
    torch.set_float32_matmul_precision(str(train_cfg.get("float32_matmul_precision", "highest")))

    kind = model_kind(config)
    model = build_model(config).to(device)
    if resume is None:
        initialize_model(model, train_cfg, device)
    freeze_parameters(model, train_cfg)
    main_parameters, auxiliary_parameters = split_parameters(model)
    if not main_parameters:
        raise ValueError("No trainable non-auxiliary model parameters.")
    base_lr = float(train_cfg.get("lr", 1.0e-4))
    optimizer = torch.optim.AdamW(
        main_parameters,
        lr=base_lr,
        betas=tuple(float(value) for value in train_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    auxiliary_optimizer = (
        torch.optim.Adam(auxiliary_parameters, lr=float(train_cfg.get("aux_lr", 1.0e-3)))
        if auxiliary_parameters
        else None
    )

    global_step = 0
    if resume:
        checkpoint = load_checkpoint(
            resume,
            model,
            optimizer=None if model_only else optimizer,
            map_location=device,
            strict=True,
        )
        if not model_only:
            global_step = int(checkpoint.get("step", 0))
            if auxiliary_optimizer is not None and "auxiliary_optimizer" in checkpoint:
                auxiliary_optimizer.load_state_dict(checkpoint["auxiliary_optimizer"])
        print(f"resume={resume} model_only={model_only} start_step={global_step}", flush=True)

    criterion = RateDistortionLoss.from_config(config).to(device) if kind == "joint_codec" else None
    teacher, teacher_kind, teacher_config = build_teacher(config, device)
    train_loader = build_loader(config, "train", device_type=device.type)
    validation_loaders = build_validation_loaders(config, device_type=device.type)
    iterator = iter(train_loader)

    max_steps = int(train_cfg.get("max_steps", 1))
    log_every = int(train_cfg.get("log_every", 100))
    val_every = int(train_cfg.get("val_every", 0))
    save_every = int(train_cfg.get("save_every", 0))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    warmup_steps = int(train_cfg.get("lr_warmup_steps", 0))
    start_ratio = float(train_cfg.get("lr_start_ratio", 0.1))
    min_ratio = float(
        train_cfg.get(
            "lr_min_ratio",
            float(train_cfg.get("eta_min", 0.0)) / max(base_lr, 1.0e-12),
        )
    )
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    dtype = amp_dtype(str(train_cfg.get("amp_dtype", "bfloat16")))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and dtype == torch.float16)
    checkpoint_dir = Path(str(train_cfg.get("checkpoint_dir", "checkpoints/run")))
    output_dir = Path(str(train_cfg.get("output_dir", "outputs/run")))
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    while global_step < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        target = target_from_batch(batch)
        current_lr = set_learning_rate(
            optimizer,
            step=global_step,
            max_steps=max_steps,
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            start_ratio=start_ratio,
            min_ratio=min_ratio,
        )
        if criterion is not None:
            criterion.set_step(global_step)
        optimizer.zero_grad(set_to_none=True)
        if auxiliary_optimizer is not None:
            auxiliary_optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp, dtype):
            recon, output = model_forward(model, kind, batch, target, config)
            if kind == "joint_codec":
                assert output is not None and criterion is not None
                loss_items = criterion(output, target, batch=batch)
                loss = loss_items["loss"]
                bpp = float(loss_items["bpp"].item())
                auxiliary_loss = output.get("aux_loss")
            else:
                loss = F.l1_loss(recon, target)
                bpp = float("nan")
                auxiliary_loss = None
            distill_items = None
            if teacher is not None and teacher_kind is not None and teacher_config is not None:
                if output is None:
                    raise ValueError("Distillation is only supported for joint codec models.")
                with torch.no_grad():
                    _, teacher_output = model_forward(
                        teacher,
                        teacher_kind,
                        batch,
                        target,
                        teacher_config,
                    )
                assert teacher_output is not None
                distill_items = distillation_losses(output, teacher_output, config)
                loss = loss + distill_items["loss"]

        auxiliary_term = auxiliary_loss if torch.is_tensor(auxiliary_loss) else loss.new_zeros(())
        scaler.scale(loss + auxiliary_term).backward()
        if max_grad_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(main_parameters, max_grad_norm)
        scaler.step(optimizer)
        if auxiliary_optimizer is not None and torch.is_tensor(auxiliary_loss):
            scaler.step(auxiliary_optimizer)
        scaler.update()

        aux_value = (
            float(auxiliary_loss.detach().item())
            if torch.is_tensor(auxiliary_loss)
            else 0.0
        )

        global_step += 1
        if global_step == 1 or (log_every > 0 and global_step % log_every == 0):
            quality = calculate_quality_metrics(recon.detach().float(), target.detach().float())
            payload = {
                "event": "train",
                "step": global_step,
                "lr": current_lr,
                "loss": float(loss.detach().item()),
                "aux_loss": aux_value,
                "bpp": bpp,
                "psnr": float(quality["psnr"]),
                "ms_ssim": float(quality["ms_ssim"]),
            }
            if distill_items is not None:
                payload["distill_latent"] = float(distill_items["latent"].item())
                payload["distill_reconstruction"] = float(
                    distill_items["reconstruction"].item()
                )
            print(json.dumps(payload, sort_keys=True), flush=True)
            append_jsonl(metrics_path, payload)

        if val_every > 0 and global_step % val_every == 0:
            for name, loader in validation_loaders.items():
                metrics = validate(model, kind, loader, device, config, criterion, use_amp, dtype)
                payload = {"event": "validation", "step": global_step, "dataset": name, **metrics}
                print(json.dumps(payload, sort_keys=True), flush=True)
                append_jsonl(metrics_path, payload)

        if save_every > 0 and global_step % save_every == 0:
            latest = checkpoint_dir / "latest.pth"
            if latest.exists():
                shutil.copy2(latest, checkpoint_dir / "previous.pth")
            extra = (
                {"auxiliary_optimizer": auxiliary_optimizer.state_dict()}
                if auxiliary_optimizer is not None
                else None
            )
            save_checkpoint(
                latest,
                model,
                optimizer=optimizer,
                step=global_step,
                config=config,
                extra_state=extra,
            )
            save_checkpoint(
                checkpoint_dir / f"step_{global_step:07d}.pth",
                model,
                optimizer=optimizer,
                step=global_step,
                config=config,
                extra_state=extra,
            )

    save_checkpoint(
        checkpoint_dir / "latest.pth",
        model,
        optimizer=optimizer,
        step=global_step,
        config=config,
        extra_state=(
            {"auxiliary_optimizer": auxiliary_optimizer.state_dict()}
            if auxiliary_optimizer is not None
            else None
        ),
    )
    print(f"complete step={global_step} checkpoint={checkpoint_dir / 'latest.pth'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train paper-scope RAW restoration/compression models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-model-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="Useful for smoke tests.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = int(args.max_steps)
    if args.device is not None:
        config.setdefault("train", {})["device"] = str(args.device)
    run(config, resume=args.resume, model_only=bool(args.resume_model_only))


if __name__ == "__main__":
    main()
