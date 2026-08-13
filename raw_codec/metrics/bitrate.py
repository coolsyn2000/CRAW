from __future__ import annotations

import math

import torch


def estimate_bitrate(
    quantized_latent: torch.Tensor,
    raw_shape: torch.Size | tuple[int, ...],
    mode: str = "fixed",
    fixed_bits_per_symbol: float = 6.0,
    raw_bit_depth: int = 12,
    quant_step_size: float | torch.Tensor = 1.0,
) -> dict[str, torch.Tensor]:
    """Estimate latent bitrate and RAW compression ratio."""
    if len(raw_shape) != 4:
        raise ValueError(f"Expected raw_shape [B, C, H, W], got {raw_shape}.")

    device = quantized_latent.device
    dtype = quantized_latent.dtype if quantized_latent.is_floating_point() else torch.float32
    raw_samples = int(raw_shape[0]) * int(raw_shape[1]) * int(raw_shape[2]) * int(raw_shape[3])
    raw_samples_per_sample = int(raw_shape[1]) * int(raw_shape[2]) * int(raw_shape[3])
    original_bits = float(raw_samples * raw_bit_depth)
    num_symbols = quantized_latent.numel()
    symbols_per_sample = int(quantized_latent[0].numel())
    batch_size = int(quantized_latent.shape[0])
    mode = mode.lower()
    quant_step = torch.as_tensor(quant_step_size, device=device, dtype=dtype)
    if torch.any(quant_step <= 0):
        raise ValueError("quant_step_size must be positive.")

    with torch.no_grad():
        symbols = torch.round((quantized_latent.detach() / quant_step.detach()).flatten())
        unique_symbols, counts = torch.unique(symbols, return_counts=True)
        probs = counts.to(device=device, dtype=dtype) / counts.sum().to(device=device, dtype=dtype).clamp_min(1.0)
        empirical_entropy = -(probs * torch.log2(probs.clamp_min(1e-12))).sum()
        unique_count = torch.tensor(float(unique_symbols.numel()), device=device, dtype=dtype)
        sample_entropies = []
        for sample_symbols in torch.round((quantized_latent.detach() / quant_step.detach()).flatten(1)):
            _, sample_counts = torch.unique(sample_symbols, return_counts=True)
            sample_probs = sample_counts.to(device=device, dtype=dtype) / sample_counts.sum().to(
                device=device,
                dtype=dtype,
            ).clamp_min(1.0)
            sample_entropies.append(-(sample_probs * torch.log2(sample_probs.clamp_min(1e-12))).sum())
        empirical_entropy_per_sample = torch.stack(sample_entropies) if sample_entropies else quantized_latent.new_zeros(0)

    if mode == "fixed":
        bits_per_symbol_per_sample = torch.full(
            (batch_size,),
            float(fixed_bits_per_symbol),
            device=device,
            dtype=dtype,
        )
    elif mode == "empirical":
        bits_per_symbol_per_sample = empirical_entropy_per_sample
    elif mode in {"proxy_gaussian", "gaussian"}:
        scaled = quantized_latent / quant_step
        scaled_flat = scaled.flatten(1)
        centered = scaled_flat - scaled_flat.mean(dim=1, keepdim=True)
        variance = centered.pow(2).mean(dim=1)
        entropy_argument = (2.0 * math.pi * math.e) * (variance + 1.0 / 12.0)
        bits_per_symbol_per_sample = 0.5 * torch.log2(entropy_argument.clamp_min(1e-12)).clamp_min(0.0)
    else:
        raise ValueError(f"Unknown bitrate mode: {mode}")

    bits_per_symbol = bits_per_symbol_per_sample.mean()
    sample_total_bits = bits_per_symbol_per_sample * float(symbols_per_sample)
    sample_bpp = sample_total_bits / float(max(raw_samples_per_sample, 1))
    total_bits = sample_total_bits.sum()
    bpp = sample_bpp.mean()
    compression_ratio = torch.tensor(original_bits, device=device, dtype=dtype) / total_bits.clamp_min(1e-9)

    return {
        "bits_per_symbol": bits_per_symbol,
        "bits_per_symbol_per_sample": bits_per_symbol_per_sample,
        "total_bits": total_bits,
        "bpp": bpp,
        "sample_bpp": sample_bpp,
        "compression_ratio": compression_ratio,
        "estimated_entropy": empirical_entropy,
        "quantized_latent_unique_count": unique_count,
    }
