from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch


def inference_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
) -> OrderedDict[str, torch.Tensor]:
    exported: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in state_dict.items():
        tensor = value.detach().cpu().contiguous()
        if torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=dtype)
        exported[name] = tensor
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip a training checkpoint to a compact inference-only payload."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--method", required=True)
    parser.add_argument("--paper-step", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    checkpoint: dict[str, Any] = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Expected a model state mapping in {source}.")

    source_step = int(checkpoint.get("step", checkpoint.get("iteration", 0)))
    if args.paper_step is not None and source_step != args.paper_step:
        raise ValueError(
            f"Checkpoint step {source_step} does not match --paper-step {args.paper_step}."
        )
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    payload = {
        "format": "raw-codec-inference-v1",
        "method": args.method,
        "step": source_step,
        "weight_dtype": args.dtype,
        "model": inference_state_dict(state_dict, dtype),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    size_mb = output.stat().st_size / 1_000_000.0
    print(
        f"exported={output} method={args.method} step={source_step} "
        f"dtype={args.dtype} size_mb={size_mb:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
