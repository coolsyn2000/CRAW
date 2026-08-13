from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raw_codec import build_model, load_config, model_kind


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public repository surface.")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/tmp/raw-codec-data"))
    args = parser.parse_args()

    os.environ["DATA_ROOT"] = str(Path(args.data_root).expanduser())
    models: dict[str, dict[str, int | str]] = {}
    for path in sorted((ROOT / "configs/models").glob("*.yaml")):
        config = load_config(path)
        model = build_model(config)
        name = str(config["model"]["name"])
        models[name] = {
            "kind": model_kind(config),
            "parameters": parameter_count(model),
        }
        del model
        gc.collect()

    experiments = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "configs/experiments").glob("*.yaml"))
        if load_config(path)
    ]
    payload = {
        "models": models,
        "experiment_configs": experiments,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
