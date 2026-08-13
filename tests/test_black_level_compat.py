from __future__ import annotations

import torch

from raw_codec.engine.data import build_dataset


def test_target_black_level_is_forwarded_to_synthetic_dataset():
    black_code = 512.0
    white_code = 16383.0
    config = {
        "target": {"include_black_level": True},
        "noise": {
            "enabled": False,
            "black_level": black_code,
            "white_level": white_code,
        },
        "unprocess": {"enabled": False},
        "data": {
            "val": {
                "mode": "random",
                "patch_size": 8,
                "length": 1,
                "simulate_noise": False,
            }
        },
    }

    dataset = build_dataset(config, "val")
    assert dataset.unprocess_cfg["linear_rgb_include_black_level"] is True

    camera_linear = torch.tensor([0.0, 0.25, 0.5, 1.0])
    normalized_black = black_code / white_code
    expected = camera_linear * (1.0 - normalized_black) + normalized_black

    actual = dataset._prepare_linear_rgb_target(camera_linear)
    torch.testing.assert_close(actual, expected)
