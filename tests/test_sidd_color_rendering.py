from __future__ import annotations

from types import SimpleNamespace

import torch

from raw_codec.data.sidd_color import sidd_render_profile_from_metadata
from raw_codec.data.unprocess import linear_camera_rgb_to_srgb


def test_sidd_render_profile_normalizes_green_awb_gain():
    metadata = SimpleNamespace(
        AsShotNeutral=[0.5, 1.0, 0.25],
        ColorMatrix1=torch.eye(3).numpy(),
        ColorMatrix2=torch.eye(3).numpy(),
        CalibrationIlluminant1=17,
        CalibrationIlluminant2=21,
    )

    profile = sidd_render_profile_from_metadata(metadata, cct=5000.0)

    torch.testing.assert_close(
        torch.tensor(profile["awb_gains"]),
        torch.tensor([2.0, 1.0, 4.0]),
    )
    assert torch.tensor(profile["camera_to_srgb"]).shape == (3, 3)


def test_linear_camera_rendering_accepts_per_sample_ccm():
    camera_rgb = torch.tensor(
        [
            [[[0.2]], [[0.3]], [[0.4]]],
            [[[0.2]], [[0.3]], [[0.4]]],
        ]
    )
    matrices = torch.stack(
        [
            torch.eye(3),
            torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ]
    )

    rendered = linear_camera_rgb_to_srgb(
        camera_rgb,
        gamma=1.0,
        ccm=matrices,
    )

    torch.testing.assert_close(rendered[0], camera_rgb[0])
    torch.testing.assert_close(
        rendered[1, :, 0, 0],
        torch.tensor([0.3, 0.2, 0.4]),
    )
