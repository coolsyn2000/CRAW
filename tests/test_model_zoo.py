from __future__ import annotations

import pytest
import torch

from raw_codec.model_zoo import OursM, OursS


@pytest.mark.parametrize(
    ("model_class", "widths"),
    [(OursM, (224, 448, 640)), (OursS, (160, 256, 320))],
)
def test_ours_sender_and_latent_shapes(model_class, widths):
    model = model_class()
    assert tuple(model.g_a.channels) == widths
    assert model.latent_channels == 320
    assert model.hyper_channels == 192
    assert model.use_scctx is False

    with torch.no_grad():
        y = model.analysis_latent(torch.rand(1, 4, 32, 32))
        z = model.h_a(y)
    assert y.shape == (1, 320, 4, 4)
    assert z.shape == (1, 192, 1, 1)


def test_ours_variants_do_not_share_parameters():
    medium = OursM()
    small = OursS()
    medium_decoder = next(medium.g_s.parameters())
    small_decoder = next(small.g_s.parameters())
    assert medium_decoder.data_ptr() != small_decoder.data_ptr()


def test_ours_s_end_to_end_forward():
    model = OursS().eval()
    with torch.no_grad():
        output = model(
            torch.rand(1, 4, 32, 32),
            bitrate_shape=torch.Size((1, 3, 64, 64)),
        )
    assert output["recon_linear_rgb"].shape == (1, 3, 64, 64)
    assert torch.isfinite(output["bitrate"]["bpp"])
