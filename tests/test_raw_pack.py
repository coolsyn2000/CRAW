from __future__ import annotations

import torch

from raw_codec.data.raw_pack import pack_bayer, unpack_packed_raw


def test_pack_round_trip():
    raw = torch.arange(2 * 3 * 8 * 10, dtype=torch.float32).reshape(2, 3, 8, 10)
    packed = pack_bayer(raw)
    restored = unpack_packed_raw(packed)
    torch.testing.assert_close(restored, raw)


def test_pack_crops_odd_spatial_edges():
    raw = torch.arange(7 * 9, dtype=torch.float32).reshape(7, 9)
    restored = unpack_packed_raw(pack_bayer(raw))
    torch.testing.assert_close(restored, raw[:6, :8])

