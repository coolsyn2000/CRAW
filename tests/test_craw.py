from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from raw_codec.io import load_raw_uint16, normalize_and_pack_raw, read_craw, write_craw


def test_craw_round_trip_and_integrity(tmp_path):
    path = tmp_path / "sample.craw"
    geometry = {
        "image_height": 64,
        "image_width": 96,
        "cfa_pattern": "RGGB",
        "y_height": 4,
        "y_width": 6,
        "z_height": 1,
        "z_width": 2,
    }
    write_craw(
        path,
        decoder_id="ours_s",
        geometry=geometry,
        capture_metadata={"iso": 3200},
        streams=(("z", b"hyper"), ("y", b"main-latent")),
    )
    decoded = read_craw(path)
    assert decoded.decoder_id == "ours_s"
    assert decoded.header["geometry"] == geometry
    assert decoded.header["capture_metadata"] == {"iso": 3200}
    assert decoded.streams == {"z": b"hyper", "y": b"main-latent"}

    corrupted = bytearray(path.read_bytes())
    corrupted[-1] ^= 0x01
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="CRC mismatch"):
        read_craw(path)


@pytest.mark.parametrize("include_black", (False, True))
def test_uint16_raw_normalization_and_cfa_canonicalization(tmp_path, include_black):
    raw_path = tmp_path / "frame.raw"
    metadata_path = tmp_path / "frame.json"
    signal = np.arange(24, dtype=np.float32).reshape(4, 6) / 23.0
    codes = np.rint(signal * (1023 - 64) + 64).astype("<u2")
    codes.tofile(raw_path)
    metadata_path.write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "endianness": "little",
                "width": 6,
                "height": 4,
                "cfa_pattern": "GRBG",
                "bit_depth": 10,
                "black_level": [64, 64, 64, 64],
                "white_level": 1023,
                "model_input_include_black_level": include_black,
            }
        ),
        encoding="utf-8",
    )
    loaded, metadata = load_raw_uint16(raw_path, metadata_path)
    packed = normalize_and_pack_raw(loaded, metadata)
    if include_black:
        normalized = codes.astype(np.float32) / 1023.0
    else:
        normalized = (codes.astype(np.float32) - 64.0) / (1023.0 - 64.0)
    expected = torch.from_numpy(normalized)
    expected = torch.flip(expected, dims=(-1,))
    expected = torch.stack(
        (
            expected[0::2, 0::2],
            expected[0::2, 1::2],
            expected[1::2, 0::2],
            expected[1::2, 1::2],
        )
    )
    torch.testing.assert_close(packed, expected)
