from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_MAGIC = b"ERICCRW1"
_PREFIX = struct.Struct("<8sI")
_SUPPORTED_DECODERS = {"ours_m", "ours_s"}


@dataclass(frozen=True)
class CrawContainer:
    """Decoded ERIC compressed-RAW container.

    Stream offsets in the JSON header are relative to the start of the binary
    payload. The payload currently contains one factorized-hyperprior stream
    and one Gaussian main-latent stream.
    """

    header: dict[str, Any]
    streams: dict[str, bytes]

    @property
    def decoder_id(self) -> str:
        return str(self.header["decoder_id"])

    @property
    def payload_bytes(self) -> int:
        return sum(len(stream) for stream in self.streams.values())


def _validate_decoder_id(decoder_id: str) -> str:
    normalized = str(decoder_id).lower().replace("-", "_")
    if normalized not in _SUPPORTED_DECODERS:
        raise ValueError(
            f"Unsupported decoder_id={decoder_id!r}; expected one of "
            f"{sorted(_SUPPORTED_DECODERS)}."
        )
    return normalized


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def write_craw(
    path: str | Path,
    *,
    decoder_id: str,
    geometry: Mapping[str, Any],
    capture_metadata: Mapping[str, Any],
    streams: Sequence[tuple[str, bytes]],
) -> dict[str, Any]:
    """Write an ERIC `.craw` file and return its JSON header."""

    decoder_id = _validate_decoder_id(decoder_id)
    names = [str(name) for name, _ in streams]
    if names != ["z", "y"]:
        raise ValueError("ERIC v1 requires streams in the exact order ['z', 'y'].")
    if any(not isinstance(payload, bytes) for _, payload in streams):
        raise TypeError("Every compressed stream must be bytes.")

    offset = 0
    descriptors: list[dict[str, Any]] = []
    for name, payload in streams:
        descriptors.append(
            {
                "name": name,
                "offset": offset,
                "length": len(payload),
                "crc32": f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
            }
        )
        offset += len(payload)

    header: dict[str, Any] = {
        "format": "eric-craw",
        "version": 1,
        "decoder_id": decoder_id,
        "geometry": dict(geometry),
        "capture_metadata": dict(capture_metadata),
        "streams": descriptors,
    }
    encoded_header = _json_bytes(header)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(_PREFIX.pack(_MAGIC, len(encoded_header)))
        handle.write(encoded_header)
        for _, payload in streams:
            handle.write(payload)
    return header


def read_craw(path: str | Path) -> CrawContainer:
    """Read and integrity-check an ERIC `.craw` file."""

    source = Path(path)
    with source.open("rb") as handle:
        prefix = handle.read(_PREFIX.size)
        if len(prefix) != _PREFIX.size:
            raise ValueError(f"Truncated .craw prefix in {source}.")
        magic, header_length = _PREFIX.unpack(prefix)
        if magic != _MAGIC:
            raise ValueError(f"Invalid .craw magic in {source}.")
        encoded_header = handle.read(header_length)
        if len(encoded_header) != header_length:
            raise ValueError(f"Truncated .craw JSON header in {source}.")
        header = json.loads(encoded_header.decode("utf-8"))
        payload = handle.read()

    if header.get("format") != "eric-craw" or int(header.get("version", -1)) != 1:
        raise ValueError(f"Unsupported .craw format/version in {source}.")
    _validate_decoder_id(str(header.get("decoder_id", "")))

    descriptors = header.get("streams")
    if not isinstance(descriptors, list) or [item.get("name") for item in descriptors] != ["z", "y"]:
        raise ValueError("ERIC v1 .craw must contain z and y streams in that order.")

    streams: dict[str, bytes] = {}
    expected_offset = 0
    for descriptor in descriptors:
        name = str(descriptor["name"])
        offset = int(descriptor["offset"])
        length = int(descriptor["length"])
        if offset != expected_offset or length < 0:
            raise ValueError(f"Invalid {name} stream layout in {source}.")
        stream = payload[offset : offset + length]
        if len(stream) != length:
            raise ValueError(f"Truncated {name} stream in {source}.")
        checksum = f"{zlib.crc32(stream) & 0xFFFFFFFF:08x}"
        if checksum != str(descriptor["crc32"]).lower():
            raise ValueError(f"CRC mismatch for {name} stream in {source}.")
        streams[name] = stream
        expected_offset += length
    if expected_offset != len(payload):
        raise ValueError(f"Unexpected trailing bytes in {source}.")
    return CrawContainer(header=header, streams=streams)
