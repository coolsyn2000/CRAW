# SIDD whole-frame uint16 RAW example

This directory contains one validation frame from SIDD Medium Raw:

- scene: `0104_005_S6_03200_01600_4400_L`;
- frame: `010`;
- camera: Samsung S6 (`SM-G925I`);
- ISO: 3200;
- native CFA: GRBG;
- dimensions: 5328 x 3000 sensor pixels.

`sidd_0104_s6_iso3200_noisy.raw` is headerless, unpacked, little-endian
`uint16`. The JSON sidecar is required to interpret it. SIDD's normalized,
black-level-subtracted signal is mapped to a 10-bit sensor-code domain with
black level 32 and white level 1023 using

```text
code = round(signal * (1023 - 32) + 32).
```

Quick inference preserves that black level with `model_input = code / 1023`.
Its normalized black level closely matches the paper training convention.

SHA-256:

```text
89b7a40b1c4ac2be6c609735c96d3da4fb5e6967aa8364f5d5f9829c45fa94c2  sidd_0104_s6_iso3200_noisy.raw
60828725d0399cacf74e9e3c86c28eb6bda21e0a3218c9517c53f00c52a739e9  sidd_0104_s6_iso3200_noisy.json
```

The sample remains subject to the original SIDD dataset terms and is included
for non-commercial research reproducibility.
