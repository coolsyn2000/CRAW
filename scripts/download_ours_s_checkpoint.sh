#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${1:-"$ROOT_DIR/checkpoints/ours_s_lambda6e4_176k/inference_fp16.pth"}
URL="https://drive.usercontent.google.com/download?id=1_e8Oi28LN9lqUwekWumcBm2Enlu7yIeh&export=download&confirm=t"
EXPECTED_SHA256="31d5966026cedf7773bafe62e67d47cdcd00968b805e1840a2040e00c8acb55c"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

mkdir -p "$(dirname "$OUTPUT")"
if [[ -f "$OUTPUT" ]] && [[ "$(sha256_file "$OUTPUT")" == "$EXPECTED_SHA256" ]]; then
  echo "checkpoint already verified: $OUTPUT"
  exit 0
fi

TEMP_FILE="${OUTPUT}.part"
curl --fail --location --retry 3 "$URL" --output "$TEMP_FILE"
ACTUAL_SHA256=$(sha256_file "$TEMP_FILE")
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "checksum mismatch: expected $EXPECTED_SHA256, got $ACTUAL_SHA256" >&2
  exit 1
fi
mv "$TEMP_FILE" "$OUTPUT"
echo "downloaded and verified: $OUTPUT"
