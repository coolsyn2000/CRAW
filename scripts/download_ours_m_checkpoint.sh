#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${1:-"$ROOT_DIR/checkpoints/ours_m_lambda0300_100k/inference_fp16.pth"}
URL="https://drive.usercontent.google.com/download?id=1opwk4OF1jhN2cFblA17XmkFczdeJLh1O&export=download&confirm=t"
EXPECTED_SHA256="4d7c03371dbe19a9cdb74b832373d8f9eabd18c1d98329c788549ad550613a6a"

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
