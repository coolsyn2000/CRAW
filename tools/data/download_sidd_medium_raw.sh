#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${DATA_ROOT:-${PWD}/data}/SIDD}"
ARCHIVE="${ROOT}/downloads/SIDD_Medium_Raw.zip"
EXTRACT_ROOT="${ROOT}/medium_raw"
URL="http://130.63.97.225/share/SIDD_Medium_Raw.zip"
EXPECTED_BYTES=21525313012

mkdir -p "${ROOT}/downloads" "${EXTRACT_ROOT}"
wget -c -O "${ARCHIVE}" "${URL}"

actual_bytes="$(stat -c '%s' "${ARCHIVE}")"
if [[ "${actual_bytes}" != "${EXPECTED_BYTES}" ]]; then
  echo "Unexpected SIDD archive size: ${actual_bytes}, expected ${EXPECTED_BYTES}." >&2
  exit 1
fi

unzip -q -n "${ARCHIVE}" -d "${EXTRACT_ROOT}"
wget -q -O "${ROOT}/bayer_patterns.csv" \
  "https://abdokamel.github.io/sidd/files/bayer_patterns.csv"
wget -q -O "${ROOT}/noise_level_functions.csv" \
  "https://abdokamel.github.io/sidd/files/noise_level_functions.csv"

scene_count="$(find "${EXTRACT_ROOT}/SIDD_Medium_Raw/Data" -mindepth 1 -maxdepth 1 -type d | wc -l)"
pair_count="$(find "${EXTRACT_ROOT}/SIDD_Medium_Raw/Data" -type f -name '*_NOISY_RAW_*.MAT' | wc -l)"
echo "SIDD Medium RAW ready: ${scene_count} scenes, ${pair_count} noisy/GT pairs."
