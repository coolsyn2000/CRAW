#!/usr/bin/env bash
set -euo pipefail

LSDIR_ROOT="${1:-${DATA_ROOT:-${PWD}/data}/LSDIR}"
OUT_DIR="${2:-${LSDIR_ROOT}/HR/train}"

mkdir -p "${OUT_DIR}"

shopt -s nullglob
shards=("${LSDIR_ROOT}"/shard-*.tar.gz)
if (( ${#shards[@]} == 0 )); then
  echo "No HR shards found under ${LSDIR_ROOT}/shard-*.tar.gz" >&2
  exit 1
fi

for shard in "${shards[@]}"; do
  marker="${OUT_DIR}/.extracted_$(basename "${shard}").ok"
  if [[ -f "${marker}" ]]; then
    echo "skip $(basename "${shard}")"
    continue
  fi
  echo "extract $(basename "${shard}") -> ${OUT_DIR}"
  tar -xzf "${shard}" -C "${OUT_DIR}"
  touch "${marker}"
done

echo "LSDIR HR ready at ${OUT_DIR}"
