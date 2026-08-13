#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "Set DATA_ROOT to the directory containing SIDD/ and LSDIR/." >&2
  exit 1
fi
teacher="${repo_root}/checkpoints/ours_m_lambda6e4_100k/previous.pth"
if [[ ! -f "${teacher}" ]]; then
  echo "Missing rate-matched Ours-M teacher: ${teacher}" >&2
  echo "Run scripts/reproduce_ours_m_lambda6e4.sh first." >&2
  exit 1
fi

"${python_bin}" "${repo_root}/train.py" \
  --config "${repo_root}/configs/experiments/ours_s_lambda6e4_align20k.yaml"
"${python_bin}" "${repo_root}/train.py" \
  --config "${repo_root}/configs/experiments/ours_s_lambda6e4_joint180k.yaml"
