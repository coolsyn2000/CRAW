#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "Set DATA_ROOT to the directory containing SIDD/ and LSDIR/." >&2
  exit 1
fi

base_config="${repo_root}/configs/experiments/ours_m_base_200k.yaml"
rate_config="${repo_root}/configs/experiments/ours_m_lambda6e4_100k.yaml"
base_checkpoint="${BASE_CHECKPOINT:-${repo_root}/checkpoints/ours_m_base_200k/latest.pth}"
rate_checkpoint="${repo_root}/checkpoints/ours_m_lambda6e4_100k/latest.pth"

if [[ ! -f "${base_checkpoint}" ]]; then
  base_args=(--config "${base_config}")
  if [[ -n "${BASE_MAX_STEPS:-}" ]]; then
    base_args+=(--max-steps "${BASE_MAX_STEPS}")
  fi
  "${python_bin}" "${repo_root}/train.py" "${base_args[@]}"
  base_checkpoint="${repo_root}/checkpoints/ours_m_base_200k/latest.pth"
fi

rate_args=(
  --config "${rate_config}"
  --resume "${base_checkpoint}"
  --resume-model-only
)
if [[ -n "${RATE_MAX_STEPS:-}" ]]; then
  rate_args+=(--max-steps "${RATE_MAX_STEPS}")
fi
"${python_bin}" "${repo_root}/train.py" "${rate_args[@]}"

if [[ "${RUN_EVALUATION:-1}" == "1" ]]; then
  "${python_bin}" "${repo_root}/evaluate.py" \
    --config "${rate_config}" \
    --checkpoint "${rate_checkpoint}" \
    --dataset sidd_real \
    --dataset lsdir_high_texture \
    --output "${repo_root}/outputs/ours_m_lambda6e4_100k/evaluation.json"
fi
