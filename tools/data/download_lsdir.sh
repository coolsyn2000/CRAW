#!/usr/bin/env bash
set -euo pipefail

root="${1:-${DATA_ROOT:-${PWD}/data}/LSDIR}"
include_pattern="${LSDIR_INCLUDE:-shard-*.tar.gz}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install requirements.txt first." >&2
  exit 1
fi

mkdir -p "${root}"
cat <<'EOF'
LSDIR is gated for academic use. Accept its access conditions on Hugging Face
and run `hf auth login` before this command if your token is not configured.
EOF
hf download ofsoundof/LSDIR \
  --include "${include_pattern}" \
  --local-dir "${root}"

echo "LSDIR archives downloaded to ${root}."
echo "Run tools/data/extract_lsdir_hr.sh ${root} next."
