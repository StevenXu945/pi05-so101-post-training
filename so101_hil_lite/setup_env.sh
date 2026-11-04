#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export GIT_LFS_SKIP_SMUDGE=1
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
