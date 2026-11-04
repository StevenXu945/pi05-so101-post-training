#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt -c constraints-tested.txt
uv pip check --python .venv/bin/python
# Vendor uses RECAP's Gemma patches. Fill its omitted data package from matching release.
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
import gemma
source = Path(gemma.__file__).parent / 'gm/data'
target = Path('vendor/gemma/gemma/gm/data')
if not target.exists():
    shutil.copytree(source, target)
PY
