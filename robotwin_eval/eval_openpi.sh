#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${OPENPI_ROOT}/../RoboTwin}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "RoboTwin" || ! -x "${CONDA_PREFIX:-}/bin/python" ]]; then
    echo "Activate the RoboTwin environment first: conda activate RoboTwin" >&2
    exit 1
fi
if [[ ! -d "${OPENPI_ROOT}/packages/openpi-client/src/openpi_client" ]]; then
    echo "openpi-client not found under OPENPI_ROOT=${OPENPI_ROOT}" >&2
    exit 1
fi

export OPENPI_ROOT ROBOTWIN_ROOT
export PYTHONPATH="${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
exec "${CONDA_PREFIX}/bin/python" "${SCRIPT_DIR}/eval_openpi.py" "$@"
