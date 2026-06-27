#!/usr/bin/env bash
# Works on DFTWS4 and NERSC Perlmutter (no system pip required).
#
# Usage:
#   bash scripts/install.sh
#
# Prerequisites:
#   - runtime/symmcd_pretrained/mp_20/ must exist (checkpoint files)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "==> SPARC install — PROJECT_ROOT=${PROJECT_ROOT}"

# 1. Bootstrap uv (no pip required)
if ! command -v uv &>/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" ]]; then
    echo "uv not found on PATH after bootstrap" >&2
    exit 1
fi

# 2. Create venv with Python 3.10
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "==> Creating venv at ${VENV_DIR}"
    "${UV_BIN}" venv "${VENV_DIR}" --python 3.10
fi

echo "==> Installing requirements.txt (full one-command setup)"
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" -r "${PROJECT_ROOT}/requirements.txt" \
    --index-strategy unsafe-best-match

mkdir -p "${PROJECT_ROOT}/runtime/wandb" \
         "${PROJECT_ROOT}/runtime/wandb_cache" \
         "${PROJECT_ROOT}/exp_res"

echo ""
echo "==> Install complete."
echo "    Run: source scripts/env.sh"
echo "    Checkpoint must be at: runtime/symmcd_pretrained/mp_20/"
