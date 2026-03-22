#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[Electra-Core] Starting setup..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 is not installed or not in PATH."
  echo "Install Python 3 and re-run this script."
  exit 1
fi

if [[ ! -f "requirements.txt" ]]; then
  echo "[ERROR] requirements.txt not found in $ROOT_DIR"
  exit 1
fi

if [[ ! -d "venv" ]]; then
  echo "[Electra-Core] Creating virtual environment (venv)..."
  python3 -m venv venv
fi

# shellcheck disable=SC1091
source "venv/bin/activate"

echo "[Electra-Core] Upgrading pip..."
python -m pip install --upgrade pip

echo "[Electra-Core] Installing dependencies..."
python -m pip install -r requirements.txt

echo "[Electra-Core] Launching dashboard..."
python -m streamlit run app.py
