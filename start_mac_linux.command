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

if [[ -d ".venv" ]]; then
  VENV_DIR=".venv"
elif [[ -d "venv" ]]; then
  VENV_DIR="venv"
else
  VENV_DIR=".venv"
  echo "[Electra-Core] Creating virtual environment ($VENV_DIR)..."
  python3 -m venv "$VENV_DIR"
fi

PYTHON_EXE="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "[ERROR] Python executable not found in virtual environment: $PYTHON_EXE"
  exit 1
fi

echo "[Electra-Core] Upgrading pip..."
"$PYTHON_EXE" -m pip install --upgrade pip

echo "[Electra-Core] Installing dependencies..."
"$PYTHON_EXE" -m pip install -r requirements.txt

# Pick PDF automatically from argument, file-picker, or first local .pdf.
if [[ $# -ge 1 ]]; then
  PDF_PATH="$1"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  PDF_PATH="$(osascript -e 'POSIX path of (choose file of type {"com.adobe.pdf"} with prompt "Select voter-roll PDF")' 2>/dev/null || true)"
else
  PDF_PATH=""
  if command -v zenity >/dev/null 2>&1; then
    PDF_PATH="$(zenity --file-selection --title='Select voter-roll PDF' --file-filter='*.pdf' 2>/dev/null || true)"
  fi
fi

if [[ -z "${PDF_PATH:-}" ]]; then
  shopt -s nullglob
  pdf_candidates=("$ROOT_DIR"/*.pdf)
  shopt -u nullglob
  if [[ ${#pdf_candidates[@]} -ge 1 ]]; then
    PDF_PATH="${pdf_candidates[0]}"
    echo "[Electra-Core] No file selected. Using first PDF: $PDF_PATH"
  fi
fi

if [[ -z "${PDF_PATH:-}" ]]; then
  echo "[ERROR] No PDF selected/found. Re-run and choose a PDF file."
  exit 1
fi

if [[ ! -f "$PDF_PATH" ]]; then
  echo "[ERROR] PDF not found: $PDF_PATH"
  exit 1
fi

echo "[Electra-Core] Running extraction for: $PDF_PATH"
"$PYTHON_EXE" main.py "$PDF_PATH"

echo
echo "[Electra-Core] Completed. Press Enter to close."
read -r _
