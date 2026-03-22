"""
Electra-Core global constants.

All configuration that changes between deployments lives here.
Override tool paths via environment variables to avoid hard-coding
machine-specific paths.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Forensic constraint ────────────────────────────────────────────────────────
EXPECTED_CARDS_PER_PAGE: int = 30
GRID_ROWS: int = 10
GRID_COLS: int = 3

# ── PDF rendering ──────────────────────────────────────────────────────────────
DEFAULT_DPI: int = 300

# ── External tools (override via env to keep paths portable) ──────────────────
TESSERACT_EXE: str = os.getenv("TESSERACT_EXE", "/opt/homebrew/bin/tesseract")
POPPLER_PATH: str = os.getenv("POPPLER_PATH", "/opt/homebrew/bin")

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
LOG_DIR: Path = PROJECT_ROOT / "logs"
