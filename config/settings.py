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

# ── OCR deskew controls ────────────────────────────────────────────────────────
# If average OCR confidence on a card ROI drops below this, retry OCR once on
# a deskewed version of the ROI.
OCR_RETRY_CONFIDENCE_THRESHOLD: float = 45.0
OCR_MIN_VALID_CONFIDENCE: float = 0.0

# Double-anchor visual debugging. Enable via env var DOUBLE_ANCHOR_DEBUG=1.
DOUBLE_ANCHOR_DEBUG: bool = os.getenv("DOUBLE_ANCHOR_DEBUG", "0").strip().lower() in {
	"1",
	"true",
	"yes",
	"on",
}

# ── External tools (override via env to keep paths portable) ──────────────────
TESSERACT_EXE: str = os.getenv("TESSERACT_EXE", "/opt/homebrew/bin/tesseract")
POPPLER_PATH: str = os.getenv("POPPLER_PATH", "/opt/homebrew/bin")

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
LOG_DIR: Path = PROJECT_ROOT / "logs"
DOUBLE_ANCHOR_DEBUG_DIR: Path = OUTPUT_DIR / "debug" / "double_anchor"
