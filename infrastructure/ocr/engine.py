"""
Tesseract OCR engine wrapper.

Responsibilities:
  1. Crop each CardRegion from the full page image.
  2. Pre-process each crop (via preprocessor.py).
  3. Run Tesseract and emit a raw text string.
  4. Parse the text with regex into a structured VoterCard.

Field-level parsing is ported from PDF_To_Excel/grid_chop.py and
hardened with explicit gender normalisation and parse-status flags.
"""
from __future__ import annotations

import logging
import re
from typing import List

import numpy as np
import pytesseract
from PIL.Image import Image as PILImage

from config.settings import TESSERACT_EXE
from domain.models import CardRegion, VoterCard
from infrastructure.ocr.preprocessor import preprocess_card_roi

log = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# ── Field regex patterns ────────────────────────────────────────────────────────
_EPIC_RE = re.compile(r"([A-Z]{3}\d{7})", re.ASCII)
_HOUSE_RE = re.compile(
    r"House\s*Number\s*[:\-\.]\s*([0-9A-Za-z\-/]+)", re.IGNORECASE
)
_AGE_RE = re.compile(r"Age\s*[:\-\.]\s*(\d+)", re.IGNORECASE)
_GENDER_RE = re.compile(r"Gender\s*[:\-\.]\s*([A-Za-z]+)", re.IGNORECASE)
_SERIAL_RE = re.compile(r"^\s*(\d{1,3})\s*$")

# Words that indicate we are inside a header row, not a voter card
_HEADER_KEYWORDS = frozenset(
    {
        "assembly constituency",
        "part no",
        "namerole",
        "relative name",
        "house number",
        "photo",
        "available",
        "deleted",
        "section",
    }
)

_EPIC_TOLERANT_RE = re.compile(
    r"\b([A-Z]{3})\s*[-: ]?\s*([0-9OISBZL]{7})\b",
    re.IGNORECASE
)

_DIGIT_FIX = str.maketrans({
    "O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"
})

def _normalize_epic_candidate(prefix: str, suffix: str) -> str | None:
    p = re.sub(r"[^A-Za-z]", "", prefix).upper()
    s = re.sub(r"[^0-9A-Za-z]", "", suffix).upper().translate(_DIGIT_FIX)
    if len(p) == 3 and len(s) == 7 and s.isdigit():
        return f"{p}{s}"
    return None

def _extract_epic(text: str) -> str | None:
    cleaned = text.upper().replace(" ", "")
    strict = re.search(r"\b[A-Z]{3}\d{7}\b", cleaned)
    if strict:
        return strict.group(0)

    for m in _EPIC_TOLERANT_RE.finditer(text.upper()):
        epic = _normalize_epic_candidate(m.group(1), m.group(2))
        if epic:
            return epic
    return None

def _clean_text(raw: str) -> str:
    """Remove OCR noise characters and fix common misreads."""
    text = raw.replace("*", "").replace("?", "").replace("'", "").replace('"', "")
    text = text.replace("Narne", "Name").replace("Nare", "Name")
    return text


def _is_header(text: str) -> bool:
    lower = text.lower()
    return sum(1 for kw in _HEADER_KEYWORDS if kw in lower) >= 2


def _normalise_gender(raw: str) -> str:
    """Map OCR gender noise onto canonical 'Male' / 'Female'."""
    lower = raw.lower()
    if "fem" in lower:
        return "Female"
    if "mal" in lower:
        return "Male"
    return raw.capitalize()


def _parse_card_text(text: str, card_index: int) -> VoterCard:
    """
    Extract structured fields from a single card's raw OCR text.

    Uses a layered approach:
      Layer 1 — regex for unambiguous fields (EPIC, HouseNo, Age, Gender).
      Layer 2 — line scan for Name and Relation (positional heuristics).
    """
    if _is_header(text):
        return VoterCard(
            card_index=card_index,
            raw_ocr_text=text,
            parse_status=["skipped_header"],
        )

    text = _clean_text(text)

    # ── Layer 1: regex extraction ───────────────────────────────────────────────
    epic_m = _EPIC_RE.search(text)
    epic = _extract_epic(text)
    house_m = _HOUSE_RE.search(text)
    age_m = _AGE_RE.search(text)
    gender_m = _GENDER_RE.search(text)
    serial_m = _SERIAL_RE.search(text.splitlines()[0]) if text.strip() else None

    # ── Layer 2: line scan for Name / Relation ─────────────────────────────────
    name: str | None = None
    relation_type: str | None = None
    relation_name: str | None = None

    skip_prefixes = ("House Number", "Age:", "Gender:", "Photo", "Available")
    relation_keywords = ("Father", "Husband", "Mother", "Other")

    for line in (ln.strip() for ln in text.splitlines() if ln.strip()):
        if any(line.startswith(p) for p in skip_prefixes):
            continue

        matched_relation = False
        for rel in relation_keywords:
            if rel in line:
                parts = re.split(r"[:\-]", line, maxsplit=1)
                if len(parts) > 1:
                    relation_type = parts[0].strip()
                    relation_name = parts[1].strip()
                matched_relation = True
                break

        if matched_relation:
            continue

        if "Name" in line and ":" in line:
            parts = re.split(r"[:\-]", line, maxsplit=1)
            if len(parts) > 1:
                name = parts[1].strip()
        elif not name and len(line) > 3 and not re.search(r"\d", line):
            if "Avail" not in line and "Delet" not in line:
                name = line

    # ── Normalise age ─────────────────────────────────────────────────────────
    age_val: int | None = None
    if age_m:
        try:
            age_val = int(age_m.group(1))
        except ValueError:
            pass

    # ── Normalise gender ──────────────────────────────────────────────────────
    gender: str | None = None
    if gender_m:
        gender = _normalise_gender(gender_m.group(1))

    # ── Build parse_status flags ──────────────────────────────────────────────
    parse_status: list[str] = []
    if not name:
        parse_status.append("missing_name")
    if not epic_m:
        parse_status.append("missing_epic")
    if age_val is None:
        parse_status.append("missing_age")
    if not gender:
        parse_status.append("missing_gender")

    return VoterCard(
        card_index=card_index,
        serial_no=serial_m.group(1) if serial_m else None,
        epic_id=epic_m.group(1) if epic_m else None,
        name=name,
        relation_type=relation_type,
        relation_name=relation_name,
        house_no=house_m.group(1) if house_m else None,
        age=age_val,
        gender=gender,
        raw_ocr_text=text,
        parse_status=parse_status,
    )


class OcrEngine:
    """
    Crops each CardRegion from the full-page image, pre-processes it,
    runs Tesseract (PSM 6 — assume uniform block of text), and returns
    a list of parsed VoterCard objects.
    """

    def extract_cards(
        self,
        page_image: PILImage,
        regions: List[CardRegion],
        page_no: int,
    ) -> List[VoterCard]:
        """
        Run OCR on every region and return one VoterCard per region.

        Never raises — individual card failures are captured in
        ``VoterCard.parse_status`` so the batch continues.
        """
        page_arr = np.array(page_image)
        cards: List[VoterCard] = []

        for idx, region in enumerate(regions, start=1):
            # Clamp crop to image bounds with 2-px padding
            y0 = max(0, region.y - 2)
            y1 = min(page_arr.shape[0], region.y + region.h + 2)
            x0 = max(0, region.x - 2)
            x1 = min(page_arr.shape[1], region.x + region.w + 2)

            roi = page_arr[y0:y1, x0:x1]
            if roi.size == 0:
                log.warning("page=%d card=%d empty ROI — skipping", page_no, idx)
                cards.append(
                    VoterCard(
                        card_index=idx,
                        parse_status=["empty_roi"],
                        region=region,
                    )
                )
                continue

            preprocessed = preprocess_card_roi(roi)

            try:
                raw_text: str = pytesseract.image_to_string(
                    preprocessed, config="--psm 6"
                )
            except Exception as exc:
                log.warning(
                    "page=%d card=%d OCR exception: %s", page_no, idx, exc
                )
                raw_text = ""

            card = _parse_card_text(raw_text, card_index=idx)
            # Attach the source region for audit purposes
            cards.append(card.model_copy(update={"region": region}))

        return cards
