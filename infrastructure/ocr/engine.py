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
from rapidfuzz import fuzz

from config.settings import (
    OCR_CLAHE_BINARIZE,
    OCR_CLAHE_CLIP_LIMIT,
    OCR_CLAHE_TILE_GRID_X,
    OCR_CLAHE_TILE_GRID_Y,
    OCR_ENABLE_CLAHE_PREPROCESS,
    OCR_MIN_VALID_CONFIDENCE,
    OCR_RETRY_CONFIDENCE_THRESHOLD,
    TESSERACT_EXE,
)
from domain.models import CardRegion, VoterCard
from infrastructure.ocr.preprocessor import (
    deskew_image,
    enhance_contrast_clahe,
    preprocess_card_roi,
)

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
_AGE_VALUE_RE = re.compile(r"[:\-\.]\s*(\d{1,3})\b", re.IGNORECASE)
_GENDER_VALUE_RE = re.compile(r"[:\-\.]\s*([A-Za-z]+)\b", re.IGNORECASE)
_NAME_VALUE_RE = re.compile(r"[:\-\.]\s*([A-Za-z][A-Za-z .']{1,80})", re.IGNORECASE)

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


def extract_value_fuzzy(
    ocr_text: str,
    target_keyword: str,
    value_pattern: str | re.Pattern,
    threshold: float = 85.0,
) -> str | None:
    """Extract a value from OCR lines using fuzzy keyword anchoring.

    This function mitigates OCR typos on static anchors such as "Name" and
    "Age" by combining fuzzy keyword matching and regex extraction on a
    per-line basis.

    Args:
        ocr_text: Raw OCR text from one card/region.
        target_keyword: Anchor keyword to locate (for example, "Name").
        value_pattern: Regex string or precompiled regex used to extract the
            target value from a matched line.
        threshold: Minimum fuzzy score required to treat a line as anchored.

    Returns:
        Extracted and cleaned value on first successful match, otherwise None.
    """
    if not isinstance(ocr_text, str) or not ocr_text.strip():
        return None
    if not isinstance(target_keyword, str) or not target_keyword.strip():
        return None

    pattern: re.Pattern
    if isinstance(value_pattern, str):
        pattern = re.compile(value_pattern, re.IGNORECASE)
    elif isinstance(value_pattern, re.Pattern):
        pattern = value_pattern
    else:
        raise ValueError(
            "extract_value_fuzzy: 'value_pattern' must be str or re.Pattern"
        )

    # Keep only letters/digits/spaces for stable fuzzy scoring.
    def _norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _anchor_score(target: str, line: str) -> float:
        """
        Compute a conservative fuzzy anchor score with false-positive checks.

        Mitigations:
        1) Penalize huge length disparities (single-token target vs long line).
        2) Penalize matches whose best signal is far from line start, to avoid
           cases like target='name' matching "father's name".
        """
        target_n = _norm(target)
        line_n = _norm(line)
        if not target_n or not line_n:
            return 0.0

        base = float(fuzz.partial_ratio(target_n, line_n))

        len_ratio = len(line_n) / max(1, len(target_n))
        if len_ratio > 4.0:
            base -= min(20.0, (len_ratio - 4.0) * 3.0)

        # Prefix alignment: anchors usually appear near line start.
        line_tokens = line_n.split()
        target_tokens = target_n.split()
        if not line_tokens or not target_tokens:
            return 0.0

        # Score first-token alignment (robust against 1-2 OCR character flips).
        start_tok_score = float(fuzz.ratio(target_tokens[0], line_tokens[0]))
        if start_tok_score < 70.0:
            base -= (70.0 - start_tok_score) * 0.6

        # If exact target phrase exists but starts late, penalize.
        phrase_pos = line_n.find(target_n)
        if phrase_pos > max(2, len(target_n) // 2):
            base -= 12.0

        return max(0.0, min(100.0, base))

    for raw_line in ocr_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        score = _anchor_score(target_keyword, line)
        if score < threshold:
            continue

        match = pattern.search(line)
        if match is None:
            # Anchor found, but extraction missing on this line. Keep scanning.
            continue

        try:
            if match.lastindex and match.lastindex >= 1:
                value = match.group(1)
            else:
                value = match.group(0)
        except (IndexError, AttributeError):
            continue

        if value is None:
            continue

        cleaned = value.strip()
        cleaned = re.sub(r"^[\s|_:\-\.]+", "", cleaned)
        cleaned = re.sub(r"[\s|_:\-\.]+$", "", cleaned)
        if cleaned:
            return cleaned

    return None

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

    # Fuzzy anchor extraction for OCR-typo resilience.
    fuzzy_name = extract_value_fuzzy(
        text,
        target_keyword="Name",
        value_pattern=_NAME_VALUE_RE,
        threshold=86.0,
    )
    fuzzy_age = extract_value_fuzzy(
        text,
        target_keyword="Age",
        value_pattern=_AGE_VALUE_RE,
        threshold=84.0,
    )
    fuzzy_gender = extract_value_fuzzy(
        text,
        target_keyword="Gender",
        value_pattern=_GENDER_VALUE_RE,
        threshold=84.0,
    )

    if fuzzy_name and not re.search(r"\d", fuzzy_name):
        name = fuzzy_name

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
    elif fuzzy_age:
        try:
            age_val = int(fuzzy_age)
        except ValueError:
            pass

    # ── Normalise gender ──────────────────────────────────────────────────────
    gender: str | None = None
    if gender_m:
        gender = _normalise_gender(gender_m.group(1))
    elif fuzzy_gender:
        gender = _normalise_gender(fuzzy_gender)

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

            preprocessed = self._prepare_roi_for_ocr(roi)

            raw_text, avg_conf = self._ocr_text_with_confidence(preprocessed)

            # Retry once on a deskewed ROI when confidence is low.
            if avg_conf < OCR_RETRY_CONFIDENCE_THRESHOLD:
                deskewed_roi = deskew_image(roi)
                preprocessed_retry = self._prepare_roi_for_ocr(deskewed_roi)
                retry_text, retry_conf = self._ocr_text_with_confidence(preprocessed_retry)
                old_conf = avg_conf
                delta = retry_conf - old_conf
                accepted = retry_conf > old_conf

                log.debug(
                    (
                        "page=%d card=%d ocr_retry_attempt "
                        "threshold=%.1f pre=%.1f post=%.1f delta=%.1f accepted=%s"
                    ),
                    page_no,
                    idx,
                    OCR_RETRY_CONFIDENCE_THRESHOLD,
                    old_conf,
                    retry_conf,
                    delta,
                    accepted,
                )

                if retry_conf > avg_conf:
                    raw_text = retry_text
                    avg_conf = retry_conf

            card = _parse_card_text(raw_text, card_index=idx)
            # Attach the source region for audit purposes
            cards.append(
                card.model_copy(
                    update={
                        "region": region,
                        "ocr_confidence": avg_conf,
                    }
                )
            )

        return cards

    @staticmethod
    def _prepare_roi_for_ocr(roi: np.ndarray) -> np.ndarray:
        """Apply selected OCR preprocessing pipeline to a card ROI."""
        if OCR_ENABLE_CLAHE_PREPROCESS:
            try:
                return enhance_contrast_clahe(
                    roi,
                    clip_limit=OCR_CLAHE_CLIP_LIMIT,
                    tile_grid=(OCR_CLAHE_TILE_GRID_X, OCR_CLAHE_TILE_GRID_Y),
                    apply_binarization=OCR_CLAHE_BINARIZE,
                )
            except (ValueError, RuntimeError) as exc:
                log.warning("CLAHE preprocessing failed; falling back to baseline: %s", exc)
        return preprocess_card_roi(roi)

    @staticmethod
    def _ocr_text_with_confidence(image: np.ndarray) -> tuple[str, float]:
        """
        Run OCR and estimate average word-level confidence in [0, 100].

        Returns empty text + 0.0 confidence on OCR exceptions.
        """
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 6")
            text = pytesseract.image_to_string(image, config="--psm 6")
        except Exception as exc:
            log.warning("OCR exception while reading ROI: %s", exc)
            return "", 0.0

        conf_values: list[float] = []
        for raw in data.get("conf", []):
            try:
                conf = float(raw)
            except (TypeError, ValueError):
                continue
            if conf >= OCR_MIN_VALID_CONFIDENCE:
                conf_values.append(conf)

        avg_conf = float(np.mean(conf_values)) if conf_values else 0.0
        return text, avg_conf
