"""Parser helpers for OCR text normalization and VoterCard extraction."""
from __future__ import annotations

import re

import numpy as np
import pytesseract
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from domain.models import CardRegion, RawOcrResult, VoterCard

# ── Field regex patterns ────────────────────────────────────────────────────────
_EPIC_RE = re.compile(r"([A-Z]{3}\d{7})", re.ASCII)
_HOUSE_RE = re.compile(
    r"House\s*Number\s*[:\-\.]\s*((?:[0-9]|[A-Za-z]|/|-)*)",
    re.IGNORECASE,
)
_AGE_RE = re.compile(r"Age\s*[:\-\.]\s*(\d+)", re.IGNORECASE)
_GENDER_RE = re.compile(
    r"Gender\s*[:\-\.]\s*((?:[A-Za-z])+)",
    re.IGNORECASE,
)
_SERIAL_RE = re.compile(r"^\s*(\d{1,3})\s*$")
_AGE_VALUE_RE = re.compile(r"[:\-\.]\s*(\d{1,3})\b", re.IGNORECASE)
_GENDER_VALUE_RE = re.compile(
    r"[:\-\.]\s*((?:[A-Za-z])+?)\b",
    re.IGNORECASE,
)
_NAME_VALUE_RE = re.compile(
    r"[:\-\.]\s*((?:[A-Za-z])(?:[A-Za-z]| |\.|'){1,80})",
    re.IGNORECASE,
)

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
    re.IGNORECASE,
)

_DIGIT_FIX = str.maketrans({
    "O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"
})

# ── EPIC ID hallucination correction tables ────────────────────────────────────
# Prefix fix: translate common digit→letter hallucinations (e.g., 5→S, 0→O)
_PREFIX_FIX = str.maketrans({
    "0": "O", "1": "I", "5": "S", "8": "B"
})

# Suffix fix: translate common letter→digit hallucinations (e.g., O→0, S→5)
_SUFFIX_FIX = str.maketrans({
    "O": "0", "o": "0", "I": "1", "i": "1", "L": "1", "l": "1",
    "S": "5", "s": "5", "B": "8", "b": "8", "Z": "2", "z": "2"
})

_STATIC_ANCHOR_BANK = (
    "Name",
    "Age",
    "Gender",
    "Husband",
    "Father",
    "Mother",
    "House Number",
    "Male",
    "Female",
)

_START_ANCHOR_RE = re.compile(r"^(\s*)([A-Za-z][A-Za-z ]{0,30})(\s*[:\-|].*)$")
_TOKEN_DELIM_RE = re.compile(r"\b([A-Za-z][A-Za-z]{1,30})\s*([:\-|])")


def clean_epic_id(raw_ocr_string: str | None) -> str | None:
    r"""Auto-correct Indian Voter EPIC ID from OCR hallucinations.

    This function deterministically recovers EPIC IDs mangled by Tesseract
    optical confusion errors (e.g., Tesseract reading a `5` as an `S`, or a
    `0` as an `O`). It uses a relaxed regex capture followed by positional
    character translation to guarantee maximum data retention for human review.

    The standard EPIC ID format is: 3 uppercase letters + 7 digits
    (e.g., `ABC1234567` for state code + sequential voter ID).
    """
    if not raw_ocr_string or not isinstance(raw_ocr_string, str):
        return None

    raw_ocr_string = raw_ocr_string.strip()
    if not raw_ocr_string:
        return None

    relaxed_match = re.search(r"[A-Z0-9]{10}", raw_ocr_string.upper())
    if not relaxed_match:
        return None

    captured_10char = relaxed_match.group(0)
    prefix_raw = captured_10char[:3]
    suffix_raw = captured_10char[3:10]
    prefix_fixed = prefix_raw.translate(_PREFIX_FIX)
    suffix_fixed = suffix_raw.translate(_SUFFIX_FIX)
    candidate = prefix_fixed + suffix_fixed

    if re.fullmatch(r"[A-Z]{3}\d{7}", candidate):
        return candidate

    return None


def extract_value_fuzzy(
    ocr_text: str,
    target_keyword: str,
    value_pattern: str | re.Pattern,
    threshold: float = 85.0,
) -> str | None:
    """Extract a value from OCR lines using fuzzy keyword anchoring."""
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

    def _norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _anchor_score(target: str, line: str) -> float:
        target_n = _norm(target)
        line_n = _norm(line)
        if not target_n or not line_n:
            return 0.0

        base = float(fuzz.partial_ratio(target_n, line_n))

        len_ratio = len(line_n) / max(1, len(target_n))
        if len_ratio > 4.0:
            base -= min(20.0, (len_ratio - 4.0) * 3.0)

        line_tokens = line_n.split()
        target_tokens = target_n.split()
        if not line_tokens or not target_tokens:
            return 0.0

        start_tok_score = float(fuzz.ratio(target_tokens[0], line_tokens[0]))
        if start_tok_score < 70.0:
            base -= (70.0 - start_tok_score) * 0.6

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


def extract_with_telemetry(
    image: np.ndarray,
    target_regex: re.Pattern,
    min_confidence: float = 75.0,
) -> dict | None:
    """Extract a regex-matched value and route by token-level OCR confidence."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    if not isinstance(target_regex, re.Pattern):
        raise ValueError("extract_with_telemetry: 'target_regex' must be re.Pattern")

    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        return None

    if not data or not isinstance(data, dict) or not data.get("text"):
        return None

    texts = data.get("text", [])
    confs = data.get("conf", [])
    n = min(len(texts), len(confs))
    if n == 0:
        return None

    valid_tokens: list[tuple[str, float]] = []
    for i in range(n):
        token = str(texts[i] if texts[i] is not None else "").strip()
        if not token:
            continue

        raw_conf = confs[i]
        try:
            conf_val = float(raw_conf)
        except (TypeError, ValueError):
            continue

        if conf_val == -1.0:
            continue

        valid_tokens.append((token, conf_val))

    if not valid_tokens:
        return None

    full_text_parts: list[str] = []
    token_spans: list[tuple[int, int, float]] = []
    cursor = 0
    for idx, (token, conf_val) in enumerate(valid_tokens):
        if idx > 0:
            full_text_parts.append(" ")
            cursor += 1
        start = cursor
        full_text_parts.append(token)
        cursor += len(token)
        token_spans.append((start, cursor, conf_val))

    full_text = "".join(full_text_parts)
    match = target_regex.search(full_text)
    if match is None:
        return None

    extracted = match.group(0).strip()
    if not extracted:
        return None

    m_start, m_end = match.span()
    matched_token_confs = [
        conf
        for t_start, t_end, conf in token_spans
        if t_end > m_start and t_start < m_end
    ]

    if not matched_token_confs:
        compact_tokens = [re.sub(r"\W+", "", tok) for tok, _ in valid_tokens]
        compact_text = "".join(compact_tokens)
        compact_match = target_regex.search(compact_text)
        if compact_match is None:
            return None

        c_start, c_end = compact_match.span()
        compact_cursor = 0
        for idx, token in enumerate(compact_tokens):
            tok_start = compact_cursor
            tok_end = compact_cursor + len(token)
            compact_cursor = tok_end
            if tok_end > c_start and tok_start < c_end:
                matched_token_confs.append(valid_tokens[idx][1])

        extracted = compact_match.group(0).strip() or extracted

    if not matched_token_confs:
        return None

    min_token_conf = float(min(matched_token_confs))
    status = (
        "AUTO_APPROVED"
        if min_token_conf >= float(min_confidence)
        else "FLAGGED_FOR_HUMAN"
    )

    return {
        "value": extracted,
        "confidence": min_token_conf,
        "status": status,
    }


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

    cleaned_epic = clean_epic_id(text)
    if cleaned_epic:
        return cleaned_epic

    return None


def _clean_text(raw: str) -> str:
    text = raw.replace("*", "").replace("?", "").replace("'", "").replace('"', "")
    text = text.replace("Narne", "Name").replace("Nare", "Name")
    return text


def normalize_ocr_anchors(ocr_text: str, threshold: float = 85.0) -> str:
    """Snap noisy OCR structural anchors to a static approved vocabulary."""
    if not isinstance(ocr_text, str) or not ocr_text.strip():
        return ""

    corrected_lines: list[str] = []

    def _best_anchor(candidate: str) -> tuple[str | None, float]:
        best = process.extractOne(
            candidate,
            _STATIC_ANCHOR_BANK,
            scorer=fuzz.WRatio,
        )
        if not best:
            return None, 0.0

        anchor = str(best[0])
        raw_score = float(best[1])

        c_compact = re.sub(r"\s+", "", candidate.strip().lower())
        a_compact = re.sub(r"\s+", "", anchor.strip().lower())
        edit_dist = Levenshtein.distance(c_compact, a_compact)
        dist_score = max(0.0, 100.0 - (15.0 * float(edit_dist)))

        adjusted = max(raw_score, dist_score)
        return anchor, adjusted

    for line in ocr_text.splitlines():
        corrected = line

        start_match = _START_ANCHOR_RE.match(corrected)
        if start_match:
            lead_ws, candidate, tail = start_match.groups()
            candidate_clean = re.sub(r"\s+", " ", candidate.strip())
            anchor, score = _best_anchor(candidate_clean)
            if anchor and score >= float(threshold):
                corrected = f"{lead_ws}{anchor}{tail}"

        def _replace_token(match: re.Match) -> str:
            token = match.group(1)
            delim = match.group(2)
            anchor, score = _best_anchor(token)
            if anchor and score >= float(threshold):
                return f"{anchor}{delim}"
            return match.group(0)

        corrected = _TOKEN_DELIM_RE.sub(_replace_token, corrected)
        corrected_lines.append(corrected)

    return "\n".join(corrected_lines)


def _is_header(text: str) -> bool:
    lower = text.lower()
    return sum(1 for kw in _HEADER_KEYWORDS if kw in lower) >= 2


def _normalise_gender(raw: str) -> str:
    lower = raw.lower()
    if "fem" in lower:
        return "Female"
    if "mal" in lower:
        return "Male"
    return raw.capitalize()


def _parse_card_text(text: str, card_index: int) -> VoterCard:
    if _is_header(text):
        return VoterCard(
            card_index=card_index,
            raw_ocr_text=text,
            parse_status=["skipped_header"],
        )

    text = normalize_ocr_anchors(_clean_text(text))

    epic_m = _EPIC_RE.search(text)
    house_m = _HOUSE_RE.search(text)
    age_m = _AGE_RE.search(text)
    gender_m = _GENDER_RE.search(text)
    serial_m = _SERIAL_RE.search(text.splitlines()[0]) if text.strip() else None

    name: str | None = None
    relation_type: str | None = None
    relation_name: str | None = None

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

    gender: str | None = None
    if gender_m:
        gender = _normalise_gender(gender_m.group(1))
    elif fuzzy_gender:
        gender = _normalise_gender(fuzzy_gender)

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


class VoterCardParser:
    def parse(self, result: RawOcrResult) -> VoterCard:
        card = _parse_card_text(result.raw_text, card_index=result.card_index)
        return card.model_copy(
            update={
                "region": result.region,
                "ocr_confidence": result.confidence,
            }
        )
