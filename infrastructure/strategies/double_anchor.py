"""
OCR-driven fallback strategy using a "Double Anchor" technique.

When grid lines and blob morphology fail, this strategy performs page-level
OCR and uses text anchors to estimate voter-card bounding boxes.

Anchor 1: EPIC ID-like tokens (e.g., ABC1234567)
Anchor 2: nearby semantic keywords (Name / Age / Husband / Father / Mother)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import pytesseract
from PIL.Image import Image as PILImage
from pytesseract import Output

from config.settings import DOUBLE_ANCHOR_DEBUG, DOUBLE_ANCHOR_DEBUG_DIR
from domain.exceptions import StrategyError
from domain.models import CardRegion
from infrastructure.strategies.base import BaseStrategy

# Standard EPIC format: 3 letters + 7 digits.
_EPIC_RE = re.compile(r"^[A-Z]{3}[0-9]{7}$", re.ASCII)

# Common OCR-visible English labels near personal details.
_KEYWORDS = {
    "NAME",
    "AGE",
    "HUSBAND",
    "FATHER",
    "MOTHER",
    "WIFE",
}


def _to_bgr_array(page_image: Union[PILImage, np.ndarray]) -> np.ndarray:
    """Convert PIL RGB/RGBA or ndarray image to OpenCV BGR ndarray."""
    if isinstance(page_image, np.ndarray):
        arr = page_image
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr.copy()
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        raise ValueError("Unsupported ndarray image shape for page_image")

    rgb = np.array(page_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _safe_int(value: object, default: int = 0) -> int:
    """Best-effort cast from OCR output values to int."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clip_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int] | None:
    """Clip coordinates to image bounds and reject empty/invalid boxes."""
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _normalize_token(token: str) -> str:
    """Uppercase + strip non-alphanumerics so noisy OCR still matches."""
    return re.sub(r"[^A-Z0-9]", "", token.upper())


def _iter_ocr_words(data: Dict[str, Sequence[object]]) -> Sequence[Dict[str, int | str]]:
    """Yield OCR word records with cleaned geometry and text."""
    n = len(data.get("text", []))
    for i in range(n):
        raw_text = str(data.get("text", [""])[i] or "").strip()
        if not raw_text:
            continue

        x = _safe_int(data.get("left", [0])[i])
        y = _safe_int(data.get("top", [0])[i])
        w = max(0, _safe_int(data.get("width", [0])[i]))
        h = max(0, _safe_int(data.get("height", [0])[i]))
        if w <= 0 or h <= 0:
            continue

        conf = _safe_int(data.get("conf", ["-1"])[i], default=-1)
        yield {
            "text": raw_text,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": conf,
        }


def _dedupe_regions(
    regions: List[Tuple[int, int, int, int]],
    x_threshold: int = 45,
    y_threshold: int = 35,
) -> List[Tuple[int, int, int, int]]:
    """Merge near-identical regions produced by duplicate OCR tokens."""
    deduped: List[Tuple[int, int, int, int]] = []
    for candidate in sorted(regions, key=lambda r: (r[1], r[0])):
        x1, y1, x2, y2 = candidate
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        is_duplicate = False
        for prev in deduped:
            px1, py1, px2, py2 = prev
            pcx = (px1 + px2) // 2
            pcy = (py1 + py2) // 2
            if abs(cx - pcx) <= x_threshold and abs(cy - pcy) <= y_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            deduped.append(candidate)
    return deduped


def _compute_double_anchor_boxes(
    page_image: Union[PILImage, np.ndarray],
) -> List[Tuple[int, int, int, int]]:
    """
    Compute card candidate boxes from EPIC + keyword anchors.

    Returns list of clipped boxes as (x1, y1, x2, y2).
    """
    _, _, _, boxes = _collect_double_anchor_artifacts(page_image)
    return boxes


def _collect_double_anchor_artifacts(
    page_image: Union[PILImage, np.ndarray],
) -> tuple[
    np.ndarray,
    List[Dict[str, int | str]],
    List[Dict[str, int | str]],
    List[Tuple[int, int, int, int]],
]:
    """
    Build all intermediate artifacts needed by both crop generation and debug.

    Returns:
        (page_bgr, epic_anchors, keyword_anchors, clipped_candidate_boxes)
    """
    img_bgr = _to_bgr_array(page_image)
    img_h, img_w = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)

    data = pytesseract.image_to_data(gray, output_type=Output.DICT, config="--psm 6")
    if not data or not data.get("text"):
        return img_bgr, [], [], []

    epic_anchors: List[Dict[str, int | str]] = []
    keyword_anchors: List[Dict[str, int | str]] = []

    for word in _iter_ocr_words(data):
        token = _normalize_token(str(word["text"]))
        if not token:
            continue

        if _EPIC_RE.fullmatch(token):
            epic_anchors.append(word)

        if token in _KEYWORDS:
            keyword_anchors.append(word)

    if not epic_anchors:
        return img_bgr, [], keyword_anchors, []

    left_pad = 50
    top_pad = 50
    right_pad = 400
    bottom_pad = 250

    candidate_boxes: List[Tuple[int, int, int, int]] = []
    for epic in epic_anchors:
        ex = int(epic["x"])
        ey = int(epic["y"])
        ew = int(epic["w"])
        eh = int(epic["h"])

        x1 = ex - left_pad
        y1 = ey - top_pad
        x2 = ex + ew + right_pad
        y2 = ey + eh + bottom_pad

        if keyword_anchors:
            nearest = min(
                keyword_anchors,
                key=lambda k: abs(int(k["x"]) - ex) + abs(int(k["y"]) - ey),
            )
            kx = int(nearest["x"])
            ky = int(nearest["y"])
            if abs(ky - ey) <= 180 and abs(kx - ex) <= 300:
                x1 = min(x1, kx - 70)
                y1 = min(y1, ky - 40)

        clipped = _clip_box(x1, y1, x2, y2, img_w, img_h)
        if clipped is not None:
            candidate_boxes.append(clipped)

    return img_bgr, epic_anchors, keyword_anchors, _dedupe_regions(candidate_boxes)


def _save_debug_overlay(
    page_bgr: np.ndarray,
    epic_anchors: List[Dict[str, int | str]],
    keyword_anchors: List[Dict[str, int | str]],
    boxes: List[Tuple[int, int, int, int]],
    out_dir: Path,
) -> None:
    """Render and persist an overlay for quick visual tuning."""
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = page_bgr.copy()

    # EPIC anchors: red rectangles with label.
    for anchor in epic_anchors:
        x, y, w, h = int(anchor["x"]), int(anchor["y"]), int(anchor["w"]), int(anchor["h"])
        txt = str(anchor["text"])
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"EPIC:{txt}",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # Keyword anchors: blue rectangles with label.
    for anchor in keyword_anchors:
        x, y, w, h = int(anchor["x"]), int(anchor["y"]), int(anchor["w"]), int(anchor["h"])
        txt = str(anchor["text"])
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            canvas,
            f"KW:{txt}",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Final crop boxes: green rectangles with index.
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(
            canvas,
            f"BOX:{idx}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"double_anchor_overlay_{ts}.png"
    cv2.imwrite(str(out_path), canvas)


def crop_via_double_anchor(page_image: Union[PILImage, np.ndarray]) -> List[np.ndarray]:
    """
    Crop voter-card candidates using OCR anchor geometry.

    Args:
        page_image: PIL image or OpenCV-style ndarray of the full page.

    Returns:
        List of cropped image arrays (BGR). Returns empty list when anchors
        cannot be detected.
    """
    img_bgr = _to_bgr_array(page_image)
    boxes = _compute_double_anchor_boxes(page_image)

    crops: List[np.ndarray] = []
    for x1, y1, x2, y2 in boxes:
        # Safe indexing; _clip_box already guarantees non-empty bounds.
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size:
            crops.append(crop)
    return crops


class DoubleAnchorStrategy(BaseStrategy):
    """Fallback strategy that estimates card boxes via OCR text anchors."""

    name = "double_anchor"

    def detect_cards(self, page_image: PILImage) -> List[CardRegion]:
        page_bgr, epic_anchors, keyword_anchors, boxes = _collect_double_anchor_artifacts(page_image)

        if DOUBLE_ANCHOR_DEBUG:
            _save_debug_overlay(
                page_bgr=page_bgr,
                epic_anchors=epic_anchors,
                keyword_anchors=keyword_anchors,
                boxes=boxes,
                out_dir=DOUBLE_ANCHOR_DEBUG_DIR,
            )

        if not boxes:
            raise StrategyError(
                self.name,
                "no EPIC/keyword anchors detected from full-page OCR",
            )

        regions: List[CardRegion] = []
        for x1, y1, x2, y2 in boxes:
            regions.append(CardRegion(x=x1, y=y1, w=x2 - x1, h=y2 - y1))
        return regions
