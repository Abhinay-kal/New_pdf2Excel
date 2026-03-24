"""
PageProcessor — the Quality-Gated Chain-of-Responsibility orchestrator.

Design
------
Every page is first *classified* (METADATA / VOTER_LIST / SUMMARY) by running
a fast OCR scan on the top 25 % of the rendered image.  Only VOTER_LIST pages
enter the strategy loop.  All others are logged and skipped, producing no
voter data.

For VOTER_LIST pages the processor iterates strategies in priority order.
Each candidate must pass two sequential gates before it is accepted:

  Gate 1 – Forensic count:  exactly 30 CardRegions  (fail-fast, no OCR cost).
  Gate 2 – Quality ratio:   >= QUALITY_RATIO_THRESHOLD of those 30 cards must
                             contain a valid EPIC ID after OCR.  This rejects
                             noise detections from low-confidence strategies.

Strategy priority (callers must supply the list in this order):
    CvGridChopStrategy -> BlobClusteringStrategy
    -> DoubleAnchorStrategy -> GridProjectionStrategy

If every strategy fails either gate the page is placed in ``human_review_queue``
and the batch continues.

The system is:
  - Non-crashing:  exceptions never bubble up and abort the batch.
  - Transparent:   every gate failure is recorded with strategy name + reason.
  - Auditable:     HumanReviewItem carries the full attempt trail.
"""
from __future__ import annotations

import logging
import re
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from PIL.Image import Image as PILImage

from config.settings import DEFAULT_DPI, POPPLER_PATH, TESSERACT_EXE
from domain.exceptions import ForensicValidationError, StrategyError
from domain.interfaces import LayoutStrategy
from domain.models import CardRegion, PageType, VoterCard
from infrastructure.ocr.engine import OcrEngine
from infrastructure.ocr.preprocessor import (
    deskew_image_with_angle,
    preprocess_for_ocr,
)
from infrastructure.strategies import (
    BlobClusteringStrategy,
    CvGridChopStrategy,
    DoubleAnchorStrategy,
    GridProjectionStrategy,
)
from pipeline.validator import LayoutValidator

log = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# ── Quality gate constant ──────────────────────────────────────────────────────
QUALITY_RATIO_THRESHOLD: float = 0.80   # 80 % of cards must carry a valid EPIC ID

# EPIC ID: 3 uppercase letters + 7 digits (India voter roll standard)
_EPIC_RE = re.compile(r"[A-Z]{3}\d{7}", re.ASCII)


def _is_ghost_card(epic_id: object, name: object) -> bool:
    """Return True when both EPIC and name are blank after whitespace removal."""
    # Remove all whitespace/newline characters before emptiness checks.
    epic_clean = "".join(str(epic_id or "").split())
    name_clean = "".join(str(name or "").split())
    return not epic_clean and not name_clean


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class PageResult:
    """Successful outcome for a single VOTER_LIST page."""

    page_no: int
    page_type: PageType
    strategy_used: str
    cards: List[VoterCard]
    validity_ratio: float   # fraction of cards whose EPIC ID passed the regex

    @property
    def card_count(self) -> int:
        return len(self.cards)

    @property
    def flagged_cards(self) -> List[VoterCard]:
        """Cards that have at least one parse_status flag."""
        return [c for c in self.cards if c.parse_status]


@dataclass
class SkippedPageResult:
    """
    A page intentionally skipped because it is not a VOTER_LIST page.
    Stored so callers can audit page-type distribution.
    """

    page_no: int
    page_type: PageType   # METADATA or SUMMARY


@dataclass
class HumanReviewItem:
    """
    A VOTER_LIST page that exhausted all strategies without producing a
    quality-passing result.

    ``attempts`` records each gate failure in order:
      ``"<strategy>: gate_1_count(found=N)"``
      ``"<strategy>: gate_2_quality(ratio=0.65)"``
      ``"<strategy>: strategy_error(<reason>)"``
    """

    page_no: int
    last_error: str
    attempts: List[str] = field(default_factory=list)
    # Best partial extraction captured from gate-2 failures (may be empty when
    # all strategies failed earlier at gate-1 or raised a StrategyError).
    best_cards: List[VoterCard] = field(default_factory=list)
    best_strategy: Optional[str] = None
    best_ratio: float = 0.0


# ── Orchestrator ───────────────────────────────────────────────────────────────

class PageProcessor:
    """
    Quality-gated Chain-of-Responsibility page processor.

    Args:
        strategies:  Priority-ordered LayoutStrategy implementations.
                     Recommended order:
                                             CvGridChopStrategy -> BlobClusteringStrategy
                                                 -> DoubleAnchorStrategy -> GridProjectionStrategy
        ocr_engine:  Shared OcrEngine instance.
        validator:   LayoutValidator (a default instance is created if omitted).
    """

    def __init__(
        self,
        strategies: Optional[List[LayoutStrategy]] = None,
        ocr_engine: Optional[OcrEngine] = None,
        validator: Optional[LayoutValidator] = None,
    ) -> None:
        self._strategies: List[LayoutStrategy] = strategies or [
            CvGridChopStrategy(),
            BlobClusteringStrategy(),
            DoubleAnchorStrategy(),
            GridProjectionStrategy(),
        ]
        if not self._strategies:
            raise ValueError("PageProcessor requires at least one strategy.")
        self._ocr: OcrEngine = ocr_engine or OcrEngine()
        self._validator = validator or LayoutValidator()
        self.human_review_queue: List[HumanReviewItem] = []
        self.skipped_pages: List[SkippedPageResult] = []

    # ------------------------------------------------------------------ #
    # Helper: page classification                                         #
    # ------------------------------------------------------------------ #

    def _classify_page(self, page_image: PILImage) -> PageType:
        """
        Fail-open page classification using lightweight OCR frequency heuristics.

        Rules:
          1) BLANK when page variance is near-zero or OCR yields no text.
          2) VOTER_LIST when repeated voter anchors are detected.
          3) METADATA only when anchor frequency is low and explicit summary
             markers are present.
          4) Default to VOTER_LIST (fail-open).
        """
        page_arr = np.array(page_image)
        if page_arr.size == 0:
            return PageType.BLANK

        gray = page_arr
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        # Circuit breaker: practically flat pages are blank/scanner artifacts.
        if float(np.std(gray)) < 2.0:
            return PageType.BLANK

        h, w = gray.shape[:2]
        if h <= 0 or w <= 0:
            return PageType.BLANK

        # Downscale for faster OCR during classification.
        sampled = cv2.resize(gray, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        preprocessed = preprocess_for_ocr(sampled)

        try:
            text: str = pytesseract.image_to_string(preprocessed, config="--psm 11")
        except Exception as exc:
            log.warning(
                "page classification OCR failed — defaulting to VOTER_LIST: %s", exc
            )
            return PageType.VOTER_LIST

        text = _norm_ocr(text)
        if not text:
            return PageType.BLANK

        voter_anchor_count = sum(text.count(token) for token in _VOTER_ANCHORS)
        if voter_anchor_count >= 5:
            return PageType.VOTER_LIST

        has_metadata_marker = any(_fuzzy_contains(text, m) for m in _META_MARKERS)
        if voter_anchor_count < 5 and has_metadata_marker:
            return PageType.METADATA

        return PageType.VOTER_LIST

    # ------------------------------------------------------------------ #
    # Helper: quality ratio                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calculate_validity_ratio(cards: List[VoterCard]) -> float:
        """
        Fraction of cards whose ``epic_id`` field passes the EPIC regex.

        A ratio below QUALITY_RATIO_THRESHOLD means the strategy fragmented
        the page into noise rectangles rather than genuine voter card cells.

        Args:
            cards: OCR-parsed VoterCard list (any length).

        Returns:
            Float in [0.0, 1.0].  Returns 0.0 for an empty list.
        """
        if not cards:
            return 0.0
        valid = sum(
            1
            for c in cards
            if c.epic_id and _EPIC_RE.fullmatch(c.epic_id)
        )
        return valid / len(cards)

    # ------------------------------------------------------------------ #
    # Main entry: single page                                             #
    # ------------------------------------------------------------------ #

    def process_page(
        self, page_image: PILImage, page_no: int
    ) -> Optional[PageResult]:
        """
        Classify then attempt quality-gated extraction on a single page.

        Returns:
            PageResult  -- VOTER_LIST page that passed both quality gates.
            None        -- page skipped (METADATA/SUMMARY) or all strategies
                          failed; see ``skipped_pages`` and
                          ``human_review_queue`` for details.
        """
        # ── Step 0: Deskew full page once (shared by all downstream stages) ──
        page_arr = np.array(page_image)
        deskewed_arr, skew_angle = deskew_image_with_angle(page_arr)
        if skew_angle != 0.0:
            log.info("page=%d deskew_applied angle_deg=%.2f", page_no, skew_angle)
        else:
            log.debug("page=%d deskew_skipped angle_deg=0.00", page_no)
        page_image = Image.fromarray(deskewed_arr)

        # ── Step 1: Classify ──────────────────────────────────────────────────
        page_type = self._classify_page(page_image)
        log.debug("page=%d classified as %s", page_no, page_type.name)

        if page_type is not PageType.VOTER_LIST:
            log.info(
                "page=%d type=%s — skipped, no extraction attempted",
                page_no,
                page_type.name,
            )
            self.skipped_pages.append(
                SkippedPageResult(page_no=page_no, page_type=page_type)
            )
            return None

        # ── Step 2: Strategy loop (VOTER_LIST only) ───────────────────────────
        attempts: List[str] = []
        last_error = "no strategies configured"
        # Track the best partial result across all strategies (gate-2 failures
        # only; gate-1 failures never reach OCR so produce no cards).
        # CRITICAL FIX: Initialize _best_ratio to -1.0 so the first strategy that
        # passes gate-1 is guaranteed to be saved, even if its EPIC ratio is 0.00.
        # This fixes the "Zero-Ratio Data Wipe" bug where partial data was silently
        # dropped when Tesseract hallucinated entirely on all strategies.
        _best_cards: List[VoterCard] = []
        _best_strategy: Optional[str] = None
        _best_ratio: float = -1.0

        for strategy in self._strategies:
            try:
                regions: List[CardRegion] = strategy.detect_cards(page_image)

                # ── Gate 1: Forensic count ─────────────────────────────────────
                self._validator.validate(regions, page_no)

                # ── Gate 2: Quality ratio — run OCR then measure ───────────────
                cards = self._ocr.extract_cards(page_image, regions, page_no)
                cards = [
                    c for c in cards
                    if not _is_ghost_card(c.epic_id, c.name)
                ]
                self._validator.validate_quality(cards, page_no)
                ratio = self._calculate_validity_ratio(cards)

                if ratio < QUALITY_RATIO_THRESHOLD:
                    last_error = (
                        f"{strategy.name}: quality_ratio={ratio:.2f} "
                        f"< threshold={QUALITY_RATIO_THRESHOLD}"
                    )
                    attempts.append(
                        f"{strategy.name}: gate_2_quality(ratio={ratio:.2f})"
                    )
                    log.warning(
                        "page=%d strategy=%s REJECTED — "
                        "quality_ratio=%.2f threshold=%.2f",
                        page_no,
                        strategy.name,
                        ratio,
                        QUALITY_RATIO_THRESHOLD,
                    )
                    # CRITICAL FIX: Keep the least-bad attempt.
                    # Tie-breaker: if multiple strategies entirely fail the Regex
                    # check (ratio == 0.00), prefer the one that extracted more
                    # physical bounding boxes (cards). This maximizes partial data
                    # preservation for human review.
                    if ratio > _best_ratio or (
                        ratio == _best_ratio and len(cards) > len(_best_cards)
                    ):
                        _best_cards = cards
                        _best_strategy = strategy.name
                        _best_ratio = ratio
                    continue  # try the next strategy

                # ── Both gates passed ──────────────────────────────────────────
                log.info(
                    "page=%d strategy=%s ACCEPTED — "
                    "cards=%d validity=%.0f%% flagged=%d",
                    page_no,
                    strategy.name,
                    len(cards),
                    ratio * 100,
                    sum(1 for c in cards if c.parse_status),
                )
                return PageResult(
                    page_no=page_no,
                    page_type=page_type,
                    strategy_used=strategy.name,
                    cards=cards,
                    validity_ratio=ratio,
                )

            except ForensicValidationError as exc:
                last_error = str(exc)
                attempts.append(
                    f"{strategy.name}: gate_1_count(found={exc.found})"
                )
                log.warning(
                    "page=%d strategy=%s gate_1_FAILED found=%d expected=%d",
                    page_no,
                    strategy.name,
                    exc.found,
                    exc.expected,
                )

            except StrategyError as exc:
                last_error = str(exc)
                attempts.append(f"{strategy.name}: strategy_error({exc.reason})")
                log.warning(
                    "page=%d strategy=%s strategy_error reason=%r",
                    page_no,
                    strategy.name,
                    exc.reason,
                )

            except Exception as exc:  # noqa: BLE001 — intentional catch-all
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append(
                    f"{strategy.name}: unexpected={type(exc).__name__}"
                )
                log.exception(
                    "page=%d strategy=%s raised unexpected exception",
                    page_no,
                    strategy.name,
                )

        # ── All strategies exhausted ────────────────────────────────────────────
        log.error(
            "page=%d VOTER_LIST: all %d strategies failed quality gates — "
            "queued for human review",
            page_no,
            len(self._strategies),
        )
        self.human_review_queue.append(
            HumanReviewItem(
                page_no=page_no,
                last_error=last_error,
                attempts=attempts,
                best_cards=_best_cards,
                best_strategy=_best_strategy,
                best_ratio=_best_ratio,
            )
        )
        return None

    # ------------------------------------------------------------------ #
    # Batch entry: full PDF                                               #
    # ------------------------------------------------------------------ #

    def process_pdf(
        self,
        source: Union[str, Path, List[PILImage]],
        start_page: int = 1,
        dpi: int = DEFAULT_DPI,
    ) -> List[PageResult]:
        """
        Classify and quality-gate every page in the rendered PDF.

        Args:
            source:      Either a path to a PDF file (str or Path) **or** a
                         pre-rendered list of PIL Image objects (e.g. from a
                         prior ``pdf2image.convert_from_path`` call).
            start_page:  1-based page number for the first image.
            dpi:         Rendering DPI — only used when *source* is a path.

        Returns:
            List of PageResult for pages that passed all quality gates.
            Skipped pages -> ``self.skipped_pages``.
            Failed pages  -> ``self.human_review_queue``.
        """
        # ── Resolve source to a list of PIL Images ─────────────────────────────
        if isinstance(source, (str, Path)):
            pdf_path = Path(source)
            if not pdf_path.is_file():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            log.info("Rendering PDF: %s at %d DPI", pdf_path, dpi)
            images: List[PILImage] = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                poppler_path=POPPLER_PATH,
            )
        else:
            images = list(source)  # accept any iterable of PIL Images

        results: List[PageResult] = []
        total = len(images)

        for offset, image in enumerate(images):
            page_no = start_page + offset
            log.debug("page=%d/%d processing…", page_no, total)
            result = self.process_page(image, page_no)
            if result is not None:
                results.append(result)

        log.info(
            "batch complete: voter_list=%d skipped=%d review_queue=%d total=%d",
            len(results),
            len(self.skipped_pages),
            len(self.human_review_queue),
            total,
        )
        return results


def _norm_ocr(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fuzzy_contains(text: str, phrase: str, cutoff: float = 0.82) -> bool:
    if phrase in text:
        return True
    tokens = text.split()
    n = len(phrase.split())
    for i in range(0, max(1, len(tokens) - n + 1)):
        win = " ".join(tokens[i:i+n])
        if difflib.SequenceMatcher(None, win, phrase).ratio() >= cutoff:
            return True
    return False


_META_MARKERS = [
    "summary of electors",
    "no of electors",
    "additions",
    "deletions",
    "details of revision",
]
_SUMMARY_MARKERS = [
    "summary of electors",
    "net electors",
    "male female third gender",
]

_VOTER_ANCHORS = [
    "name",
    "age",
    "gender",
    "photo",
]
