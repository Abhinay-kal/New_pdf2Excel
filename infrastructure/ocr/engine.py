"""
Tesseract OCR engine wrapper.

Responsibilities:
  1. Crop each CardRegion from the full page image.
  2. Pre-process each crop (via preprocessor.py).
  3. Run Tesseract and emit a raw text string.
  4. Parse the text with regex into a structured VoterCard.

This layer is intentionally dumb: it only crops, preprocesses, and runs
Tesseract. Parsing and business rules live in ``domain.parsers`` and
``domain.rules``.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pytesseract
from PIL.Image import Image as PILImage

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
from domain.models import CardRegion, RawOcrResult
from infrastructure.ocr.preprocessor import (
    deskew_image,
    enforce_card_segmentation,
    enhance_contrast_clahe,
    is_valid_voter_card_crop,
    preprocess_card_roi,
)

log = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
class OcrEngine:
    """
    Crops each CardRegion from the full-page image, pre-processes it,
    runs Tesseract (PSM 6 — assume uniform block of text), and returns
    a list of raw OCR results.
    """

    def extract_raw_text(
        self,
        page_image: PILImage,
        regions: List[CardRegion],
        page_no: int,
    ) -> List[RawOcrResult]:
        """
        Run OCR on every region and return one RawOcrResult per region.

        Never raises — individual card failures are captured in the raw result
        payload so the batch continues.
        """
        page_arr = np.array(page_image)
        cards: List[RawOcrResult] = []

        for idx, region in enumerate(regions, start=1):
            # Clamp crop to image bounds with 2-px padding
            y0 = max(0, region.y - 2)
            y1 = min(page_arr.shape[0], region.y + region.h + 2)
            x0 = max(0, region.x - 2)
            x1 = min(page_arr.shape[1], region.x + region.w + 2)

            roi = page_arr[y0:y1, x0:x1]
            if roi.size == 0:
                log.warning("page=%d card=%d empty ROI — skipping", page_no, idx)
                cards.append(self._empty_raw_result(idx, region))
                continue

            segmented_rois = enforce_card_segmentation([roi], expected_ratio=3.0)
            if len(segmented_rois) > 1:
                log.debug(
                    "page=%d card=%d under-segmentation fallback produced %d slices",
                    page_no,
                    idx,
                    len(segmented_rois),
                )

            for segmented_roi in segmented_rois:
                card_idx = len(cards) + 1
                cards.append(
                    self._extract_raw_result_from_segmented_roi(
                        segmented_roi=segmented_roi,
                        region=region,
                        page_no=page_no,
                        card_idx=card_idx,
                    )
                )

        return cards

    def extract_cards(
        self,
        page_image: PILImage,
        regions: List[CardRegion],
        page_no: int,
    ) -> List[RawOcrResult]:
        return self.extract_raw_text(page_image, regions, page_no)

    @staticmethod
    def _empty_raw_result(card_idx: int, region: CardRegion) -> RawOcrResult:
        return RawOcrResult(
            card_index=card_idx,
            raw_text="",
            confidence=0.0,
            region=region,
            crop_rejected=True,
        )

    def _extract_raw_result_from_segmented_roi(
        self,
        segmented_roi: np.ndarray,
        region: CardRegion,
        page_no: int,
        card_idx: int,
    ) -> RawOcrResult:
        # Fail-open crop gate: tag weak crops, but still attempt OCR to avoid
        # silent attrition when the validity heuristic is overly strict.
        crop_rejected = not is_valid_voter_card_crop(segmented_roi)
        if crop_rejected:
            log.debug(
                "page=%d card=%d crop flagged by circuit breaker (fail-open OCR)",
                page_no,
                card_idx,
            )

        preprocessed = self._prepare_roi_for_ocr(segmented_roi)
        raw_text, avg_conf = self._ocr_text_with_confidence(preprocessed)

        # Retry once on a deskewed ROI when confidence is low.
        if avg_conf < OCR_RETRY_CONFIDENCE_THRESHOLD:
            deskewed_roi = deskew_image(segmented_roi)
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
                card_idx,
                OCR_RETRY_CONFIDENCE_THRESHOLD,
                old_conf,
                retry_conf,
                delta,
                accepted,
            )

            if retry_conf > avg_conf:
                raw_text = retry_text
                avg_conf = retry_conf

        return RawOcrResult(
            card_index=card_idx,
            raw_text=raw_text,
            confidence=avg_conf,
            region=region,
            crop_rejected=crop_rejected,
        )

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
