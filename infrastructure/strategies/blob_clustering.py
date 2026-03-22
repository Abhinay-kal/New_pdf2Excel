"""
Strategy 3 — BlobClustering (last-resort fallback, grid-free).

Ported from  PDF_To_Excel/blob_detection.py
             (Project A's most resilient variant).

Algorithm
---------
Instead of detecting printed grid lines, this strategy **dilates** the
binarised page with a wide (25×15) rectangular kernel so that the
individual text tokens within each voter card merge into a single blob.
Contours of those blobs are then filtered to the expected width range
of a single voter card column (20–40% of page width).

This strategy works even when:
  - grid lines are completely absent (e.g. photocopied or watermarked pages)
  - the page is skewed beyond what grid morphology can recover

It is the **most resilient** but least precise strategy, and is only
attempted after GridProjection and CvGridChop have both failed validation.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from PIL.Image import Image as PILImage

from domain.exceptions import StrategyError
from domain.models import CardRegion
from infrastructure.strategies.base import BaseStrategy


class BlobClusteringStrategy(BaseStrategy):
    """Dilation blob-clustering fallback strategy (ported from Project A)."""

    name = "blob_clustering"

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def detect_cards(self, page_image: PILImage) -> List[CardRegion]:
        img_bgr = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)

        thresh = self._preprocess(img_bgr)
        boxes = self._get_blobs(thresh, img_bgr.shape)

        if not boxes:
            raise StrategyError(
                self.name,
                "dilation blob detection found no candidate card regions",
            )

        return [CardRegion(x=x, y=y, w=w, h=h) for x, y, w, h in boxes]

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
        """Global Otsu binarisation — simple and fast, good for blobs."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return thresh

    @staticmethod
    def _get_blobs(
        thresh: np.ndarray, img_shape: tuple
    ) -> list[tuple[int, int, int, int]]:
        """
        Dilate the threshold image so text tokens merge into card-sized blobs,
        then extract and filter contours.

        Width filter  : 20–40% of page width  (one of 3 columns ≈ 33%)
        Height filter : > 50 px               (exclude noise)
        Sort order    : row-first (Y // 50), then left-to-right within row
        """
        img_h, img_w = img_shape[:2]

        # (25, 15): connect glyphs that are 25 px apart horizontally
        # and 15 px apart vertically — collapses Name+ID+address into one blob
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_w = img_w * 0.20
        max_w = img_w * 0.40

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if min_w < w < max_w and h > 50:
                boxes.append((x, y, w, h))

        # Stable row-first sort (same as the original blob_detection.py)
        return sorted(boxes, key=lambda b: (b[1] // 50, b[0]))
