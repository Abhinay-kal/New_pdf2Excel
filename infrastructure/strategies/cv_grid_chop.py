"""
Strategy 2 — CvGridChop (robust, OpenCV morphological grid detection).

Ported from  PDF_To_Excel/grid_chop.py
             (Project A's most battle-tested variant).

Algorithm
---------
1. Adaptive Gaussian thresholding  (handles uneven illumination / faint ink).
2. Morphological OPEN with scale=60 kernels extracts horizontal and
   vertical line segments independently.
3. Three dilation passes bridge gaps in broken grid lines.
4. Contours of the resulting grid mask are filtered by size (must be at
   least 8% page-width) and sorted into reading order using a 50-px
   row-grouping tolerance.

This strategy is slower than GridProjection but more resilient to scans
where the physical ink lines are faint, smudged, or partially missing.
It is the fallback when GridProjection's count fails validation.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from PIL.Image import Image as PILImage

from domain.exceptions import StrategyError
from domain.models import CardRegion
from infrastructure.strategies.base import BaseStrategy


class CvGridChopStrategy(BaseStrategy):
    """
    Adaptive-threshold morphological grid strategy (ported from Project A).
    """

    name = "cv_grid_chop"

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def detect_cards(self, page_image: PILImage) -> List[CardRegion]:
        img_bgr = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)

        thresh = self._preprocess(img_bgr)
        grid_mask = self._detect_grid(thresh)
        contours = self._get_voter_contours(grid_mask, img_bgr.shape)

        if not contours:
            raise StrategyError(
                self.name, "no valid contours found after morphological grid detection"
            )

        sorted_contours = self._sort_contours(contours)

        regions: List[CardRegion] = []
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append(CardRegion(x=x, y=y, w=w, h=h))

        return regions

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
        """
        Adaptive Gaussian thresholding on grayscale.
        Inverted output: ink/lines are white, background is black.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=5,
        )

    @staticmethod
    def _detect_grid(thresh: np.ndarray) -> np.ndarray:
        """
        Isolate the grid skeleton with morphological OPEN + dilation.

        scale=60 means a segment must span at least 1/60th of the page
        dimension — very sensitive to short/broken lines.
        """
        scale = 60
        v_len = max(1, thresh.shape[0] // scale)
        h_len = max(1, thresh.shape[1] // scale)

        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))

        vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ver_kernel, iterations=2)
        horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel, iterations=2)

        # Combine both orientations
        grid = cv2.addWeighted(vertical, 0.5, horizontal, 0.5, 0.0)

        # Bridge gaps in faint/broken lines
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid = cv2.dilate(grid, close_kernel, iterations=3)

        _, grid = cv2.threshold(grid, 0, 255, cv2.THRESH_BINARY)
        return grid

    @staticmethod
    def _get_voter_contours(
        grid_mask: np.ndarray, img_shape: tuple
    ) -> list:
        """
        Keep only contours whose width suggests a voter card
        (≥ 8% of page width, ≤ 90% of page width, height ≥ 35 px).
        """
        contours, _ = cv2.findContours(
            grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        img_h, img_w = img_shape[:2]
        min_width = img_w * 0.08

        valid = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > min_width and h > 35 and w < img_w * 0.9:
                valid.append(cnt)
        return valid

    @staticmethod
    def _sort_contours(contours: list) -> list:
        """
        Sort contours in reading order: top-to-bottom rows, then
        left-to-right within each row.

        Contours within 50 px of each other vertically are grouped into
        the same row — this tolerates slight skew between card rows.
        """
        bboxes = [cv2.boundingRect(c) for c in contours]
        paired = sorted(zip(contours, bboxes), key=lambda p: p[1][1])

        rows: list[list] = []
        current_row: list = []
        last_y: int = 0
        row_threshold = 50

        for i, (cnt, box) in enumerate(paired):
            _, y, _, _ = box
            if i == 0:
                current_row.append((cnt, box))
                last_y = y
            elif abs(y - last_y) <= row_threshold:
                current_row.append((cnt, box))
            else:
                current_row.sort(key=lambda p: p[1][0])
                rows.append(current_row)
                current_row = [(cnt, box)]
                last_y = y

        if current_row:
            current_row.sort(key=lambda p: p[1][0])
            rows.append(current_row)

        return [cnt for row in rows for cnt, _ in row]
