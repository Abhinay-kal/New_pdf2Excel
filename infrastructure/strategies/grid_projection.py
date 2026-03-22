"""
Strategy 1 — GridProjection (fast, pure-NumPy / pure-PIL).

Ported from  chunawin-import/scripts/goa_layout_boxes.py
             (the "Gatekeeper" from Project B).

Algorithm
---------
Projects a binarised page image onto its row axis (horizontal density)
and column axis (vertical density).  Dense runs correspond to the printed
separator lines of the 10-row × 3-column voter card grid.

Centers of those runs are extracted, merged when close, and reduced to
exactly 11 horizontal lines + 4 vertical lines.  Intersecting them
produces 30 CardRegions in deterministic top→bottom, left→right order.

When the density peak count doesn't match (faint or missing lines), the
strategy falls back to percentage-based grid slicing.  It therefore
always produces exactly 30 regions and never raises StrategyError.

This is the **cheapest** strategy (no cv2, no morphology) and should
be attempted first.
"""
from __future__ import annotations

from typing import List

import numpy as np
from PIL.Image import Image as PILImage

from domain.exceptions import StrategyError
from domain.models import CardRegion
from infrastructure.strategies.base import BaseStrategy


class GridProjectionStrategy(BaseStrategy):
    """Fast pixel-density projection strategy (ported from Project B)."""

    name = "grid_projection"

    _EXPECTED_H_LINES: int = 11   # 10 rows → 11 separator lines
    _EXPECTED_V_LINES: int = 4    # 3  cols →  4 separator lines

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def detect_cards(self, page_image: PILImage) -> List[CardRegion]:
        binary = self._binarize(page_image)
        height, width = binary.shape

        row_density = binary.mean(axis=1)
        col_density = binary.mean(axis=0)

        h_lines = self._detect_line_positions(row_density, self._EXPECTED_H_LINES)
        v_lines = self._detect_line_positions(col_density, self._EXPECTED_V_LINES)

        if len(h_lines) != self._EXPECTED_H_LINES or len(v_lines) != self._EXPECTED_V_LINES:
            h_lines, v_lines = self._percentage_fallback(
                width=width,
                height=height,
                h_detected=h_lines,
                v_detected=v_lines,
            )

        h_lines = sorted(h_lines)
        v_lines = sorted(v_lines)

        inset = 2
        regions: List[CardRegion] = []
        for row_idx in range(10):
            y0 = h_lines[row_idx] + inset
            y1 = h_lines[row_idx + 1] - inset
            for col_idx in range(3):
                x0 = v_lines[col_idx] + inset
                x1 = v_lines[col_idx + 1] - inset
                regions.append(
                    CardRegion(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0))
                )

        if len(regions) != 30:
            raise StrategyError(
                self.name,
                f"grid intersection produced {len(regions)} boxes, expected 30",
            )

        return regions

    # ------------------------------------------------------------------ #
    # Private helpers  (all static — pure functions, easy to unit-test)  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _binarize(image: PILImage) -> np.ndarray:
        """Convert to grayscale and apply an adaptive percentile threshold."""
        gray = np.array(image.convert("L"))
        percentile = float(np.percentile(gray, 20))
        # Clamp threshold to a sensible range so very dark/light scans still work
        threshold = min(200.0, max(60.0, percentile + 20.0))
        return gray < threshold   # bool ndarray — True where ink/lines exist

    @staticmethod
    def _group_dense(
        density: np.ndarray, threshold: float
    ) -> list[tuple[int, int]]:
        """Find contiguous runs of density values at or above *threshold*."""
        groups: list[tuple[int, int]] = []
        start: int | None = None
        for idx, val in enumerate(density):
            if val >= threshold and start is None:
                start = idx
            elif val < threshold and start is not None:
                groups.append((start, idx - 1))
                start = None
        if start is not None:
            groups.append((start, len(density) - 1))
        return groups

    @staticmethod
    def _merge_close(positions: list[int], min_gap: int = 5) -> list[int]:
        """Average positions that are within *min_gap* pixels of each other."""
        if not positions:
            return []
        positions = sorted(positions)
        merged: list[list[int]] = [[positions[0]]]
        for pos in positions[1:]:
            if pos - merged[-1][-1] <= min_gap:
                merged[-1].append(pos)
            else:
                merged.append([pos])
        return [int(round(sum(group) / len(group))) for group in merged]

    @staticmethod
    def _reduce_to_expected(positions: list[int], expected: int) -> list[int]:
        """
        Iteratively remove the member of the smallest gap until
        len(positions) == expected.
        """
        positions = sorted(positions)
        while len(positions) > expected and len(positions) >= 2:
            gaps = [
                (i, positions[i + 1] - positions[i])
                for i in range(len(positions) - 1)
            ]
            min_idx, _ = min(gaps, key=lambda g: g[1])
            # Remove the interior point of the smallest gap
            if min_idx == 0:
                remove_at = 1
            elif min_idx == len(positions) - 2:
                remove_at = min_idx
            else:
                remove_at = min_idx + 1
            positions.pop(remove_at)
        return positions

    def _detect_line_positions(
        self, density: np.ndarray, expected: int
    ) -> list[int]:
        """Full pipeline: threshold → runs → centers → merge → reduce."""
        max_d = float(density.max()) if density.size else 0.0
        threshold = max(0.2, max_d * 0.6)
        groups = self._group_dense(density, threshold)
        centers = [int(round((s + e) / 2.0)) for s, e in groups]
        centers = self._merge_close(centers, min_gap=5)
        if len(centers) > expected:
            centers = self._reduce_to_expected(centers, expected)
        return centers

    @staticmethod
    def _percentage_fallback(
        *,
        width: int,
        height: int,
        h_detected: list[int],
        v_detected: list[int],
    ) -> tuple[list[int], list[int]]:
        """
        When density detection misses lines, fall back to evenly-spaced
        percentage-based slicing within the extremes of what was detected.
        """
        if len(v_detected) >= 2:
            left, right = min(v_detected), max(v_detected)
        else:
            left, right = int(width * 0.05), int(width * 0.95)

        if len(h_detected) >= 2:
            top, bottom = min(h_detected), max(h_detected)
        else:
            top, bottom = int(height * 0.25), int(height * 0.95)

        h_lines = [
            int(round(top + (bottom - top) * i / 10.0)) for i in range(11)
        ]
        v_lines = [
            int(round(left + (right - left) * i / 3.0)) for i in range(4)
        ]
        return h_lines, v_lines
