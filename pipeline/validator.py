"""
Forensic layout validator — the "Gatekeeper" rules engine.

Rule 1 (Fail-Fast, mandatory):
    A page MUST yield exactly EXPECTED_CARDS_PER_PAGE CardRegions before
    OCR is invoked.  If any strategy returns the wrong count, it raises
    ForensicValidationError immediately.  The Orchestrator then tries the
    next strategy in the chain.

This is the strategic check described in the Electra-Core architecture:
- cheap (just a length check — no image processing)
- fast (runs before expensive Tesseract calls)
- strict (zero tolerance on count deviation)
"""
from __future__ import annotations

from typing import List

from config.settings import EXPECTED_CARDS_PER_PAGE
from domain.exceptions import ForensicValidationError
from domain.models import CardRegion


class LayoutValidator:
    """
    Stateless rules engine.  Instantiate once per PageProcessor.

    Only the card-count rule is mandatory.  Future rules (e.g. minimum
    card area, no wildly-overlapping boxes) can be added as additional
    ``_check_*`` methods called from ``validate``.
    """

    def validate(self, regions: List[CardRegion], page_no: int) -> None:
        """
        Assert that forensic constraints are satisfied.

        Args:
            regions:  CardRegions returned by a LayoutStrategy.
            page_no:  1-based page number (used in the exception message).

        Raises:
            ForensicValidationError: if any rule is violated.
        """
        self._check_count(regions, page_no)

    def validate_quality(self, cards, page_no: int) -> None:
        total = len(cards) or 1
        missing_epic = sum(1 for c in cards if not c.epic_id)
        missing_ratio = missing_epic / total
        ok_rows = sum(1 for c in cards if not c.parse_status)

        if missing_ratio > MAX_MISSING_EPIC_RATIO:
            raise ForensicValidationError(
                page_no=page_no,
                found=0,
                expected=30
            )
        if ok_rows < MIN_OK_ROWS:
            raise ForensicValidationError(
                page_no=page_no,
                found=ok_rows,
                expected=MIN_OK_ROWS
            )

    # ------------------------------------------------------------------ #
    # Individual rules                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_count(regions: List[CardRegion], page_no: int) -> None:
        """Rule 1: exact card count."""
        if len(regions) != EXPECTED_CARDS_PER_PAGE:
            raise ForensicValidationError(
                page_no=page_no,
                found=len(regions),
                expected=EXPECTED_CARDS_PER_PAGE,
            )
