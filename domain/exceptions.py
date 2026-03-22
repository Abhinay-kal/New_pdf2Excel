"""
Domain-level custom exceptions.

Keeping exceptions here (not in infrastructure) means the orchestrator
can catch them without importing cv2 or any heavy library.
"""
from __future__ import annotations


class ElectraCoreError(Exception):
    """Base class for all Electra-Core errors."""


class ForensicValidationError(ElectraCoreError):
    """
    Raised by LayoutValidator when a page does not yield exactly
    EXPECTED_CARDS_PER_PAGE regions.

    Attributes:
        page_no:  1-based page number that failed.
        found:    Number of card regions actually detected.
        expected: Number required (default 30).
    """

    def __init__(self, page_no: int, found: int, expected: int = 30) -> None:
        super().__init__(
            f"Page {page_no}: forensic validation failed — "
            f"expected {expected} cards, got {found}."
        )
        self.page_no = page_no
        self.found = found
        self.expected = expected


class StrategyError(ElectraCoreError):
    """
    Raised by a LayoutStrategy when it cannot produce any card regions
    (e.g. no contours found, image too dark, etc.).

    Attributes:
        strategy_name: The ``name`` attribute of the failing strategy.
        reason:        Human-readable explanation.
    """

    def __init__(self, strategy_name: str, reason: str) -> None:
        super().__init__(f"Strategy '{strategy_name}' failed: {reason}")
        self.strategy_name = strategy_name
        self.reason = reason
