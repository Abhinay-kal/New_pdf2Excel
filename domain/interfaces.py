"""
Domain protocols (interfaces).

Using ``typing.Protocol`` (structural sub-typing) means concrete strategy
classes do NOT need to import from this module — duck-typing is enough.
Heavy libraries (cv2, numpy) are kept out via TYPE_CHECKING.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

from domain.models import CardRegion


@runtime_checkable
class LayoutStrategy(Protocol):
    """
    Contract every card-detection strategy must satisfy.

    A strategy is responsible for locating the bounding boxes of all
    voter cards on a single page image.  It must NOT perform OCR.
    Validation (card-count check) is handled by the pipeline layer.
    """

    #: Unique human-readable identifier used in logs and reports.
    name: str

    def detect_cards(self, page_image: PILImage) -> List[CardRegion]:
        """
        Detect voter card bounding boxes on *page_image*.

        Args:
            page_image: A PIL ``Image`` of the full rendered page.

        Returns:
            A list of ``CardRegion`` objects in reading order
            (top-to-bottom, left-to-right).

        Raises:
            StrategyError: If the strategy cannot produce any result
                           (e.g. blank page, OpenCV exception).
        """
        ...
