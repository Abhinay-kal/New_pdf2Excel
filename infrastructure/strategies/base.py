"""
Abstract base class for all layout extraction strategies.

Concrete strategies only need to implement ``detect_cards``.
They do NOT call the validator — that is the orchestrator's job.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from PIL.Image import Image as PILImage

from domain.models import CardRegion


class BaseStrategy(ABC):
    """Shared interface and repr for every extraction strategy."""

    #: Human-readable strategy identifier — override in subclasses.
    name: str = "base"

    @abstractmethod
    def detect_cards(self, page_image: PILImage) -> List[CardRegion]:
        """
        Detect voter card bounding boxes on *page_image*.

        Returns:
            List of CardRegion in reading order.

        Raises:
            StrategyError: if the strategy cannot produce any regions.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
