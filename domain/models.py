"""
Pure domain models — no cv2, pandas, or other heavy imports allowed here.

PageType    : classification of a PDF page (METADATA / VOTER_LIST / SUMMARY).
CardRegion  : the raw pixel bounding-box returned by a LayoutStrategy.
VoterCard   : the fully-parsed output after OCR.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional

from pydantic import BaseModel, Field


from enum import Enum

class PageType(str, Enum):
    METADATA = "METADATA"
    VOTER_LIST = "VOTER_LIST"
    SUMMARY = "SUMMARY"


class CardRegion(BaseModel):
    """
    Pixel bounding-box of a single voter card on a page image.

    Coordinates follow the OpenCV convention (origin = top-left).
    """

    x: int = Field(..., ge=0, description="Left edge in pixels")
    y: int = Field(..., ge=0, description="Top edge in pixels")
    w: int = Field(..., gt=0, description="Width in pixels")
    h: int = Field(..., gt=0, description="Height in pixels")

    model_config = {"frozen": True}


class VoterCard(BaseModel):
    """
    Parsed voter record extracted from a single CardRegion.

    ``parse_status`` accumulates human-readable flags for any field that
    could not be confidently extracted (e.g. ``"missing_epic"``).
    These cards are still emitted — they are never silently dropped.
    """

    card_index: int = Field(..., ge=1, le=30, description="1-based index on the page")

    # ── Identification ──────────────────────────────────────────────────────────
    serial_no: Optional[str] = None
    epic_id: Optional[str] = None

    # ── Personal details ────────────────────────────────────────────────────────
    name: Optional[str] = None
    relation_type: Optional[str] = None   # Father / Husband / Mother / Other
    relation_name: Optional[str] = None
    house_no: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None          # "Male" | "Female"

    # ── OCR provenance ──────────────────────────────────────────────────────────
    raw_ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None

    # ── Quality flags ───────────────────────────────────────────────────────────
    parse_status: List[str] = Field(default_factory=list)
    region: Optional[CardRegion] = None
