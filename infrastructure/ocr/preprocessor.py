"""
OpenCV image pre-processing helpers for Tesseract OCR.

Goal: maximise character recognition accuracy on scanned voter roll pages.
The pipeline is intentionally conservative — every step is reversible and
can be disabled by the caller if it hurts a particular scan type.
"""
from __future__ import annotations

import cv2
import numpy as np


def preprocess_card_roi(roi: np.ndarray) -> np.ndarray:
    """
    Deskew, denoise, and binarize a single card region for Tesseract.

    Args:
        roi: HxWx3 BGR (or HxW grayscale) NumPy array of the cropped card.

    Returns:
        Binary (grayscale, 0/255) image suitable for ``pytesseract``.
    """
    # 1. Ensure grayscale
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # 2. Mild denoising — removes salt-and-pepper from scanner noise
    #    fastNlMeansDenoising is slow on large images but fine for a card crop.
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 3. Adaptive binarisation — handles uneven illumination and faint ink
    #    better than global Otsu on scanned pages.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )

    # 4. Morphological close — reconnects broken character strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary
