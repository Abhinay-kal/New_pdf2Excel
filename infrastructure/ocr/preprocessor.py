"""
OpenCV image pre-processing helpers for Tesseract OCR.

Goal: maximise character recognition accuracy on scanned voter roll pages.
The pipeline is intentionally conservative — every step is reversible and
can be disabled by the caller if it hurts a particular scan type.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def enhance_contrast_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
    apply_binarization: bool = False,
) -> np.ndarray:
    """Enhance local text contrast using CLAHE for watermark-heavy scans.

    Args:
        image: Input image as a NumPy array. Supports grayscale (HxW),
            3-channel color (HxWx3), or 4-channel color (HxWx4).
        clip_limit: CLAHE clip limit. Higher values increase local contrast
            but can amplify noise.
        tile_grid: CLAHE tile grid size as (cols, rows). Typical values are
            (8, 8) or (16, 16).
        apply_binarization: If True, apply adaptive Gaussian thresholding and
            return a binary (0/255) image.

    Returns:
        Enhanced grayscale image, or binarized image when
        ``apply_binarization`` is True.

    Raises:
        ValueError: If the input is invalid or parameters are out of range.
        RuntimeError: If OpenCV processing fails.
    """
    if image is None:
        raise ValueError("enhance_contrast_clahe: 'image' must not be None")
    if not isinstance(image, np.ndarray):
        raise ValueError("enhance_contrast_clahe: 'image' must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError("enhance_contrast_clahe: 'image' must not be empty")

    if clip_limit <= 0:
        raise ValueError("enhance_contrast_clahe: 'clip_limit' must be > 0")
    if (
        not isinstance(tile_grid, tuple)
        or len(tile_grid) != 2
        or tile_grid[0] <= 0
        or tile_grid[1] <= 0
    ):
        raise ValueError(
            "enhance_contrast_clahe: 'tile_grid' must be a tuple of two positive integers"
        )

    try:
        # Convert to single-channel grayscale with minimal copying.
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            channels = image.shape[2]
            if channels == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                raise ValueError(
                    "enhance_contrast_clahe: 3D images must have 3 or 4 channels"
                )
        else:
            raise ValueError("enhance_contrast_clahe: image must be 2D or 3D")

        # CLAHE expects 8-bit/16-bit single-channel input; cast safely when needed.
        if gray.dtype != np.uint8:
            if np.issubdtype(gray.dtype, np.integer):
                gray = np.clip(gray, 0, 255).astype(np.uint8, copy=False)
            else:
                # For float inputs, map dynamic range to [0, 255] once.
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                gray = gray.astype(np.uint8, copy=False)

        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(int(tile_grid[0]), int(tile_grid[1])),
        )
        gray = clahe.apply(gray)

        # High-throughput denoise: Gaussian 3x3 is much cheaper than bilateral.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if apply_binarization:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15,
                3,
            )

        return gray

    except cv2.error as exc:
        raise RuntimeError(f"enhance_contrast_clahe: OpenCV failure: {exc}") from exc


def is_valid_voter_card_crop(
    crop: np.ndarray,
    expected_ratio: float = 3.0,
    ratio_tolerance: float = 1.0,
) -> bool:
    """Fast circuit-breaker to reject non-text/garbage OCR crops.

    The function applies fail-fast gates in a fixed order to avoid expensive
    OCR on crops that are implausible voter-card regions.

    Args:
        crop: Candidate crop image as a NumPy array (grayscale or color).
        expected_ratio: Expected width/height aspect ratio for voter cards.
        ratio_tolerance: Allowed deviation from ``expected_ratio``.

    Returns:
        True if the crop passes all gates; otherwise False.
    """
    # Guard against null/empty/non-array inputs.
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return False

    if crop.ndim < 2:
        return False

    height, width = crop.shape[:2]

    # Gate 1: absolute minimum dimensions.
    if width < 150 or height < 50:
        return False

    # Gate 2: aspect-ratio plausibility.
    if height == 0:
        return False
    aspect_ratio = float(width) / float(height)
    ratio_min = expected_ratio - ratio_tolerance
    ratio_max = expected_ratio + ratio_tolerance
    if aspect_ratio < ratio_min or aspect_ratio > ratio_max:
        return False

    try:
        # Convert to grayscale only when needed.
        if crop.ndim == 2:
            gray = crop
        elif crop.ndim == 3:
            channels = crop.shape[2]
            if channels == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
            else:
                return False
        else:
            return False

        # Gate 3: low-variance rejection (blank/solid crops).
        _, stddev = cv2.meanStdDev(gray)
        std_value = float(stddev[0, 0]) if stddev.size else 0.0
        if std_value < 10.0:
            return False

        # Gate 4: edge-density range check (blank/logo/noise rejection).
        edges = cv2.Canny(gray, 70, 180)
        active = float(np.count_nonzero(edges))
        total = float(edges.size)
        if total <= 0.0:
            return False
        edge_density = active / total
        if edge_density < 0.01 or edge_density > 0.40:
            return False

    except cv2.error:
        return False

    return True


def _extract_text_block_angles(gray: np.ndarray) -> list[float]:
    """Extract normalized text-block angles suitable for robust median skew."""
    # Invert so text/ink becomes white (foreground) and paper black.
    inv = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Merge neighboring characters into horizontal blocks so minAreaRect sees
    # stable line-level contours rather than noisy per-character contours.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    img_h, img_w = gray.shape[:2]
    img_area = float(img_h * img_w)
    min_area = max(80.0, img_area * 0.00015)
    min_w = max(12.0, img_w * 0.015)
    min_h = max(8.0, img_h * 0.005)

    # Analyze larger contours first; this improves robustness on noisy scans.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    angles: list[float] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_w or h < min_h:
            continue

        # Skip near-square blobs since their orientation is ambiguous.
        aspect = w / float(max(h, 1))
        if aspect < 1.2:
            continue

        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        raw_angle = float(rect[2])

        # ------------------ minAreaRect angle normalization ------------------
        # OpenCV returns angle in [-90, 0) tied to which edge is treated as
        # width vs height. This causes a discontinuity around -45 degrees.
        #
        # Typical cases:
        #   - A nearly horizontal text line may appear with raw angle near 0.
        #   - The same geometry can flip representation and report ~-90.
        #
        # To normalize into a stable true tilt in (-45, +45], we first correct
        # for rectangle orientation and then wrap values beyond +/-45:
        #   1) If rectangle is "taller" (rw < rh), add +90 to move to the
        #      equivalent long-edge orientation.
        #   2) If angle > +45, subtract 90.
        #   3) If angle <= -45, add 90.
        # This keeps all usable text-block angles centered around horizontal.
        if rw < rh:
            raw_angle += 90.0

        norm_angle = raw_angle
        if norm_angle > 45.0:
            norm_angle -= 90.0
        elif norm_angle <= -45.0:
            norm_angle += 90.0

        # Reject pathological values; expected output is strictly (-45, +45].
        if -45.0 < norm_angle <= 45.0:
            angles.append(norm_angle)

    return angles


def estimate_skew_angle(image: np.ndarray) -> float:
    """
    Estimate median page skew angle in degrees.

    Returns 0.0 when skew cannot be estimated reliably.
    """
    if image is None or image.size == 0:
        return 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    angles = _extract_text_block_angles(gray)
    if not angles:
        return 0.0

    median_angle = float(np.median(np.array(angles, dtype=np.float32)))
    if abs(median_angle) < 0.05:
        return 0.0
    return median_angle


def deskew_image_with_angle(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Deskew image and return (deskewed_image, estimated_angle_degrees)."""
    if image is None or image.size == 0:
        return image, 0.0

    angle = estimate_skew_angle(image)
    if angle == 0.0:
        return image, 0.0

    img_h, img_w = image.shape[:2]
    center = (img_w / 2.0, img_h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    border_value = 255 if image.ndim == 2 else (255, 255, 255)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (img_w, img_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated, angle


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Estimate page skew from clustered text contours and rotate to deskew.

    Steps:
      1) grayscale -> invert -> Otsu threshold
      2) horizontal dilation to merge glyphs into text-line blocks
      3) minAreaRect angle extraction on significant contours
      4) median angle aggregation for outlier robustness
      5) affine rotation with white background fill

    Args:
        image: HxW (grayscale) or HxWxC NumPy image.

    Returns:
        Deskewed image with the same shape as input. If angle cannot be
        estimated reliably, returns the original image unchanged.
    """
    deskewed, _ = deskew_image_with_angle(image)
    return deskewed


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
