"""
HSV color histogram extraction for team clustering.

Per CLAUDE.md Section 7 (Utilities):
- Extract 8x8x8 HSV histogram (512 bins) from bbox region
- Normalize histogram to sum to 1.0
- Used for Pass 3 K-means team clustering
- Pure extraction (no team assignment logic)
"""

import numpy as np
import cv2
from typing import Optional, List
from ..core.types import BBox, HSVHistogram


def extract_hsv_histogram(
    frame: np.ndarray,
    bbox: BBox,
    bins: tuple = (8, 8, 8),
    normalize: bool = True,
) -> Optional[HSVHistogram]:
    """
    Extract HSV histogram from a bounding box region.

    Per CLAUDE.md Section 4 (Pass 1):
    - 8x8x8 bins = 512 total bins
    - Histogram normalized to sum to 1.0
    - Returns None if bbox is invalid or extraction fails

    Args:
        frame: BGR image (OpenCV format)
        bbox: [x1, y1, x2, y2] format
        bins: (H_bins, S_bins, V_bins) - default (8, 8, 8)
        normalize: Whether to normalize histogram to sum to 1.0

    Returns:
        512-element list (HSV histogram), or None if extraction fails

    Examples:
        >>> frame = cv2.imread('frame.jpg')
        >>> hist = extract_hsv_histogram(frame, [100, 100, 200, 300])
        >>> len(hist)
        512
        >>> abs(sum(hist) - 1.0) < 1e-6  # Normalized
        True
    """
    from .geometry import bbox_width, bbox_height, clip_bbox_to_frame

    # Validate bbox
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    # Clip to frame bounds
    frame_height, frame_width = frame.shape[:2]
    bbox = clip_bbox_to_frame(bbox, frame_width, frame_height)
    x1, y1, x2, y2 = [int(x) for x in bbox]

    # Check if bbox has area after clipping
    if x2 <= x1 or y2 <= y1:
        return None

    # Extract region
    try:
        region = frame[y1:y2, x1:x2]
    except Exception:
        return None

    if region.size == 0:
        return None

    # Convert BGR to HSV
    try:
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    except Exception:
        return None

    # Calculate histogram
    # OpenCV ranges: H [0, 180], S [0, 256], V [0, 256]
    h_bins, s_bins, v_bins = bins
    hist = cv2.calcHist(
        [hsv_region],
        channels=[0, 1, 2],  # H, S, V
        mask=None,
        histSize=[h_bins, s_bins, v_bins],
        ranges=[0, 180, 0, 256, 0, 256],  # OpenCV HSV ranges
    )

    # Flatten to 1D
    hist = hist.flatten()

    # Normalize
    if normalize:
        total = hist.sum()
        if total > 0:
            hist = hist / total
        else:
            # Empty histogram - uniform distribution
            hist = np.ones_like(hist) / len(hist)

    # Convert to list
    return hist.tolist()


def compare_hsv_histograms(hist1: HSVHistogram, hist2: HSVHistogram) -> float:
    """
    Compare two HSV histograms using correlation.

    Per CLAUDE.md Section 5 (Pass 2A):
    - Used for appearance drift detection
    - Correlation metric: higher = more similar
    - Returns value in [-1, 1], where 1 = identical

    Args:
        hist1: First histogram (512-element list)
        hist2: Second histogram (512-element list)

    Returns:
        Correlation coefficient in [-1, 1]

    Examples:
        >>> hist1 = [1/512] * 512  # Uniform
        >>> hist2 = [1/512] * 512  # Uniform
        >>> compare_hsv_histograms(hist1, hist2)
        1.0  # Identical
    """
    if len(hist1) != 512 or len(hist2) != 512:
        raise ValueError("Histograms must be 512 elements (8x8x8 bins)")

    h1 = np.array(hist1, dtype=np.float32)
    h2 = np.array(hist2, dtype=np.float32)

    # Use OpenCV correlation
    correlation = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

    return float(correlation)


def histogram_distance(hist1: HSVHistogram, hist2: HSVHistogram, method: str = "correlation") -> float:
    """
    Calculate distance between two HSV histograms.

    Args:
        hist1: First histogram (512-element list)
        hist2: Second histogram (512-element list)
        method: Distance metric - "correlation", "chi_square", "intersection", "bhattacharyya"

    Returns:
        Distance value (metric-dependent range)

    Examples:
        >>> hist1 = [1/512] * 512
        >>> hist2 = [1/512] * 512
        >>> histogram_distance(hist1, hist2, method="correlation")
        1.0  # Identical
    """
    if len(hist1) != 512 or len(hist2) != 512:
        raise ValueError("Histograms must be 512 elements (8x8x8 bins)")

    h1 = np.array(hist1, dtype=np.float32)
    h2 = np.array(hist2, dtype=np.float32)

    method_map = {
        "correlation": cv2.HISTCMP_CORREL,
        "chi_square": cv2.HISTCMP_CHISQR,
        "intersection": cv2.HISTCMP_INTERSECT,
        "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
    }

    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Use {list(method_map.keys())}")

    distance = cv2.compareHist(h1, h2, method_map[method])

    return float(distance)


def is_histogram_valid(hist: Optional[HSVHistogram]) -> bool:
    """
    Check if a histogram is valid.

    A valid histogram:
    - Is not None
    - Has exactly 512 elements (8x8x8 bins)
    - Is normalized (sums to ~1.0)
    - Has no NaN or infinite values

    Args:
        hist: Histogram to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> is_histogram_valid(None)
        False
        >>> is_histogram_valid([1/512] * 512)
        True
        >>> is_histogram_valid([0.5] * 512)  # Not normalized
        False
    """
    if hist is None:
        return False

    if len(hist) != 512:
        return False

    h = np.array(hist, dtype=np.float32)

    # Check for NaN or inf
    if not np.isfinite(h).all():
        return False

    # Check normalization (should sum to ~1.0)
    total = h.sum()
    if not (0.99 <= total <= 1.01):  # Allow small floating point error
        return False

    return True


def batch_extract_hsv_histograms(
    frame: np.ndarray,
    bboxes: List[BBox],
    bins: tuple = (8, 8, 8),
) -> List[Optional[HSVHistogram]]:
    """
    Extract HSV histograms for multiple bboxes in parallel.

    Args:
        frame: BGR image (OpenCV format)
        bboxes: List of [x1, y1, x2, y2] bboxes
        bins: (H_bins, S_bins, V_bins) - default (8, 8, 8)

    Returns:
        List of histograms (same length as bboxes, None for failed extractions)

    Examples:
        >>> frame = cv2.imread('frame.jpg')
        >>> bboxes = [[100, 100, 200, 300], [300, 100, 400, 300]]
        >>> hists = batch_extract_hsv_histograms(frame, bboxes)
        >>> len(hists)
        2
    """
    histograms = []
    for bbox in bboxes:
        hist = extract_hsv_histogram(frame, bbox, bins=bins, normalize=True)
        histograms.append(hist)
    return histograms


def average_histograms(histograms: List[HSVHistogram]) -> Optional[HSVHistogram]:
    """
    Calculate average of multiple histograms.

    Used for computing representative histogram for a fragment.

    Args:
        histograms: List of valid histograms (512-element lists)

    Returns:
        Average histogram (512-element list), or None if no valid histograms

    Examples:
        >>> hist1 = [1/512] * 512
        >>> hist2 = [1/512] * 512
        >>> avg = average_histograms([hist1, hist2])
        >>> len(avg)
        512
        >>> abs(sum(avg) - 1.0) < 1e-6  # Normalized
        True
    """
    valid_hists = [h for h in histograms if is_histogram_valid(h)]

    if not valid_hists:
        return None

    # Stack and average
    hists_array = np.array(valid_hists, dtype=np.float32)
    avg_hist = hists_array.mean(axis=0)

    # Re-normalize
    total = avg_hist.sum()
    if total > 0:
        avg_hist = avg_hist / total

    return avg_hist.tolist()


def histogram_std(histograms: List[HSVHistogram]) -> float:
    """
    Calculate standard deviation across multiple histograms.

    Used for HSV consistency scoring in Pass 2B.

    Args:
        histograms: List of valid histograms

    Returns:
        Standard deviation (lower = more consistent)

    Examples:
        >>> hist1 = [1/512] * 512
        >>> hist2 = [1/512] * 512
        >>> histogram_std([hist1, hist2])
        0.0  # Identical histograms
    """
    valid_hists = [h for h in histograms if is_histogram_valid(h)]

    if not valid_hists:
        return float('inf')

    if len(valid_hists) == 1:
        return 0.0

    hists_array = np.array(valid_hists, dtype=np.float32)
    std = hists_array.std(axis=0).mean()

    return float(std)
