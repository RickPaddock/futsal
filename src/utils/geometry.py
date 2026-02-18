"""
Geometric operations for bounding boxes and spatial calculations.

Per CLAUDE.md Section 7 (Utilities):
- Bbox operations: area, IoU, centroid, distance
- Multi-layer defense: bbox size validation
- No interpretation or identity logic (pure geometry)
"""

from typing import List, Tuple, Optional
from ..core.types import BBox, Centroid


def bbox_area(bbox: BBox) -> float:
    """
    Calculate the area of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2] format

    Returns:
        Area in pixels^2

    Examples:
        >>> bbox_area([10, 20, 50, 80])
        2400.0
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return max(0.0, width * height)


def bbox_width(bbox: BBox) -> float:
    """
    Calculate the width of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2] format

    Returns:
        Width in pixels
    """
    return max(0.0, bbox[2] - bbox[0])


def bbox_height(bbox: BBox) -> float:
    """
    Calculate the height of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2] format

    Returns:
        Height in pixels
    """
    return max(0.0, bbox[3] - bbox[1])


def bbox_centroid(bbox: BBox) -> Centroid:
    """
    Calculate the centroid of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2] format

    Returns:
        [x, y] centroid coordinates

    Examples:
        >>> bbox_centroid([10, 20, 50, 80])
        [30.0, 50.0]
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return [cx, cy]


def iou(bbox1: BBox, bbox2: BBox) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Per CLAUDE.md: Used for deduplication (threshold 0.55), track splitting.

    Args:
        bbox1: [x1, y1, x2, y2] format
        bbox2: [x1, y1, x2, y2] format

    Returns:
        IoU value in [0, 1]

    Examples:
        >>> iou([0, 0, 10, 10], [5, 5, 15, 15])
        0.142857...  # Overlapping boxes
        >>> iou([0, 0, 10, 10], [20, 20, 30, 30])
        0.0  # No overlap
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Check if there's no intersection
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def centroid_distance(c1: Centroid, c2: Centroid) -> float:
    """
    Calculate Euclidean distance between two centroids.

    Args:
        c1: (x, y) centroid
        c2: (x, y) centroid

    Returns:
        Distance in pixels

    Examples:
        >>> centroid_distance((0, 0), (3, 4))
        5.0
    """
    import math
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    return math.sqrt(dx * dx + dy * dy)


def bbox_is_valid(bbox: BBox, frame_width: int, frame_height: int) -> bool:
    """
    Check if a bbox is valid (within frame bounds, non-negative dimensions).

    Args:
        bbox: [x1, y1, x2, y2] format
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        True if bbox is valid

    Examples:
        >>> bbox_is_valid([10, 20, 50, 80], 1920, 1080)
        True
        >>> bbox_is_valid([10, 20, 5, 80], 1920, 1080)  # x2 < x1
        False
        >>> bbox_is_valid([-10, 20, 50, 80], 1920, 1080)  # x1 < 0
        False
    """
    x1, y1, x2, y2 = bbox

    # Check dimensions
    if x2 <= x1 or y2 <= y1:
        return False

    # Check bounds
    if x1 < 0 or y1 < 0:
        return False

    if x2 > frame_width or y2 > frame_height:
        return False

    return True


def filter_huge_bboxes(
    bboxes: List[BBox],
    frame_width: int,
    frame_height: int,
    max_width_px: Optional[int] = None,
    max_height_px: Optional[int] = None,
    max_area_fraction: Optional[float] = None,
) -> List[BBox]:
    """
    Filter out pathologically large bounding boxes.

    Per CLAUDE.md Section 9 (Multi-Layer Defense):
    - Layer 1: Absolute limits (800px height, 600px width)
    - Layer 2: Relative limit (25% of frame area)

    This is a CRITICAL defense against YOLO hallucinations (Track 14-style bugs).

    Args:
        bboxes: List of bboxes to filter
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        max_width_px: Max width in pixels (default from constants)
        max_height_px: Max height in pixels (default from constants)
        max_area_fraction: Max area as fraction of frame (default from constants)

    Returns:
        Filtered list of bboxes (huge bboxes removed)

    Examples:
        >>> # Normal player bbox (OK)
        >>> filter_huge_bboxes([[100, 100, 200, 300]], 1920, 1080)
        [[100, 100, 200, 300]]

        >>> # Pathological bbox (87% of frame - REJECTED)
        >>> filter_huge_bboxes([[32, 129, 3412, 779]], 3840, 900)
        []
    """
    from ..core.constants import (
        MAX_BBOX_WIDTH_PX,
        MAX_BBOX_HEIGHT_PX,
        MAX_BBOX_AREA_FRACTION,
    )

    max_width_px = max_width_px or MAX_BBOX_WIDTH_PX
    max_height_px = max_height_px or MAX_BBOX_HEIGHT_PX
    max_area_fraction = max_area_fraction or MAX_BBOX_AREA_FRACTION

    frame_area = frame_width * frame_height
    max_area = frame_area * max_area_fraction

    filtered = []
    for bbox in bboxes:
        width = bbox_width(bbox)
        height = bbox_height(bbox)
        area = bbox_area(bbox)

        # Layer 1: Absolute limits
        if width > max_width_px:
            continue
        if height > max_height_px:
            continue

        # Layer 2: Relative limit
        if area > max_area:
            continue

        filtered.append(bbox)

    return filtered


def clip_bbox_to_frame(bbox: BBox, frame_width: int, frame_height: int) -> BBox:
    """
    Clip a bounding box to frame boundaries.

    Args:
        bbox: [x1, y1, x2, y2] format
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Clipped bbox within frame bounds

    Examples:
        >>> clip_bbox_to_frame([-10, 20, 50, 1100], 1920, 1080)
        [0, 20, 50, 1080]
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, frame_width))
    y1 = max(0, min(y1, frame_height))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))

    return [x1, y1, x2, y2]


def expand_bbox(bbox: BBox, margin: float) -> BBox:
    """
    Expand a bounding box by a margin (in pixels).

    Args:
        bbox: [x1, y1, x2, y2] format
        margin: Margin to add on all sides (pixels)

    Returns:
        Expanded bbox

    Examples:
        >>> expand_bbox([10, 20, 50, 80], 5)
        [5, 15, 55, 85]
    """
    x1, y1, x2, y2 = bbox
    return [x1 - margin, y1 - margin, x2 + margin, y2 + margin]


def apply_fisheye_bbox_correction(
    bbox: BBox,
    frame_width: int,
    frame_height: int,
    expansion_strength: Optional[float] = None,
) -> BBox:
    """
    Apply radial bbox expansion to account for fisheye lens distortion.

    Problem:
    - Fisheye lenses cause radial distortion - players far from center appear tilted
    - YOLO produces axis-aligned bboxes that cut off tilted players
    - Incomplete bbox → incomplete jersey crop → wrong HSV → false splits in Pass 2A

    Solution:
    - Expand bboxes based on distance from frame center
    - Bboxes far from center get expanded more (radial expansion)
    - This captures the full player even when tilted by fisheye distortion

    Per CLAUDE.md P0 (Root-Cause Fixes Only):
    - Fix bbox quality at source (Pass 1), don't patch downstream
    - Prevents false "jersey color change" detections → prevents false splits

    Args:
        bbox: [x1, y1, x2, y2] in pixel coordinates
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        expansion_strength: Expansion factor strength (0.15 = 15% expansion at corners)
                           If None, uses FISHEYE_EXPANSION_STRENGTH from constants

    Returns:
        Corrected bbox with radial expansion, clipped to frame bounds

    Examples:
        >>> # Player at frame center (no expansion needed)
        >>> apply_fisheye_bbox_correction([900, 500, 1020, 700], 1920, 1080, 0.15)
        [900, 500, 1020, 700]  # Minimal expansion

        >>> # Player at frame edge (significant expansion)
        >>> apply_fisheye_bbox_correction([100, 100, 200, 300], 1920, 1080, 0.15)
        [85, 85, 215, 315]  # ~15% expansion due to distance from center
    """
    if expansion_strength is None:
        from ..core.constants import FISHEYE_EXPANSION_STRENGTH
        expansion_strength = FISHEYE_EXPANSION_STRENGTH

    x1, y1, x2, y2 = bbox

    # Compute bbox centroid
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Frame center
    frame_cx = frame_width / 2.0
    frame_cy = frame_height / 2.0

    # Normalized distance from center (0 = center, 1 = corner)
    # Use normalized coordinates so expansion is proportional to frame size
    dx_norm = (cx - frame_cx) / frame_cx
    dy_norm = (cy - frame_cy) / frame_cy
    distance_from_center = (dx_norm**2 + dy_norm**2) ** 0.5

    # Radial expansion factor
    # Quadratic falloff: expansion is stronger at edges
    # expansion_factor = 1 + k * distance^2
    # At center (distance=0): factor = 1.0 (no expansion)
    # At corner (distance=√2): factor = 1 + k*2 (maximum expansion)
    expansion_factor = 1.0 + expansion_strength * (distance_from_center ** 2)

    # Expand bbox symmetrically around centroid
    width = x2 - x1
    height = y2 - y1
    new_width = width * expansion_factor
    new_height = height * expansion_factor

    new_x1 = cx - new_width / 2.0
    new_y1 = cy - new_height / 2.0
    new_x2 = cx + new_width / 2.0
    new_y2 = cy + new_height / 2.0

    # Clip to frame bounds (critical: prevents bbox going out of frame)
    new_bbox = [new_x1, new_y1, new_x2, new_y2]
    clipped_bbox = clip_bbox_to_frame(new_bbox, frame_width, frame_height)

    return clipped_bbox


def bbox_overlap_1d(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    """
    Calculate 1D overlap between two intervals.

    Args:
        a_min: Start of interval A
        a_max: End of interval A
        b_min: Start of interval B
        b_max: End of interval B

    Returns:
        Length of overlap (0 if no overlap)

    Examples:
        >>> bbox_overlap_1d(0, 10, 5, 15)
        5.0
        >>> bbox_overlap_1d(0, 10, 20, 30)
        0.0
    """
    overlap_min = max(a_min, b_min)
    overlap_max = min(a_max, b_max)
    return max(0.0, overlap_max - overlap_min)


def bbox_intersection(bbox1: BBox, bbox2: BBox) -> Optional[BBox]:
    """
    Calculate the intersection bounding box of two bboxes.

    Args:
        bbox1: [x1, y1, x2, y2] format
        bbox2: [x1, y1, x2, y2] format

    Returns:
        Intersection bbox, or None if no intersection

    Examples:
        >>> bbox_intersection([0, 0, 10, 10], [5, 5, 15, 15])
        [5, 5, 10, 10]
        >>> bbox_intersection([0, 0, 10, 10], [20, 20, 30, 30])
        None
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return None

    return [inter_x_min, inter_y_min, inter_x_max, inter_y_max]


def bbox_union(bbox1: BBox, bbox2: BBox) -> BBox:
    """
    Calculate the union (minimum enclosing) bounding box of two bboxes.

    Args:
        bbox1: [x1, y1, x2, y2] format
        bbox2: [x1, y1, x2, y2] format

    Returns:
        Union bbox (smallest bbox containing both)

    Examples:
        >>> bbox_union([0, 0, 10, 10], [5, 5, 15, 15])
        [0, 0, 15, 15]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    return [
        min(x1_min, x2_min),
        min(y1_min, y2_min),
        max(x1_max, x2_max),
        max(y1_max, y2_max),
    ]


def bbox_aspect_ratio(bbox: BBox) -> float:
    """
    Calculate aspect ratio of a bounding box (width / height).

    Args:
        bbox: [x1, y1, x2, y2] format

    Returns:
        Aspect ratio (width / height)

    Examples:
        >>> bbox_aspect_ratio([0, 0, 100, 200])
        0.5
        >>> bbox_aspect_ratio([0, 0, 200, 100])
        2.0
    """
    width = bbox_width(bbox)
    height = bbox_height(bbox)

    if height == 0:
        return float('inf')

    return width / height
