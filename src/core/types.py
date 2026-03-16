"""
Type definitions for the futsal tracking system.

Enums and type aliases per CLAUDE.md contract Section 2 (Entity Model) and throughout.
"""

from enum import Enum
from typing import List

# ============================================================================
# ENTITY ENUMS
# ============================================================================

class TeamID(str, Enum):
    """
    Team identifier enum.

    Per CLAUDE.md R2: Every player MUST have team_a or team_b after Pass 3C.
    "unknown" is only allowed BEFORE Pass 3C.
    """
    TEAM_A = "team_a"
    TEAM_B = "team_b"
    UNKNOWN = "unknown"  # FORBIDDEN after Pass 3C


class FragmentQuality(str, Enum):
    """
    Fragment quality enum.

    Assigned in Pass 2B based on appearance stability, jersey observability,
    motion smoothness, and occlusion ratio.
    """
    HIGH = "high"      # Good detections throughout, stable appearance
    MEDIUM = "medium"  # Some issues but generally trackable
    LOW = "low"        # Short fragment or problematic detections
    GHOST = "ghost"    # Estimated position during occlusion (assigned in Pass 2C)


class ConstraintType(str, Enum):
    """
    Constraint types for Pass 3B constraint graph.

    Per CLAUDE.md Section 5 (Pass 3B):
    - MUST_SAME: Hard identity constraint (track adjacency, ghost continuity)
    - CANNOT_SAME: Hard exclusion constraint (jersey temporal exclusivity)
    - SOFT_SAME: Soft preference constraint (track continuity)
    """
    MUST_SAME = "must_same"      # Track adjacency + ghost continuity
    CANNOT_SAME = "cannot_same"  # Jersey temporal exclusivity violations
    SOFT_SAME = "soft_same"      # Track continuity preferences


class AssignmentMethod(str, Enum):
    """
    Method used to assign player identity in Pass 3C.

    Used for audit trail in CommittedIdentity.
    """
    KMEANS = "kmeans"              # Team assigned via K-means clustering
    INHERITED = "inherited"        # Jersey/team inherited from adjacent fragment
    CONSTRAINT_SOLVED = "constraint_solved"  # Resolved via constraint satisfaction
    GHOST_INHERITED = "ghost_inherited"      # Ghost inherits from source fragment


class InterpolationMethod(str, Enum):
    """
    Ball interpolation method.

    Per CLAUDE.md Section 8: BALL_INTERPOLATION_METHOD = "linear"
    """
    LINEAR = "linear"
    KALMAN = "kalman"


class BallState(str, Enum):
    """
    Ball state enum.

    Per CLAUDE.md R5: Ball state exists at every frame.
    State ∈ {real, interpolated, unknown}
    """
    REAL = "real"                    # YOLO detection with bbox and confidence
    INTERPOLATED = "interpolated"    # Gap ≤ 30 frames, position interpolated
    UNKNOWN = "unknown"              # No trusted ball position available for this frame
    OUT_OF_PLAY = "unknown"          # Legacy alias for backward compatibility


def normalize_ball_state_value(value: object) -> str:
    """Normalize legacy ball state labels to the current semantic name."""
    if isinstance(value, BallState):
        return value.value
    if isinstance(value, str):
        return BallState.UNKNOWN.value if value == "out_of_play" else value
    return str(value)


# ============================================================================
# TYPE ALIASES
# ============================================================================

# Bbox format: [x1, y1, x2, y2] (top-left, bottom-right)
BBox = List[float]

# Centroid format: [x, y]
Centroid = List[float]

# HSV histogram: 512-element list (8x8x8 bins, normalized)
HSVHistogram = List[float]

# Jersey probability vector: {jersey_number: confidence}
JerseyProbs = dict[int, float]

# Frame index
FrameIndex = int

# Track ID (ByteTrack temporary ID)
TrackID = int

# Fragment ID (immutable, format: F{counter:06d})
FragmentID = str

# Player ID (immutable after Pass 3C, format: P{jersey:02d}_{team})
PlayerID = str

# Detection ID (frame-local hash, format: {frame_idx}_{track_id}_{bbox_hash})
DetectionID = str
