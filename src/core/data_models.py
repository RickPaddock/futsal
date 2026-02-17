"""
Pydantic data models for the futsal tracking system.

All entity models per CLAUDE.md contract Sections 2, 5.
These models define the structure of JSON artifacts at each pass.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from .types import (
    TeamID,
    FragmentQuality,
    ConstraintType,
    AssignmentMethod,
    InterpolationMethod,
    BallState,
    BBox,
    Centroid,
    HSVHistogram,
    JerseyProbs,
    FrameIndex,
    TrackID,
    FragmentID,
    PlayerID,
    DetectionID,
)

# ============================================================================
# PASS 1 MODELS (RAW EVIDENCE)
# ============================================================================

class Detection(BaseModel):
    """
    A single player detection from Pass 1.

    Per CLAUDE.md R1: Pass 1 is raw truth only.
    - MUST contain: bbox, confidence, track_id, jersey probs, HSV histogram
    - MUST NOT contain: team, player_id, or any interpretation
    """
    detection_id: DetectionID  # Format: {frame_idx}_{track_id}_{bbox_hash}
    frame_idx: FrameIndex
    bbox: BBox  # [x1, y1, x2, y2]
    centroid: Centroid  # [x, y]
    confidence: float  # YOLO detection confidence

    track_id: TrackID  # ByteTrack temporary ID (mutable, meaningless beyond Pass 1)

    # Raw observations (no interpretation)
    jersey_number: Optional[int] = None  # Detected if conf >= JERSEY_CONF_THRESHOLD
    jersey_confidence: float = 0.0
    jersey_probs: Optional[JerseyProbs] = None  # Full probability distribution

    hsv_histogram: Optional[HSVHistogram] = None  # 512 bins (8x8x8), normalized

    # SAM recovery markers (if detection recovered via segmentation)
    is_sam_recovered: bool = False
    sam_bbox: Optional[BBox] = None

    class Config:
        # Validation: MUST NOT have team or player_id (R1)
        extra = "forbid"  # Reject unknown fields


class BallDetection(BaseModel):
    """
    A single ball detection from Pass 1.
    """
    frame_idx: FrameIndex
    bbox: BBox  # [x1, y1, x2, y2]
    centroid: Centroid  # [x, y]
    confidence: float  # YOLO detection confidence


class Pass1Output(BaseModel):
    """
    Output from Pass 1: Raw Evidence Collection.

    Per CLAUDE.md Section 5 (Pass 1).
    """
    video_name: str
    fps: float
    width: int  # Frame width in pixels
    height: int  # Frame height in pixels
    total_frames: int

    detections: List[Detection]  # Player detections
    ball_detections: List[BallDetection]  # Ball detections


# ============================================================================
# PASS 2A MODELS (MECHANICAL FRAGMENTATION)
# ============================================================================

class Fragment(BaseModel):
    """
    A contiguous fragment of detections.

    Per CLAUDE.md Section 2: fragment_id is immutable.
    Created in Pass 2A via mechanical splitting.
    """
    fragment_id: FragmentID  # Format: F{counter:06d} (globally unique, immutable)
    original_track_id: TrackID  # ByteTrack ID that created this fragment
    start_frame: FrameIndex
    end_frame: FrameIndex
    detection_ids: List[DetectionID]  # References to Pass 1 detections

    # Mechanical split metadata
    split_reason: Optional[str] = None  # "appearance_drift", "jersey_conflict", etc.
    parent_fragment_id: Optional[FragmentID] = None  # If split from another fragment


class Pass2AOutput(BaseModel):
    """
    Output from Pass 2A: Mechanical Fragmentation.
    """
    fragments: List[Fragment]
    split_log: List[Dict[str, Any]]  # Audit trail of all splits


# ============================================================================
# PASS 2B MODELS (FRAGMENT QUALITY SCORING)
# ============================================================================

class ScoredFragment(Fragment):
    """
    Fragment with quality scoring added in Pass 2B.

    Extends Fragment with quality metadata.
    """
    quality: FragmentQuality
    quality_score: float  # 0-1 continuous score
    quality_reasons: List[str] = Field(default_factory=list)  # Why this score?

    # Quality indicators
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    avg_bbox_stability: float = 0.0  # Low jitter = high stability
    jersey_consistency: float = 0.0  # How often same jersey appears
    hsv_consistency: float = 0.0  # HSV histogram similarity across frames


class Pass2BOutput(BaseModel):
    """
    Output from Pass 2B: Fragment Quality Scoring.
    """
    fragments: List[ScoredFragment]
    quality_distribution: Dict[str, int] = Field(default_factory=dict)  # quality -> count


# ============================================================================
# PASS 2C MODELS (GHOST GENERATION)
# ============================================================================

class GhostFragment(ScoredFragment):
    """
    Ghost fragment created in Pass 2C to maintain player count continuity.

    Per CLAUDE.md R4: Players never disappear.
    Ghosts are excluded from K-means clustering (Pass 3C).
    """
    quality: FragmentQuality = FragmentQuality.GHOST  # Always GHOST
    is_ghost: bool = True
    ghost_reason: str  # "occlusion", "off_screen", etc.
    estimated_position: BBox  # Last known position (held, no interpolation)
    estimated_centroid: Centroid  # Last known centroid
    source_fragment_id: FragmentID  # Which fragment created this ghost
    source_track_id: TrackID  # Original track this ghost belongs to


class Pass2COutput(BaseModel):
    """
    Output from Pass 2C: Ghost Generation.
    """
    fragments: List[ScoredFragment]  # Real fragments (from Pass 2B)
    ghosts: List[GhostFragment]  # Ghost fragments
    ghost_creation_log: List[Dict[str, Any]] = Field(default_factory=list)
    level_timeline: List[Dict[str, Any]] = Field(default_factory=list)  # Frame-by-frame level


# ============================================================================
# PASS 3A MODELS (IDENTITY CANDIDATE GENERATION)
# ============================================================================

class IdentityCandidate(BaseModel):
    """
    Identity candidates for a fragment (Pass 3A).

    Per CLAUDE.md Section 5 (Pass 3A): NO LOCKING (candidates only).
    """
    fragment_id: FragmentID
    candidate_team: Optional[TeamID] = None
    candidate_jersey: Optional[int] = None
    candidate_player_id: Optional[PlayerID] = None

    # Evidence scores (source -> score)
    team_evidence: Dict[str, float] = Field(default_factory=dict)
    jersey_evidence: Dict[str, float] = Field(default_factory=dict)
    player_evidence: Dict[str, float] = Field(default_factory=dict)


class Pass3AOutput(BaseModel):
    """
    Output from Pass 3A: Identity Candidate Generation.
    """
    candidates: List[IdentityCandidate]


# ============================================================================
# PASS 3B MODELS (CONSTRAINT GRAPH)
# ============================================================================

class Constraint(BaseModel):
    """
    A constraint in the identity resolution graph (Pass 3B).

    Per CLAUDE.md Section 5 (Pass 3B):
    - MUST_SAME: Track adjacency + ghost continuity (hard identity constraint)
    - CANNOT_SAME: Jersey temporal exclusivity (hard exclusion)
    - SOFT_SAME: Track continuity preferences (soft)
    """
    constraint_id: str  # Unique constraint identifier
    constraint_type: ConstraintType
    fragment_ids: List[FragmentID]  # Fragments involved in this constraint
    value: Optional[Any] = None  # Team/jersey value for MUST constraints
    weight: float = 1.0  # For SOFT constraints
    reason: str  # Why this constraint exists


class Pass3BOutput(BaseModel):
    """
    Output from Pass 3B: Constraint Graph Construction.
    """
    constraints: List[Constraint]
    constraint_graph: Dict[FragmentID, List[str]] = Field(default_factory=dict)  # fragment_id -> constraint_ids


# ============================================================================
# PASS 3C MODELS (IDENTITY COMMIT - LOCK POINT)
# ============================================================================

class CommittedIdentity(BaseModel):
    """
    Committed identity for a fragment (Pass 3C - LOCK POINT).

    Per CLAUDE.md P3: Identity is immutable after Pass 3C.
    Per CLAUDE.md R2: team MUST be team_a or team_b (no "unknown").
    """
    fragment_id: FragmentID
    player_id: PlayerID  # Format: P{jersey:02d}_{team} (immutable)
    team: TeamID  # LOCKED (MUST be team_a or team_b, never "unknown")
    jersey_number: int  # LOCKED

    # Decision audit trail
    assignment_method: AssignmentMethod
    assignment_confidence: float
    assignment_reasons: List[str] = Field(default_factory=list)

    # Internal locked team (single source of truth)
    # Per CLAUDE.md Section 9: ALWAYS use _locked_team, NEVER use team field directly
    _locked_team: Optional[TeamID] = None


class Pass3COutput(BaseModel):
    """
    Output from Pass 3C: Identity Commit (LOCK POINT).

    This is the ONLY place identity is decided.
    """
    identities: List[CommittedIdentity]
    solver_log: Dict[str, Any] = Field(default_factory=dict)  # CSP solver decisions
    unresolved_conflicts: List[Dict[str, Any]] = Field(default_factory=list)  # Should be empty (fail-fast if not)


# ============================================================================
# BALL INTERPOLATION MODELS
# ============================================================================

class BallPosition(BaseModel):
    """
    Ball state at a frame.

    Per CLAUDE.md R5: Ball state exists at every frame.
    State ∈ {real, interpolated, out_of_play}
    """
    frame_idx: FrameIndex
    state: BallState  # real, interpolated, or out_of_play
    centroid: Optional[Centroid] = None  # [x, y] - None if out_of_play
    bbox: Optional[BBox] = None  # None if interpolated or out_of_play
    confidence: float = 1.0  # 1.0 for real, < 1.0 for interpolated, 0.0 for out_of_play


class BallInterpolationOutput(BaseModel):
    """
    Output from Ball Interpolation.
    """
    ball_positions: List[BallPosition]
    interpolation_method: InterpolationMethod
    total_frames: int
    interpolated_frames: List[FrameIndex] = Field(default_factory=list)
    gap_summary: List[Dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# VALIDATION MODELS
# ============================================================================

class ValidationViolation(BaseModel):
    """
    A single validation violation.
    """
    rule: str  # R1, R2, etc.
    severity: str  # "error", "warning"
    message: str
    frame_idx: Optional[FrameIndex] = None
    fragment_id: Optional[FragmentID] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """
    Result of validation check.
    """
    passed: bool
    violations: List[ValidationViolation] = Field(default_factory=list)
    warnings: List[ValidationViolation] = Field(default_factory=list)
    timestamp: str  # ISO 8601
    pass_name: str  # "pass1", "pass2", "pass3", etc.


# ============================================================================
# DEBUG METRICS MODELS
# ============================================================================

class FrameMetrics(BaseModel):
    """
    Debug metrics for a single frame.
    """
    frame_idx: FrameIndex
    player_count: int  # Total players (tracked + ghosts)
    tracked_count: int  # Real detections
    ghost_count: int  # Ghosts
    team_a_count: int
    team_b_count: int
    unknown_count: int  # Should be 0 after Pass 3C
    jersey_conflicts: List[str] = Field(default_factory=list)  # List of conflicts


class DebugMetrics(BaseModel):
    """
    Frame-by-frame debug metrics.

    Per CLAUDE.md Section 6: Everything auditable without watching video.
    """
    video_name: str
    total_frames: int
    frame_metrics: List[FrameMetrics]

    # Summary statistics
    total_identity_changes: int = 0  # MUST be 0 after Pass 3C
    total_jersey_conflicts: int = 0
    total_unknown_frames: int = 0
    avg_player_count: float = 0.0
