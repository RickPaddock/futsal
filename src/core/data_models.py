"""
Pydantic data models for the futsal tracking system.

All entity models per CLAUDE.md contract Sections 2, 5.
These models define the structure of JSON artifacts at each pass.
"""

from pydantic import BaseModel, Field, ConfigDict, AliasChoices, field_validator
from typing import List, Optional, Dict, Any
from .types import (
    TeamID,
    FragmentQuality,
    ConstraintType,
    AssignmentMethod,
    InterpolationMethod,
    BallState,
    normalize_ball_state_value,
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
    - MUST contain: bbox, confidence, track_id, jersey probs
    - MUST NOT contain: team, player_id, or any interpretation

    Field meanings:
    - detection_id: Stable row identity within Pass 1 for audit/join operations.
    - frame_idx: Frame index where this detection occurred.
    - bbox: Player bounding box [x1, y1, x2, y2] in frame coordinates.
    - centroid: Geometric center of bbox, used for motion/proximity reasoning.
    - confidence: Detector confidence for the player bbox.
    - track_id: Temporary tracker id (Pass 1-local, not final identity).
    - jersey_number: Observed jersey digit if classifier confidence passes threshold.
    - jersey_confidence: Confidence for jersey_number.
    - jersey_probs: Full jersey probability distribution for downstream evidence.
    - hsv_histogram_jersey: Jersey-ROI HSV evidence for team clustering.
    - jersey_color_sampled: Whether HSV extraction was intentionally sampled this frame.
    - jersey_roi_valid: Whether jersey ROI extraction succeeded for this detection.
    - jersey_roi_bbox: Exact ROI box used for jersey HSV extraction.
    - jersey_crop_quality: Heuristic crop quality score in [0,1] for jersey visibility.
    - is_sam_recovered: Marker that detection came from SAM-based recovery path.
    - sam_bbox: SAM-derived bbox when recovery is used.
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

    hsv_histogram_jersey: Optional[HSVHistogram] = None  # PRIMARY for team assignment
    jersey_color_sampled: bool = False
    jersey_roi_valid: bool = False
    jersey_roi_bbox: Optional[BBox] = None
    jersey_crop_quality: Optional[float] = None

    # SAM recovery markers (if detection recovered via segmentation)
    is_sam_recovered: bool = False
    sam_bbox: Optional[BBox] = None

    class Config:
        # Validation: MUST NOT have team or player_id (R1)
        extra = "forbid"  # Reject unknown fields


class BallDetection(BaseModel):
    """
    A single ball detection from Pass 1.

    Field meanings:
    - frame_idx: Frame index where ball was detected.
    - bbox: Ball bounding box [x1, y1, x2, y2].
    - centroid: Ball center point.
    - confidence: Detector confidence for this ball observation.
    """
    frame_idx: FrameIndex
    bbox: BBox  # [x1, y1, x2, y2]
    centroid: Centroid  # [x, y]
    confidence: float  # YOLO detection confidence


class Pass1Output(BaseModel):
    """
    Output from Pass 1: Raw Evidence Collection.

    Per CLAUDE.md Section 5 (Pass 1).

    Field meanings:
    - video_name: Clip stem used for artifact naming and audit.
    - fps: Source frame rate used for timing and interpolation limits.
    - width: Frame width in pixels.
    - height: Frame height in pixels.
    - total_frames: Total frame count for the source clip.
    - processed_start_frame: Inclusive start frame processed in this run.
    - processed_end_frame_exclusive: Exclusive end frame processed in this run.
    - detections: Player detection evidence rows.
    - ball_detections: Ball detection evidence rows.
    """
    video_name: str
    fps: float
    width: int  # Frame width in pixels
    height: int  # Frame height in pixels
    total_frames: int
    processed_start_frame: int = 0
    processed_end_frame_exclusive: Optional[int] = None

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

    Field meanings:
    - fragment_id: Immutable fragment identifier.
    - track_id: Pass 1 track that produced this fragment.
    - start_frame: First frame covered by this fragment.
    - end_frame: Last frame covered by this fragment.
    - detection_ids: Ordered references to Pass 1 detection rows.
    - split_reason: Mechanical reason this fragment was created/split.
    - split_trigger_frame: Frame where split trigger occurred (non-initial fragments).
    - split_rule_id: Stable split rule id (e.g., JERSEY_CHANGE, TRACK_COLLISION).
    - parent_fragment_id: Source fragment when created by split operation.
    """
    fragment_id: FragmentID  # Format: F{counter:06d} (globally unique, immutable)
    track_id: TrackID = Field(
        validation_alias=AliasChoices("track_id", "original_track_id"),
        serialization_alias="track_id",
    )  # ByteTrack ID that created this fragment
    start_frame: FrameIndex
    end_frame: FrameIndex
    detection_ids: List[DetectionID]  # References to Pass 1 detections

    # Mechanical split metadata
    split_reason: Optional[str] = None  # "appearance_drift", "jersey_conflict", etc.
    split_trigger_frame: Optional[FrameIndex] = None
    split_rule_id: Optional[str] = None
    parent_fragment_id: Optional[FragmentID] = None  # If split from another fragment

    # Fragment-level jersey summary (computed in Pass 2A)
    dominant_jersey_number: Optional[int] = None  # Mode of observed jersey numbers

    # Fragment-level color/team-cluster hints from Pass 2A (for Pass 3 reuse)
    dominant_color_cluster_id: Optional[int] = None
    dominant_team_cluster_id: Optional[int] = None
    dominant_team_cluster_confidence: Optional[float] = None

    # Pass 2A metadata fields (computed during mechanical fragmentation)
    jersey_visible_ratio: Optional[float] = None
    occlusion_ratio: Optional[float] = None
    mean_velocity: Optional[float] = None
    appearance_stability_score: Optional[float] = None
    quality: Optional[FragmentQuality] = None

    # Ghost metadata (Pass 2C) - available at Fragment level for uniform access
    is_ghost: bool = False

    @property
    def original_track_id(self) -> TrackID:
        """Backward-compatible alias for legacy code paths."""
        return self.track_id

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


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

    Field meanings:
    - quality: Discrete quality label (high/medium/low/ghost).
    - quality_score: Continuous overall quality score in [0, 1].
    - quality_reasons: Audit reasons explaining the quality_score.
    - avg_confidence: Mean detection confidence over fragment lifespan.
    - min_confidence: Minimum detection confidence over fragment lifespan.
    - occlusion_ratio: Proxy metric from bbox smoothness/stability.
    - jersey_visible_ratio: Fraction of consistent jersey observations.
    - jersey_observability_score: Alias metric equal to jersey_visible_ratio.
    - appearance_stability_score: Stability of HSV appearance within fragment.
    - mean_velocity: Mean centroid velocity in px/frame.
    - motion_smoothness_score: Inverse velocity variance in [0, 1].

    Ghost-specific fields (Pass 2C):
    - is_ghost: True if this is a ghost fragment (default False).
    - ghost_last_known_bbox: Last known bbox before occlusion (ghosts only).
    - ghost_last_known_centroid: Last known centroid before occlusion (ghosts only).
    - ghost_reason: Why ghost was created (ghosts only).
    """
    quality: FragmentQuality
    quality_score: float  # 0-1 continuous score
    quality_reasons: List[str] = Field(default_factory=list)  # Why this score?

    # Quality indicators
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    occlusion_ratio: float = Field(
        default=0.0,
        validation_alias=AliasChoices("occlusion_ratio", "avg_bbox_stability"),
        serialization_alias="occlusion_ratio",
    )  # Low jitter = low occlusion risk
    jersey_visible_ratio: float = Field(
        default=0.0,
        validation_alias=AliasChoices("jersey_visible_ratio", "jersey_consistency"),
        serialization_alias="jersey_visible_ratio",
    )  # How often same jersey appears
    appearance_stability_score: float = Field(
        default=0.0,
        validation_alias=AliasChoices("appearance_stability_score", "hsv_consistency"),
        serialization_alias="appearance_stability_score",
    )  # HSV histogram similarity
    mean_velocity: float = 0.0  # px/frame based on centroid displacement
    jersey_observability_score: float = 0.0
    motion_smoothness_score: float = 0.0

    # Ghost-specific fields (Pass 2C)
    is_ghost: bool = False
    exclude_from_clustering: bool = False
    ghost_last_known_bbox: Optional[BBox] = None
    ghost_last_known_centroid: Optional[Centroid] = None
    ghost_reason: Optional[str] = None

    # Pass 2B binary presence classification (identity-agnostic)
    # Allowed values: "real" | "occlusion_candidate"
    presence_class: str = "real"

    @property
    def avg_bbox_stability(self) -> float:
        """Backward-compatible alias for legacy code paths."""
        return self.occlusion_ratio

    @property
    def jersey_consistency(self) -> float:
        """Backward-compatible alias for legacy code paths."""
        return self.jersey_visible_ratio

    @property
    def hsv_consistency(self) -> float:
        """Backward-compatible alias for legacy code paths."""
        return self.appearance_stability_score

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


class Pass2BOutput(BaseModel):
    """
    Output from Pass 2B: Fragment Quality Scoring.
    """
    fragments: List[ScoredFragment]
    quality_distribution: Dict[str, int] = Field(default_factory=dict)  # quality -> count
    split_log: List[Dict[str, Any]] = Field(default_factory=list)  # Preserved from Pass 2A


# ============================================================================
# PASS 2C MODELS (GHOST GENERATION)
# ============================================================================

class GhostFragment(ScoredFragment):
    """
    Ghost fragment created in Pass 2C to maintain player count continuity.

    Per CLAUDE.md R4: Players never disappear.
    Ghosts are excluded from K-means clustering (Pass 3C).

    Field meanings:
    - quality: Always ghost for ghost rows.
    - is_ghost: Explicit marker for downstream exclusion/inference rules.
    - ghost_reason: Why the ghost was created (occlusion/off-screen/etc).
    - estimated_position: Held bbox estimate while player is missing.
    - estimated_centroid: Held centroid estimate while player is missing.
    - source_fragment_id: Fragment that spawned this ghost.
    - source_track_id: Original track lineage for this ghost.
    """
    quality: FragmentQuality = FragmentQuality.GHOST  # Always GHOST
    is_ghost: bool = True
    exclude_from_clustering: bool = True
    ghost_reason: str  # "occlusion", "off_screen", etc.
    estimated_position: BBox  # Last known position (held, no interpolation)
    estimated_centroid: Centroid  # Last known centroid
    source_fragment_id: FragmentID  # Which fragment created this ghost
    source_track_id: TrackID  # Original track this ghost belongs to


class Pass2COutput(BaseModel):
    """
    Output from Pass 2C: Ghost Generation.

    Unified list of fragments (real + ghosts) for downstream processing.
    Ghosts marked with is_ghost=True and quality=GHOST.
    """
    fragments: List[ScoredFragment]  # Real fragments + ghosts (unified)
    split_log: List[Dict[str, Any]] = Field(default_factory=list)  # From Pass 2A (preserved)
    ghost_creation_log: List[Dict[str, Any]] = Field(default_factory=list)  # Ghost audit trail
    level_timeline: List[Dict[str, Any]] = Field(default_factory=list)  # Frame-by-frame level


# ============================================================================
# PASS 3A MODELS (IDENTITY CANDIDATE GENERATION)
# ============================================================================

class IdentityCandidate(BaseModel):
    """
    Identity candidates for a fragment (Pass 3A).

    Per CLAUDE.md Section 5 (Pass 3A): NO LOCKING (candidates only).

    Field meanings:
    - fragment_id: Target fragment being evaluated.
    - candidate_team: Proposed team label (not committed).
    - candidate_jersey: Proposed jersey number (not committed).
    - candidate_player_id: Proposed player id (not committed).
    - team_evidence: Evidence breakdown for team scoring.
    - jersey_evidence: Evidence breakdown for jersey scoring.
    - player_evidence: Evidence breakdown for player scoring.
    """
    fragment_id: FragmentID
    candidate_team: Optional[TeamID] = None
    candidate_jersey: Optional[int] = None
    candidate_player_id: Optional[PlayerID] = None

    # Evidence scores
    # team_evidence: source -> score (float)
    # jersey_evidence: jersey_key (e.g. "04") -> {"count": int, "ratio": float}
    # player_evidence: source -> score (float)
    team_evidence: Dict[str, float] = Field(default_factory=dict)
    jersey_evidence: Dict[str, Any] = Field(default_factory=dict)
    player_evidence: Dict[str, float] = Field(default_factory=dict)


class Pass3AOutput(BaseModel):
    """
    Output from Pass 3A: Identity Candidate Generation.
    """
    candidates: List[IdentityCandidate]


class IdentityCandidateEdge(BaseModel):
    """
    Pairwise candidate edge emitted by Pass 3A.

    Each edge represents a possible identity continuation from fragment_a to
    fragment_b under temporal and spatial feasibility constraints.
    """
    fragment_a: FragmentID
    fragment_b: FragmentID
    temporal_gap: int
    spatial_distance: float
    track_continuity_score: float = 0.0
    velocity_consistency_score: float
    appearance_similarity: float
    jersey_similarity: float
    temporal_gap_score: float = 0.0
    overall_candidate_score: float


class Pass3AEdgesOutput(BaseModel):
    """
    Output artifact for contract-compliant Pass 3A pairwise candidates.
    """
    candidates: List[IdentityCandidateEdge]


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

    Field meanings:
    - constraint_id: Unique identifier for this constraint.
    - constraint_type: MUST_SAME / CANNOT_SAME / SOFT_SAME.
    - fragment_ids: Fragment ids participating in this constraint.
    - value: Optional payload associated with the constraint.
    - weight: Weight for soft optimization constraints.
    - reason: Human-readable reason for why constraint exists.
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

    Field meanings:
    - fragment_id: Fragment receiving final committed identity.
    - player_id: Final immutable player id.
    - team: Final immutable team assignment.
    - jersey_number: Final immutable jersey number (nullable when unresolved).
    - assignment_method: Method used to produce assignment.
    - assignment_confidence: Confidence score for assignment.
    - assignment_reasons: Explainability trail for the assignment.
    - _locked_team: Internal lock source used to enforce team immutability.
    """
    fragment_id: FragmentID
    player_id: PlayerID  # Format: P{jersey:02d}_{team} (immutable)
    team: TeamID  # LOCKED (MUST be team_a or team_b, never "unknown")
    jersey_number: Optional[int] = None  # LOCKED (nullable when unresolved)

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
    State ∈ {real, interpolated, unknown}

    Field meanings:
    - frame_idx: Frame index for this ball state row.
    - state: real / interpolated / unknown.
    - centroid: Ball position if state has on-court position.
    - bbox: Detection bbox for real observations.
    - confidence: Confidence/proxy confidence for this state.
    """
    frame_idx: FrameIndex
    state: BallState  # real, interpolated, or unknown
    centroid: Optional[Centroid] = None  # [x, y] - None if unknown
    bbox: Optional[BBox] = None  # None if interpolated or unknown
    confidence: float = 1.0  # 1.0 for real, < 1.0 for interpolated, 0.0 for unknown

    @field_validator("state", mode="before")
    @classmethod
    def _normalize_legacy_state(cls, value):
        return normalize_ball_state_value(value)


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
# BIRD'S-EYE PROJECTION MODELS
# ============================================================================

class BirdseyePlayerPosition(BaseModel):
    """
    Projected player position for a single frame.

    Uses committed Pass 3 identity and preserves whether the source span is
    estimated from a ghost window.
    """
    frame_idx: FrameIndex
    fragment_id: FragmentID
    player_id: PlayerID
    team: TeamID
    jersey_number: Optional[int] = None
    track_id: TrackID
    is_ghost: bool = False
    is_estimated: bool = False
    image_bbox: BBox
    image_anchor: Centroid
    raw_image_anchor: Optional[Centroid] = None
    raw_court_position: Optional[Centroid] = None
    raw_render_position: Optional[Centroid] = None
    stabilization_trust: float = 1.0
    court_position: Centroid
    render_position: Centroid


class BirdseyeBallFrame(BaseModel):
    """
    Projected ball state for a single frame.
    """
    frame_idx: FrameIndex
    state: BallState
    confidence: float = 0.0
    image_bbox: Optional[BBox] = None
    image_position: Optional[Centroid] = None
    court_position: Optional[Centroid] = None
    render_position: Optional[Centroid] = None

    @field_validator("state", mode="before")
    @classmethod
    def _normalize_legacy_state(cls, value):
        return normalize_ball_state_value(value)


class BirdseyeFrame(BaseModel):
    """
    Bird's-eye projection state for a single frame.
    """
    frame_idx: FrameIndex
    players: List[BirdseyePlayerPosition] = Field(default_factory=list)
    ball: BirdseyeBallFrame


class BirdseyeProjectionOutput(BaseModel):
    """
    Output from the bird's-eye projection stage.
    """
    video_name: str
    fps: float
    total_frames: int
    processed_start_frame: int = 0
    processed_end_frame_exclusive: Optional[int] = None
    court_length_m: float
    court_width_m: float
    output_pixel_scale: int
    frames: List[BirdseyeFrame]
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# ANALYTICS MODELS
# ============================================================================

class PossessionCandidate(BaseModel):
    """
    Candidate controller for a single frame in analytics possession output.
    """
    player_id: PlayerID
    team: TeamID
    jersey_number: Optional[int] = None
    distance_to_ball: float
    source_space: str  # court or image
    confidence: float


class PossessionFrame(BaseModel):
    """
    Possession state for a single frame.

    This is a downstream analytics artifact built from committed identities,
    ball interpolation, and bird's-eye projection.
    """
    frame_idx: FrameIndex
    player_id: Optional[PlayerID] = None
    team: Optional[TeamID] = None
    jersey_number: Optional[int] = None
    confidence: float = 0.0
    is_ambiguous: bool = False
    ball_state: BallState
    source_space: Optional[str] = None
    candidates: List[PossessionCandidate] = Field(default_factory=list)

    @field_validator("ball_state", mode="before")
    @classmethod
    def _normalize_possession_ball_state(cls, value):
        return normalize_ball_state_value(value)


class AnalyticsPossessionOutput(BaseModel):
    """
    Frame-indexed possession artifact for downstream analytics.
    """
    video_name: str
    fps: float
    total_frames: int
    processed_start_frame: int = 0
    processed_end_frame_exclusive: Optional[int] = None
    frames: List[PossessionFrame]
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class AnalyticsEvent(BaseModel):
    """
    Event record for downstream analytics reporting.

    Initial implementation emits only pass events.
    """
    event_id: str
    event_type: str
    start_frame: FrameIndex
    end_frame: FrameIndex
    team_id: TeamID
    outcome: str
    event_confidence: float = 0.0
    is_audit_only: bool = False
    passer_player_id: Optional[PlayerID] = None
    passer_jersey_number: Optional[int] = None
    receiver_player_id: Optional[PlayerID] = None
    receiver_team_id: Optional[TeamID] = None
    receiver_jersey_number: Optional[int] = None


class AnalyticsEventsOutput(BaseModel):
    """
    Event artifact for downstream analytics reporting.

    Initial implementation contains pass events only.
    """
    video_name: str
    fps: float
    total_frames: int
    processed_start_frame: int = 0
    processed_end_frame_exclusive: Optional[int] = None
    events: List[AnalyticsEvent]
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class PlayerDistanceSummaryRow(BaseModel):
    """
    Reportable distance summary for one committed player identity.

    Distance summaries are restricted to jersey-resolved identities and must
    state their units explicitly.
    """
    player_id: PlayerID
    jersey_number: int
    team_id: TeamID
    display_name: Optional[str] = None
    distance_value: float
    distance_unit: str
    observed_frame_count: int = 0
    estimated_frame_count: int = 0
    distance_confidence: float = 0.0


class PlayerDistanceSummaryOutput(BaseModel):
    """
    Per-player distance summary artifact for downstream reporting.
    """
    video_name: str
    fps: float
    total_frames: int
    processed_start_frame: int = 0
    processed_end_frame_exclusive: Optional[int] = None
    players: List[PlayerDistanceSummaryRow]
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class NamedPlayerStatus(BaseModel):
    """
    Resolution status for named jersey mappings used in downstream reports.
    """
    jersey_number: int
    display_name: str
    resolved: bool
    player_id: Optional[PlayerID] = None
    team_id: Optional[TeamID] = None


class AnalyticsPlayerSummary(BaseModel):
    """
    Lightweight per-player rollup built from possession, events, and distance.
    """
    player_id: PlayerID
    jersey_number: int
    team_id: TeamID
    display_name: Optional[str] = None
    confirmed_possession_frames: int = 0
    confirmed_possession_seconds: float = 0.0
    passes_attempted: int = 0
    passes_completed: int = 0
    passes_received: int = 0
    unsuccessful_passes: int = 0
    loose_ball_releases: int = 0
    shots_attempted: int = 0
    goals_scored: int = 0
    distance_value: float = 0.0
    distance_unit: str = "m"
    distance_confidence: float = 0.0


class AnalyticsSummaryOutput(BaseModel):
    """
    Lightweight report artifact built from committed analytics artifacts only.
    """
    video_name: str
    fps: float
    total_frames: int
    processed_start_frame: int = 0
    processed_end_frame_exclusive: Optional[int] = None
    players: List[AnalyticsPlayerSummary]
    named_players: List[NamedPlayerStatus] = Field(default_factory=list)
    event_totals: Dict[str, int] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# VALIDATION MODELS
# ============================================================================

class ValidationViolation(BaseModel):
    """
    A single validation violation.

    Field meanings:
    - rule: Rule identifier (global or pass-specific).
    - severity: error or warning.
    - message: Human-readable violation message.
    - frame_idx: Optional frame context.
    - fragment_id: Optional fragment context.
    - details: Structured violation payload for debugging.
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

    Field meanings:
    - passed: True only when no blocking violations exist.
    - violations: Blocking rule failures.
    - warnings: Non-blocking issues.
    - timestamp: ISO8601 time validation completed.
    - pass_name: Validation scope (pass1/pass2/pass3/ball/etc).
    - diagnostics: Optional pass-level diagnostics payload for auditability.
    """
    passed: bool
    violations: List[ValidationViolation] = Field(default_factory=list)
    warnings: List[ValidationViolation] = Field(default_factory=list)
    timestamp: str  # ISO 8601
    pass_name: str  # "pass1", "pass2", "pass3", etc.
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# DEBUG METRICS MODELS
# ============================================================================

class FrameMetrics(BaseModel):
    """
    Debug metrics for a single frame.

    Field meanings:
    - frame_idx: Frame index for this metrics row.
    - player_count: Total players present (tracked + ghosts).
    - tracked_count: Real tracked players (non-ghost).
    - ghost_count: Ghost player count.
    - team_a_count: Team A players in frame.
    - team_b_count: Team B players in frame.
    - unknown_count: Unknown-team players (must be 0 after Pass 3C).
    - jersey_conflicts: Jersey exclusivity conflicts detected in frame.
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

    Field meanings:
    - video_name: Clip identifier.
    - total_frames: Number of frames represented in frame_metrics.
    - frame_metrics: Per-frame metric records.
    - total_identity_changes: Identity changes after lock point (must be 0 after Pass 3C).
    - total_jersey_conflicts: Aggregate jersey conflicts across clip.
    - total_unknown_frames: Frames containing unknown-team assignments.
    - avg_player_count: Mean player count across frames.
    - cluster_compactness_a: Team A compactness diagnostic from Pass 3C.
    - cluster_compactness_b: Team B compactness diagnostic from Pass 3C.
    - compactness_ratio: Ratio between diffuse and compact cluster scores.
    """
    video_name: str
    total_frames: int
    frame_metrics: List[FrameMetrics]

    # Summary statistics
    total_identity_changes: int = 0  # MUST be 0 after Pass 3C
    total_jersey_conflicts: int = 0
    total_unknown_frames: int = 0
    avg_player_count: float = 0.0

    # Pass 3C clustering diagnostics
    cluster_compactness_a: Optional[float] = None
    cluster_compactness_b: Optional[float] = None
    compactness_ratio: Optional[float] = None
