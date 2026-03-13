"""
Configuration constants for the futsal tracking system.

All magic numbers and thresholds are defined here per CLAUDE.md contract Section 8.
These values are NON-NEGOTIABLE and must not be changed without contract update.
"""

from pathlib import Path

# ============================================================================
# MODEL PATHS
# ============================================================================

MODELS_DIR = Path("models")

PLAYER_MODEL_PATH = MODELS_DIR / "PLAYER_MODEL_best_v1.pt"
BALL_MODEL_PATH = MODELS_DIR / "BALL_MODEL_best_v2.pt"
JERSEY_MODEL_PATH = MODELS_DIR / "JERSEY_MODEL_best_v1.pt"

# ============================================================================
# DETECTION THRESHOLDS
# ============================================================================

# YOLO confidence thresholds
PLAYER_CONF_THRESHOLD = 0.5
BALL_CONF_THRESHOLD = 0.3
JERSEY_CONF_THRESHOLD = 0.3  # Lower threshold per memory learnings

# Jersey number classification optimization
JERSEY_NUMBER_CLASSIFY_EVERY_N_FRAMES = 5  # Only classify jersey number every N frames

# Jersey color (HSV) sampling optimization
JERSEY_COLOR_SAMPLE_EVERY_N_FRAMES = 5  # Only extract jersey HSV every N frames

# Jersey ROI crop geometry (relative to player bbox)
# Focus torso/jersey and reduce shorts contamination in HSV signal.
JERSEY_ROI_X_MIN_FRAC = 0.25
JERSEY_ROI_X_MAX_FRAC = 0.75
JERSEY_ROI_Y_MIN_FRAC = 0.20
JERSEY_ROI_Y_MAX_FRAC = 0.50  # Raised from 0.55 to avoid shorts contamination

# ============================================================================
# BBOX FILTERING (MULTI-LAYER DEFENSE)
# ============================================================================

# Layer 1: Absolute size limits
# Rationale: YOLO occasionally hallucinates huge bboxes (floor, shadows)
MAX_BBOX_HEIGHT_PX = 800  # Players shouldn't exceed 800px even in 4K
MAX_BBOX_WIDTH_PX = 600
MAX_BBOX_AREA_FRACTION = 0.25  # Layer 2: Reject detections > 25% of frame area

# ============================================================================
# FISHEYE LENS CORRECTION
# ============================================================================

# Adaptive bbox expansion to account for fisheye distortion
# Players far from frame center appear tilted → need larger bbox for complete jersey crop
FISHEYE_CORRECTION_ENABLED = True  # Enable fisheye jersey ROI correction
FISHEYE_EXPANSION_STRENGTH = 0.15  # Shift strength (0.15 = ~15% shift at frame edges)

# Rationale:
# - Fisheye distortion causes players at frame edges to appear tilted
# - Axis-aligned bboxes cut off tilted players → incomplete jersey crops
# - Incomplete crops → wrong HSV → false "appearance discontinuity" splits in Pass 2A
# - Solution: Radially expand bboxes based on distance from frame center

# ============================================================================
# BYTETRACK PARAMETERS
# ============================================================================

TRACK_HIGH_THRESH = 0.6  # High confidence threshold for track initialization
TRACK_LOW_THRESH = 0.1   # Low confidence threshold for track continuation
TRACK_BUFFER = 30        # Number of frames to keep lost tracks
MIN_TRACK_LENGTH = 5     # Minimum track length in frames
PLAYER_TRACKER_CONFIG = "config/bytetrack_fast.yaml"  # Disable GMC for faster tracking

# ============================================================================
# FRAGMENT PARAMETERS
# ============================================================================

MIN_FRAGMENT_LENGTH = 15  # Frames - but keep shorter ones, mark as low_quality
MAX_FRAGMENT_GAP = 60     # Legacy gap constant (not used for Pass 2C ghost duration)
MERGE_CONSECUTIVE_SHORT = False  # Fragments must never merge (CLAUDE.md Section 8)

# ============================================================================
# HSV CLUSTERING
# ============================================================================

KMEANS_N_CLUSTERS = 2  # Team A vs Team B
HSV_BINS = 8           # 8x8x8 = 512 bins for histogram
HSV_HISTOGRAM_SIZE = HSV_BINS ** 3  # 512 bins total

# Pass 3C compactness diagnostics and ambiguity guardrails.
# Compactness is measured as mean L2 distance to cluster centroid in HSV space.
COMPACT_CLUSTER_MAX_MEAN_DISTANCE = 0.35
COMPACTNESS_DIFF_MIN = 0.05

# Optional quality gates for Pass 3C K-means input selection.
# Applied only when quality metadata exists on fragments.
KMEANS_MIN_FRAGMENT_QUALITY_SCORE = 0.40
KMEANS_MIN_HSV_CONSISTENCY = 0.35

# ============================================================================
# PASS 3A JERSEY EVIDENCE THRESHOLDS
# ============================================================================

# Pass 3A candidate-edge eligibility constraints.
MAX_IDENTITY_GAP = 300  # Maximum temporal gap (frames) for identity continuation edges
MAX_PLAYER_SPEED = 50.0  # Max plausible player speed (px/frame) for Pass 3 spatial feasibility gate
TRACK_CONTINUITY_GAP = 120  # Max gap for same-track continuity candidate edges
MIN_CANDIDATE_SCORE = 0.35  # Minimum overall score required to keep a Pass 3A edge

# Pass 3B score thresholds for hard/soft constraints.
PASS3B_SOFT_THRESHOLD = 0.35
PASS3B_MUST_SAME_THRESHOLD = 0.85

# Pass 3C short-fragment team stabilization.
# Very short fragments are prone to noisy team assignment and may momentarily
# flicker before a nearby long continuation fragment appears.
PASS3_SHORT_FRAGMENT_MAX_FRAMES = 45
PASS3_SHORT_FRAGMENT_NEIGHBOR_MIN_FRAMES = 180
PASS3_SHORT_FRAGMENT_MAX_TEMPORAL_GAP = 40
PASS3_SHORT_FRAGMENT_MAX_SPATIAL_DISTANCE = 50.0
PASS3_SHORT_FRAGMENT_MIN_SOFT_SCORE = 0.36

# Pass 3C experiment mode: lock teams from fragment color/subcluster evidence only.
# When enabled, Pass 3C skips team-cap rebalance, GT calibration, and SOFT_SAME merges.
PASS3_COLOR_ONLY_TEAM_ASSIGNMENT = True

# Minimum support required to emit jersey evidence for a fragment.
# Fragments below either threshold produce no jersey candidate — classifier noise suppressed.
MIN_JERSEY_DETECTIONS = 10   # Minimum raw detection count
MIN_JERSEY_RATIO = 0.02      # Minimum fraction of fragment detections
PASS3A_MIN_JERSEY_AVG_CONF = 0.60  # Min mean confidence for emitted jersey evidence.
PASS3_USE_JERSEY_PROBS_FALLBACK = False  # Prefer explicit jersey detections for stability.
PASS3_JERSEY_PROB_FALLBACK_MIN = 0.92
PASS3_JERSEY_PROB_FALLBACK_MARGIN = 0.20

# Global jersey allocation (Pass 3C)
PASS3_GLOBAL_JERSEY_MIN_SUPPORT = 0.55
PASS3_GLOBAL_JERSEY_MIN_DOMINANCE = 1.20
PASS3_GLOBAL_JERSEY_CURRENT_LABEL_BONUS = 0.20
# Raised evidence quality gate for Pass 3A legacy candidates.
# Prevents low-confidence jersey bleed-through from contaminating downstream constraints.

# ============================================================================
# JERSEY TEMPORAL EXCLUSIVITY
# ============================================================================

# PoC model/classes are restricted to observed jerseys in current clips.
JERSEY_NUMBERS = [4, 7, 10]
MAX_CONCURRENT_PLAYERS = 12  # Futsal regulation: 6v6

# Jersey temporal exclusivity thresholds (Pass 2A - Trigger 3 ownership voting)
# Per IMPLEMENTATION_PLAN.md: Use voting/consensus, not first appearance
# Tightened (2026-02-20): raised from 3/0.5/0.5 to filter cross-player crop contamination
JERSEY_MIN_OBSERVATIONS = 5   # Minimum jersey observations to consider it "owned" by a fragment
JERSEY_MIN_CONFIDENCE = 0.6   # Minimum confidence for jersey observations to count
JERSEY_MAJORITY_THRESHOLD = 0.85  # Fragment must have ≥85% observations with same jersey to "own" it
# Raised from 0.70: requires clear jersey dominance before fragment "owns" a number.
# Reduces false temporal conflicts where jersey classifier is inconsistent.

# Jersey change split thresholds (Pass 2A - Trigger 2)
# Both sides must exceed JERSEY_CHANGE_MIN_CONF; new jersey confirmed by N consecutive sampled obs
JERSEY_CHANGE_MIN_CONF = 0.65          # Min confidence on BOTH sampled obs to trigger jersey-change split
JERSEY_CHANGE_CONFIRM_OBSERVATIONS = 2  # Consecutive confirming sampled obs needed before split fires

# Jersey temporal conflict minimum overlap (Pass 2A - Trigger 3)
# Only split if fragments overlap by at least this many frames (brief boundary noise is skipped)
JERSEY_TEMPORAL_MIN_OVERLAP_FRAMES = 150  # Min frame overlap to trigger a temporal conflict split
# Raised from 5: jersey classifier noise causes same jersey on different tracks for many frames.
# Only a genuine sustained conflict (150+ overlapping frames) triggers a split. ~6s at 25fps.
JERSEY_TEMPORAL_MIN_CONFIDENCE = 0.85  # Stricter confidence for T3 ownership voting.
JERSEY_TEMPORAL_MIN_DENSITY = 0.10     # Min confident-jersey observation density within fragment.

# Optional jersey-to-player display labels for visualization overlays.
# Used only by visualization layer (does NOT affect identity inference or validation).
PLAYER_NAME_BY_JERSEY = {
	4: "Spyros",
	7: "Rick",
	10: "Kiki",
}

# ============================================================================
# GHOST PARAMETERS
# ============================================================================

# Dynamic level (high water mark) - only increases, never decreases
INITIAL_LEVEL_FRAMES = 10  # Frames to establish initial level
DYNAMIC_LEVEL_MAX = 12     # Cap at 12 for futsal
MAX_GHOST_COUNT = 6        # Maximum number of ghosts at any time

# ============================================================================
# BALL DETECTION & INTERPOLATION
# ============================================================================

PASS1_BALL_BATCH_SIZE = 8  # Batch size for Pass 1 ball detector inference

# InferenceSlicer configuration for small-object recall
# Matches training data tiling strategy (see utils/tile_dataset.py)
USE_INFERENCE_SLICER = True   # Enable tiling for improved ball recall
SLICER_OVERLAP_PX = 200       # Pixel overlap between tiles (matches training augmentation)
SLICER_IOU_THRESHOLD = 0.1    # NMS threshold for merging tile detections (same as model IOU)

MAX_BALL_GAP_FRAMES = 30  # Maximum gap for interpolation (1 second at 30 FPS)
BALL_INTERPOLATION_METHOD = "linear"  # or "kalman"
BALL_MAX_SPEED_PX_PER_FRAME = 100  # Physical limit for validation

# ============================================================================
# VALIDATION TOLERANCES
# ============================================================================

# R2: Every player has a team
MAX_UNKNOWN_FRAGMENTS = 0  # After Pass 3C, no "unknown" allowed

# Team size constraints
MIN_TEAM_SIZE = 5  # Allow 5-7 players per team
MAX_TEAM_SIZE = 7
MAX_TEAM_SIZE_VIOLATION_FRAMES = 5  # Allow brief violations

# R3: One jersey = one player
MAX_CONCURRENT_JERSEY_VIOLATIONS = 0  # Strict temporal exclusivity

# R4: Player continuity (duration-aware in Pass 2)
# Tolerates: tracker jitter (1-2 frames) + occlusion-based track re-IDs (up to ~60 frames)
# Rationale: ByteTrack can fragment one player into multiple track_ids during occlusions
# Pass 3 will merge these based on jersey/team identity, reducing count back to ≤12
MAX_R4_VIOLATION_CONSECUTIVE_FRAMES = 60  # ~2 seconds at 30 FPS

# ============================================================================
# OUTPUT DIRECTORY STRUCTURE
# ============================================================================

VIDEOS_INPUT_DIR = Path("videos/input")
VIDEOS_OUTPUT_DIR = Path("videos/output")

# Artifact filenames
PASS1_RAW_JSON = "pass1_raw.json"
PASS1_VALIDATION_JSON = "pass1_validation.json"

PASS2_FRAGMENTS_JSON = "pass2_fragments.json"
PASS2B_SCORED_FRAGMENTS_JSON = "pass2b_scored_fragments.json"
PASS2_GHOSTS_JSON = "pass2_ghosts.json"
PASS2_VALIDATION_JSON = "pass2_validation.json"

PASS3A_CANDIDATES_JSON = "pass3a_candidates.json"
PASS3_CANDIDATES_JSON = "pass3_candidates.json"
PASS3_CONSTRAINTS_JSON = "pass3_constraints.json"
PASS3_IDENTITY_COMMIT_JSON = "pass3_identity_commit.json"
PASS3_VALIDATION_JSON = "pass3_validation.json"

BALL_INTERPOLATION_JSON = "ball_interpolation.json"
DEBUG_METRICS_JSON = "debug_metrics.json"
VISUALIZATION_VIDEO = "visualization.mp4"

# ============================================================================
# SPLIT TRIGGERS (PASS 2A) — per IMPLEMENTATION_PLAN.md contract
# ============================================================================

# T2: Jersey Change Detection
# New jersey number must persist for >= this many frames before a split fires.
# At JERSEY_NUMBER_CLASSIFY_EVERY_N_FRAMES=5, 15 frames = 3 consecutive samples.
JERSEY_CHANGE_PERSISTENCE_FRAMES = 15

# T4: Team Assignment Discontinuity (windowed HSV + K-means clustering)
TEAM_SWITCH_WINDOW = 30           # Frames before/after split candidate
TEAM_SWITCH_HSV_THRESHOLD = 0.25  # Min 1-correlation between window means to fire.
TEAM_SWITCH_MIN_SAMPLES = 3       # Min valid HSV samples required in each window
TEAM_SWITCH_CONFIDENCE = 0.60     # Min dominant-cluster ratio required on both windows
TEAM_SWITCH_MIN_CROP_QUALITY = 0.55  # Min mean jersey-crop quality in each window
TEAM_SWITCH_SCAN_STEP = 10        # Scan a split candidate every N frames
TEAM_SWITCH_EDGE_MARGIN = 30      # Do not evaluate T4 too close to track start/end
TEAM_SWITCH_TEAM_CLUSTERS = 2     # Collapse colour clusters into 2 team-level groups
TEAM_SWITCH_PROXIMITY_BBOX_WIDTHS = 1.8  # Require nearby-player crossing evidence
TEAM_SWITCH_CROSSING_FRAME_TOLERANCE = 10  # Nearby crossing may occur a few frames around split.
TEAM_SWITCH_EDGE_RESCUE_MAX_TRACK_FRAMES = 300  # Only run edge rescue on short tracks.
TEAM_SWITCH_EDGE_RESCUE_MIN_MAIN_CONF = 0.95    # Strong dominant cluster before split.
TEAM_SWITCH_EDGE_RESCUE_PROXIMITY_BBOX_WIDTHS = 0.90  # Stricter crossing gate at edges.
TEAM_SWITCH_TRACK_MINORITY_RATIO = 0.15  # Track must show meaningful support for both teams.
TEAM_SWITCH_TRACK_MINORITY_SAMPLES = 6   # Minimum observations supporting the minority team.
TEAM_SWITCH_MIN_TEAM_MARGIN = 0.05       # Minimum per-observation team-separation margin.
TEAM_SWITCH_EDGE_RELAX_FRAMES = 45       # Near track boundary: allow split without global two-team support.
TEAM_SWITCH_EDGE_RELAX_MIN_TEAM_CONF = 0.90    # Require strong team vote confidence on both windows.
TEAM_SWITCH_EDGE_RELAX_MIN_CLUSTER_CONF = 0.50  # Require at least moderate dominant cluster confidence.

# T5: Impossible Motion Spike
PASS2_MAX_PLAYER_SPEED = 150      # Max pixels/frame for single-frame centroid jump (Pass 2A T5)

# Global HSV K-means for T4 cluster comparison
N_COLOR_CLUSTERS = 4              # 1 bib colour + ~3 shirt colours
KMEANS_SUBSAMPLE_SIZE = 3000      # Max histograms sampled for K-means fit

# Appearance drift thresholds (legacy — kept for reference, not used in new T4)
HSV_DRIFT_THRESHOLD = 0.58  # Min HSV distance to split on colour discontinuity.
# Natural player movement (angle/lighting) peaks at ~p95=0.32 across clip7/clip2.
# Genuine jersey colour swaps (e.g. orange vs black) produce 0.58+.
# Set above noise ceiling to suppress false splits from natural variation.
# Calibrated at 0.58: catches clip2 track4@500 bilateral swap (dist=0.598).
# Gap vs noise: 0.58 - 0.32 = 0.26 (comfortable margin).
HSV_WINDOW_FRAMES = 5      # ±frames to search for nearest HSV sample around a boundary
HSV_SPLIT_COOLDOWN_FRAMES = 5   # Minimal guard between HSV splits; streak persistence is the primary filter.
# Reduced from 25: the 3-frame streak requirement prevents double-splits on the same event.
# A small cooldown still prevents the exact same boundary firing twice in adjacent samples.
HSV_CHAOS_COOLDOWN_FRAMES = 5   # Chaos cooldown removed as a state-consuming suppressor.
# Previously 60 frames; reduced to match normal cooldown. The streak requirement handles
# post-split noise without blocking legitimate later splits on the same track.
HSV_SPLIT_MIN_FRAGMENT_AGE = 20  # Min frames a fragment must exist before an HSV split fires.
# Very young fragments have unstable jersey ROIs (player entering frame, partial view).
# Suppresses false splits on tracks that start and immediately show jersey noise.
# Set to 20: safely below track 9's first real split at age=25, above track 15's false at age=17.
HSV_SPLIT_STREAK_REQUIRED = 3    # Consecutive HSV samples all above threshold required before split fires.
# Clip 7 joe/armen swap sustains exactly 3 samples — verified minimum, cannot increase.
HSV_SPATIAL_CENTROID_RATIO = 0.30  # Centroid shift as fraction of bbox width for spatial sanity check.
# HSV split requires centroid shift > ratio*bbox_width OR bbox area change > 30%.
# Filters lighting-induced HSV changes where the same player doesn't move.
HSV_BASELINE_SAMPLES = 15        # Number of HSV samples used to build the per-fragment median baseline.
# Median is robust to a few early anomalous samples.  15 samples ≈ 75 frames at sample-every-5.
HSV_BASELINE_PERSIST_COUNT = 3   # Consecutive above-threshold baseline distances before drift split fires.
# Matched to HSV_SPLIT_STREAK_REQUIRED for consistency.
HSV_SPLIT_COOLDOWN_PROGRESSIVE = [5, 60, 150, 150]
# Cooldown frames after the Nth HSV split on the same track.
# [0]=5: initial (first split uses minimal cooldown to prevent same-boundary re-fire).
# [1]=60: after 1st split, 60 frames before 2nd can fire (prevents anchor-noise cascade).
# [2]=150: after 2nd split, 150 frames before 3rd. Calibrated for clip 7 joe/armen swap:
#   2nd HSV split fires at frame 635; swap at 800 → needs 635 + cooldown + 15 ≤ 800 → ≤151.
# [3]=150: same cap for subsequent splits.
# The progressive cooldown reduces false cascade splits without blocking genuine late swaps.
HSV_SPLIT_STREAK_PROGRESSIVE = [3, 3, 3, 3]
# Unused — kept for future use. Progressive streak was found to break T2 in clip 7.
# Requires sustained deviation from the baseline median, not a single-frame spike.
HSV_PROXIMITY_THRESHOLD = 2.0  # Max distance in bbox-widths to count another track as "nearby"
# Proximity gate for HSV splits: genuine ID swaps require player crossing (tracks nearby).
# Lighting changes happen to isolated players (no nearby track) → reject the split.
# 3.0 bbox-widths ≈ 300px at typical player scale. Players cross within ~1-2 widths but
# may have drifted slightly apart by the time the 3-frame streak completes (~15 frames).
# Check at curr_frame (streak completion); at 30fps players move ~50px in 15 frames → safe.

# Velocity spike thresholds
VELOCITY_SPIKE_THRESHOLD = 50  # Pixels per frame

# Confidence drop thresholds
CONFIDENCE_DROP_THRESHOLD = 0.3  # Relative drop

# ============================================================================
# FRAGMENT QUALITY THRESHOLDS
# ============================================================================

# Quality score boundaries (0-1)
QUALITY_HIGH_THRESHOLD = 0.7
QUALITY_MEDIUM_THRESHOLD = 0.4
# Below QUALITY_MEDIUM_THRESHOLD = LOW quality

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
