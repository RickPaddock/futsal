# PROV: FUSBAL.PIPELINE.DIAGNOSTICS_KEYS.01
# REQ: SYS-ARCH-15, FUSBAL-V1-TRUST-001
# WHY: Centralize all diagnostics key constants to prevent drift and ensure consistency.

"""
Canonical diagnostics key constants used across all pipeline modules.

Import and use these constants instead of hardcoded strings to ensure consistency
between code and documentation.
"""

# Detection-related diagnostics
FRAME_INDEX = "frame_index"
GATING_REASON = "gating_reason"
NUM_CANDIDATES = "num_candidates"
NUM_EMITTED = "num_emitted"

# MOT-related diagnostics
ASSOCIATION_SCORE = "association_score"
REASON = "reason"
DISTANCE_PX = "distance_px"

# Team assignment diagnostics
SMOOTHING = "smoothing"
WINDOW_FRAMES = "window_frames"
HYSTERESIS = "hysteresis"
COLOR_EVIDENCE = "color_evidence"
P_A = "p_a"
P_B = "p_b"
UNKNOWN_REASON = "unknown_reason"

# Ball tracking diagnostics
MISSING_REASON = "missing_reason"
JUMP_PX = "jump_px"

# Unknown/ambiguity reasons
UNKNOWN_REASON_NO_COLOR = "no_color_evidence"
UNKNOWN_REASON_LOW_CONFIDENCE = "low_confidence"
UNKNOWN_REASON_AMBIGUOUS = "ambiguous_association"

# Break reasons (also used in diagnostics)
BREAK_OCCLUSION = "occlusion"
BREAK_AMBIGUOUS = "ambiguous_association"
BREAK_OUT_OF_VIEW = "out_of_view"
BREAK_DETECTOR_MISSING = "detector_missing"
BREAK_MANUAL_RESET = "manual_reset"

# Gating reasons
GATING_NONE = "none"
GATING_LOW_CONFIDENCE = "low_confidence_suppressed"

# Missing reasons (ball tracker)
MISSING_DETECTOR = "detector_missing"
MISSING_LOW_CONF = "low_confidence"
MISSING_JUMP_REJECTED = "jump_rejected"