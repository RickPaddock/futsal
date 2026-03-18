"""
JSON schema definitions for validation.

Per CLAUDE.md P5: JSON is source of truth.
These schemas validate the structure of all JSON artifacts.
"""

from typing import Dict, Any

# ============================================================================
# SCHEMA HELPERS
# ============================================================================

def get_bbox_schema() -> Dict[str, Any]:
    """Schema for bbox: [x1, y1, x2, y2]"""
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "Bounding box [x1, y1, x2, y2]"
    }


def get_centroid_schema() -> Dict[str, Any]:
    """Schema for centroid: [x, y]"""
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 2,
        "description": "Centroid [x, y]"
    }


def get_hsv_histogram_schema() -> Dict[str, Any]:
    """Schema for HSV histogram: 512-element array"""
    return {
        "type": "array",
        "items": {"type": "number", "minimum": 0, "maximum": 1},
        "minItems": 512,
        "maxItems": 512,
        "description": "HSV histogram (8x8x8 bins, normalized)"
    }


# ============================================================================
# PASS 1 SCHEMAS
# ============================================================================

DETECTION_SCHEMA = {
    "type": "object",
    "required": ["detection_id", "frame_idx", "bbox", "centroid", "confidence", "track_id"],
    "properties": {
        "detection_id": {"type": "string"},
        "frame_idx": {"type": "integer", "minimum": 0},
        "bbox": get_bbox_schema(),
        "centroid": get_centroid_schema(),
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "track_id": {"type": "integer"},
        "jersey_number": {"type": ["integer", "null"], "minimum": 1, "maximum": 12},
        "jersey_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "jersey_probs": {"type": ["object", "null"]},
        "hsv_histogram_jersey": {
            "anyOf": [
                {"type": "null"},
                get_hsv_histogram_schema()
            ]
        },
        "jersey_color_sampled": {"type": "boolean"},
        "jersey_roi_valid": {"type": "boolean"},
        "jersey_roi_bbox": {
            "anyOf": [
                {"type": "null"},
                get_bbox_schema()
            ]
        },
        "jersey_crop_quality": {
            "anyOf": [
                {"type": "null"},
                {"type": "number", "minimum": 0, "maximum": 1}
            ]
        },
        "is_sam_recovered": {"type": "boolean"},
        "sam_bbox": {
            "anyOf": [
                {"type": "null"},
                get_bbox_schema()
            ]
        }
    },
    "additionalProperties": False  # R1: Pass 1 is raw truth only
}


BALL_DETECTION_SCHEMA = {
    "type": "object",
    "required": ["frame_idx", "bbox", "centroid", "confidence"],
    "properties": {
        "frame_idx": {"type": "integer", "minimum": 0},
        "bbox": get_bbox_schema(),
        "centroid": get_centroid_schema(),
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    }
}


PASS1_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["video_name", "fps", "width", "height", "total_frames", "detections", "ball_detections"],
    "properties": {
        "video_name": {"type": "string"},
        "fps": {"type": "number", "minimum": 1},
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
        "total_frames": {"type": "integer", "minimum": 1},
        "processed_start_frame": {"type": "integer", "minimum": 0},
        "processed_end_frame_exclusive": {
            "anyOf": [
                {"type": "null"},
                {"type": "integer", "minimum": 0}
            ]
        },
        "detections": {
            "type": "array",
            "items": DETECTION_SCHEMA
        },
        "ball_detections": {
            "type": "array",
            "items": BALL_DETECTION_SCHEMA
        }
    }
}


# ============================================================================
# PASS 2A SCHEMAS
# ============================================================================

FRAGMENT_SCHEMA = {
    "type": "object",
    "required": ["fragment_id", "original_track_id", "start_frame", "end_frame", "detection_ids"],
    "properties": {
        "fragment_id": {"type": "string", "pattern": "^F\\d{6}$"},  # Format: F000001
        "original_track_id": {"type": "integer"},
        "start_frame": {"type": "integer", "minimum": 0},
        "end_frame": {"type": "integer", "minimum": 0},
        "detection_ids": {"type": "array", "items": {"type": "string"}},
        "split_reason": {"type": ["string", "null"]},
        "split_trigger_frame": {"type": ["integer", "null"], "minimum": 0},
        "split_rule_id": {"type": ["string", "null"]},
        "parent_fragment_id": {"type": ["string", "null"]},
        "dominant_color_cluster_id": {"type": ["integer", "null"]},
        "dominant_team_cluster_id": {"type": ["integer", "null"]},
        "dominant_team_cluster_confidence": {"type": ["number", "null"]}
    }
}


PASS2A_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["fragments", "split_log"],
    "properties": {
        "fragments": {"type": "array", "items": FRAGMENT_SCHEMA},
        "split_log": {"type": "array", "items": {"type": "object"}}
    }
}


# ============================================================================
# PASS 2B SCHEMAS
# ============================================================================

SCORED_FRAGMENT_SCHEMA = {
    "type": "object",
    "required": [
        "fragment_id", "original_track_id", "start_frame", "end_frame", "detection_ids",
        "quality", "quality_score"
    ],
    "properties": {
        **FRAGMENT_SCHEMA["properties"],
        "quality": {"type": "string", "enum": ["high", "medium", "low", "ghost"]},
        "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
        "quality_reasons": {"type": "array", "items": {"type": "string"}},
        "avg_confidence": {"type": "number"},
        "min_confidence": {"type": "number"},
        "avg_bbox_stability": {"type": "number"},
        "jersey_consistency": {"type": "number"},
        "hsv_consistency": {"type": "number"},
        "jersey_observability_score": {"type": "number", "minimum": 0, "maximum": 1},
        "motion_smoothness_score": {"type": "number", "minimum": 0, "maximum": 1}
    }
}


# ============================================================================
# PASS 2C SCHEMAS
# ============================================================================

GHOST_FRAGMENT_SCHEMA = {
    "type": "object",
    "required": [
        "fragment_id", "original_track_id", "start_frame", "end_frame",
        "is_ghost", "ghost_reason", "estimated_position", "estimated_centroid",
        "source_fragment_id", "source_track_id"
    ],
    "properties": {
        **SCORED_FRAGMENT_SCHEMA["properties"],
        "is_ghost": {"type": "boolean", "const": True},
        "ghost_reason": {"type": "string"},
        "estimated_position": get_bbox_schema(),
        "estimated_centroid": get_centroid_schema(),
        "source_fragment_id": {"type": "string"},
        "source_track_id": {"type": "integer"}
    }
}


PASS2C_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["fragments", "ghosts"],
    "properties": {
        "fragments": {"type": "array", "items": SCORED_FRAGMENT_SCHEMA},
        "ghosts": {"type": "array", "items": GHOST_FRAGMENT_SCHEMA},
        "ghost_creation_log": {"type": "array"},
        "level_timeline": {"type": "array"}
    }
}


# ============================================================================
# PASS 3A / 3B SCHEMAS
# ============================================================================

IDENTITY_CANDIDATE_SCHEMA = {
    "type": "object",
    "required": ["fragment_id", "team_evidence", "jersey_evidence", "player_evidence"],
    "properties": {
        "fragment_id": {"type": "string"},
        "candidate_team": {
            "anyOf": [
                {"type": "null"},
                {"type": "string", "enum": ["team_a", "team_b", "unknown"]},
            ]
        },
        "candidate_jersey": {
            "anyOf": [
                {"type": "null"},
                {"type": "integer", "minimum": 1, "maximum": 12},
            ]
        },
        "candidate_player_id": {
            "anyOf": [
                {"type": "null"},
                {"type": "string"},
            ]
        },
        "team_evidence": {"type": "object", "additionalProperties": {"type": "number"}},
        "jersey_evidence": {"type": "object", "additionalProperties": {"type": "number"}},
        "player_evidence": {"type": "object", "additionalProperties": {"type": "number"}},
    },
}


PASS3A_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["candidates"],
    "properties": {
        "candidates": {"type": "array", "items": IDENTITY_CANDIDATE_SCHEMA},
    },
}


IDENTITY_CANDIDATE_EDGE_SCHEMA = {
    "type": "object",
    "required": [
        "fragment_a",
        "fragment_b",
        "temporal_gap",
        "spatial_distance",
        "track_continuity_score",
        "velocity_consistency_score",
        "appearance_similarity",
        "jersey_similarity",
        "temporal_gap_score",
        "overall_candidate_score",
    ],
    "properties": {
        "fragment_a": {"type": "string", "pattern": "^F\\d{6}$"},
        "fragment_b": {"type": "string", "pattern": "^F\\d{6}$"},
        "temporal_gap": {"type": "integer", "minimum": 1},
        "spatial_distance": {"type": "number", "minimum": 0},
        "track_continuity_score": {"type": "number", "minimum": 0, "maximum": 1},
        "velocity_consistency_score": {"type": "number", "minimum": 0, "maximum": 1},
        "appearance_similarity": {"type": "number", "minimum": 0, "maximum": 1},
        "jersey_similarity": {"type": "number", "minimum": 0, "maximum": 1},
        "temporal_gap_score": {"type": "number", "minimum": 0, "maximum": 1},
        "overall_candidate_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


PASS3A_EDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["candidates"],
    "properties": {
        "candidates": {"type": "array", "items": IDENTITY_CANDIDATE_EDGE_SCHEMA},
    },
}


CONSTRAINT_SCHEMA = {
    "type": "object",
    "required": ["constraint_id", "constraint_type", "fragment_ids", "weight", "reason"],
    "properties": {
        "constraint_id": {"type": "string"},
        "constraint_type": {"type": "string", "enum": ["must_same", "cannot_same", "soft_same"]},
        "fragment_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
        },
        "value": {},
        "weight": {"type": "number"},
        "reason": {"type": "string"},
    },
}


PASS3B_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["constraints", "constraint_graph"],
    "properties": {
        "constraints": {"type": "array", "items": CONSTRAINT_SCHEMA},
        "constraint_graph": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
}


# ============================================================================
# PASS 3C SCHEMAS
# ============================================================================

COMMITTED_IDENTITY_SCHEMA = {
    "type": "object",
    "required": [
        "fragment_id", "player_id", "team", "jersey_number",
        "assignment_method", "assignment_confidence"
    ],
    "properties": {
        "fragment_id": {"type": "string"},
        "player_id": {"type": "string", "pattern": "^P\\d{2}_(team_a|team_b)$"},  # Format: P07_team_a
        "team": {"type": "string", "enum": ["team_a", "team_b"]},  # R2: No "unknown"
        "jersey_number": {
            "anyOf": [
                {"type": "null"},
                {"type": "integer", "minimum": 1, "maximum": 12}
            ]
        },
        "assignment_method": {
            "type": "string",
            "enum": ["kmeans", "inherited", "constraint_solved", "ghost_inherited"]
        },
        "assignment_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "assignment_reasons": {"type": "array", "items": {"type": "string"}},
        "_locked_team": {
            "anyOf": [
                {"type": "null"},
                {"type": "string", "enum": ["team_a", "team_b", "unknown"]}
            ]
        }
    }
}


PASS3C_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["identities"],
    "properties": {
        "identities": {"type": "array", "items": COMMITTED_IDENTITY_SCHEMA},
        "solver_log": {"type": "object"},
        "unresolved_conflicts": {"type": "array"}  # Should be empty (fail-fast if not)
    }
}


# ============================================================================
# BALL INTERPOLATION SCHEMAS
# ============================================================================

BALL_POSITION_SCHEMA = {
    "type": "object",
    "required": ["frame_idx", "state", "confidence"],
    "properties": {
        "frame_idx": {"type": "integer", "minimum": 0},
        "state": {"type": "string", "enum": ["real", "interpolated", "unknown", "out_of_play"]},
        "centroid": {
            "anyOf": [
                {"type": "null"},
                get_centroid_schema()
            ]
        },
        "bbox": {
            "anyOf": [
                {"type": "null"},
                get_bbox_schema()
            ]
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    }
}


BALL_INTERPOLATION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["ball_positions", "interpolation_method", "total_frames"],
    "properties": {
        "ball_positions": {"type": "array", "items": BALL_POSITION_SCHEMA},
        "interpolation_method": {"type": "string", "enum": ["linear", "kalman"]},
        "total_frames": {"type": "integer", "minimum": 1},
        "interpolated_frames": {"type": "array", "items": {"type": "integer"}},
        "gap_summary": {"type": "array"}
    }
}


# ============================================================================
# BIRD'S-EYE PROJECTION SCHEMAS
# ============================================================================

BIRDSEYE_PLAYER_POSITION_SCHEMA = {
    "type": "object",
    "required": [
        "frame_idx",
        "fragment_id",
        "player_id",
        "team",
        "track_id",
        "is_ghost",
        "is_estimated",
        "image_bbox",
        "image_anchor",
        "court_position",
        "render_position",
    ],
    "properties": {
        "frame_idx": {"type": "integer", "minimum": 0},
        "fragment_id": {"type": "string"},
        "player_id": {"type": "string", "pattern": "^P\\d{2}_(team_a|team_b)$"},
        "team": {"type": "string", "enum": ["team_a", "team_b"]},
        "jersey_number": {
            "anyOf": [
                {"type": "null"},
                {"type": "integer", "minimum": 1, "maximum": 12}
            ]
        },
        "track_id": {"type": "integer"},
        "is_ghost": {"type": "boolean"},
        "is_estimated": {"type": "boolean"},
        "image_bbox": get_bbox_schema(),
        "image_anchor": get_centroid_schema(),
        "raw_image_anchor": {
            "anyOf": [
                {"type": "null"},
                get_centroid_schema(),
            ]
        },
        "raw_court_position": {
            "anyOf": [
                {"type": "null"},
                get_centroid_schema(),
            ]
        },
        "raw_render_position": {
            "anyOf": [
                {"type": "null"},
                get_centroid_schema(),
            ]
        },
        "stabilization_trust": {"type": "number", "minimum": 0, "maximum": 1},
        "court_position": get_centroid_schema(),
        "render_position": get_centroid_schema(),
    },
}


BIRDSEYE_BALL_FRAME_SCHEMA = {
    "type": "object",
    "required": ["frame_idx", "state", "confidence"],
    "properties": {
        "frame_idx": {"type": "integer", "minimum": 0},
        "state": {"type": "string", "enum": ["real", "interpolated", "unknown", "out_of_play"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "image_bbox": {
            "anyOf": [
                {"type": "null"},
                get_bbox_schema(),
            ]
        },
        "image_position": {
            "anyOf": [
                {"type": "null"},
                get_centroid_schema(),
            ]
        },
        "court_position": {
            "anyOf": [
                {"type": "null"},
                get_centroid_schema(),
            ]
        },
        "render_position": {
            "anyOf": [
                {"type": "null"},
                get_centroid_schema(),
            ]
        },
    },
}


BIRDSEYE_FRAME_SCHEMA = {
    "type": "object",
    "required": ["frame_idx", "players", "ball"],
    "properties": {
        "frame_idx": {"type": "integer", "minimum": 0},
        "players": {"type": "array", "items": BIRDSEYE_PLAYER_POSITION_SCHEMA},
        "ball": BIRDSEYE_BALL_FRAME_SCHEMA,
    },
}


BIRDSEYE_PROJECTION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "video_name",
        "fps",
        "total_frames",
        "court_length_m",
        "court_width_m",
        "output_pixel_scale",
        "frames",
    ],
    "properties": {
        "video_name": {"type": "string"},
        "fps": {"type": "number", "minimum": 1},
        "total_frames": {"type": "integer", "minimum": 1},
        "processed_start_frame": {"type": "integer", "minimum": 0},
        "processed_end_frame_exclusive": {
            "anyOf": [
                {"type": "null"},
                {"type": "integer", "minimum": 0},
            ]
        },
        "court_length_m": {"type": "number", "exclusiveMinimum": 0},
        "court_width_m": {"type": "number", "exclusiveMinimum": 0},
        "output_pixel_scale": {"type": "integer", "minimum": 1},
        "frames": {"type": "array", "items": BIRDSEYE_FRAME_SCHEMA},
        "diagnostics": {"type": "object"},
    },
}


# ============================================================================
# DEBUG METRICS SCHEMAS
# ============================================================================

FRAME_METRICS_SCHEMA = {
    "type": "object",
    "required": [
        "frame_idx", "player_count", "tracked_count", "ghost_count",
        "team_a_count", "team_b_count", "unknown_count", "jersey_conflicts"
    ],
    "properties": {
        "frame_idx": {"type": "integer", "minimum": 0},
        "player_count": {"type": "integer", "minimum": 0},
        "tracked_count": {"type": "integer", "minimum": 0},
        "ghost_count": {"type": "integer", "minimum": 0},
        "team_a_count": {"type": "integer", "minimum": 0},
        "team_b_count": {"type": "integer", "minimum": 0},
        "unknown_count": {"type": "integer", "minimum": 0},
        "jersey_conflicts": {"type": "array", "items": {"type": "string"}},
    }
}


DEBUG_METRICS_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["video_name", "total_frames", "frame_metrics"],
    "properties": {
        "video_name": {"type": "string"},
        "total_frames": {"type": "integer", "minimum": 0},
        "frame_metrics": {"type": "array", "items": FRAME_METRICS_SCHEMA},
        "total_identity_changes": {"type": "integer", "minimum": 0},
        "total_jersey_conflicts": {"type": "integer", "minimum": 0},
        "total_unknown_frames": {"type": "integer", "minimum": 0},
        "avg_player_count": {"type": "number", "minimum": 0},
        "cluster_compactness_a": {"type": ["number", "null"]},
        "cluster_compactness_b": {"type": ["number", "null"]},
        "compactness_ratio": {"type": ["number", "null"]},
    }
}


# ============================================================================
# VALIDATION SCHEMAS
# ============================================================================

VALIDATION_VIOLATION_SCHEMA = {
    "type": "object",
    "required": ["rule", "severity", "message"],
    "properties": {
        "rule": {"type": "string"},
        "severity": {"type": "string", "enum": ["error", "warning"]},
        "message": {"type": "string"},
        "frame_idx": {"type": ["integer", "null"]},
        "fragment_id": {"type": ["string", "null"]},
        "details": {"type": "object"}
    }
}


VALIDATION_RESULT_SCHEMA = {
    "type": "object",
    "required": ["passed", "timestamp", "pass_name"],
    "properties": {
        "passed": {"type": "boolean"},
        "violations": {"type": "array", "items": VALIDATION_VIOLATION_SCHEMA},
        "warnings": {"type": "array", "items": VALIDATION_VIOLATION_SCHEMA},
        "timestamp": {"type": "string"},  # ISO 8601
        "pass_name": {"type": "string"},
        "diagnostics": {"type": "object"},
    }
}


# ============================================================================
# SCHEMA REGISTRY
# ============================================================================

SCHEMA_REGISTRY = {
    "pass1_raw": PASS1_OUTPUT_SCHEMA,
    "pass2_fragments": PASS2A_OUTPUT_SCHEMA,
    "pass2b_scored_fragments": PASS2A_OUTPUT_SCHEMA,
    "pass2_ghosts": PASS2C_OUTPUT_SCHEMA,
    "pass3a_candidates": PASS3A_EDGE_OUTPUT_SCHEMA,
    "pass3_candidates": PASS3A_OUTPUT_SCHEMA,
    "pass3_constraints": PASS3B_OUTPUT_SCHEMA,
    "pass3_identity_commit": PASS3C_OUTPUT_SCHEMA,
    "ball_interpolation": BALL_INTERPOLATION_OUTPUT_SCHEMA,
    "birdseye_projection": BIRDSEYE_PROJECTION_OUTPUT_SCHEMA,
    "debug_metrics": DEBUG_METRICS_OUTPUT_SCHEMA,
    "validation_result": VALIDATION_RESULT_SCHEMA,
}


def get_schema(artifact_name: str) -> Dict[str, Any]:
    """
    Get JSON schema for a specific artifact.

    Args:
        artifact_name: Name of the artifact (e.g., "pass1_raw")

    Returns:
        JSON schema dictionary

    Raises:
        KeyError: If artifact_name not found in registry
    """
    return SCHEMA_REGISTRY[artifact_name]
