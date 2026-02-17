"""
Validation module for futsal tracking system.

Per CLAUDE.md Section 6 (Failure Policy):
- Fail-fast on validation errors
- Validation runs BEFORE writing JSON
- Failed pass writes nothing (not even partial artifacts)
"""

from .validator import Validator, ValidationResult
from .global_rules import (
    validate_r1_pass1_raw_truth,
    validate_r2_no_unknown_teams,
    validate_r3_jersey_temporal_exclusivity,
    validate_r4_player_continuity,
    validate_r5_ball_never_disappears,
)

__all__ = [
    "Validator",
    "ValidationResult",
    "validate_r1_pass1_raw_truth",
    "validate_r2_no_unknown_teams",
    "validate_r3_jersey_temporal_exclusivity",
    "validate_r4_player_continuity",
    "validate_r5_ball_never_disappears",
]
