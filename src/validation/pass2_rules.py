"""
Pass 2 validation rules.

Per CLAUDE.md Section 5 (Pass 2: Fragmentation, Scoring, Ghosts):
- Pass 2A: Fragment validity (no overlaps, complete coverage)
- Pass 2B: Quality scoring validity
- Pass 2C: Ghost fragment validity (high water mark, max 12 concurrent)
"""

from typing import List, Dict, Set, Tuple
from ..core.data_models import (
    Fragment,
    ScoredFragment,
    GhostFragment,
    Pass2AOutput,
    Pass2BOutput,
    Pass2COutput,
    ValidationViolation,
)
from ..core.types import FragmentQuality


def validate_pass2a_fragments(pass2a_output: Pass2AOutput) -> List[ValidationViolation]:
    """
    Validate Pass 2A fragments.

    Checks:
    - No temporal overlaps within same track
    - Fragment IDs are unique
    - start_frame <= end_frame
    - detection_ids are not empty

    Args:
        pass2a_output: Pass 2A output data

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    seen_fragment_ids: Set[str] = set()

    # Group fragments by track_id
    track_fragments: Dict[int, List[Tuple[str, int, int]]] = {}  # track_id -> [(fragment_id, start, end)]

    for fragment in pass2a_output.fragments:
        # Check fragment_id uniqueness
        if fragment.fragment_id in seen_fragment_ids:
            violations.append(
                ValidationViolation(
                    rule="PASS2A_DUPLICATE_ID",
                    severity="error",
                    message=f"Duplicate fragment_id: {fragment.fragment_id}",
                    fragment_id=fragment.fragment_id,
                    details={"fragment_id": fragment.fragment_id},
                )
            )
        seen_fragment_ids.add(fragment.fragment_id)

        # Check start_frame <= end_frame
        if fragment.start_frame > fragment.end_frame:
            violations.append(
                ValidationViolation(
                    rule="PASS2A_INVALID_RANGE",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} has start_frame={fragment.start_frame} > end_frame={fragment.end_frame}",
                    fragment_id=fragment.fragment_id,
                    details={
                        "fragment_id": fragment.fragment_id,
                        "start_frame": fragment.start_frame,
                        "end_frame": fragment.end_frame,
                    },
                )
            )

        # Check detection_ids not empty
        if len(fragment.detection_ids) == 0:
            violations.append(
                ValidationViolation(
                    rule="PASS2A_EMPTY_DETECTIONS",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} has no detection_ids",
                    fragment_id=fragment.fragment_id,
                    details={"fragment_id": fragment.fragment_id},
                )
            )

        # Group by track
        track_id = fragment.original_track_id
        if track_id not in track_fragments:
            track_fragments[track_id] = []

        track_fragments[track_id].append((
            fragment.fragment_id,
            fragment.start_frame,
            fragment.end_frame,
        ))

    # Check for temporal overlaps within same track
    for track_id, frags in track_fragments.items():
        # Sort by start frame
        frags = sorted(frags, key=lambda x: x[1])

        for i in range(len(frags) - 1):
            frag_a_id, start_a, end_a = frags[i]
            frag_b_id, start_b, end_b = frags[i + 1]

            # Check overlap
            if start_b <= end_a:
                violations.append(
                    ValidationViolation(
                        rule="PASS2A_TEMPORAL_OVERLAP",
                        severity="error",
                        message=f"Track {track_id}: fragments {frag_a_id} and {frag_b_id} overlap in time",
                        details={
                            "track_id": track_id,
                            "fragment_a": frag_a_id,
                            "fragment_b": frag_b_id,
                            "overlap_start": start_b,
                            "overlap_end": min(end_a, end_b),
                        },
                    )
                )

    return violations


def validate_pass2b_quality_scoring(pass2b_output: Pass2BOutput) -> List[ValidationViolation]:
    """
    Validate Pass 2B quality scoring.

    Checks:
    - Quality enum valid (HIGH, MEDIUM, LOW, GHOST)
    - Quality score in [0, 1]
    - Quality consistency metrics are valid

    Args:
        pass2b_output: Pass 2B output data

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    for fragment in pass2b_output.fragments:
        # Check quality enum
        quality_str = fragment.quality.value if isinstance(fragment.quality, FragmentQuality) else fragment.quality
        if quality_str not in ["high", "medium", "low", "ghost"]:
            violations.append(
                ValidationViolation(
                    rule="PASS2B_INVALID_QUALITY",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} has invalid quality='{quality_str}'",
                    fragment_id=fragment.fragment_id,
                    details={
                        "fragment_id": fragment.fragment_id,
                        "quality": quality_str,
                    },
                )
            )

        # Check quality score range
        if not (0 <= fragment.quality_score <= 1):
            violations.append(
                ValidationViolation(
                    rule="PASS2B_INVALID_SCORE",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} quality_score={fragment.quality_score} out of range [0, 1]",
                    fragment_id=fragment.fragment_id,
                    details={
                        "fragment_id": fragment.fragment_id,
                        "quality_score": fragment.quality_score,
                    },
                )
            )

        # Check consistency metrics are valid (0-1 range)
        metrics = {
            "avg_confidence": fragment.avg_confidence,
            "min_confidence": fragment.min_confidence,
            "avg_bbox_stability": fragment.avg_bbox_stability,
            "jersey_consistency": fragment.jersey_consistency,
            "hsv_consistency": fragment.hsv_consistency,
        }

        for metric_name, value in metrics.items():
            if not (0 <= value <= 1):
                violations.append(
                    ValidationViolation(
                        rule="PASS2B_INVALID_METRIC",
                        severity="warning",
                        message=f"Fragment {fragment.fragment_id} {metric_name}={value} out of range [0, 1]",
                        fragment_id=fragment.fragment_id,
                        details={
                            "fragment_id": fragment.fragment_id,
                            "metric_name": metric_name,
                            "metric_value": value,
                        },
                    )
                )

    return violations


def validate_pass2c_ghosts(pass2c_output: Pass2COutput) -> List[ValidationViolation]:
    """
    Validate Pass 2C ghost fragments.

    Checks:
    - Ghost quality = GHOST
    - is_ghost = True
    - Ghost has source_fragment_id and source_track_id
    - Ghost position is valid
    - R4: Max 12 concurrent players (tracked + ghosts)

    Args:
        pass2c_output: Pass 2C output data

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    # Check each ghost
    for ghost in pass2c_output.ghosts:
        # Check quality = GHOST
        quality_str = ghost.quality.value if isinstance(ghost.quality, FragmentQuality) else ghost.quality
        if quality_str != "ghost":
            violations.append(
                ValidationViolation(
                    rule="PASS2C_GHOST_QUALITY",
                    severity="error",
                    message=f"Ghost {ghost.fragment_id} has quality='{quality_str}' (expected 'ghost')",
                    fragment_id=ghost.fragment_id,
                    details={
                        "fragment_id": ghost.fragment_id,
                        "quality": quality_str,
                    },
                )
            )

        # Check is_ghost = True
        if not ghost.is_ghost:
            violations.append(
                ValidationViolation(
                    rule="PASS2C_GHOST_FLAG",
                    severity="error",
                    message=f"Ghost {ghost.fragment_id} has is_ghost=False",
                    fragment_id=ghost.fragment_id,
                    details={"fragment_id": ghost.fragment_id},
                )
            )

        # Check source fields exist
        if not ghost.source_fragment_id:
            violations.append(
                ValidationViolation(
                    rule="PASS2C_GHOST_NO_SOURCE",
                    severity="error",
                    message=f"Ghost {ghost.fragment_id} missing source_fragment_id",
                    fragment_id=ghost.fragment_id,
                    details={"fragment_id": ghost.fragment_id},
                )
            )

        # Check estimated position is valid
        from ..utils.geometry import bbox_area

        if bbox_area(ghost.estimated_position) <= 0:
            violations.append(
                ValidationViolation(
                    rule="PASS2C_GHOST_INVALID_POSITION",
                    severity="error",
                    message=f"Ghost {ghost.fragment_id} has invalid estimated_position (zero area)",
                    fragment_id=ghost.fragment_id,
                    details={
                        "fragment_id": ghost.fragment_id,
                        "estimated_position": ghost.estimated_position,
                    },
                )
            )

    # R4: Player continuity (max 12 concurrent)
    all_fragments = list(pass2c_output.fragments) + list(pass2c_output.ghosts)

    # Find total_frames (max end_frame)
    total_frames = max((f.end_frame for f in all_fragments), default=0) + 1

    from .global_rules import validate_r4_player_continuity
    violations.extend(validate_r4_player_continuity(all_fragments, total_frames))

    return violations


def validate_pass2(pass2c_output: Pass2COutput) -> List[ValidationViolation]:
    """
    Run all Pass 2 validations.

    Args:
        pass2c_output: Pass 2C output data (includes fragments from 2A/2B)

    Returns:
        List of all violations
    """
    violations = []

    # Pass 2A checks (fragments)
    pass2a_mock = type('obj', (object,), {
        'fragments': pass2c_output.fragments,
        'split_log': []
    })()
    violations.extend(validate_pass2a_fragments(pass2a_mock))

    # Pass 2B checks (quality scoring)
    pass2b_mock = type('obj', (object,), {
        'fragments': pass2c_output.fragments,
        'quality_distribution': {}
    })()
    violations.extend(validate_pass2b_quality_scoring(pass2b_mock))

    # Pass 2C checks (ghosts)
    violations.extend(validate_pass2c_ghosts(pass2c_output))

    return violations
