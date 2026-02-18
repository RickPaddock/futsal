"""
Pass 2 validation rules.

Per CLAUDE.md Section 5 (Pass 2: Fragmentation, Scoring, Ghosts):
- Pass 2A: Fragment validity (no overlaps, complete coverage)
- Pass 2B: Quality scoring validity
- Pass 2C: Ghost fragment validity (high water mark, max 12 concurrent)
"""

from typing import List, Dict, Set, Tuple
from collections import Counter
from ..core.data_models import (
    Fragment,
    ScoredFragment,
    GhostFragment,
    Pass1Output,
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

        # Check detection_ids not empty (skip ghosts - they have no real detections)
        is_ghost = getattr(fragment, 'is_ghost', False)
        if not is_ghost and len(fragment.detection_ids) == 0:
            violations.append(
                ValidationViolation(
                    rule="PASS2A_EMPTY_DETECTIONS",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} has no detection_ids",
                    fragment_id=fragment.fragment_id,
                    details={"fragment_id": fragment.fragment_id},
                )
            )

        # Enforce split metadata on non-initial fragments
        is_non_initial = fragment.split_reason is not None
        if is_non_initial:
            if fragment.split_trigger_frame is None:
                violations.append(
                    ValidationViolation(
                        rule="PASS2A_MISSING_SPLIT_TRIGGER_FRAME",
                        severity="error",
                        message=f"Fragment {fragment.fragment_id} missing split_trigger_frame",
                        fragment_id=fragment.fragment_id,
                        details={"fragment_id": fragment.fragment_id},
                    )
                )
            if fragment.split_rule_id is None:
                violations.append(
                    ValidationViolation(
                        rule="PASS2A_MISSING_SPLIT_RULE_ID",
                        severity="error",
                        message=f"Fragment {fragment.fragment_id} missing split_rule_id",
                        fragment_id=fragment.fragment_id,
                        details={"fragment_id": fragment.fragment_id},
                    )
                )

            if fragment.split_trigger_frame is not None:
                if not (fragment.start_frame <= fragment.split_trigger_frame <= fragment.end_frame):
                    violations.append(
                        ValidationViolation(
                            rule="PASS2A_INVALID_SPLIT_TRIGGER_FRAME",
                            severity="error",
                            message=(
                                f"Fragment {fragment.fragment_id} split_trigger_frame={fragment.split_trigger_frame} "
                                f"outside fragment range [{fragment.start_frame}, {fragment.end_frame}]"
                            ),
                            fragment_id=fragment.fragment_id,
                            details={
                                "fragment_id": fragment.fragment_id,
                                "split_trigger_frame": fragment.split_trigger_frame,
                                "start_frame": fragment.start_frame,
                                "end_frame": fragment.end_frame,
                            },
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
    # STRICT: No overlaps allowed (fragments now split on detection gaps)
    for track_id, frags in track_fragments.items():
        # Sort by start frame
        frags = sorted(frags, key=lambda x: x[1])

        for i in range(len(frags) - 1):
            frag_a_id, start_a, end_a = frags[i]
            frag_b_id, start_b, end_b = frags[i + 1]

            # Check overlap (no exceptions - fragments and ghosts should never overlap)
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


def validate_pass2a_frame_coverage(
    pass2a_output: Pass2AOutput,
    pass1_output: Pass1Output,
) -> List[ValidationViolation]:
    """
    Validate that Pass 2A fragments cover 100% of Pass 1 detections.

    Per CLAUDE.md Section 5 (Pass 2A):
    - Exact detection-id set equality vs Pass 1
    - No missing detection_ids
    - No unknown detection_ids
    - Cross-track exclusivity: each detection_id appears in exactly one fragment

    Args:
        pass2a_output: Pass 2A output data
        pass1_output: Pass 1 output data

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    pass1_detection_ids = {det.detection_id for det in pass1_output.detections}

    fragment_detection_ids: List[str] = []
    for fragment in pass2a_output.fragments:
        fragment_detection_ids.extend(fragment.detection_ids)

    fragment_detection_id_set = set(fragment_detection_ids)

    # Missing Pass 1 detection_ids
    missing_detection_ids = pass1_detection_ids - fragment_detection_id_set
    if missing_detection_ids:
        violations.append(
            ValidationViolation(
                rule="PASS2A_MISSING_DETECTIONS",
                severity="error",
                message=f"Pass 2A fragments missing {len(missing_detection_ids)} detection_ids from Pass 1",
                details={
                    "missing_detection_ids_count": len(missing_detection_ids),
                    "missing_detection_ids_sample": sorted(list(missing_detection_ids))[:10],
                    "total_pass1_detections": len(pass1_detection_ids),
                },
            )
        )

    # Unknown detection_ids fabricated in Pass 2A
    unknown_detection_ids = fragment_detection_id_set - pass1_detection_ids
    if unknown_detection_ids:
        violations.append(
            ValidationViolation(
                rule="PASS2A_EXTRA_COVERAGE",
                severity="error",
                message=f"Pass 2A contains {len(unknown_detection_ids)} unknown detection_ids not present in Pass 1",
                details={
                    "unknown_detection_ids_count": len(unknown_detection_ids),
                    "unknown_detection_ids_sample": sorted(list(unknown_detection_ids))[:10],
                },
            )
        )

    # Cross-track exclusivity: no detection_id may be assigned to multiple fragments
    counts = Counter(fragment_detection_ids)
    duplicated_detection_ids = [det_id for det_id, count in counts.items() if count > 1]
    if duplicated_detection_ids:
        violations.append(
            ValidationViolation(
                rule="PASS2A_DETECTION_ASSIGNED_MULTIPLE_FRAGMENTS",
                severity="error",
                message=(
                    f"Pass 2A assigns {len(duplicated_detection_ids)} detection_ids to multiple fragments"
                ),
                details={
                    "duplicate_detection_ids_count": len(duplicated_detection_ids),
                    "duplicate_detection_ids_sample": sorted(duplicated_detection_ids)[:10],
                },
            )
        )

    # Exact set equality check (summary guard)
    if pass1_detection_ids != fragment_detection_id_set:
        violations.append(
            ValidationViolation(
                rule="PASS2A_DETECTION_SET_MISMATCH",
                severity="error",
                message="Pass 2A detection_id set does not exactly match Pass 1 detection_id set",
                details={
                    "pass1_detection_count": len(pass1_detection_ids),
                    "pass2_detection_count": len(fragment_detection_id_set),
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
        # Skip ghosts (they always have quality=GHOST)
        if fragment.is_ghost:
            continue

        # Check that quality fields exist (CRITICAL: catches when Pass 2B wasn't called)
        if not hasattr(fragment, 'quality') or fragment.quality is None:
            violations.append(
                ValidationViolation(
                    rule="PASS2B_MISSING_QUALITY",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} missing quality field (Pass 2B not called?)",
                    fragment_id=fragment.fragment_id,
                    details={"fragment_id": fragment.fragment_id},
                )
            )
            continue

        if not hasattr(fragment, 'quality_score') or fragment.quality_score is None:
            violations.append(
                ValidationViolation(
                    rule="PASS2B_MISSING_SCORE",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} missing quality_score field (Pass 2B not called?)",
                    fragment_id=fragment.fragment_id,
                    details={"fragment_id": fragment.fragment_id},
                )
            )
            continue

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

    Per unified model: pass2c_output.fragments contains both real + ghost fragments.
    Ghosts marked with is_ghost=True and quality=GHOST.

    Checks:
    - Ghost quality = GHOST
    - is_ghost = True
    - Ghost has parent_fragment_id (source fragment)
    - Ghost position is valid (if available)
    - R4: Max 12 concurrent players (tracked + ghosts)

    Args:
        pass2c_output: Pass 2C output data

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    # Filter ghosts from unified list
    all_fragments = pass2c_output.fragments
    ghosts = [f for f in all_fragments if f.is_ghost]

    # Check each ghost
    for ghost in ghosts:
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

        # Check is_ghost = True (should always be true since we filtered, but defensive check)
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

        # Check parent_fragment_id exists (source fragment that spawned this ghost)
        if not ghost.parent_fragment_id:
            violations.append(
                ValidationViolation(
                    rule="PASS2C_GHOST_NO_SOURCE",
                    severity="warning",  # Warning not error - ghosts may not always have clear parent
                    message=f"Ghost {ghost.fragment_id} missing parent_fragment_id",
                    fragment_id=ghost.fragment_id,
                    details={"fragment_id": ghost.fragment_id},
                )
            )

        # Check ghost position is valid (if provided)
        if ghost.ghost_last_known_bbox:
            from ..utils.geometry import bbox_area

            if bbox_area(ghost.ghost_last_known_bbox) <= 0:
                violations.append(
                    ValidationViolation(
                        rule="PASS2C_GHOST_INVALID_POSITION",
                        severity="error",
                        message=f"Ghost {ghost.fragment_id} has invalid ghost_last_known_bbox (zero area)",
                        fragment_id=ghost.fragment_id,
                        details={
                            "fragment_id": ghost.fragment_id,
                            "ghost_last_known_bbox": ghost.ghost_last_known_bbox,
                        },
                    )
                )

    # R4: Player continuity (max 12 concurrent) - DURATION-AWARE ENFORCEMENT
    # Brief violations (1-2 frames) = tracker jitter → tolerate
    # Sustained violations (3+ frames) = detector failure → FAIL HARD
    all_fragments = pass2c_output.fragments  # Already includes real + ghosts

    # Find total_frames (max end_frame)
    total_frames = max((f.end_frame for f in all_fragments), default=0) + 1

    from .global_rules import validate_r4_player_continuity_duration_aware
    violations.extend(validate_r4_player_continuity_duration_aware(all_fragments, total_frames))

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
