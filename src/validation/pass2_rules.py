"""
Pass 2 validation rules.

Per CLAUDE.md Section 5 (Pass 2: Fragmentation, Scoring, Ghosts):
- Pass 2A: Fragment validity (no overlaps, complete coverage)
- Pass 2B: Quality scoring validity
- Pass 2C: Ghost fragment validity (presence continuity, chain integrity)
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from collections import Counter
import math
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
from ..core import constants as const


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

        is_ghost = getattr(fragment, 'is_ghost', False)

        track_fragments[track_id].append((
            fragment.fragment_id,
            fragment.start_frame,
            fragment.end_frame,
            is_ghost,
        ))

    # Check for temporal overlaps within same track
    for track_id, frags in track_fragments.items():
        # Sort by start frame
        frags = sorted(frags, key=lambda x: x[1])

        for i in range(len(frags) - 1):
            frag_a_id, start_a, end_a, is_ghost_a = frags[i]
            frag_b_id, start_b, end_b, is_ghost_b = frags[i + 1]

            # Check overlap
            # ALLOW: Ghost-real overlap on same track (ghost fills detection gaps)
            # FAIL: Real-real or ghost-ghost overlap
            if start_b <= end_a:
                if (is_ghost_a and not is_ghost_b) or (not is_ghost_a and is_ghost_b):
                    continue

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
                            "is_ghost_a": is_ghost_a,
                            "is_ghost_b": is_ghost_b,
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

    # Explicit per-track frame-span invariant (geometry-only fragmentation contract):
    # For each original track_id, union of fragment detection frames must equal
    # Pass 1 detection frames for that track (no missing, no duplication).
    det_by_id = {det.detection_id: det for det in pass1_output.detections}

    pass1_track_frames: Dict[int, Set[int]] = {}
    for det in pass1_output.detections:
        if det.track_id not in pass1_track_frames:
            pass1_track_frames[det.track_id] = set()
        pass1_track_frames[det.track_id].add(det.frame_idx)

    fragment_track_frame_counts: Dict[int, Dict[int, int]] = {}
    for fragment in pass2a_output.fragments:
        track_id = fragment.original_track_id
        if track_id not in fragment_track_frame_counts:
            fragment_track_frame_counts[track_id] = {}

        for det_id in fragment.detection_ids:
            det = det_by_id.get(det_id)
            if det is None:
                continue

            frame_idx = det.frame_idx
            fragment_track_frame_counts[track_id][frame_idx] = (
                fragment_track_frame_counts[track_id].get(frame_idx, 0) + 1
            )

    all_track_ids = set(pass1_track_frames.keys()) | set(fragment_track_frame_counts.keys())
    for track_id in sorted(all_track_ids):
        expected = pass1_track_frames.get(track_id, set())
        observed_counts = fragment_track_frame_counts.get(track_id, {})
        observed = set(observed_counts.keys())

        missing_frames = sorted(expected - observed)
        extra_frames = sorted(observed - expected)
        duplicated_frames = sorted([f for f, c in observed_counts.items() if c > 1])

        if missing_frames or extra_frames or duplicated_frames:
            violations.append(
                ValidationViolation(
                    rule="PASS2A_TRACK_FRAME_SPAN_MISMATCH",
                    severity="error",
                    message=(
                        f"Track {track_id}: fragment frame union does not match Pass 1 track frame set"
                    ),
                    details={
                        "track_id": track_id,
                        "missing_frames_count": len(missing_frames),
                        "missing_frames_sample": missing_frames[:20],
                        "extra_frames_count": len(extra_frames),
                        "extra_frames_sample": extra_frames[:20],
                        "duplicated_frames_count": len(duplicated_frames),
                        "duplicated_frames_sample": duplicated_frames[:20],
                    },
                )
            )

    return violations


def validate_pass2a_jersey_ambiguity(
    pass2a_output: Pass2AOutput,
    pass1_output: Pass1Output,
) -> List[ValidationViolation]:
    """
    Diagnostic (non-fatal) check: detect fragments that may span two physical players.

    If a fragment shows ≥2 disjoint dominant jerseys (each meeting JERSEY_MIN_OBSERVATIONS
    and JERSEY_MIN_CONFIDENCE), it is identity-ambiguous — the tracker may have bridged
    two real players without triggering a split.

    This does NOT cause a pipeline failure. It is a warning signal for audit purposes.
    Per CLAUDE.md Pass 2A Responsibility Boundary: Pass 2A may produce identity-ambiguous
    fragments; Pass 3 is responsible for global identity feasibility.

    Rule ID: PASS2A_JERSEY_AMBIGUOUS (severity=warning, non-blocking)
    """
    violations = []
    det_by_id = {det.detection_id: det for det in pass1_output.detections}
    min_conf = const.JERSEY_MIN_CONFIDENCE
    min_obs = const.JERSEY_MIN_OBSERVATIONS

    for frag in pass2a_output.fragments:
        if frag.is_ghost:
            continue

        # Collect high-confidence jersey observations
        jersey_obs: Dict[int, int] = {}  # jersey_number -> count
        for det_id in frag.detection_ids:
            det = det_by_id.get(det_id)
            if det and det.jersey_number is not None:
                if det.jersey_confidence >= min_conf:
                    jersey_obs[det.jersey_number] = jersey_obs.get(det.jersey_number, 0) + 1

        # Find dominant jerseys: each appearing >= JERSEY_MIN_OBSERVATIONS times
        dominant = [j for j, cnt in jersey_obs.items() if cnt >= min_obs]

        if len(dominant) >= 2:
            violations.append(
                ValidationViolation(
                    rule="PASS2A_JERSEY_AMBIGUOUS",
                    severity="warning",
                    message=(
                        f"Fragment {frag.fragment_id} (track {frag.original_track_id}) "
                        f"shows {len(dominant)} dominant jerseys — may span two physical players"
                    ),
                    details={
                        "fragment_id": frag.fragment_id,
                        "track_id": frag.original_track_id,
                        "dominant_jerseys": sorted(dominant),
                        "jersey_observation_counts": jersey_obs,
                        "fragment_length_frames": frag.end_frame - frag.start_frame + 1,
                    },
                )
            )

    return violations


def _is_finite_number(value: Any) -> bool:
    """Return True when value is a finite int/float (bool excluded)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def validate_pass2b_quality_scoring(
    pass2b_output: Pass2BOutput,
    pass2a_output: Optional[Pass2AOutput] = None,
) -> List[ValidationViolation]:
    """
    Validate Pass 2B quality scoring.

    Checks (blocking):
    - Required metadata fields present and non-null
    - Required numeric fields are finite (no NaN/Inf)
    - Range checks for jersey/occlusion/appearance metrics
    - mean_velocity >= 0
    - quality enum valid
    - Temporal integrity (start_frame <= end_frame)
    - Fragment immutability vs Pass 2A (count + fragment_id set/sequence)

    Args:
        pass2b_output: Pass 2B output data

    Returns:
        List of violations (empty if valid)
    """
    violations: List[ValidationViolation] = []

    if pass2a_output is not None:
        expected_ids = [frag.fragment_id for frag in pass2a_output.fragments if not frag.is_ghost]
        actual_ids = [frag.fragment_id for frag in pass2b_output.fragments if not frag.is_ghost]

        if len(actual_ids) != len(expected_ids):
            violations.append(
                ValidationViolation(
                    rule="PASS2B_FRAGMENT_COUNT_CHANGED",
                    severity="error",
                    message=(
                        f"Pass 2B changed fragment count: expected {len(expected_ids)} from Pass 2A, "
                        f"got {len(actual_ids)}"
                    ),
                    details={
                        "expected_count": len(expected_ids),
                        "actual_count": len(actual_ids),
                    },
                )
            )

        expected_set = set(expected_ids)
        actual_set = set(actual_ids)
        if actual_set != expected_set:
            violations.append(
                ValidationViolation(
                    rule="PASS2B_FRAGMENT_ID_CHANGED",
                    severity="error",
                    message="Pass 2B changed fragment_id membership relative to Pass 2A",
                    details={
                        "missing_ids_sample": sorted(list(expected_set - actual_set))[:20],
                        "added_ids_sample": sorted(list(actual_set - expected_set))[:20],
                    },
                )
            )

        if actual_ids != expected_ids:
            violations.append(
                ValidationViolation(
                    rule="PASS2B_FRAGMENT_ORDER_CHANGED",
                    severity="error",
                    message="Pass 2B changed fragment_id sequence relative to Pass 2A",
                    details={
                        "expected_first10": expected_ids[:10],
                        "actual_first10": actual_ids[:10],
                    },
                )
            )

    required_fields = [
        "fragment_id",
        "track_id",
        "start_frame",
        "end_frame",
        "jersey_visible_ratio",
        "jersey_observability_score",
        "occlusion_ratio",
        "mean_velocity",
        "motion_smoothness_score",
        "appearance_stability_score",
        "quality",
    ]

    numeric_required_fields = {
        "track_id",
        "start_frame",
        "end_frame",
        "jersey_visible_ratio",
        "jersey_observability_score",
        "occlusion_ratio",
        "mean_velocity",
        "motion_smoothness_score",
        "appearance_stability_score",
    }

    for fragment in pass2b_output.fragments:
        for field_name in required_fields:
            if not hasattr(fragment, field_name):
                violations.append(
                    ValidationViolation(
                        rule="PASS2B_REQUIRED_FIELD_MISSING",
                        severity="error",
                        message=f"Fragment {fragment.fragment_id} missing required field '{field_name}'",
                        fragment_id=fragment.fragment_id,
                        details={"fragment_id": fragment.fragment_id, "field": field_name},
                    )
                )
                continue

            value = getattr(fragment, field_name)
            if value is None:
                violations.append(
                    ValidationViolation(
                        rule="PASS2B_REQUIRED_FIELD_NULL",
                        severity="error",
                        message=f"Fragment {fragment.fragment_id} has null required field '{field_name}'",
                        fragment_id=fragment.fragment_id,
                        details={"fragment_id": fragment.fragment_id, "field": field_name},
                    )
                )
                continue

            if field_name in numeric_required_fields and not _is_finite_number(value):
                violations.append(
                    ValidationViolation(
                        rule="PASS2B_REQUIRED_FIELD_NONFINITE",
                        severity="error",
                        message=(
                            f"Fragment {fragment.fragment_id} field '{field_name}' is invalid "
                            "(None/NaN/Inf/non-numeric)"
                        ),
                        fragment_id=fragment.fragment_id,
                        details={"fragment_id": fragment.fragment_id, "field": field_name, "value": value},
                    )
                )

        if not isinstance(fragment.fragment_id, str) or not fragment.fragment_id.strip():
            violations.append(
                ValidationViolation(
                    rule="PASS2B_INVALID_FRAGMENT_ID",
                    severity="error",
                    message=f"Fragment has invalid fragment_id='{fragment.fragment_id}'",
                    fragment_id=fragment.fragment_id if isinstance(fragment.fragment_id, str) else None,
                    details={"fragment_id": fragment.fragment_id},
                )
            )

        if isinstance(fragment.track_id, bool) or not isinstance(fragment.track_id, int):
            violations.append(
                ValidationViolation(
                    rule="PASS2B_INVALID_TRACK_ID",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} has invalid track_id='{fragment.track_id}'",
                    fragment_id=fragment.fragment_id,
                    details={"fragment_id": fragment.fragment_id, "track_id": fragment.track_id},
                )
            )

        if isinstance(fragment.start_frame, bool) or not isinstance(fragment.start_frame, int):
            violations.append(
                ValidationViolation(
                    rule="PASS2B_INVALID_START_FRAME_TYPE",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} start_frame must be int",
                    fragment_id=fragment.fragment_id,
                    details={"start_frame": fragment.start_frame},
                )
            )

        if isinstance(fragment.end_frame, bool) or not isinstance(fragment.end_frame, int):
            violations.append(
                ValidationViolation(
                    rule="PASS2B_INVALID_END_FRAME_TYPE",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} end_frame must be int",
                    fragment_id=fragment.fragment_id,
                    details={"end_frame": fragment.end_frame},
                )
            )

        if fragment.start_frame > fragment.end_frame:
            violations.append(
                ValidationViolation(
                    rule="PASS2B_INVALID_RANGE",
                    severity="error",
                    message=(
                        f"Fragment {fragment.fragment_id} has start_frame={fragment.start_frame} "
                        f"> end_frame={fragment.end_frame}"
                    ),
                    fragment_id=fragment.fragment_id,
                    details={
                        "fragment_id": fragment.fragment_id,
                        "start_frame": fragment.start_frame,
                        "end_frame": fragment.end_frame,
                    },
                )
            )

        if not fragment.is_ghost:
            presence_class = getattr(fragment, "presence_class", None)
            if presence_class not in {"real", "occlusion_candidate"}:
                violations.append(
                    ValidationViolation(
                        rule="PASS2B_INVALID_PRESENCE_CLASS",
                        severity="error",
                        message=(
                            f"Fragment {fragment.fragment_id} has invalid presence_class='{presence_class}'. "
                            "Expected one of {'real', 'occlusion_candidate'}."
                        ),
                        fragment_id=fragment.fragment_id,
                        details={"fragment_id": fragment.fragment_id, "presence_class": presence_class},
                    )
                )

        quality_value = fragment.quality.value if isinstance(fragment.quality, FragmentQuality) else fragment.quality
        if isinstance(quality_value, (int, float)) and not isinstance(quality_value, bool):
            if not _is_finite_number(quality_value) or not (0 <= float(quality_value) <= 1):
                violations.append(
                    ValidationViolation(
                        rule="PASS2B_INVALID_QUALITY",
                        severity="error",
                        message=(
                            f"Fragment {fragment.fragment_id} numeric quality={quality_value} "
                            "must be finite and in range [0, 1]"
                        ),
                        fragment_id=fragment.fragment_id,
                        details={"fragment_id": fragment.fragment_id, "quality": quality_value},
                    )
                )
        elif quality_value not in {"high", "medium", "low", "ghost"}:
            violations.append(
                ValidationViolation(
                    rule="PASS2B_INVALID_QUALITY",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} has invalid quality='{quality_value}'",
                    fragment_id=fragment.fragment_id,
                    details={"fragment_id": fragment.fragment_id, "quality": quality_value},
                )
            )

        if not hasattr(fragment, "quality_score") or fragment.quality_score is None:
            violations.append(
                ValidationViolation(
                    rule="PASS2B_MISSING_SCORE",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} missing quality_score",
                    fragment_id=fragment.fragment_id,
                    details={"fragment_id": fragment.fragment_id},
                )
            )
        elif not _is_finite_number(fragment.quality_score) or not (0 <= float(fragment.quality_score) <= 1):
            violations.append(
                ValidationViolation(
                    rule="PASS2B_INVALID_SCORE",
                    severity="error",
                    message=(
                        f"Fragment {fragment.fragment_id} quality_score={fragment.quality_score} "
                        "must be finite and in range [0, 1]"
                    ),
                    fragment_id=fragment.fragment_id,
                    details={"fragment_id": fragment.fragment_id, "quality_score": fragment.quality_score},
                )
            )

        numeric_ranges = {
            "jersey_visible_ratio": (fragment.jersey_visible_ratio, 0.0, 1.0),
            "jersey_observability_score": (fragment.jersey_observability_score, 0.0, 1.0),
            "occlusion_ratio": (fragment.occlusion_ratio, 0.0, 1.0),
            "appearance_stability_score": (fragment.appearance_stability_score, 0.0, 1.0),
            "mean_velocity": (fragment.mean_velocity, 0.0, None),
            "motion_smoothness_score": (fragment.motion_smoothness_score, 0.0, 1.0),
        }

        for metric_name, (value, low, high) in numeric_ranges.items():
            if not _is_finite_number(value):
                violations.append(
                    ValidationViolation(
                        rule="PASS2B_INVALID_METRIC",
                        severity="error",
                        message=(
                            f"Fragment {fragment.fragment_id} {metric_name}={value} is invalid "
                            "(must be finite, non-null numeric)"
                        ),
                        fragment_id=fragment.fragment_id,
                        details={
                            "fragment_id": fragment.fragment_id,
                            "metric_name": metric_name,
                            "metric_value": value,
                        },
                    )
                )
                continue

            numeric_value = float(value)
            if numeric_value < low or (high is not None and numeric_value > high):
                range_text = f"[{low}, {high}]" if high is not None else f">= {low}"
                violations.append(
                    ValidationViolation(
                        rule="PASS2B_INVALID_METRIC",
                        severity="error",
                        message=(
                            f"Fragment {fragment.fragment_id} {metric_name}={numeric_value} out of range "
                            f"{range_text}"
                        ),
                        fragment_id=fragment.fragment_id,
                        details={
                            "fragment_id": fragment.fragment_id,
                            "metric_name": metric_name,
                            "metric_value": numeric_value,
                            "expected_range": range_text,
                        },
                    )
                )

    return violations


def validate_pass2c_ghosts(
    pass2c_output: Pass2COutput,
    pass1_output: Optional[Pass1Output] = None,
) -> List[ValidationViolation]:
    """
    Validate Pass 2C ghost fragments.

    Per unified model: pass2c_output.fragments contains both real + ghost fragments.
    Ghosts marked with is_ghost=True and quality=GHOST.

    Checks:
    - Ghost quality = GHOST
    - is_ghost = True
    - Ghost has parent_fragment_id (source fragment)
    - Ghost position is valid (if available)
    - Per-track lifespan continuity: exactly one of {real, ghost} at each frame

    Args:
        pass2c_output: Pass 2C output data

    Returns:
        List of violations (empty if valid)
    """
    from ..core import constants as const

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

        # Ghosts must be excluded from clustering inputs in later passes.
        if not bool(getattr(ghost, "exclude_from_clustering", False)):
            violations.append(
                ValidationViolation(
                    rule="PASS2C_GHOST_CLUSTER_FLAG",
                    severity="error",
                    message=(
                        f"Ghost {ghost.fragment_id} missing exclude_from_clustering=True"
                    ),
                    fragment_id=ghost.fragment_id,
                    details={
                        "fragment_id": ghost.fragment_id,
                        "exclude_from_clustering": getattr(ghost, "exclude_from_clustering", None),
                    },
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

    all_fragments = pass2c_output.fragments

    # Build frame-level real and ghost presence maps
    frame_real_tracks: Dict[int, Set[int]] = {}
    if pass1_output is not None:
        for det in pass1_output.detections:
            if det.frame_idx not in frame_real_tracks:
                frame_real_tracks[det.frame_idx] = set()
            frame_real_tracks[det.frame_idx].add(det.track_id)

    frame_ghost_tracks: Dict[int, Set[int]] = {}
    for ghost in ghosts:
        track_id = ghost.original_track_id
        for frame_idx in range(ghost.start_frame, ghost.end_frame + 1):
            if frame_idx not in frame_ghost_tracks:
                frame_ghost_tracks[frame_idx] = set()
            frame_ghost_tracks[frame_idx].add(track_id)

    # Explicit per-identity lifespan audit (same-track only).
    # For each original_track_id lifespan [first_real_frame, last_real_frame],
    # require exactly one presence source at each frame: real XOR ghost.
    real_by_track: Dict[int, List[ScoredFragment]] = {}
    ghost_by_track: Dict[int, List[ScoredFragment]] = {}
    for fragment in all_fragments:
        track_id = fragment.original_track_id
        if fragment.is_ghost:
            if track_id not in ghost_by_track:
                ghost_by_track[track_id] = []
            ghost_by_track[track_id].append(fragment)
        else:
            if track_id not in real_by_track:
                real_by_track[track_id] = []
            real_by_track[track_id].append(fragment)

    for track_id, real_frags in real_by_track.items():
        lifespan_start = min(f.start_frame for f in real_frags)
        lifespan_end = max(f.end_frame for f in real_frags)
        ghost_frags = ghost_by_track.get(track_id, [])

        real_detection_frames = set()
        if pass1_output is not None:
            for frame_idx, track_ids in frame_real_tracks.items():
                if track_id in track_ids:
                    real_detection_frames.add(frame_idx)

        gap_frames: List[int] = []
        overlap_frames: List[int] = []

        for frame_idx in range(lifespan_start, lifespan_end + 1):
            if pass1_output is not None:
                has_real = frame_idx in real_detection_frames
            else:
                has_real = any(f.start_frame <= frame_idx <= f.end_frame for f in real_frags)
            has_ghost = any(f.start_frame <= frame_idx <= f.end_frame for f in ghost_frags)
            sources = int(has_real) + int(has_ghost)

            if sources == 0:
                gap_frames.append(frame_idx)
            elif sources > 1:
                overlap_frames.append(frame_idx)

        if gap_frames:
            violations.append(
                ValidationViolation(
                    rule="PASS2C_TRACK_LIFESPAN_GAP",
                    severity="error",
                    message=(
                        f"Track {track_id} has {len(gap_frames)} gap frames in lifespan "
                        f"[{lifespan_start}, {lifespan_end}] with neither real nor ghost presence."
                    ),
                    details={
                        "track_id": track_id,
                        "lifespan_start": lifespan_start,
                        "lifespan_end": lifespan_end,
                        "gap_count": len(gap_frames),
                        "gap_frames_sample": gap_frames[:20],
                    },
                )
            )

        if overlap_frames:
            violations.append(
                ValidationViolation(
                    rule="PASS2C_TRACK_LIFESPAN_OVERLAP",
                    severity="error",
                    message=(
                        f"Track {track_id} has {len(overlap_frames)} frames in lifespan "
                        f"[{lifespan_start}, {lifespan_end}] with overlapping real+ghost presence."
                    ),
                    details={
                        "track_id": track_id,
                        "lifespan_start": lifespan_start,
                        "lifespan_end": lifespan_end,
                        "overlap_count": len(overlap_frames),
                        "overlap_frames_sample": overlap_frames[:20],
                    },
                )
            )

    # Resolve validation frame range
    if pass1_output is not None:
        start_frame = pass1_output.processed_start_frame
        end_frame = (
            pass1_output.processed_end_frame_exclusive - 1
            if pass1_output.processed_end_frame_exclusive is not None
            else pass1_output.total_frames - 1
        )
    else:
        start_frame = min((f.start_frame for f in all_fragments), default=0)
        end_frame = max((f.end_frame for f in all_fragments), default=-1)

    # Pass 2C validation semantics (identity-agnostic):
    # - BLOCKING: Presence completeness and chain continuity (R4 presence layer)
    # - NON-BLOCKING: >12 concurrent presence (identity collision signal for Pass 3)
    if end_frame >= start_frame:
        init_end = min(end_frame, start_frame + const.INITIAL_LEVEL_FRAMES - 1)
        level = 0
        for frame_idx in range(start_frame, init_end + 1):
            level = max(level, len(frame_real_tracks.get(frame_idx, set())))
        level = min(level, const.DYNAMIC_LEVEL_MAX)

        too_many_frames: List[Tuple[int, int]] = []
        r4_deficit_frames: List[Tuple[int, int, int]] = []  # frame, count, level

        for frame_idx in range(start_frame, end_frame + 1):
            real_count = len(frame_real_tracks.get(frame_idx, set()))
            level = min(max(level, real_count), const.DYNAMIC_LEVEL_MAX)

            presence_tracks = frame_real_tracks.get(frame_idx, set()) | frame_ghost_tracks.get(frame_idx, set())
            presence_count = len(presence_tracks)

            if presence_count > const.DYNAMIC_LEVEL_MAX:
                too_many_frames.append((frame_idx, presence_count))

            # Blocking R4 condition in Pass 2C: any deficit is a hard failure.
            # Overages are reported separately as identity-collision evidence.
            if presence_count < level:
                r4_deficit_frames.append((frame_idx, presence_count, level))

        if too_many_frames:
            max_count = max(c for _, c in too_many_frames)
            violations.append(
                ValidationViolation(
                    rule="PASS2C_PHYSICAL_MAX_EXCEEDED_DEFERRED",
                    severity="warning",
                    message=(
                        f"{len(too_many_frames)} frames have >12 concurrent presence (max={max_count}). "
                        f"This indicates unresolved identity collisions and is deferred to Pass 3 identity resolution."
                    ),
                    details={
                        "violation_count": len(too_many_frames),
                        "max_concurrent_count": max_count,
                        "deferred_to_pass": "pass3",
                        "sample_violations": [
                            {"frame": f, "count": c} for f, c in too_many_frames[:20]
                        ],
                    },
                )
            )

        if r4_deficit_frames:
            violations.append(
                ValidationViolation(
                    rule="PASS2C_R4_INVARIANT_VIOLATED",
                    severity="error",
                    message=(
                        f"R4 presence continuity violated on {len(r4_deficit_frames)} frames: "
                        f"tracked + ghosts < dynamic level. Per CLAUDE.md this is ZERO TOLERANCE."
                    ),
                    details={
                        "violation_count": len(r4_deficit_frames),
                        "sample_violations": [
                            {
                                "frame": f,
                                "presence_count": c,
                                "expected_level": lv,
                                "deficit": lv - c,
                            }
                            for f, c, lv in r4_deficit_frames[:20]
                        ],
                    },
                )
            )

    # Ghost must terminate when same-track real detection is present.
    if pass1_output is not None:
        for ghost in ghosts:
            track_id = ghost.original_track_id
            overlap_frames = [
                frame_idx
                for frame_idx in range(ghost.start_frame, ghost.end_frame + 1)
                if track_id in frame_real_tracks.get(frame_idx, set())
            ]

            if overlap_frames:
                violations.append(
                    ValidationViolation(
                        rule="PASS2C_GHOST_OVERLAPS_REAL",
                        severity="error",
                        message=(
                            f"Ghost {ghost.fragment_id} (track {track_id}) overlaps real detections "
                            f"on {len(overlap_frames)} frames. Ghost must end before reappearance."
                        ),
                        fragment_id=ghost.fragment_id,
                        details={
                            "ghost_fragment": ghost.fragment_id,
                            "track_id": track_id,
                            "first_overlap_frame": overlap_frames[0],
                            "last_overlap_frame": overlap_frames[-1],
                            "overlap_count": len(overlap_frames),
                        },
                    )
                )

        # CLAUDE.md R4 ghost chaining: if a ghost ends before clip end,
        # the next frame must contain either real reappearance for that track
        # or another ghost for that track (zero-gap chaining).
        for ghost in ghosts:
            next_frame = ghost.end_frame + 1
            if next_frame > end_frame:
                continue

            track_id = ghost.original_track_id
            has_real = track_id in frame_real_tracks.get(next_frame, set())
            has_ghost = track_id in frame_ghost_tracks.get(next_frame, set())

            if not has_real and not has_ghost:
                violations.append(
                    ValidationViolation(
                        rule="PASS2C_GHOST_CHAIN_BROKEN",
                        severity="error",
                        message=(
                            f"Ghost {ghost.fragment_id} (track {track_id}) ends at frame {ghost.end_frame} "
                            f"with no real or ghost continuation at frame {next_frame}."
                        ),
                        fragment_id=ghost.fragment_id,
                        details={
                            "ghost_fragment": ghost.fragment_id,
                            "track_id": track_id,
                            "ghost_end_frame": ghost.end_frame,
                            "missing_continuation_frame": next_frame,
                        },
                    )
                )

    return violations


def validate_pass2(
    pass2c_output: Pass2COutput,
    pass1_output: Optional[Pass1Output] = None,
) -> List[ValidationViolation]:
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

    # Pass 2A explicit frame-span coverage checks when Pass 1 context is available.
    if pass1_output is not None:
        violations.extend(validate_pass2a_frame_coverage(pass2a_mock, pass1_output))

    # Pass 2B checks (quality scoring)
    pass2b_mock = type('obj', (object,), {
        'fragments': pass2c_output.fragments,
        'quality_distribution': {}
    })()
    violations.extend(validate_pass2b_quality_scoring(pass2b_mock))

    # Pass 2C checks (ghosts)
    violations.extend(validate_pass2c_ghosts(pass2c_output, pass1_output))

    return violations
