"""
Pass 1 validation rules.

Per CLAUDE.md Section 4 (Pass 1: Raw Evidence Collection):
- Bbox size validation (multi-layer defense)
- Jersey probability validation
- HSV histogram validation
- No duplicate detections
- Frame coverage checks
"""

from typing import List, Dict, Set
from ..core.data_models import Pass1Output, Detection, ValidationViolation
from ..core.constants import (
    PLAYER_CONF_THRESHOLD,
    JERSEY_CONF_THRESHOLD,
    JERSEY_ROI_X_MIN_FRAC,
    JERSEY_ROI_X_MAX_FRAC,
    JERSEY_ROI_Y_MIN_FRAC,
    JERSEY_ROI_Y_MAX_FRAC,
)


def validate_pass1_detections(pass1_output: Pass1Output) -> List[ValidationViolation]:
    """
    Validate Pass 1 detections.

    Checks:
    - Detection confidence >= threshold
    - Bbox validity (within frame bounds, non-zero area)
    - No duplicate detection_ids
    - Jersey confidence consistency
    - HSV histogram validity

    Args:
        pass1_output: Pass 1 output data

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    seen_detection_ids: Set[str] = set()

    for detection in pass1_output.detections:
        # Check detection confidence
        if detection.confidence < PLAYER_CONF_THRESHOLD:
            violations.append(
                ValidationViolation(
                    rule="PASS1_DETECTION",
                    severity="warning",
                    message=f"Detection {detection.detection_id} confidence {detection.confidence:.3f} below threshold {PLAYER_CONF_THRESHOLD}",
                    frame_idx=detection.frame_idx,
                    details={
                        "detection_id": detection.detection_id,
                        "confidence": detection.confidence,
                        "threshold": PLAYER_CONF_THRESHOLD,
                    },
                )
            )

        # Check bbox validity
        from ..utils.geometry import bbox_is_valid

        if not bbox_is_valid(detection.bbox, pass1_output.width, pass1_output.height):
            violations.append(
                ValidationViolation(
                    rule="PASS1_BBOX",
                    severity="error",
                    message=f"Detection {detection.detection_id} has invalid bbox",
                    frame_idx=detection.frame_idx,
                    details={
                        "detection_id": detection.detection_id,
                        "bbox": detection.bbox,
                        "frame_width": pass1_output.width,
                        "frame_height": pass1_output.height,
                    },
                )
            )

        # Check for duplicate detection_ids
        if detection.detection_id in seen_detection_ids:
            violations.append(
                ValidationViolation(
                    rule="PASS1_DUPLICATE",
                    severity="error",
                    message=f"Duplicate detection_id: {detection.detection_id}",
                    frame_idx=detection.frame_idx,
                    details={"detection_id": detection.detection_id},
                )
            )
        seen_detection_ids.add(detection.detection_id)

        # Check jersey confidence consistency
        if detection.jersey_number is not None:
            if detection.jersey_confidence < JERSEY_CONF_THRESHOLD:
                violations.append(
                    ValidationViolation(
                        rule="PASS1_JERSEY",
                        severity="error",
                        message=f"Detection {detection.detection_id} has jersey_number={detection.jersey_number} but jersey_confidence={detection.jersey_confidence:.3f} < {JERSEY_CONF_THRESHOLD}",
                        frame_idx=detection.frame_idx,
                        details={
                            "detection_id": detection.detection_id,
                            "jersey_number": detection.jersey_number,
                            "jersey_confidence": detection.jersey_confidence,
                            "threshold": JERSEY_CONF_THRESHOLD,
                        },
                    )
                )

        from ..utils.hsv_color import is_histogram_valid

        # Jersey HSV validity contract (primary team signal)
        if detection.jersey_roi_valid:
            # Geometry contract: jersey ROI must be torso sub-box inside player bbox
            if detection.jersey_roi_bbox is None:
                violations.append(
                    ValidationViolation(
                        rule="PASS1_JERSEY_ROI_MISSING",
                        severity="error",
                        message=f"Detection {detection.detection_id} has jersey_roi_valid=true but missing jersey_roi_bbox",
                        frame_idx=detection.frame_idx,
                        details={"detection_id": detection.detection_id},
                    )
                )
            else:
                b = detection.bbox
                r = detection.jersey_roi_bbox

                # ROI must be contained in player bbox
                if not (r[0] >= b[0] and r[1] >= b[1] and r[2] <= b[2] and r[3] <= b[3]):
                    violations.append(
                        ValidationViolation(
                            rule="PASS1_JERSEY_ROI_OUTSIDE_BBOX",
                            severity="error",
                            message=f"Detection {detection.detection_id} jersey ROI is outside player bbox",
                            frame_idx=detection.frame_idx,
                            details={"detection_id": detection.detection_id, "bbox": b, "jersey_roi_bbox": r},
                        )
                    )
                else:
                    bw = b[2] - b[0]
                    bh = b[3] - b[1]
                    rw = r[2] - r[0]
                    rh = r[3] - r[1]

                    if bw > 0 and bh > 0:
                        width_ratio = rw / bw
                        height_ratio = rh / bh

                        expected_width_ratio = JERSEY_ROI_X_MAX_FRAC - JERSEY_ROI_X_MIN_FRAC
                        expected_height_ratio = JERSEY_ROI_Y_MAX_FRAC - JERSEY_ROI_Y_MIN_FRAC

                        # Tolerance allows minor clipping near frame edges.
                        width_tol = 0.05
                        height_tol = 0.06

                        if not (
                            (expected_width_ratio - width_tol) <= width_ratio <= (expected_width_ratio + width_tol)
                            and (expected_height_ratio - height_tol) <= height_ratio <= (expected_height_ratio + height_tol)
                        ):
                            violations.append(
                                ValidationViolation(
                                    rule="PASS1_JERSEY_ROI_RATIO",
                                    severity="error",
                                    message=f"Detection {detection.detection_id} jersey ROI ratio out of expected torso range",
                                    frame_idx=detection.frame_idx,
                                    details={
                                        "detection_id": detection.detection_id,
                                        "expected_width_ratio": expected_width_ratio,
                                        "expected_height_ratio": expected_height_ratio,
                                        "width_ratio": width_ratio,
                                        "height_ratio": height_ratio,
                                    },
                                )
                            )

            if detection.hsv_histogram_jersey is None:
                violations.append(
                    ValidationViolation(
                        rule="PASS1_HSV_JERSEY_MISSING",
                        severity="error",
                        message=f"Detection {detection.detection_id} has jersey_roi_valid=true but missing hsv_histogram_jersey",
                        frame_idx=detection.frame_idx,
                        details={"detection_id": detection.detection_id},
                    )
                )
            elif not is_histogram_valid(detection.hsv_histogram_jersey):
                violations.append(
                    ValidationViolation(
                        rule="PASS1_HSV_JERSEY_INVALID",
                        severity="error",
                        message=f"Detection {detection.detection_id} has invalid jersey HSV histogram",
                        frame_idx=detection.frame_idx,
                        details={"detection_id": detection.detection_id},
                    )
                )
        else:
            if detection.hsv_histogram_jersey is not None:
                violations.append(
                    ValidationViolation(
                        rule="PASS1_HSV_JERSEY_INCONSISTENT",
                        severity="warning",
                        message=f"Detection {detection.detection_id} has jersey_roi_valid=false but non-null hsv_histogram_jersey",
                        frame_idx=detection.frame_idx,
                        details={"detection_id": detection.detection_id},
                    )
                )

    return violations


def validate_pass1_ball_detections(pass1_output: Pass1Output) -> List[ValidationViolation]:
    """
    Validate Pass 1 ball detections.

    Checks:
    - Ball detection confidence >= threshold
    - Bbox validity
    - No more than 1 ball per frame (warning if violated)

    Args:
        pass1_output: Pass 1 output data

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    from ..core.constants import BALL_CONF_THRESHOLD

    # Count balls per frame
    balls_per_frame: Dict[int, int] = {}

    for ball_det in pass1_output.ball_detections:
        # Check ball confidence
        if ball_det.confidence < BALL_CONF_THRESHOLD:
            violations.append(
                ValidationViolation(
                    rule="PASS1_BALL",
                    severity="warning",
                    message=f"Ball detection at frame {ball_det.frame_idx} confidence {ball_det.confidence:.3f} below threshold {BALL_CONF_THRESHOLD}",
                    frame_idx=ball_det.frame_idx,
                    details={
                        "frame_idx": ball_det.frame_idx,
                        "confidence": ball_det.confidence,
                        "threshold": BALL_CONF_THRESHOLD,
                    },
                )
            )

        # Check bbox validity
        from ..utils.geometry import bbox_is_valid

        if not bbox_is_valid(ball_det.bbox, pass1_output.width, pass1_output.height):
            violations.append(
                ValidationViolation(
                    rule="PASS1_BALL_BBOX",
                    severity="error",
                    message=f"Ball detection at frame {ball_det.frame_idx} has invalid bbox",
                    frame_idx=ball_det.frame_idx,
                    details={
                        "frame_idx": ball_det.frame_idx,
                        "bbox": ball_det.bbox,
                    },
                )
            )

        # Count balls per frame
        balls_per_frame[ball_det.frame_idx] = balls_per_frame.get(ball_det.frame_idx, 0) + 1

    # Warn if multiple balls in same frame
    for frame_idx, count in balls_per_frame.items():
        if count > 1:
            violations.append(
                ValidationViolation(
                    rule="PASS1_BALL_MULTIPLE",
                    severity="warning",
                    message=f"Frame {frame_idx}: {count} ball detections (expected 1)",
                    frame_idx=frame_idx,
                    details={"frame_idx": frame_idx, "ball_count": count},
                )
            )

    return violations


def validate_pass1_frame_coverage(pass1_output: Pass1Output) -> List[ValidationViolation]:
    """
    Validate Pass 1 frame coverage.

    Checks:
    - All detections have frame_idx within [0, total_frames)
    - Detections exist for at least some frames (not all empty)

    Args:
        pass1_output: Pass 1 output data

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    # Check frame indices are valid
    for detection in pass1_output.detections:
        if detection.frame_idx < 0 or detection.frame_idx >= pass1_output.total_frames:
            violations.append(
                ValidationViolation(
                    rule="PASS1_FRAME_RANGE",
                    severity="error",
                    message=f"Detection {detection.detection_id} frame_idx={detection.frame_idx} out of range [0, {pass1_output.total_frames})",
                    frame_idx=detection.frame_idx,
                    details={
                        "detection_id": detection.detection_id,
                        "frame_idx": detection.frame_idx,
                        "total_frames": pass1_output.total_frames,
                    },
                )
            )

    # Check we have at least some detections
    if len(pass1_output.detections) == 0:
        violations.append(
            ValidationViolation(
                rule="PASS1_NO_DETECTIONS",
                severity="error",
                message="No player detections found in Pass 1",
                details={"total_frames": pass1_output.total_frames},
            )
        )

    processed_start = max(0, pass1_output.processed_start_frame or 0)
    processed_end_exclusive = pass1_output.processed_end_frame_exclusive
    if processed_end_exclusive is None:
        processed_end_exclusive = pass1_output.total_frames
    processed_end_exclusive = min(pass1_output.total_frames, processed_end_exclusive)

    expected_frames = max(0, processed_end_exclusive - processed_start)

    # Count frames with detections inside processed window only
    frames_with_detections = {
        d.frame_idx for d in pass1_output.detections
        if processed_start <= d.frame_idx < processed_end_exclusive
    }
    coverage = len(frames_with_detections) / expected_frames if expected_frames > 0 else 0

    if coverage < 0.5:
        violations.append(
            ValidationViolation(
                rule="PASS1_LOW_COVERAGE",
                severity="warning",
                message=f"Low frame coverage: {coverage*100:.1f}% of frames have detections",
                details={
                    "coverage_pct": coverage * 100,
                    "frames_with_detections": len(frames_with_detections),
                    "expected_frames": expected_frames,
                    "processed_start_frame": processed_start,
                    "processed_end_frame_exclusive": processed_end_exclusive,
                    "total_frames": pass1_output.total_frames,
                },
            )
        )

    return violations


def validate_pass1(pass1_output: Pass1Output) -> List[ValidationViolation]:
    """
    Run all Pass 1 validations.

    Args:
        pass1_output: Pass 1 output data

    Returns:
        List of all violations
    """
    violations = []

    # R1: Pass 1 is raw truth only (global rule)
    from .global_rules import validate_r1_pass1_raw_truth
    violations.extend(validate_r1_pass1_raw_truth(pass1_output))

    # Pass 1 specific checks
    violations.extend(validate_pass1_detections(pass1_output))
    violations.extend(validate_pass1_ball_detections(pass1_output))
    violations.extend(validate_pass1_frame_coverage(pass1_output))

    return violations
