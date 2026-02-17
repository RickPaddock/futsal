"""
Ball interpolation validation rules.

Per CLAUDE.md Section 8 (Ball Interpolation):
- R5: Ball state exists at every frame
- Gaps <= 30 frames: interpolated
- Gaps > 30 frames: out_of_play
- Interpolation speed limits (physical constraints)
"""

from typing import List, Optional
from ..core.data_models import BallPosition, BallInterpolationOutput, ValidationViolation
from ..core.types import BallState
from ..core.constants import MAX_BALL_GAP_FRAMES


def validate_ball_interpolation(
    ball_output: BallInterpolationOutput,
    total_frames: int,
) -> List[ValidationViolation]:
    """
    Validate ball interpolation.

    Checks:
    - R5: Ball state exists at every frame
    - Interpolation gaps <= MAX_BALL_GAP_FRAMES
    - Interpolation speed is physically plausible
    - State consistency (real has bbox, out_of_play has no position)

    Args:
        ball_output: Ball interpolation output
        total_frames: Total frames in video

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    # R5: Ball never disappears
    from .global_rules import validate_r5_ball_never_disappears
    violations.extend(validate_r5_ball_never_disappears(ball_output.ball_positions, total_frames))

    # Build frame index map
    ball_map = {bp.frame_idx: bp for bp in ball_output.ball_positions}

    # Check interpolation gaps
    real_frames = [
        bp.frame_idx for bp in ball_output.ball_positions
        if bp.state == BallState.REAL or (isinstance(bp.state, str) and bp.state == "real")
    ]

    if len(real_frames) > 1:
        real_frames = sorted(real_frames)

        for i in range(len(real_frames) - 1):
            gap = real_frames[i + 1] - real_frames[i] - 1

            if gap > 0:
                # Check if gap is filled with interpolation or out_of_play
                gap_start = real_frames[i] + 1
                gap_end = real_frames[i + 1] - 1

                for frame_idx in range(gap_start, gap_end + 1):
                    if frame_idx not in ball_map:
                        violations.append(
                            ValidationViolation(
                                rule="BALL_GAP_UNFILLED",
                                severity="error",
                                message=f"Frame {frame_idx}: Ball gap not filled (no interpolation or out_of_play state)",
                                frame_idx=frame_idx,
                                details={
                                    "frame_idx": frame_idx,
                                    "previous_real_frame": real_frames[i],
                                    "next_real_frame": real_frames[i + 1],
                                },
                            )
                        )
                        continue

                    bp = ball_map[frame_idx]
                    state_value = bp.state.value if isinstance(bp.state, BallState) else bp.state

                    # Gap <= threshold: should be interpolated
                    if gap <= MAX_BALL_GAP_FRAMES:
                        if state_value != "interpolated":
                            violations.append(
                                ValidationViolation(
                                    rule="BALL_SHOULD_INTERPOLATE",
                                    severity="error",
                                    message=f"Frame {frame_idx}: Gap={gap} <= {MAX_BALL_GAP_FRAMES} but state='{state_value}' (expected 'interpolated')",
                                    frame_idx=frame_idx,
                                    details={
                                        "frame_idx": frame_idx,
                                        "gap_size": gap,
                                        "state": state_value,
                                        "threshold": MAX_BALL_GAP_FRAMES,
                                    },
                                )
                            )

                    # Gap > threshold: should be out_of_play
                    else:
                        if state_value != "out_of_play":
                            violations.append(
                                ValidationViolation(
                                    rule="BALL_SHOULD_OUT_OF_PLAY",
                                    severity="error",
                                    message=f"Frame {frame_idx}: Gap={gap} > {MAX_BALL_GAP_FRAMES} but state='{state_value}' (expected 'out_of_play')",
                                    frame_idx=frame_idx,
                                    details={
                                        "frame_idx": frame_idx,
                                        "gap_size": gap,
                                        "state": state_value,
                                        "threshold": MAX_BALL_GAP_FRAMES,
                                    },
                                )
                            )

    # Check interpolation speed limits (physical plausibility)
    # Max ball speed ~30 m/s in futsal, ~100 pixels/frame at typical camera distances
    MAX_BALL_SPEED_PX_PER_FRAME = 100

    for i in range(len(ball_output.ball_positions) - 1):
        bp1 = ball_output.ball_positions[i]
        bp2 = ball_output.ball_positions[i + 1]

        # Only check consecutive frames
        if bp2.frame_idx != bp1.frame_idx + 1:
            continue

        # Skip if either is out_of_play
        state1 = bp1.state.value if isinstance(bp1.state, BallState) else bp1.state
        state2 = bp2.state.value if isinstance(bp2.state, BallState) else bp2.state

        if state1 == "out_of_play" or state2 == "out_of_play":
            continue

        # Check speed
        if bp1.centroid and bp2.centroid:
            from ..utils.geometry import centroid_distance

            distance = centroid_distance(bp1.centroid, bp2.centroid)

            if distance > MAX_BALL_SPEED_PX_PER_FRAME:
                violations.append(
                    ValidationViolation(
                        rule="BALL_SPEED_IMPLAUSIBLE",
                        severity="warning",
                        message=f"Frame {bp1.frame_idx}-{bp2.frame_idx}: Ball moved {distance:.1f} pixels (max {MAX_BALL_SPEED_PX_PER_FRAME})",
                        frame_idx=bp1.frame_idx,
                        details={
                            "frame_idx": bp1.frame_idx,
                            "next_frame_idx": bp2.frame_idx,
                            "distance": distance,
                            "max_speed": MAX_BALL_SPEED_PX_PER_FRAME,
                        },
                    )
                )

    return violations


def validate_ball_state_consistency(ball_positions: List[BallPosition]) -> List[ValidationViolation]:
    """
    Validate ball state consistency.

    Checks:
    - real: must have bbox and centroid
    - interpolated: must have centroid, bbox optional
    - out_of_play: centroid and bbox should be None

    Args:
        ball_positions: List of ball positions

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    for bp in ball_positions:
        state_value = bp.state.value if isinstance(bp.state, BallState) else bp.state

        if state_value == "real":
            # Real detection: must have bbox and centroid
            if bp.bbox is None:
                violations.append(
                    ValidationViolation(
                        rule="BALL_REAL_NO_BBOX",
                        severity="error",
                        message=f"Frame {bp.frame_idx}: state='real' but bbox is None",
                        frame_idx=bp.frame_idx,
                        details={"frame_idx": bp.frame_idx},
                    )
                )

            if bp.centroid is None:
                violations.append(
                    ValidationViolation(
                        rule="BALL_REAL_NO_CENTROID",
                        severity="error",
                        message=f"Frame {bp.frame_idx}: state='real' but centroid is None",
                        frame_idx=bp.frame_idx,
                        details={"frame_idx": bp.frame_idx},
                    )
                )

            # Real detection: confidence should be > 0
            if bp.confidence <= 0:
                violations.append(
                    ValidationViolation(
                        rule="BALL_REAL_ZERO_CONFIDENCE",
                        severity="warning",
                        message=f"Frame {bp.frame_idx}: state='real' but confidence={bp.confidence}",
                        frame_idx=bp.frame_idx,
                        details={
                            "frame_idx": bp.frame_idx,
                            "confidence": bp.confidence,
                        },
                    )
                )

        elif state_value == "interpolated":
            # Interpolated: must have centroid
            if bp.centroid is None:
                violations.append(
                    ValidationViolation(
                        rule="BALL_INTERPOLATED_NO_CENTROID",
                        severity="error",
                        message=f"Frame {bp.frame_idx}: state='interpolated' but centroid is None",
                        frame_idx=bp.frame_idx,
                        details={"frame_idx": bp.frame_idx},
                    )
                )

        elif state_value == "out_of_play":
            # Out of play: centroid should be None
            if bp.centroid is not None:
                violations.append(
                    ValidationViolation(
                        rule="BALL_OUT_OF_PLAY_HAS_CENTROID",
                        severity="warning",
                        message=f"Frame {bp.frame_idx}: state='out_of_play' but centroid is not None",
                        frame_idx=bp.frame_idx,
                        details={
                            "frame_idx": bp.frame_idx,
                            "centroid": bp.centroid,
                        },
                    )
                )

    return violations
