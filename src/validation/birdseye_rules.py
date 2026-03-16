"""
Bird's-eye projection validation rules.

This stage must preserve committed player/team identity while projecting players
and the ball into court space.
"""

from typing import List

from ..core.data_models import BirdseyeProjectionOutput, ValidationViolation
from ..core.types import TeamID, normalize_ball_state_value


def validate_birdseye_projection(
    birdseye_output: BirdseyeProjectionOutput,
) -> List[ValidationViolation]:
    """
    Validate bird's-eye projection output.

    Checks:
    - frame coverage within processed range
    - projected player coordinates remain inside plausible court bounds
    - player teams remain committed teams (team_a/team_b only)
    - ball state and projected coordinates stay consistent
    """
    violations: List[ValidationViolation] = []

    expected_end = birdseye_output.processed_end_frame_exclusive
    if expected_end is None:
        expected_end = birdseye_output.total_frames
    expected_frames = list(range(birdseye_output.processed_start_frame, expected_end))
    observed_frames = [frame.frame_idx for frame in birdseye_output.frames]

    if observed_frames != expected_frames:
        violations.append(
            ValidationViolation(
                rule="BIRDSEYE_FRAME_COVERAGE",
                severity="error",
                message=(
                    "Bird's-eye frames do not match processed frame range "
                    f"[{birdseye_output.processed_start_frame}, {expected_end})"
                ),
                details={
                    "expected_frame_count": len(expected_frames),
                    "observed_frame_count": len(observed_frames),
                    "first_observed": observed_frames[0] if observed_frames else None,
                    "last_observed": observed_frames[-1] if observed_frames else None,
                },
            )
        )

    tolerance_m = 0.75
    min_x = -tolerance_m
    max_x = birdseye_output.court_length_m + tolerance_m
    min_y = -tolerance_m
    max_y = birdseye_output.court_width_m + tolerance_m

    for frame in birdseye_output.frames:
        for player in frame.players:
            if player.team == TeamID.UNKNOWN:
                violations.append(
                    ValidationViolation(
                        rule="BIRDSEYE_UNKNOWN_TEAM",
                        severity="error",
                        message=f"Frame {frame.frame_idx}: projected player {player.player_id} has unknown team",
                        frame_idx=frame.frame_idx,
                        fragment_id=player.fragment_id,
                    )
                )

            x_m, y_m = player.court_position
            if x_m < min_x or x_m > max_x or y_m < min_y or y_m > max_y:
                violations.append(
                    ValidationViolation(
                        rule="BIRDSEYE_PLAYER_OUT_OF_BOUNDS",
                        severity="warning",
                        message=(
                            f"Frame {frame.frame_idx}: player {player.player_id} projected outside court bounds "
                            f"at ({x_m:.2f}, {y_m:.2f})"
                        ),
                        frame_idx=frame.frame_idx,
                        fragment_id=player.fragment_id,
                        details={"court_position": player.court_position},
                    )
                )

        ball = frame.ball
        state = normalize_ball_state_value(ball.state)
        if state in {"real", "interpolated"} and ball.court_position is None:
            violations.append(
                ValidationViolation(
                    rule="BIRDSEYE_BALL_MISSING_POSITION",
                    severity="error",
                    message=f"Frame {frame.frame_idx}: ball state='{state}' but projected court_position is missing",
                    frame_idx=frame.frame_idx,
                )
            )
        if state == "unknown" and ball.court_position is not None:
            violations.append(
                ValidationViolation(
                    rule="BIRDSEYE_UNKNOWN_BALL_HAS_POSITION",
                    severity="warning",
                    message=f"Frame {frame.frame_idx}: unknown ball still has projected coordinates",
                    frame_idx=frame.frame_idx,
                )
            )

    return violations