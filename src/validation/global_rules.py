"""
Global validation rules (R1-R5).

Per CLAUDE.md Section 2 (Non-Negotiable Rules):
- R1: Pass 1 is raw truth only (no team, no identity)
- R2: Every player has a team (no "unknown" after Pass 3C)
- R3: Jersey temporal exclusivity (one jersey = one player at any time)
- R4: Player continuity (max 12 concurrent, high water mark)
- R5: Ball never disappears (state exists at every frame)

These are HARD constraints. Violation = pipeline failure.
"""

from typing import List, Dict, Any, Set, Tuple
from ..core.data_models import (
    Detection,
    Pass1Output,
    ScoredFragment,
    GhostFragment,
    CommittedIdentity,
    Pass3COutput,
    BallPosition,
    ValidationViolation,
)
from ..core.types import TeamID, BallState, FrameIndex


def validate_r1_pass1_raw_truth(pass1_output: Pass1Output) -> List[ValidationViolation]:
    """
    R1: Pass 1 is raw truth only.

    Per CLAUDE.md R1:
    - Pass 1 MUST NOT contain: team, player_id, or any interpretation
    - Pass 1 MUST contain: bbox, confidence, track_id, jersey probs, HSV histogram
    - Detections must pass multi-layer bbox defense (max 800px height, 600px width, 25% area)

    Args:
        pass1_output: Pass 1 output data

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    # Check each detection
    for detection in pass1_output.detections:
        # R1: No team field allowed
        if hasattr(detection, 'team'):
            violations.append(
                ValidationViolation(
                    rule="R1",
                    severity="error",
                    message=f"Detection {detection.detection_id} has forbidden 'team' field",
                    frame_idx=detection.frame_idx,
                    details={"detection_id": detection.detection_id},
                )
            )

        # R1: No player_id field allowed
        if hasattr(detection, 'player_id'):
            violations.append(
                ValidationViolation(
                    rule="R1",
                    severity="error",
                    message=f"Detection {detection.detection_id} has forbidden 'player_id' field",
                    frame_idx=detection.frame_idx,
                    details={"detection_id": detection.detection_id},
                )
            )

        # R1: Bbox size validation (multi-layer defense)
        from ..utils.geometry import bbox_width, bbox_height, bbox_area
        from ..core.constants import MAX_BBOX_WIDTH_PX, MAX_BBOX_HEIGHT_PX, MAX_BBOX_AREA_FRACTION

        width = bbox_width(detection.bbox)
        height = bbox_height(detection.bbox)
        area = bbox_area(detection.bbox)
        frame_area = pass1_output.width * pass1_output.height
        area_fraction = area / frame_area if frame_area > 0 else 0

        # Layer 1: Absolute limits
        if width > MAX_BBOX_WIDTH_PX:
            violations.append(
                ValidationViolation(
                    rule="R1",
                    severity="error",
                    message=f"Detection {detection.detection_id} bbox width {width:.0f}px exceeds limit {MAX_BBOX_WIDTH_PX}px",
                    frame_idx=detection.frame_idx,
                    details={
                        "detection_id": detection.detection_id,
                        "bbox_width": width,
                        "max_width": MAX_BBOX_WIDTH_PX,
                    },
                )
            )

        if height > MAX_BBOX_HEIGHT_PX:
            violations.append(
                ValidationViolation(
                    rule="R1",
                    severity="error",
                    message=f"Detection {detection.detection_id} bbox height {height:.0f}px exceeds limit {MAX_BBOX_HEIGHT_PX}px",
                    frame_idx=detection.frame_idx,
                    details={
                        "detection_id": detection.detection_id,
                        "bbox_height": height,
                        "max_height": MAX_BBOX_HEIGHT_PX,
                    },
                )
            )

        # Layer 2: Relative limit
        if area_fraction > MAX_BBOX_AREA_FRACTION:
            violations.append(
                ValidationViolation(
                    rule="R1",
                    severity="error",
                    message=f"Detection {detection.detection_id} bbox area {area_fraction*100:.1f}% exceeds limit {MAX_BBOX_AREA_FRACTION*100:.1f}%",
                    frame_idx=detection.frame_idx,
                    details={
                        "detection_id": detection.detection_id,
                        "bbox_area_fraction": area_fraction,
                        "max_area_fraction": MAX_BBOX_AREA_FRACTION,
                    },
                )
            )

    return violations


def validate_r2_no_unknown_teams(
    identities: List[CommittedIdentity],
    fragments: List[ScoredFragment],
) -> List[ValidationViolation]:
    """
    R2: Every player has a team (no "unknown" after Pass 3C).

    Per CLAUDE.md R2:
    - After Pass 3C, every fragment MUST have team = "team_a" OR "team_b"
    - team = "unknown" is FORBIDDEN
    - Max 6 players per team (HARD constraint)
    - Imbalances (5v6, 6v5, etc.) are allowed (SOFT metric)

    Args:
        identities: List of committed identities from Pass 3C
        fragments: List of fragments (real + ghosts)

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    # Build identity map
    identity_map = {i.fragment_id: i for i in identities}

    # Check each fragment (exclude ghosts - they inherit team from source)
    for fragment in fragments:
        if fragment.get('is_ghost', False):
            continue  # Ghosts are excluded

        fragment_id = fragment.fragment_id

        # Check if fragment has committed identity
        if fragment_id not in identity_map:
            violations.append(
                ValidationViolation(
                    rule="R2",
                    severity="error",
                    message=f"Fragment {fragment_id} has no committed identity",
                    fragment_id=fragment_id,
                    details={"fragment_id": fragment_id},
                )
            )
            continue

        identity = identity_map[fragment_id]

        # R2: No "unknown" teams
        if identity.team == TeamID.UNKNOWN:
            violations.append(
                ValidationViolation(
                    rule="R2",
                    severity="error",
                    message=f"Fragment {fragment_id} has team='unknown' (FORBIDDEN after Pass 3C)",
                    fragment_id=fragment_id,
                    details={
                        "fragment_id": fragment_id,
                        "team": identity.team,
                    },
                )
            )

    # R2: Team size constraint (max 6 per team - HARD)
    # Build frame-by-frame team counts
    frame_team_counts: Dict[int, Dict[str, Set[str]]] = {}

    for fragment in fragments:
        if fragment.get('is_ghost', False):
            continue  # Exclude ghosts

        fragment_id = fragment.fragment_id
        if fragment_id not in identity_map:
            continue  # Already flagged above

        identity = identity_map[fragment_id]
        team = identity.team.value if isinstance(identity.team, TeamID) else identity.team

        # Count this player in each frame
        for frame_idx in range(fragment.start_frame, fragment.end_frame + 1):
            if frame_idx not in frame_team_counts:
                frame_team_counts[frame_idx] = {"team_a": set(), "team_b": set()}

            frame_team_counts[frame_idx][team].add(identity.player_id)

    # Check max 6 per team at each frame
    for frame_idx, team_counts in frame_team_counts.items():
        team_a_count = len(team_counts["team_a"])
        team_b_count = len(team_counts["team_b"])

        if team_a_count > 6:
            violations.append(
                ValidationViolation(
                    rule="R2",
                    severity="error",
                    message=f"Frame {frame_idx}: team_a has {team_a_count} players (max 6 allowed)",
                    frame_idx=frame_idx,
                    details={
                        "frame_idx": frame_idx,
                        "team_a_count": team_a_count,
                        "team_a_players": sorted(team_counts["team_a"]),
                    },
                )
            )

        if team_b_count > 6:
            violations.append(
                ValidationViolation(
                    rule="R2",
                    severity="error",
                    message=f"Frame {frame_idx}: team_b has {team_b_count} players (max 6 allowed)",
                    frame_idx=frame_idx,
                    details={
                        "frame_idx": frame_idx,
                        "team_b_count": team_b_count,
                        "team_b_players": sorted(team_counts["team_b"]),
                    },
                )
            )

    return violations


def validate_r3_jersey_temporal_exclusivity(
    identities: List[CommittedIdentity],
    fragments: List[ScoredFragment],
) -> List[ValidationViolation]:
    """
    R3: Jersey temporal exclusivity (one jersey = one player at any time).

    Per CLAUDE.md R3:
    - Same jersey number cannot appear on different players simultaneously
    - Temporal overlap detection: if two fragments with same jersey overlap in time → violation
    - Ghosts inherit jersey from source, so exclusivity applies to them too

    Args:
        identities: List of committed identities from Pass 3C
        fragments: List of fragments (real + ghosts)

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    # Build identity map
    identity_map = {i.fragment_id: i for i in identities}

    # Group fragments by jersey number
    jersey_fragments: Dict[int, List[Tuple[str, int, int, str]]] = {}  # jersey -> [(fragment_id, start, end, player_id)]

    for fragment in fragments:
        fragment_id = fragment.fragment_id

        if fragment_id not in identity_map:
            continue  # No identity assigned

        identity = identity_map[fragment_id]
        jersey = identity.jersey_number
        player_id = identity.player_id

        if jersey not in jersey_fragments:
            jersey_fragments[jersey] = []

        jersey_fragments[jersey].append((
            fragment_id,
            fragment.start_frame,
            fragment.end_frame,
            player_id,
        ))

    # Check for temporal overlaps within each jersey
    for jersey, frags in jersey_fragments.items():
        # Sort by start frame
        frags = sorted(frags, key=lambda x: x[1])

        for i in range(len(frags)):
            for j in range(i + 1, len(frags)):
                frag_a_id, start_a, end_a, player_a = frags[i]
                frag_b_id, start_b, end_b, player_b = frags[j]

                # Check if different players
                if player_a == player_b:
                    continue  # Same player - OK

                # Check temporal overlap
                if start_b <= end_a:  # Overlap detected
                    overlap_start = max(start_a, start_b)
                    overlap_end = min(end_a, end_b)

                    violations.append(
                        ValidationViolation(
                            rule="R3",
                            severity="error",
                            message=(
                                f"Jersey #{jersey} temporal conflict: "
                                f"{player_a} ({frag_a_id}) and {player_b} ({frag_b_id}) "
                                f"overlap in frames {overlap_start}-{overlap_end}"
                            ),
                            frame_idx=overlap_start,
                            details={
                                "jersey_number": jersey,
                                "player_a": player_a,
                                "fragment_a": frag_a_id,
                                "player_b": player_b,
                                "fragment_b": frag_b_id,
                                "overlap_start": overlap_start,
                                "overlap_end": overlap_end,
                            },
                        )
                    )

    return violations


def validate_r4_player_continuity(
    fragments: List[ScoredFragment],
    total_frames: int,
) -> List[ValidationViolation]:
    """
    R4: Player continuity (max 12 concurrent, high water mark).

    Per CLAUDE.md R4:
    - Target level = high water mark up to 12 (futsal regulation: 2 teams × 6 players)
    - Level only increases as more players enter, never decreases
    - Ghosts maintain identity continuity (tracked + ghosts ≤ 12 per frame)
    - Players never disappear (ghosts created for occlusions)

    Args:
        fragments: List of fragments (real + ghosts)
        total_frames: Total number of frames in video

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    # Build frame-by-frame player count (tracked players by original_track_id)
    frame_players: Dict[int, Set[int]] = {}  # frame_idx -> set of track_ids

    for fragment in fragments:
        track_id = fragment.original_track_id

        for frame_idx in range(fragment.start_frame, fragment.end_frame + 1):
            if frame_idx not in frame_players:
                frame_players[frame_idx] = set()

            frame_players[frame_idx].add(track_id)

    # Check max 12 concurrent players per frame
    for frame_idx in range(total_frames):
        if frame_idx not in frame_players:
            continue  # No players in this frame (possible at start/end)

        player_count = len(frame_players[frame_idx])

        if player_count > 12:
            violations.append(
                ValidationViolation(
                    rule="R4",
                    severity="error",
                    message=f"Frame {frame_idx}: {player_count} concurrent players (max 12 allowed)",
                    frame_idx=frame_idx,
                    details={
                        "frame_idx": frame_idx,
                        "player_count": player_count,
                        "track_ids": sorted(frame_players[frame_idx]),
                    },
                )
            )

    return violations


def validate_r5_ball_never_disappears(
    ball_positions: List[BallPosition],
    total_frames: int,
) -> List[ValidationViolation]:
    """
    R5: Ball never disappears (state exists at every frame).

    Per CLAUDE.md R5:
    - Ball state MUST exist at every frame
    - State ∈ {real, interpolated, out_of_play}
    - Validator checks presence of state, not position
    - Position can be None if state = out_of_play

    Args:
        ball_positions: List of ball positions
        total_frames: Total number of frames in video

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    # Build frame index set
    ball_frames = {bp.frame_idx for bp in ball_positions}

    # Check every frame has ball state
    for frame_idx in range(total_frames):
        if frame_idx not in ball_frames:
            violations.append(
                ValidationViolation(
                    rule="R5",
                    severity="error",
                    message=f"Frame {frame_idx}: Ball state missing (FORBIDDEN)",
                    frame_idx=frame_idx,
                    details={"frame_idx": frame_idx},
                )
            )

    # Check state validity
    for bp in ball_positions:
        # Validate state enum
        if not isinstance(bp.state, (BallState, str)):
            violations.append(
                ValidationViolation(
                    rule="R5",
                    severity="error",
                    message=f"Frame {bp.frame_idx}: Ball state invalid type {type(bp.state)}",
                    frame_idx=bp.frame_idx,
                    details={"frame_idx": bp.frame_idx, "state": str(bp.state)},
                )
            )
            continue

        state_value = bp.state.value if isinstance(bp.state, BallState) else bp.state

        if state_value not in ["real", "interpolated", "out_of_play"]:
            violations.append(
                ValidationViolation(
                    rule="R5",
                    severity="error",
                    message=f"Frame {bp.frame_idx}: Ball state '{state_value}' invalid (must be real/interpolated/out_of_play)",
                    frame_idx=bp.frame_idx,
                    details={"frame_idx": bp.frame_idx, "state": state_value},
                )
            )

        # Check position consistency
        if state_value == "out_of_play":
            if bp.centroid is not None:
                violations.append(
                    ValidationViolation(
                        rule="R5",
                        severity="warning",
                        message=f"Frame {bp.frame_idx}: Ball state='out_of_play' but centroid is not None",
                        frame_idx=bp.frame_idx,
                        details={"frame_idx": bp.frame_idx, "centroid": bp.centroid},
                    )
                )
        else:
            # real or interpolated - must have centroid
            if bp.centroid is None:
                violations.append(
                    ValidationViolation(
                        rule="R5",
                        severity="error",
                        message=f"Frame {bp.frame_idx}: Ball state='{state_value}' but centroid is None",
                        frame_idx=bp.frame_idx,
                        details={"frame_idx": bp.frame_idx, "state": state_value},
                    )
                )

    return violations
