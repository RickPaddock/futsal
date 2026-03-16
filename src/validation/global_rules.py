"""
Global validation rules (R1-R5).

Per CLAUDE.md Section 2 (Non-Negotiable Rules):
- R1: Pass 1 is raw truth only (no team, no identity)
- R2: Every player has a team (no "unknown" after Pass 3C)
- R3: Jersey temporal exclusivity (one jersey = one player at any time)
- R4: Player continuity helpers (Pass 3 ownership; pre-Pass3 use is diagnostic only)
- R5: Ball never disappears (state exists at every frame)

These are HARD constraints. Violation = pipeline failure.
"""

from typing import List, Dict, Any, Set, Tuple, Optional
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
from ..core.types import TeamID, BallState, FrameIndex, normalize_ball_state_value


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
                    severity="warning",
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
                    severity="warning",
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
                    severity="warning",
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
    from ..core import constants as const

    # Build identity map
    identity_map = {i.fragment_id: i for i in identities}

    # Check each fragment (exclude ghosts - they inherit team from source)
    for fragment in fragments:
        if getattr(fragment, 'is_ghost', False):
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
    # Build frame-by-frame team counts using real detection presence for non-ghost fragments.
    frame_team_counts: Dict[int, Dict[str, Set[str]]] = {}

    for fragment in fragments:
        if getattr(fragment, 'is_ghost', False):
            continue  # Exclude ghosts

        fragment_id = fragment.fragment_id
        if fragment_id not in identity_map:
            continue  # Already flagged above

        identity = identity_map[fragment_id]
        team = identity.team.value if isinstance(identity.team, TeamID) else identity.team

        # Count this player in active presence frames only.
        active_frames: Set[int] = set()
        for detection_id in getattr(fragment, "detection_ids", []) or []:
            try:
                frame_idx = int(str(detection_id).split("_", maxsplit=1)[0])
            except (ValueError, IndexError):
                continue
            active_frames.add(frame_idx)

        for frame_idx in sorted(active_frames):
            if frame_idx not in frame_team_counts:
                frame_team_counts[frame_idx] = {"team_a": set(), "team_b": set()}

            frame_team_counts[frame_idx][team].add(identity.player_id)

    max_team_size = int(getattr(const, "MAX_TEAM_SIZE", 6))
    allowed_violation_frames = int(getattr(const, "MAX_TEAM_SIZE_VIOLATION_FRAMES", 0))

    team_a_violations = []
    team_b_violations = []
    for frame_idx, team_counts in frame_team_counts.items():
        team_a_count = len(team_counts["team_a"])
        team_b_count = len(team_counts["team_b"])

        if team_a_count > max_team_size:
            team_a_violations.append((frame_idx, team_a_count, sorted(team_counts["team_a"])))
        if team_b_count > max_team_size:
            team_b_violations.append((frame_idx, team_b_count, sorted(team_counts["team_b"])))

    def _append_team_size_violations(team_name: str, items: List[Tuple[int, int, List[str]]]) -> None:
        if not items:
            return
        severity = "warning" if len(items) <= allowed_violation_frames else "error"
        for frame_idx, count, players in items:
            violations.append(
                ValidationViolation(
                    rule="R2",
                    severity=severity,
                    message=(
                        f"Frame {frame_idx}: {team_name} has {count} players "
                        f"(max {max_team_size} allowed, tolerance={allowed_violation_frames} frames)"
                    ),
                    frame_idx=frame_idx,
                    details={
                        "frame_idx": frame_idx,
                        f"{team_name}_count": count,
                        f"{team_name}_players": players,
                        "max_team_size": max_team_size,
                        "allowed_violation_frames": allowed_violation_frames,
                    },
                )
            )

    _append_team_size_violations("team_a", team_a_violations)
    _append_team_size_violations("team_b", team_b_violations)

    return violations


def validate_r3_jersey_temporal_exclusivity(
    identities: List[CommittedIdentity],
    fragments: List[ScoredFragment],
    ghost_activity_windows: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[ValidationViolation]:
    """
    R3: Jersey temporal exclusivity within each team.

        Per implementation contract for Pass 3C:
        - Same jersey number cannot appear on different players simultaneously
            within the same team.
    - Temporal overlap detection: if two fragments with same jersey overlap in time → violation
    - Ghosts inherit jersey from source, so exclusivity applies to them too

    Args:
        identities: List of committed identities from Pass 3C
        fragments: List of fragments (real + ghosts)

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    identity_map = {i.fragment_id: i for i in identities}
    ghost_activity_windows = ghost_activity_windows or {}
    frame_owners: Dict[Tuple[str, int, int], Set[Tuple[str, str]]] = {}

    def _active_frames(fragment: ScoredFragment) -> Set[int]:
        if getattr(fragment, 'is_ghost', False):
            window = ghost_activity_windows.get(fragment.fragment_id)
            if window is None:
                return set()
            start_frame = int(window.get("start_frame", fragment.start_frame))
            end_frame = int(window.get("end_frame", fragment.end_frame))
            if end_frame < start_frame:
                return set()
            return set(range(start_frame, end_frame + 1))

        frames: Set[int] = set()
        for detection_id in getattr(fragment, "detection_ids", []) or []:
            try:
                frames.add(int(str(detection_id).split("_", maxsplit=1)[0]))
            except (ValueError, IndexError):
                continue
        if frames:
            return frames
        return set(range(fragment.start_frame, fragment.end_frame + 1))

    for fragment in fragments:
        identity = identity_map.get(fragment.fragment_id)
        if identity is None or identity.jersey_number is None:
            continue

        team = identity.team.value if hasattr(identity.team, "value") else str(identity.team)
        if team == "unknown":
            continue

        for frame_idx in _active_frames(fragment):
            frame_owners.setdefault((team, int(identity.jersey_number), frame_idx), set()).add(
                (identity.player_id, fragment.fragment_id)
            )

    pair_conflicts: Dict[Tuple[str, int, str, str, str, str], List[int]] = {}
    for (team, jersey, frame_idx), owners in frame_owners.items():
        distinct_owners = sorted(owners)
        if len({player_id for player_id, _ in distinct_owners}) <= 1:
            continue
        for i in range(len(distinct_owners)):
            for j in range(i + 1, len(distinct_owners)):
                player_a, frag_a_id = distinct_owners[i]
                player_b, frag_b_id = distinct_owners[j]
                if player_a == player_b:
                    continue
                key = (team, jersey, player_a, frag_a_id, player_b, frag_b_id)
                pair_conflicts.setdefault(key, []).append(frame_idx)

    for (team, jersey, player_a, frag_a_id, player_b, frag_b_id), frames in pair_conflicts.items():
        frames = sorted(set(frames))
        if not frames:
            continue
        run_start = frames[0]
        run_end = frames[0]
        for frame_idx in frames[1:]:
            if frame_idx == run_end + 1:
                run_end = frame_idx
                continue
            violations.append(
                ValidationViolation(
                    rule="R3",
                    severity="error",
                    message=(
                        f"Jersey #{jersey} temporal conflict: "
                        f"{player_a} ({frag_a_id}) and {player_b} ({frag_b_id}) "
                        f"on team {team} overlap in frames {run_start}-{run_end}"
                    ),
                    frame_idx=run_start,
                    details={
                        "jersey_number": jersey,
                        "player_a": player_a,
                        "fragment_a": frag_a_id,
                        "player_b": player_b,
                        "fragment_b": frag_b_id,
                        "overlap_start": run_start,
                        "overlap_end": run_end,
                    },
                )
            )
            run_start = frame_idx
            run_end = frame_idx

        violations.append(
            ValidationViolation(
                rule="R3",
                severity="error",
                message=(
                    f"Jersey #{jersey} temporal conflict: "
                    f"{player_a} ({frag_a_id}) and {player_b} ({frag_b_id}) "
                    f"on team {team} overlap in frames {run_start}-{run_end}"
                ),
                frame_idx=run_start,
                details={
                    "jersey_number": jersey,
                    "player_a": player_a,
                    "fragment_a": frag_a_id,
                    "player_b": player_b,
                    "fragment_b": frag_b_id,
                    "overlap_start": run_start,
                    "overlap_end": run_end,
                },
            )
        )

    return violations


def validate_r4_player_continuity(
    fragments: List[ScoredFragment],
    total_frames: int,
) -> List[ValidationViolation]:
    """
    R4 helper: concurrent identity diagnostics.

    NOTE:
    - Physical cap enforcement belongs to Pass 3 identity-resolved state.
    - If called on pre-Pass3 artifacts, >12 is a non-blocking diagnostic signal.

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
                    severity="warning",
                    message=(
                        f"Frame {frame_idx}: {player_count} concurrent entities (>12). "
                        "Diagnostic signal; enforce as hard failure only in Pass 3."
                    ),
                    frame_idx=frame_idx,
                    details={
                        "frame_idx": frame_idx,
                        "player_count": player_count,
                        "track_ids": sorted(frame_players[frame_idx]),
                    },
                )
            )

    return violations


def validate_r4_player_continuity_duration_aware(
    fragments: List[ScoredFragment],
    total_frames: int,
) -> List[ValidationViolation]:
    """
    R4 helper (legacy): duration-aware concurrent-entity diagnostics.

    Per architectural principle:
    - Brief violations (1-2 frames) = tracker jitter → tolerate
    - Sustained violations (3+ frames) = detector failure → FAIL HARD

    This is the correct mental model:
    - Pass 1 can violate R4 (ByteTrack can hallucinate briefly)
    - Pass 2 must detect and expose sustained violations (diagnostic constraint)
    - Pass 3 enforces strictly after identity resolution

    Args:
        fragments: List of fragments (real + ghosts)
        total_frames: Total number of frames in video

    Returns:
        List of violations (only for sustained violations)
    """
    from ..core import constants as const

    violations = []

    # Build frame-by-frame player count (tracked players by original_track_id)
    frame_players: Dict[int, Set[int]] = {}  # frame_idx -> set of track_ids

    for fragment in fragments:
        track_id = fragment.original_track_id

        for frame_idx in range(fragment.start_frame, fragment.end_frame + 1):
            if frame_idx not in frame_players:
                frame_players[frame_idx] = set()

            frame_players[frame_idx].add(track_id)

    # Find consecutive violation runs
    violation_runs: List[Tuple[int, int, int]] = []  # (start_frame, end_frame, player_count)
    current_run_start = None
    current_run_count = 0

    for frame_idx in range(total_frames):
        if frame_idx not in frame_players:
            # No players in this frame - end current run
            if current_run_start is not None:
                violation_runs.append((current_run_start, frame_idx - 1, current_run_count))
                current_run_start = None
            continue

        player_count = len(frame_players[frame_idx])

        if player_count > 12:
            # Violation detected
            if current_run_start is None:
                # Start new run
                current_run_start = frame_idx
                current_run_count = player_count
            else:
                # Continue current run (update count to max)
                current_run_count = max(current_run_count, player_count)
        else:
            # No violation - end current run if active
            if current_run_start is not None:
                violation_runs.append((current_run_start, frame_idx - 1, current_run_count))
                current_run_start = None

    # Close final run if active
    if current_run_start is not None:
        violation_runs.append((current_run_start, total_frames - 1, current_run_count))

    # Filter: only fail on sustained violations (> threshold consecutive frames)
    max_consecutive = const.MAX_R4_VIOLATION_CONSECUTIVE_FRAMES

    for start_frame, end_frame, player_count in violation_runs:
        run_length = end_frame - start_frame + 1

        if run_length > max_consecutive:
            # SUSTAINED DIAGNOSTIC EVENT (pre-Pass3 enforcement is a layering violation)
            violations.append(
                ValidationViolation(
                    rule="R4",
                    severity="warning",
                    message=(
                        f"Sustained concurrent-entity overage: {player_count} entities "
                        f"for {run_length} consecutive frames [{start_frame}-{end_frame}] "
                        f"(max {max_consecutive} frames tolerated). "
                        "Diagnostic signal; hard enforcement belongs to Pass 3."
                    ),
                    frame_idx=start_frame,
                    details={
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "run_length": run_length,
                        "player_count": player_count,
                        "max_tolerated_frames": max_consecutive,
                    },
                )
            )
        # else: brief violation, tolerated

    return violations


def validate_r5_ball_never_disappears(
    ball_positions: List[BallPosition],
    total_frames: int,
) -> List[ValidationViolation]:
    """
    R5: Ball never disappears (state exists at every frame).

    Per CLAUDE.md R5:
    - Ball state MUST exist at every frame
    - State ∈ {real, interpolated, unknown}
    - Validator checks presence of state, not position
    - Position can be None if state = unknown

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

        state_value = normalize_ball_state_value(bp.state)

        if state_value not in ["real", "interpolated", "unknown"]:
            violations.append(
                ValidationViolation(
                    rule="R5",
                    severity="error",
                    message=f"Frame {bp.frame_idx}: Ball state '{state_value}' invalid (must be real/interpolated/unknown)",
                    frame_idx=bp.frame_idx,
                    details={"frame_idx": bp.frame_idx, "state": state_value},
                )
            )

        # Check position consistency
        if state_value == "unknown":
            if bp.centroid is not None:
                violations.append(
                    ValidationViolation(
                        rule="R5",
                        severity="warning",
                        message=f"Frame {bp.frame_idx}: Ball state='unknown' but centroid is not None",
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
