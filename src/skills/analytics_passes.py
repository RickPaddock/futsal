"""Analytics event detection built from the committed possession artifact."""

from __future__ import annotations

from math import acos, degrees, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..core import constants as const
from ..core.data_models import AnalyticsEvent, AnalyticsEventsOutput, AnalyticsPossessionOutput, BirdseyeProjectionOutput, PossessionCandidate, PossessionFrame
from ..core.schemas import ANALYTICS_EVENTS_OUTPUT_SCHEMA
from ..utils.file_utils import load_json, save_json
from ..utils.logging_utils import get_logger

logger = get_logger("analytics_passes")


def _receive_window_frames(fps: float) -> int:
    return max(1, int(round(float(fps) * float(const.ANALYTICS_PASS_RECEIVE_WINDOW_SECONDS))))


def _relay_window_frames(fps: float) -> int:
    receive_window = _receive_window_frames(fps)
    relay_target = int(round(float(fps) * float(const.ANALYTICS_PASS_RELAY_LOOKAHEAD_SECONDS)))
    return max(receive_window + 3, relay_target)


def _find_possession_run_end(frames: List[PossessionFrame], start_idx: int) -> int:
    owner = frames[start_idx].player_id
    run_end = start_idx
    while run_end + 1 < len(frames) and frames[run_end + 1].player_id == owner:
        run_end += 1
    return run_end


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return float(sqrt(dx * dx + dy * dy))


def _vector_angle_degrees(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ab = (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
    bc = (float(c[0]) - float(b[0]), float(c[1]) - float(b[1]))
    ab_norm = sqrt((ab[0] * ab[0]) + (ab[1] * ab[1]))
    bc_norm = sqrt((bc[0] * bc[0]) + (bc[1] * bc[1]))
    if ab_norm <= 1e-6 or bc_norm <= 1e-6:
        return 0.0

    dot = (ab[0] * bc[0]) + (ab[1] * bc[1])
    cosine = max(-1.0, min(1.0, dot / (ab_norm * bc_norm)))
    return float(degrees(acos(cosine)))


def _candidate_to_frame(frame_idx: int, candidate: PossessionCandidate, source_frame: PossessionFrame) -> PossessionFrame:
    return PossessionFrame(
        frame_idx=frame_idx,
        player_id=candidate.player_id,
        team=candidate.team,
        jersey_number=candidate.jersey_number,
        confidence=candidate.confidence,
        ball_state=source_frame.ball_state,
        source_space=source_frame.source_space,
        candidates=source_frame.candidates,
    )


def _find_next_owner_idx(
    frames: List[PossessionFrame],
    start_idx: int,
    end_idx: int,
    same_team_only: Optional[bool] = None,
    team=None,
    exclude_player_ids: Optional[set[str]] = None,
) -> Optional[int]:
    excluded = exclude_player_ids or set()
    for probe_idx in range(start_idx, end_idx + 1):
        probe_frame = frames[probe_idx]
        if probe_frame.player_id is None or probe_frame.player_id in excluded:
            continue
        if same_team_only is True and probe_frame.team != team:
            continue
        if same_team_only is False and probe_frame.team == team:
            continue
        return probe_idx
    return None


def _find_candidate_streak(
    frames: List[PossessionFrame],
    start_idx: int,
    end_idx: int,
    team,
    exclude_player_ids: set[str],
) -> Optional[Tuple[int, int, PossessionCandidate]]:
    streak_start: Optional[int] = None
    streak_end: Optional[int] = None
    streak_candidate: Optional[PossessionCandidate] = None

    for probe_idx in range(start_idx, end_idx + 1):
        probe_frame = frames[probe_idx]
        candidate = next(
            (
                entry
                for entry in probe_frame.candidates
                if entry.team == team and entry.player_id not in exclude_player_ids
            ),
            None,
        )
        if candidate is None:
            if streak_start is not None:
                break
            continue

        if streak_candidate is None or candidate.player_id != streak_candidate.player_id:
            if streak_start is not None:
                break
            streak_start = probe_idx
            streak_candidate = candidate
        streak_end = probe_idx

    if streak_start is None or streak_end is None or streak_candidate is None:
        return None

    streak_length = (streak_end - streak_start) + 1
    if streak_length < int(const.ANALYTICS_PASS_RELAY_MIN_STREAK_FRAMES):
        return None
    return streak_start, streak_end, streak_candidate


def _find_goal_crossing_idx(
    ball_positions: Dict[int, List[float]],
    start_frame: int,
    end_frame: int,
    court_length_m: float,
    court_width_m: float,
) -> Optional[int]:
    release_ball = ball_positions.get(start_frame)
    if release_ball is None:
        return None

    center_y = float(court_width_m) / 2.0
    for frame_idx in range(start_frame + 1, end_frame + 1):
        ball_position = ball_positions.get(frame_idx)
        if ball_position is None:
            continue

        crossed_goal_line = ball_position[0] <= 0.0 or ball_position[0] >= float(court_length_m)
        in_goal_band = abs(float(ball_position[1]) - center_y) <= float(const.ANALYTICS_SHOT_GOAL_BAND_HALF_WIDTH_M)
        travelled_far_enough = _distance(release_ball, ball_position) >= float(const.ANALYTICS_SHOT_MIN_TRAVEL_M)
        release_near_goal = min(
            abs(float(release_ball[0]) - 0.0),
            abs(float(court_length_m) - float(release_ball[0])),
        ) <= float(const.ANALYTICS_SHOT_GOAL_LINE_DISTANCE_M)

        if crossed_goal_line and in_goal_band and travelled_far_enough and release_near_goal:
            return frame_idx
    return None


def _route_turn_supports_relay(
    ball_positions: Dict[int, List[float]],
    release_frame: int,
    bridge_frame: int,
    receive_frame: int,
) -> bool:
    release_ball = ball_positions.get(release_frame)
    bridge_ball = ball_positions.get(bridge_frame)
    receive_ball = ball_positions.get(receive_frame)
    if release_ball is None or bridge_ball is None or receive_ball is None:
        return False

    return _vector_angle_degrees(release_ball, bridge_ball, receive_ball) >= float(
        const.ANALYTICS_PASS_RELAY_MIN_ROUTE_TURN_DEGREES
    )


def _event_confidence(
    passer_frame: PossessionFrame,
    receiver_frame: Optional[PossessionFrame],
    outcome: str,
) -> float:
    confidence_values = [float(passer_frame.confidence)]
    if receiver_frame is not None:
        confidence_values.append(float(receiver_frame.confidence))

    base_confidence = sum(confidence_values) / len(confidence_values)
    if outcome == "loose_ball":
        base_confidence *= 0.85
    return max(0.0, min(1.0, float(base_confidence)))


def _build_pass_event(
    event_index: int,
    passer_frame: PossessionFrame,
    receiver_frame: Optional[PossessionFrame],
    outcome: str,
    end_frame: int,
) -> AnalyticsEvent:
    is_audit_only = passer_frame.jersey_number is None
    if outcome == "successful" and receiver_frame is not None and receiver_frame.jersey_number is None:
        is_audit_only = True

    return AnalyticsEvent(
        event_id=f"EV{event_index:06d}",
        event_type="pass",
        start_frame=int(passer_frame.frame_idx),
        end_frame=int(end_frame),
        team_id=passer_frame.team,
        outcome=outcome,
        event_confidence=_event_confidence(passer_frame, receiver_frame, outcome),
        is_audit_only=is_audit_only,
        passer_player_id=passer_frame.player_id,
        passer_jersey_number=passer_frame.jersey_number,
        receiver_player_id=None if receiver_frame is None else receiver_frame.player_id,
        receiver_team_id=None if receiver_frame is None else receiver_frame.team,
        receiver_jersey_number=None if receiver_frame is None else receiver_frame.jersey_number,
    )


def _build_shot_event(
    event_index: int,
    shooter_frame: PossessionFrame,
    end_frame: int,
    outcome: str,
) -> AnalyticsEvent:
    return AnalyticsEvent(
        event_id=f"EV{event_index:06d}",
        event_type="shot",
        start_frame=int(shooter_frame.frame_idx),
        end_frame=int(end_frame),
        team_id=shooter_frame.team,
        outcome=outcome,
        event_confidence=max(0.0, min(1.0, float(shooter_frame.confidence))),
        is_audit_only=shooter_frame.jersey_number is None,
        passer_player_id=shooter_frame.player_id,
        passer_jersey_number=shooter_frame.jersey_number,
        receiver_player_id=None,
        receiver_team_id=None,
        receiver_jersey_number=None,
    )


def build_analytics_events_output(
    possession_output: AnalyticsPossessionOutput,
    birdseye_output: Optional[BirdseyeProjectionOutput] = None,
) -> AnalyticsEventsOutput:
    """Build pass and shot events from committed possession output."""
    frames = possession_output.frames
    receive_window = _receive_window_frames(possession_output.fps)
    relay_window = _relay_window_frames(possession_output.fps)
    events: List[AnalyticsEvent] = []
    ball_positions: Dict[int, List[float]] = {}
    court_length_m = 0.0
    court_width_m = 0.0
    if birdseye_output is not None:
        court_length_m = float(birdseye_output.court_length_m)
        court_width_m = float(birdseye_output.court_width_m)
        ball_positions = {
            int(frame.frame_idx): frame.ball.court_position
            for frame in birdseye_output.frames
            if frame.ball.court_position is not None
        }

    diagnostics: Dict[str, float] = {
        "receive_window_frames": float(receive_window),
        "relay_window_frames": float(relay_window),
        "candidate_release_sequences": 0.0,
        "recoveries_skipped": 0.0,
        "emitted_pass_events": 0.0,
        "emitted_shot_events": 0.0,
        "successful_pass_events": 0.0,
        "unsuccessful_pass_events": 0.0,
        "loose_ball_events": 0.0,
        "goal_shot_events": 0.0,
        "relay_split_events": 0.0,
        "audit_only_events": 0.0,
        "unknown_passer_events": 0.0,
        "unknown_receiver_events": 0.0,
    }

    event_index = 1
    idx = 0
    while idx < len(frames):
        frame = frames[idx]
        if frame.player_id is None:
            idx += 1
            continue

        run_end = _find_possession_run_end(frames, idx)
        passer_frame = frames[run_end]
        diagnostics["candidate_release_sequences"] += 1.0

        search_end_idx = min(len(frames) - 1, run_end + receive_window)
        extended_search_end_idx = min(len(frames) - 1, run_end + relay_window)
        next_owner_idx: Optional[int] = None
        next_owner_frame: Optional[PossessionFrame] = None

        next_owner_idx = _find_next_owner_idx(frames, run_end + 1, search_end_idx)
        if next_owner_idx is not None:
            next_owner_frame = frames[next_owner_idx]

        if next_owner_frame is not None and next_owner_frame.player_id == passer_frame.player_id:
            diagnostics["recoveries_skipped"] += 1.0
            idx = next_owner_idx
            continue

        shot_goal_idx: Optional[int] = None
        if birdseye_output is not None:
            shot_goal_idx = _find_goal_crossing_idx(
                ball_positions=ball_positions,
                start_frame=int(passer_frame.frame_idx),
                end_frame=extended_search_end_idx,
                court_length_m=court_length_m,
                court_width_m=court_width_m,
            )

        same_team_receive_before_goal = (
            shot_goal_idx is not None
            and _find_next_owner_idx(
                frames,
                run_end + 1,
                min(len(frames) - 1, shot_goal_idx),
                same_team_only=True,
                team=passer_frame.team,
                exclude_player_ids={passer_frame.player_id},
            )
            is not None
        )
        any_owner_before_goal = (
            shot_goal_idx is not None
            and _find_next_owner_idx(
                frames,
                run_end + 1,
                min(len(frames) - 1, shot_goal_idx),
            )
            is not None
        )

        if shot_goal_idx is not None and not same_team_receive_before_goal and not any_owner_before_goal:
            event_to_emit = _build_shot_event(
                event_index=event_index,
                shooter_frame=passer_frame,
                end_frame=int(shot_goal_idx),
                outcome="goal",
            )
            events.append(event_to_emit)
            event_index += 1
            diagnostics["emitted_shot_events"] += 1.0
            diagnostics["goal_shot_events"] += 1.0
            if event_to_emit.is_audit_only:
                diagnostics["audit_only_events"] += 1.0
            if event_to_emit.passer_jersey_number is None:
                diagnostics["unknown_passer_events"] += 1.0
            idx = min(len(frames) - 1, int(shot_goal_idx) + 1)
            continue

        first_extended_owner_idx = _find_next_owner_idx(
            frames,
            run_end + 1,
            extended_search_end_idx,
        )
        late_same_team_owner_idx = None
        if first_extended_owner_idx is None or frames[first_extended_owner_idx].team == passer_frame.team:
            late_same_team_owner_idx = _find_next_owner_idx(
                frames,
                run_end + 1,
                extended_search_end_idx,
                same_team_only=True,
                team=passer_frame.team,
                exclude_player_ids={passer_frame.player_id},
            )

        relay_bridge = None
        if birdseye_output is not None and late_same_team_owner_idx is not None:
            relay_bridge = _find_candidate_streak(
                frames,
                run_end + 1,
                late_same_team_owner_idx - 1,
                passer_frame.team,
                exclude_player_ids={passer_frame.player_id, frames[late_same_team_owner_idx].player_id},
            )

        if relay_bridge is not None and late_same_team_owner_idx is not None:
            bridge_start_idx, bridge_end_idx, bridge_candidate = relay_bridge
            if _route_turn_supports_relay(
                ball_positions=ball_positions,
                release_frame=int(passer_frame.frame_idx),
                bridge_frame=bridge_start_idx,
                receive_frame=late_same_team_owner_idx,
            ):
                bridge_frame = _candidate_to_frame(
                    frame_idx=int(frames[bridge_start_idx].frame_idx),
                    candidate=bridge_candidate,
                    source_frame=frames[bridge_start_idx],
                )
                first_event = _build_pass_event(
                    event_index=event_index,
                    passer_frame=passer_frame,
                    receiver_frame=bridge_frame,
                    outcome="successful",
                    end_frame=int(frames[bridge_start_idx].frame_idx),
                )
                event_index += 1
                second_event = _build_pass_event(
                    event_index=event_index,
                    passer_frame=bridge_frame,
                    receiver_frame=frames[late_same_team_owner_idx],
                    outcome="successful",
                    end_frame=int(frames[late_same_team_owner_idx].frame_idx),
                )
                event_index += 1
                events.extend([first_event, second_event])
                diagnostics["emitted_pass_events"] += 2.0
                diagnostics["successful_pass_events"] += 2.0
                diagnostics["relay_split_events"] += 1.0
                for event_to_emit in (first_event, second_event):
                    if event_to_emit.is_audit_only:
                        diagnostics["audit_only_events"] += 1.0
                    if event_to_emit.passer_jersey_number is None:
                        diagnostics["unknown_passer_events"] += 1.0
                    if event_to_emit.receiver_player_id is not None and event_to_emit.receiver_jersey_number is None:
                        diagnostics["unknown_receiver_events"] += 1.0
                idx = late_same_team_owner_idx
                continue

        event_to_emit: Optional[AnalyticsEvent] = None
        if next_owner_frame is None:
            if late_same_team_owner_idx is not None:
                next_owner_idx = late_same_team_owner_idx
                next_owner_frame = frames[next_owner_idx]
                event_to_emit = _build_pass_event(
                    event_index=event_index,
                    passer_frame=passer_frame,
                    receiver_frame=next_owner_frame,
                    outcome="successful",
                    end_frame=int(next_owner_frame.frame_idx),
                )
            else:
                event_to_emit = _build_pass_event(
                    event_index=event_index,
                    passer_frame=passer_frame,
                    receiver_frame=None,
                    outcome="loose_ball",
                    end_frame=int(frames[search_end_idx].frame_idx),
                )
        elif next_owner_frame.team == passer_frame.team:
            event_to_emit = _build_pass_event(
                event_index=event_index,
                passer_frame=passer_frame,
                receiver_frame=next_owner_frame,
                outcome="successful",
                end_frame=int(next_owner_frame.frame_idx),
            )
        else:
            event_to_emit = _build_pass_event(
                event_index=event_index,
                passer_frame=passer_frame,
                receiver_frame=next_owner_frame,
                outcome="unsuccessful",
                end_frame=int(next_owner_frame.frame_idx),
            )

        if event_to_emit is not None:
            events.append(event_to_emit)
            event_index += 1
            diagnostics["emitted_pass_events"] += 1.0
            if event_to_emit.is_audit_only:
                diagnostics["audit_only_events"] += 1.0
            if event_to_emit.passer_jersey_number is None:
                diagnostics["unknown_passer_events"] += 1.0
            if event_to_emit.receiver_player_id is not None and event_to_emit.receiver_jersey_number is None:
                diagnostics["unknown_receiver_events"] += 1.0
            if event_to_emit.outcome == "successful":
                diagnostics["successful_pass_events"] += 1.0
            elif event_to_emit.outcome == "unsuccessful":
                diagnostics["unsuccessful_pass_events"] += 1.0
            else:
                diagnostics["loose_ball_events"] += 1.0

        idx = search_end_idx + 1 if next_owner_idx is None else next_owner_idx

    return AnalyticsEventsOutput(
        video_name=possession_output.video_name,
        fps=possession_output.fps,
        total_frames=possession_output.total_frames,
        processed_start_frame=possession_output.processed_start_frame,
        processed_end_frame_exclusive=possession_output.processed_end_frame_exclusive,
        events=events,
        diagnostics=diagnostics,
    )


def run_analytics_pass_detection(
    input_dir: Path,
    output_dir: Optional[Path] = None,
) -> AnalyticsEventsOutput:
    """Execute event detection from an artifact directory."""
    if output_dir is None:
        output_dir = input_dir

    possession_path = input_dir / const.ANALYTICS_POSSESSION_JSON
    birdseye_path = input_dir / const.BIRDSEYE_PROJECTION_JSON
    output_path = output_dir / const.ANALYTICS_EVENTS_JSON

    possession_output = load_json(possession_path, AnalyticsPossessionOutput)
    birdseye_output = load_json(birdseye_path, BirdseyeProjectionOutput)
    events_output = build_analytics_events_output(
        possession_output=possession_output,
        birdseye_output=birdseye_output,
    )
    save_json(events_output.model_dump(), str(output_path), ANALYTICS_EVENTS_OUTPUT_SCHEMA)
    logger.info(f"Analytics events artifact written: {output_path}")
    return events_output