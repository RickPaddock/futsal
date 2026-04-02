"""Analytics debug visualization for possession review."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from ..core import constants as const
from ..core.data_models import AnalyticsEvent, AnalyticsEventsOutput, AnalyticsPossessionOutput, BirdseyeFrame, BirdseyePlayerPosition, BirdseyeProjectionOutput, PossessionCandidate, PossessionFrame
from ..core.types import BallState, TeamID
from ..utils.file_utils import load_json
from ..utils.logging_utils import get_logger
from ..utils.video_io import VideoReader
from .analytics_passes import run_analytics_pass_detection
from .analytics_possession import run_analytics_possession
from .visualizer import _dedupe_players_for_render, _draw_ball_overlay, _draw_player_overlay, _draw_text_with_bg, _identity_label, _player_label, _render_pitch_inset

logger = get_logger("analytics_visualizer")

REPORTABLE_HIGHLIGHT = (0, 215, 255)
INTERNAL_ONLY_HIGHLIGHT = (200, 200, 200)
AMBIGUOUS_HIGHLIGHT = (0, 140, 255)
INFO_TEXT_COLOR = (0, 255, 255)
INFO_TOP_Y = 84
INFO_LINE_SPACING = 30
PASS_ROUTE_PREVIEW_SECONDS = 0.35
PASS_ROUTE_IMAGE_THICKNESS = 1
PASS_ROUTE_PITCH_THICKNESS = 2
SHOT_GOAL_COLOR = (0, 255, 0)
SHOT_MISS_COLOR = (0, 0, 255)
SHOT_ROUTE_IMAGE_THICKNESS = 3
SHOT_ROUTE_PITCH_THICKNESS = 4
GOAL_BANNER_TEXT = "GOAL!"


def _team_label(team: Optional[TeamID]) -> str:
    if team is None:
        return "unknown_team"
    return team.value


def _candidate_label(candidate: PossessionCandidate) -> str:
    return _identity_label(candidate.player_id, candidate.jersey_number)


def _event_participant_label(
    player_id: Optional[str],
    jersey_number: Optional[int],
    team: Optional[TeamID],
) -> str:
    if player_id is None:
        return f"unknown player ({_team_label(team)})"
    return _identity_label(player_id, jersey_number)


def _possession_owner_text(frame: PossessionFrame) -> Tuple[str, Tuple[int, int, int]]:
    if frame.is_ambiguous:
        return "poss=ambiguous", AMBIGUOUS_HIGHLIGHT

    if frame.player_id is None:
        return "poss=none", (255, 255, 255)

    if frame.jersey_number is None:
        return f"poss=unknown player ({_team_label(frame.team)}) conf={frame.confidence:.2f}", INTERNAL_ONLY_HIGHLIGHT

    label = _candidate_label(
        PossessionCandidate(
            player_id=frame.player_id,
            team=frame.team,
            jersey_number=frame.jersey_number,
            distance_to_ball=0.0,
            source_space=frame.source_space or "-",
            confidence=frame.confidence,
        )
    )
    return f"poss={label} conf={frame.confidence:.2f}", REPORTABLE_HIGHLIGHT


def _candidate_summary_text(frame: PossessionFrame) -> Optional[str]:
    if not frame.candidates:
        return None

    summary = ", ".join(
        f"{_candidate_label(candidate)}:{candidate.confidence:.2f}"
        for candidate in frame.candidates[:3]
    )
    return f"candidates {summary}"


def _highlight_player_overlay(
    canvas: np.ndarray,
    player: BirdseyePlayerPosition,
    color: Tuple[int, int, int],
    label_prefix: Optional[str] = None,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in player.image_bbox]
    inset = 4
    cv2.rectangle(
        canvas,
        (max(0, x1 - inset), max(0, y1 - inset)),
        (min(canvas.shape[1] - 1, x2 + inset), min(canvas.shape[0] - 1, y2 + inset)),
        color,
        2,
    )

    anchor_x, anchor_y = [int(round(v)) for v in player.image_anchor]
    cv2.circle(canvas, (anchor_x, anchor_y), 10, color, 2)
    if label_prefix is not None:
        _draw_text_with_bg(canvas, f"{label_prefix} {_player_label(player)}", (x1, max(18, y2 + 16)), 0.42, (255, 255, 255), 1)


def _highlight_pitch_player(
    canvas: np.ndarray,
    player: BirdseyePlayerPosition,
    color: Tuple[int, int, int],
) -> None:
    center = tuple(int(round(v)) for v in player.render_position)
    cv2.circle(canvas, center, 14, color, 2)


def _active_event_for_frame(
    active_events_by_frame: Dict[int, List[AnalyticsEvent]],
    frame_idx: int,
) -> Optional[AnalyticsEvent]:
    active_events = active_events_by_frame.get(frame_idx)
    if not active_events:
        return None

    for event in active_events:
        if int(event.start_frame) <= frame_idx <= int(event.end_frame):
            return event
    return active_events[0]


def _pass_preview_frame_count(fps: float) -> int:
    return max(3, int(round(fps * PASS_ROUTE_PREVIEW_SECONDS)))


def _build_display_events_by_frame(
    events_output: AnalyticsEventsOutput,
    preview_frames: int,
) -> Dict[int, List[AnalyticsEvent]]:
    display_events_by_frame: Dict[int, List[AnalyticsEvent]] = {}
    for event in events_output.events:
        preview_start = int(event.start_frame)
        if event.event_type == "pass":
            preview_start = max(0, int(event.start_frame) - preview_frames)
        for frame_idx in range(preview_start, int(event.end_frame) + 1):
            display_events_by_frame.setdefault(frame_idx, []).append(event)
    return display_events_by_frame


def _goal_banner_frame_count(fps: float) -> int:
    return max(1, int(round(float(fps) * float(const.ANALYTICS_GOAL_BANNER_SECONDS))))


def _build_goal_banner_by_frame(events_output: AnalyticsEventsOutput) -> Dict[int, AnalyticsEvent]:
    banner_frames = _goal_banner_frame_count(events_output.fps)
    goal_banner_by_frame: Dict[int, AnalyticsEvent] = {}
    max_frame = int(events_output.processed_end_frame_exclusive or events_output.total_frames)
    for event in events_output.events:
        if event.event_type != "shot" or event.outcome != "goal":
            continue
        banner_end = min(max_frame, int(event.end_frame) + banner_frames)
        for frame_idx in range(int(event.end_frame), banner_end):
            goal_banner_by_frame[frame_idx] = event
    return goal_banner_by_frame


def _dedupe_consecutive_points(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    deduped: List[Tuple[int, int]] = []
    for point in points:
        if not deduped or deduped[-1] != point:
            deduped.append(point)
    return deduped


def _build_event_route_lookup(
    frame_map: Dict[int, BirdseyeFrame],
    events_output: AnalyticsEventsOutput,
) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    route_lookup: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    for event in events_output.events:
        if event.event_type not in {"pass", "shot"}:
            continue

        image_points: List[Tuple[int, int]] = []
        pitch_points: List[Tuple[int, int]] = []
        for frame_idx in range(int(event.start_frame), int(event.end_frame) + 1):
            frame_projection = frame_map.get(frame_idx)
            if frame_projection is None:
                continue

            if frame_projection.ball.image_position is not None:
                image_points.append(tuple(int(round(v)) for v in frame_projection.ball.image_position))
            if frame_projection.ball.render_position is not None:
                pitch_points.append(tuple(int(round(v)) for v in frame_projection.ball.render_position))

        route_lookup[event.event_id] = {
            "image_points": _dedupe_consecutive_points(image_points),
            "pitch_points": _dedupe_consecutive_points(pitch_points),
        }
    return route_lookup


def _draw_route_polyline(
    canvas: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return

    cv2.polylines(
        canvas,
        [np.asarray(points, dtype=np.int32)],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _event_route_style(event: AnalyticsEvent) -> Tuple[Tuple[int, int, int], int, int]:
    if event.event_type == "shot":
        color = SHOT_GOAL_COLOR if event.outcome == "goal" else SHOT_MISS_COLOR
        return color, SHOT_ROUTE_IMAGE_THICKNESS, SHOT_ROUTE_PITCH_THICKNESS

    color = INTERNAL_ONLY_HIGHLIGHT if event.is_audit_only else REPORTABLE_HIGHLIGHT
    return color, PASS_ROUTE_IMAGE_THICKNESS, PASS_ROUTE_PITCH_THICKNESS


def _draw_goal_banner(overlay: np.ndarray) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_w, _), _ = cv2.getTextSize(GOAL_BANNER_TEXT, font, font_scale, thickness)
    origin_x = max(12, (overlay.shape[1] - text_w) // 2)
    _draw_text_with_bg(overlay, GOAL_BANNER_TEXT, (origin_x, 38), font_scale, INFO_TEXT_COLOR, thickness)


def _event_banner_text(event: AnalyticsEvent) -> str:
    if event.event_type == "shot":
        shooter_label = _event_participant_label(
            player_id=event.passer_player_id,
            jersey_number=event.passer_jersey_number,
            team=event.team_id,
        )
        return f"SHOT {_team_label(event.team_id)} {shooter_label} {event.outcome}"
    if event.event_type != "pass":
        return f"{event.event_type.upper()} {_team_label(event.team_id)}"

    passer_label = _event_participant_label(
        player_id=event.passer_player_id,
        jersey_number=event.passer_jersey_number,
        team=event.team_id,
    )
    if event.outcome == "loose_ball":
        return f"PASS {_team_label(event.team_id)} {passer_label} -> loose ball"

    receiver_label = _event_participant_label(
        player_id=event.receiver_player_id,
        jersey_number=event.receiver_jersey_number,
        team=event.receiver_team_id,
    )
    return f"PASS {_team_label(event.team_id)} {passer_label} -> {receiver_label}"


def _control_line_text(frame: PossessionFrame) -> str:
    if frame.is_ambiguous:
        return "CONTROL ambiguous"
    if frame.player_id is None:
        return "CONTROL none"
    if frame.jersey_number is None:
        return f"CONTROL unknown player ({_team_label(frame.team)}) conf={frame.confidence:.2f}"
    return f"CONTROL {_event_participant_label(frame.player_id, frame.jersey_number, frame.team)} conf={frame.confidence:.2f}"


def _pass_line_text(active_event: Optional[AnalyticsEvent], frame_idx: int) -> str:
    if active_event is None or active_event.event_type != "pass":
        return "PASS none"

    banner = _event_banner_text(active_event)
    suffix = " audit-only" if active_event.is_audit_only else " reportable"
    if frame_idx < int(active_event.start_frame):
        return f"PREVIEW {banner}{suffix}"
    return f"{banner}{suffix}"


def _shot_line_text(active_event: Optional[AnalyticsEvent]) -> str:
    if active_event is None or active_event.event_type != "shot":
        return "SHOT none"
    return _event_banner_text(active_event)


def _status_line_text(frame_idx: int, possession_frame: PossessionFrame, events_output: AnalyticsEventsOutput) -> str:
    ball_state = possession_frame.ball_state.value if isinstance(possession_frame.ball_state, BallState) else str(possession_frame.ball_state)
    return (
        f"frame={frame_idx} ball={ball_state} source={possession_frame.source_space or '-'} "
        f"events={len(events_output.events)}"
    )


def _draw_analytics_info_block(
    overlay: np.ndarray,
    frame_idx: int,
    possession_frame: PossessionFrame,
    active_event: Optional[AnalyticsEvent],
    events_output: AnalyticsEventsOutput,
) -> None:
    lines = [
        _control_line_text(possession_frame),
        _pass_line_text(active_event, frame_idx),
        _shot_line_text(active_event),
        _status_line_text(frame_idx, possession_frame, events_output),
    ]
    for index, line in enumerate(lines):
        _draw_text_with_bg(
            overlay,
            line,
            (12, INFO_TOP_Y + (INFO_LINE_SPACING * index)),
            0.58 if index < 3 else 0.46,
            INFO_TEXT_COLOR,
            2 if index < 3 else 1,
        )


def _build_active_events_by_frame(events_output: AnalyticsEventsOutput) -> Dict[int, List[AnalyticsEvent]]:
    active_events_by_frame: Dict[int, List[AnalyticsEvent]] = {}
    for event in events_output.events:
        for frame_idx in range(int(event.start_frame), int(event.end_frame) + 1):
            active_events_by_frame.setdefault(frame_idx, []).append(event)
    return active_events_by_frame


def render_analytics_possession_video_from_artifact(
    video_path: str,
    birdseye_output_path: str,
    analytics_possession_path: str,
    visualization_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    """Render possession review video from committed projection and possession artifacts."""
    birdseye_output = load_json(birdseye_output_path, BirdseyeProjectionOutput)
    possession_output = load_json(analytics_possession_path, AnalyticsPossessionOutput)

    frame_map = {frame.frame_idx: frame for frame in birdseye_output.frames}
    possession_map = {frame.frame_idx: frame for frame in possession_output.frames}

    # --- Cumulative stats tracking ---
    player_stats = {
        4: {"name": "Spyros", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
        7: {"name": "Rick", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
        10: {"name": "Kiki", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
    }
    jersey_to_pid = {4: "Spyros", 7: "Rick", 10: "Kiki"}

    def _draw_stats_table(overlay, stats):
        # Table position and style
        x0, y0 = 18, 18
        row_h = 28
        col_w = 110
        font_scale = 0.52
        thickness = 1
        header = ["Player", "Touches", "Passes", "Shots"]
        # Draw header
        for col, text in enumerate(header):
            _draw_text_with_bg(overlay, text, (x0 + col * col_w, y0), font_scale, (255,255,255), thickness)
        # Draw rows
        for idx, jersey in enumerate([4,7,10]):
            stat = stats[jersey]
            y = y0 + (idx+1) * row_h
            _draw_text_with_bg(overlay, f"{stat['name']} ({jersey})", (x0, y), font_scale, (255,255,255), thickness)
            _draw_text_with_bg(overlay, str(stat['touches']), (x0 + col_w, y), font_scale, (255,255,255), thickness)
            _draw_text_with_bg(overlay, f"{stat['passes_attempted']},{stat['passes_successful']}", (x0 + 2*col_w, y), font_scale, (255,255,255), thickness)
            _draw_text_with_bg(overlay, str(stat['shots']), (x0 + 3*col_w, y), font_scale, (255,255,255), thickness)

    reader = VideoReader(video_path)
    writer = None

    try:
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc_fn is None:
            fourcc_fn = cv2.VideoWriter.fourcc

        writer = cv2.VideoWriter(
            visualization_path,
            fourcc_fn(*"mp4v"),
            float(reader.fps),
            (int(reader.width), int(reader.height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open analytics possession video writer: {visualization_path}")

        artifact_start = max(
            int(birdseye_output.processed_start_frame),
            int(possession_output.processed_start_frame),
        )
        birdseye_end = birdseye_output.processed_end_frame_exclusive
        if birdseye_end is None:
            birdseye_end = birdseye_output.total_frames
        possession_end = possession_output.processed_end_frame_exclusive
        if possession_end is None:
            possession_end = possession_output.total_frames
        artifact_end_exclusive = min(int(birdseye_end), int(possession_end))

        render_start = max(start_frame, artifact_start)
        requested_end = artifact_end_exclusive if end_frame is None else end_frame
        render_end_exclusive = min(requested_end, artifact_end_exclusive)

        if render_end_exclusive <= render_start:
            raise ValueError(
                f"Invalid render window: start={render_start}, end={render_end_exclusive}. "
                f"Artifact range is [{artifact_start}, {artifact_end_exclusive})."
            )

        pad = 12
        max_inset_width = max(80, reader.width - (pad * 2))
        max_inset_height = max(60, reader.height - (pad * 2))
        inset_width = min(max_inset_width, max(200, int(reader.width * 0.30)))
        inset_height = int(inset_width * (birdseye_output.court_width_m / birdseye_output.court_length_m))
        inset_height = min(max_inset_height, max(60, inset_height))
        inset_width = min(max_inset_width, max(80, int(inset_height * (birdseye_output.court_length_m / birdseye_output.court_width_m))))

        total_render_frames = render_end_exclusive - render_start
        with tqdm(total=total_render_frames, desc="Analytics possession render", unit="frame") as pbar:
            for frame_idx, frame in reader.iter_frames():
                if frame_idx < render_start:
                    continue
                if frame_idx >= render_end_exclusive:
                    break

                frame_projection = frame_map.get(frame_idx)
                possession_frame = possession_map.get(frame_idx)
                if frame_projection is None or possession_frame is None:
                    continue

                overlay = frame.copy()
                players = _dedupe_players_for_render(frame_projection.players)
                for player in players:
                    _draw_player_overlay(overlay, player)

                owner_player = next(
                    (player for player in players if player.player_id == possession_frame.player_id),
                    None,
                )
                if owner_player is not None:
                    highlight_color = REPORTABLE_HIGHLIGHT if possession_frame.jersey_number is not None else INTERNAL_ONLY_HIGHLIGHT
                    _highlight_player_overlay(overlay, owner_player, highlight_color, label_prefix="CONTROL")

                _draw_ball_overlay(overlay, frame_projection)

                inset = _render_pitch_inset(frame_projection, birdseye_output)
                if owner_player is not None:
                    highlight_color = REPORTABLE_HIGHLIGHT if possession_frame.jersey_number is not None else INTERNAL_ONLY_HIGHLIGHT
                    _highlight_pitch_player(inset, owner_player, highlight_color)
                inset = cv2.resize(inset, (inset_width, inset_height), interpolation=cv2.INTER_AREA)

                x1 = overlay.shape[1] - inset_width - pad
                y1 = pad
                x2 = x1 + inset_width
                y2 = y1 + inset_height
                overlay[y1:y2, x1:x2] = inset
                cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)

                title = "ANALYTICS DEBUG"
                _draw_text_with_bg(overlay, title, (x1, max(20, y1 - 6)), 0.48, (255, 255, 255), 1)

                _draw_analytics_info_block(
                    overlay=overlay,
                    frame_idx=frame_idx,
                    possession_frame=possession_frame,
                    active_event=None,
                    events_output=AnalyticsEventsOutput(
                        video_name=possession_output.video_name,
                        fps=possession_output.fps,
                        total_frames=possession_output.total_frames,
                        processed_start_frame=possession_output.processed_start_frame,
                        processed_end_frame_exclusive=possession_output.processed_end_frame_exclusive,
                        events=[],
                        diagnostics={},
                    ),
                )

                writer.write(overlay)
                pbar.update(1)
    finally:
        if writer is not None:
            writer.release()
        reader.close()

    logger.info(f"Analytics debug video written: {visualization_path}")


def run_analytics_possession_visualization(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    video_path: Optional[str] = None,
) -> Path:
    """Render analytics possession debug video from an artifact directory."""
    if output_dir is None:
        output_dir = input_dir
    if video_path is None:
        raise ValueError("video_path is required for analytics possession visualization")

    birdseye_output_path = input_dir / const.BIRDSEYE_PROJECTION_JSON
    if not birdseye_output_path.exists():
        raise FileNotFoundError(f"Required artifact not found: {birdseye_output_path}")

    analytics_possession_path = output_dir / const.ANALYTICS_POSSESSION_JSON
    if not analytics_possession_path.exists():
        run_analytics_possession(input_dir=input_dir, output_dir=output_dir)

    analytics_events_path = output_dir / const.ANALYTICS_EVENTS_JSON
    if not analytics_events_path.exists():
        run_analytics_pass_detection(input_dir=input_dir, output_dir=output_dir)

    visualization_path = output_dir / const.ANALYTICS_DEBUG_VIDEO
    render_analytics_events_video_from_artifact(
        video_path=video_path,
        birdseye_output_path=str(birdseye_output_path),
        analytics_possession_path=str(analytics_possession_path),
        analytics_events_path=str(analytics_events_path),
        visualization_path=str(visualization_path),
    )
    return visualization_path


def render_analytics_events_video_from_artifact(
        # --- Cumulative stats tracking ---
        player_stats = {
            4: {"name": "Spyros", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
            7: {"name": "Rick", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
            10: {"name": "Kiki", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
        }
        jersey_to_pid = {4: "Spyros", 7: "Rick", 10: "Kiki"}

        def _draw_stats_table(overlay, stats):
            # Table position and style
            x0, y0 = 18, 18
            row_h = 28
            col_w = 110
            font_scale = 0.52
            thickness = 1
            header = ["Player", "Touches", "Passes", "Shots"]
            # Draw header
            for col, text in enumerate(header):
                _draw_text_with_bg(overlay, text, (x0 + col * col_w, y0), font_scale, (255,255,255), thickness)
            # Draw rows
            for idx, jersey in enumerate([4,7,10]):
                stat = stats[jersey]
                y = y0 + (idx+1) * row_h
                _draw_text_with_bg(overlay, f"{stat['name']} ({jersey})", (x0, y), font_scale, (255,255,255), thickness)
                _draw_text_with_bg(overlay, str(stat['touches']), (x0 + col_w, y), font_scale, (255,255,255), thickness)
                _draw_text_with_bg(overlay, f"{stat['passes_attempted']},{stat['passes_successful']}", (x0 + 2*col_w, y), font_scale, (255,255,255), thickness)
                _draw_text_with_bg(overlay, str(stat['shots']), (x0 + 3*col_w, y), font_scale, (255,255,255), thickness)
    video_path: str,
    birdseye_output_path: str,
    analytics_possession_path: str,
    analytics_events_path: str,
    visualization_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    """Render unified analytics debug video from projection, possession, and event artifacts."""
    birdseye_output = load_json(birdseye_output_path, BirdseyeProjectionOutput)
    possession_output = load_json(analytics_possession_path, AnalyticsPossessionOutput)
    events_output = load_json(analytics_events_path, AnalyticsEventsOutput)

    frame_map = {frame.frame_idx: frame for frame in birdseye_output.frames}
    possession_map = {frame.frame_idx: frame for frame in possession_output.frames}
    preview_frames = _pass_preview_frame_count(events_output.fps)
    active_events_by_frame = _build_display_events_by_frame(events_output, preview_frames)
    event_route_lookup = _build_event_route_lookup(frame_map, events_output)
    goal_banner_by_frame = _build_goal_banner_by_frame(events_output)

    # --- Cumulative stats tracking ---
    player_stats = {
        4: {"name": "Spyros", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
        7: {"name": "Rick", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
        10: {"name": "Kiki", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
    }
    jersey_to_pid = {4: "Spyros", 7: "Rick", 10: "Kiki"}

    def _draw_stats_table(overlay, stats):
        # Table position and style
        x0, y0 = 18, 18
        row_h = 28
        col_w = 110
        font_scale = 0.52
        thickness = 1
        header = ["Player", "Touches", "Passes", "Shots"]
        # Draw header
        for col, text in enumerate(header):
            _draw_text_with_bg(overlay, text, (x0 + col * col_w, y0), font_scale, (255,255,255), thickness)
        # Draw rows
        for idx, jersey in enumerate([4,7,10]):
            stat = stats[jersey]
            y = y0 + (idx+1) * row_h
            _draw_text_with_bg(overlay, f"{stat['name']} ({jersey})", (x0, y), font_scale, (255,255,255), thickness)
            _draw_text_with_bg(overlay, str(stat['touches']), (x0 + col_w, y), font_scale, (255,255,255), thickness)
            _draw_text_with_bg(overlay, f"{stat['passes_attempted']},{stat['passes_successful']}", (x0 + 2*col_w, y), font_scale, (255,255,255), thickness)
            _draw_text_with_bg(overlay, str(stat['shots']), (x0 + 3*col_w, y), font_scale, (255,255,255), thickness)

    reader = VideoReader(video_path)
    writer = None

    try:
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc_fn is None:
            fourcc_fn = cv2.VideoWriter.fourcc

        writer = cv2.VideoWriter(
            visualization_path,
            fourcc_fn(*"mp4v"),
            float(reader.fps),
            (int(reader.width), int(reader.height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open analytics events video writer: {visualization_path}")

        artifact_start = max(
            int(birdseye_output.processed_start_frame),
            int(possession_output.processed_start_frame),
            int(events_output.processed_start_frame),
        )
        birdseye_end = birdseye_output.processed_end_frame_exclusive or birdseye_output.total_frames
        possession_end = possession_output.processed_end_frame_exclusive or possession_output.total_frames
        events_end = events_output.processed_end_frame_exclusive or events_output.total_frames
        artifact_end_exclusive = min(int(birdseye_end), int(possession_end), int(events_end))

        render_start = max(start_frame, artifact_start)
        requested_end = artifact_end_exclusive if end_frame is None else end_frame
        render_end_exclusive = min(requested_end, artifact_end_exclusive)
        if render_end_exclusive <= render_start:
            raise ValueError(
                f"Invalid render window: start={render_start}, end={render_end_exclusive}. "
                f"Artifact range is [{artifact_start}, {artifact_end_exclusive})."
            )

        pad = 12
        max_inset_width = max(80, reader.width - (pad * 2))
        max_inset_height = max(60, reader.height - (pad * 2))
        inset_width = min(max_inset_width, max(200, int(reader.width * 0.30)))
        inset_height = int(inset_width * (birdseye_output.court_width_m / birdseye_output.court_length_m))
        inset_height = min(max_inset_height, max(60, inset_height))
        inset_width = min(max_inset_width, max(80, int(inset_height * (birdseye_output.court_length_m / birdseye_output.court_width_m))))

        total_render_frames = render_end_exclusive - render_start
        with tqdm(total=total_render_frames, desc="Analytics events render", unit="frame") as pbar:
            for frame_idx, frame in reader.iter_frames():
                if frame_idx < render_start:
                    continue
                if frame_idx >= render_end_exclusive:
                    break

                frame_projection = frame_map.get(frame_idx)
                possession_frame = possession_map.get(frame_idx)
                if frame_projection is None or possession_frame is None:
                    continue

                active_event = _active_event_for_frame(active_events_by_frame, frame_idx)
                overlay = frame.copy()
                players = _dedupe_players_for_render(frame_projection.players)
                for player in players:
                    _draw_player_overlay(overlay, player)

                inset = _render_pitch_inset(frame_projection, birdseye_output)
                owner_player = next(
                    (player for player in players if player.player_id == possession_frame.player_id),
                    None,
                )

                if active_event is not None:
                    event_color, image_route_thickness, pitch_route_thickness = _event_route_style(active_event)
                    if active_event.event_type in {"pass", "shot"}:
                        route = event_route_lookup.get(active_event.event_id, {})
                        _draw_route_polyline(
                            overlay,
                            route.get("image_points", []),
                            event_color,
                            image_route_thickness,
                        )
                        _draw_route_polyline(
                            inset,
                            route.get("pitch_points", []),
                            event_color,
                            pitch_route_thickness,

                        # --- Cumulative stats tracking ---
                        player_stats = {
                            4: {"name": "Spyros", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
                            7: {"name": "Rick", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
                            10: {"name": "Kiki", "touches": 0, "passes_attempted": 0, "passes_successful": 0, "shots": 0},
                        }
                        jersey_to_pid = {4: "Spyros", 7: "Rick", 10: "Kiki"}

                        def _draw_stats_table(overlay, stats):
                            # Table position and style
                            x0, y0 = 18, 18
                            row_h = 28
                            col_w = 110
                            font_scale = 0.52
                            thickness = 1
                            header = ["Player", "Touches", "Passes", "Shots"]
                            # Draw header
                            for col, text in enumerate(header):
                                _draw_text_with_bg(overlay, text, (x0 + col * col_w, y0), font_scale, (255,255,255), thickness)
                            # Draw rows
                            for idx, jersey in enumerate([4,7,10]):
                                stat = stats[jersey]
                                y = y0 + (idx+1) * row_h
                                _draw_text_with_bg(overlay, f"{stat['name']} ({jersey})", (x0, y), font_scale, (255,255,255), thickness)
                                _draw_text_with_bg(overlay, str(stat['touches']), (x0 + col_w, y), font_scale, (255,255,255), thickness)
                                _draw_text_with_bg(overlay, f"{stat['passes_attempted']},{stat['passes_successful']}", (x0 + 2*col_w, y), font_scale, (255,255,255), thickness)
                                _draw_text_with_bg(overlay, str(stat['shots']), (x0 + 3*col_w, y), font_scale, (255,255,255), thickness)
                        )
                    for player in players:
                        if player.player_id in {active_event.passer_player_id, active_event.receiver_player_id}:
                            _highlight_player_overlay(overlay, player, event_color)
                            _highlight_pitch_player(inset, player, event_color)

                if owner_player is not None:
                    highlight_color = REPORTABLE_HIGHLIGHT if possession_frame.jersey_number is not None else INTERNAL_ONLY_HIGHLIGHT
                    _highlight_player_overlay(overlay, owner_player, highlight_color, label_prefix="CONTROL")
                    _highlight_pitch_player(inset, owner_player, highlight_color)

                _draw_ball_overlay(overlay, frame_projection)

                inset = cv2.resize(inset, (inset_width, inset_height), interpolation=cv2.INTER_AREA)
                x1 = overlay.shape[1] - inset_width - pad
                y1 = pad
                x2 = x1 + inset_width
                y2 = y1 + inset_height
                overlay[y1:y2, x1:x2] = inset
                cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)
                _draw_text_with_bg(overlay, "ANALYTICS DEBUG", (x1, max(20, y1 - 6)), 0.48, (255, 255, 255), 1)
                if frame_idx in goal_banner_by_frame:
                    _draw_goal_banner(overlay)
                _draw_analytics_info_block(
                    overlay=overlay,
                    frame_idx=frame_idx,
                    possession_frame=possession_frame,
                    active_event=active_event,
                    events_output=events_output,
                )

                writer.write(overlay)
                pbar.update(1)
    finally:
        if writer is not None:
            writer.release()
        reader.close()

    logger.info(f"Analytics debug video written: {visualization_path}")


def run_analytics_events_visualization(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    video_path: Optional[str] = None,
) -> Path:
    """Render unified analytics debug video from an artifact directory."""
    if output_dir is None:
        output_dir = input_dir
    if video_path is None:
        raise ValueError("video_path is required for analytics events visualization")

    birdseye_output_path = input_dir / const.BIRDSEYE_PROJECTION_JSON
    if not birdseye_output_path.exists():
        raise FileNotFoundError(f"Required artifact not found: {birdseye_output_path}")

    analytics_possession_path = output_dir / const.ANALYTICS_POSSESSION_JSON
    if not analytics_possession_path.exists():
        run_analytics_possession(input_dir=input_dir, output_dir=output_dir)

    analytics_events_path = output_dir / const.ANALYTICS_EVENTS_JSON
    if not analytics_events_path.exists():
        run_analytics_pass_detection(input_dir=input_dir, output_dir=output_dir)

    visualization_path = output_dir / const.ANALYTICS_DEBUG_VIDEO
    render_analytics_events_video_from_artifact(
        video_path=video_path,
        birdseye_output_path=str(birdseye_output_path),
        analytics_possession_path=str(analytics_possession_path),
        analytics_events_path=str(analytics_events_path),
        visualization_path=str(visualization_path),
    )
    return visualization_path