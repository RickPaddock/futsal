"""Final visualization renderer built from committed projection artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from utils.pitch_drawing import create_court_view

from ..core import constants as const
from ..core.data_models import BirdseyeFrame, BirdseyePlayerPosition, BirdseyeProjectionOutput
from ..core.types import BallState, TeamID
from ..utils.file_utils import load_json
from ..utils.logging_utils import get_logger
from ..utils.video_io import VideoReader

logger = get_logger("visualizer")


TEAM_COLORS: Dict[TeamID, Tuple[int, int, int]] = {
    TeamID.TEAM_A: (0, 0, 0),
    TeamID.TEAM_B: (0, 140, 255),
}
BALL_REAL_COLOR = (0, 215, 255)
BALL_INTERP_COLOR = (255, 255, 0)
JERSEY_NAME_MAP = {
    4: "Spyros",
    7: "Rick",
    10: "Kiki",
}


def _identity_label(player_id: str, jersey_number: Optional[int]) -> str:
    if jersey_number is not None:
        mapped_name = JERSEY_NAME_MAP.get(int(jersey_number))
        if mapped_name is not None:
            return f"{jersey_number} - {mapped_name}"
        return str(jersey_number)
    return player_id


def _draw_text_with_bg(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad_x = 4
    pad_y = 3
    x1 = max(0, x - pad_x)
    y1 = max(0, y - text_h - pad_y)
    x2 = min(img.shape[1] - 1, x + text_w + pad_x)
    y2 = min(img.shape[0] - 1, y + baseline + pad_y)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_dashed_line(
    img: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash_length: int = 8,
) -> None:
    distance = int(np.hypot(end[0] - start[0], end[1] - start[1]))
    if distance <= 0:
        return

    for offset in range(0, distance, dash_length * 2):
        start_ratio = offset / distance
        end_ratio = min(1.0, (offset + dash_length) / distance)
        x1 = int(start[0] + (end[0] - start[0]) * start_ratio)
        y1 = int(start[1] + (end[1] - start[1]) * start_ratio)
        x2 = int(start[0] + (end[0] - start[0]) * end_ratio)
        y2 = int(start[1] + (end[1] - start[1]) * end_ratio)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def _draw_dashed_rectangle(
    img: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    _draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness)
    _draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness)
    _draw_dashed_line(img, (x2, y2), (x1, y2), color, thickness)
    _draw_dashed_line(img, (x1, y2), (x1, y1), color, thickness)


def _draw_dashed_circle(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    for start_angle in range(0, 360, 36):
        cv2.ellipse(img, center, (radius, radius), 0, start_angle, start_angle + 18, color, thickness)


def _player_label(player: BirdseyePlayerPosition) -> str:
    return _identity_label(player.player_id, player.jersey_number)


def _dedupe_players_for_render(players: List[BirdseyePlayerPosition]) -> List[BirdseyePlayerPosition]:
    deduped: List[BirdseyePlayerPosition] = []
    seen_track_ids = set()
    for player in players:
        track_id = int(player.track_id)
        if track_id in seen_track_ids:
            continue
        seen_track_ids.add(track_id)
        deduped.append(player)
    return deduped


def _draw_player_overlay(canvas: np.ndarray, player: BirdseyePlayerPosition) -> None:
    color = TEAM_COLORS.get(player.team, (200, 200, 200))
    x1, y1, x2, y2 = [int(round(v)) for v in player.image_bbox]

    if player.is_ghost:
        _draw_dashed_rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    else:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    if player.raw_image_anchor is not None:
        raw_x, raw_y = [int(round(v)) for v in player.raw_image_anchor]
        cv2.circle(canvas, (raw_x, raw_y), 4, (255, 255, 255), 1)
        anchor_x, anchor_y = [int(round(v)) for v in player.image_anchor]
        if (raw_x, raw_y) != (anchor_x, anchor_y):
            cv2.line(canvas, (raw_x, raw_y), (anchor_x, anchor_y), (255, 255, 255), 1)

    anchor_x, anchor_y = [int(round(v)) for v in player.image_anchor]
    cv2.circle(canvas, (anchor_x, anchor_y), 5, (255, 255, 255), -1)
    cv2.circle(canvas, (anchor_x, anchor_y), 3, color, -1)

    label = _player_label(player)
    label_y = y1 - 8 if y1 > 18 else y1 + 16
    _draw_text_with_bg(canvas, label, (x1, label_y), 0.45, (255, 255, 255), 1)


def _draw_ball_overlay(canvas: np.ndarray, frame_projection: BirdseyeFrame) -> str:
    ball = frame_projection.ball
    state = ball.state.value if isinstance(ball.state, BallState) else str(ball.state)
    if state == BallState.UNKNOWN.value or ball.image_position is None:
        return "ball=unknown"

    center = tuple(int(round(v)) for v in ball.image_position)
    if state == BallState.REAL.value:
        cv2.circle(canvas, center, 6, BALL_REAL_COLOR, -1)
        cv2.circle(canvas, center, 6, (0, 0, 0), 1)
        return f"ball=real conf={ball.confidence:.2f}"

    _draw_dashed_circle(canvas, center, 8, BALL_INTERP_COLOR, 2)
    return "ball=interpolated"


def _draw_player_on_pitch(canvas: np.ndarray, player: BirdseyePlayerPosition) -> None:
    color = TEAM_COLORS.get(player.team, (200, 200, 200))
    center = tuple(int(round(v)) for v in player.render_position)
    radius = 10 if player.jersey_number is not None else 9

    if player.is_ghost or player.is_estimated:
        cv2.circle(canvas, center, radius, color, 2)
    else:
        cv2.circle(canvas, center, radius, color, -1)
        cv2.circle(canvas, center, radius, (255, 255, 255), 1)

    if player.jersey_number is not None:
        text = str(player.jersey_number)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        text_org = (center[0] - text_w // 2, center[1] + text_h // 2)
        cv2.putText(canvas, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_ball_on_pitch(frame_projection: BirdseyeFrame, canvas: np.ndarray) -> None:
    ball = frame_projection.ball
    if ball.render_position is None:
        return

    center = tuple(int(round(v)) for v in ball.render_position)
    state = ball.state.value if isinstance(ball.state, BallState) else str(ball.state)
    if state == BallState.REAL.value:
        cv2.circle(canvas, center, 5, BALL_REAL_COLOR, -1)
        cv2.circle(canvas, center, 5, (0, 0, 0), 1)
    elif state == BallState.INTERPOLATED.value:
        _draw_dashed_circle(canvas, center, 6, BALL_INTERP_COLOR, 2)


def _render_pitch_inset(frame_projection: BirdseyeFrame, output: BirdseyeProjectionOutput) -> np.ndarray:
    width = int(output.court_length_m * output.output_pixel_scale)
    height = int(output.court_width_m * output.output_pixel_scale)
    pitch = create_court_view(
        width=width,
        height=height,
        court_length=output.court_length_m,
        court_width=output.court_width_m,
    )
    for player in _dedupe_players_for_render(frame_projection.players):
        _draw_player_on_pitch(pitch, player)
    _draw_ball_on_pitch(frame_projection, pitch)
    return pitch


def render_visualization_from_artifact(
    video_path: str,
    birdseye_output_path: str,
    visualization_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    """Render the final visualization video from birdseye_projection.json."""
    birdseye_output = load_json(birdseye_output_path, BirdseyeProjectionOutput)
    frame_map = {frame.frame_idx: frame for frame in birdseye_output.frames}

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
            raise RuntimeError(f"Failed to open visualization video writer: {visualization_path}")

        artifact_start = birdseye_output.processed_start_frame
        artifact_end_exclusive = birdseye_output.processed_end_frame_exclusive
        if artifact_end_exclusive is None:
            artifact_end_exclusive = birdseye_output.total_frames

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
        with tqdm(total=total_render_frames, desc="Visualization render", unit="frame") as pbar:
            for frame_idx, frame in reader.iter_frames():
                if frame_idx < render_start:
                    continue
                if frame_idx >= render_end_exclusive:
                    break

                frame_projection = frame_map.get(frame_idx)
                if frame_projection is None:
                    continue

                overlay = frame.copy()
                players = _dedupe_players_for_render(frame_projection.players)
                for player in players:
                    _draw_player_overlay(overlay, player)
                ball_label = _draw_ball_overlay(overlay, frame_projection)

                inset = _render_pitch_inset(frame_projection, birdseye_output)
                inset = cv2.resize(inset, (inset_width, inset_height), interpolation=cv2.INTER_AREA)

                x1 = overlay.shape[1] - inset_width - pad
                y1 = pad
                x2 = x1 + inset_width
                y2 = y1 + inset_height
                overlay[y1:y2, x1:x2] = inset
                cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)

                _draw_text_with_bg(overlay, "VISUALIZATION", (x1, max(20, y1 - 6)), 0.48, (255, 255, 255), 1)
                ghost_count = sum(1 for player in players if player.is_ghost)
                _draw_text_with_bg(
                    overlay,
                    f"frame={frame_idx} players={len(players)} ghosts={ghost_count} {ball_label}",
                    (12, 52),
                    0.52,
                    (255, 255, 255),
                    2,
                )

                writer.write(overlay)
                pbar.update(1)
    finally:
        if writer is not None:
            writer.release()
        reader.close()

    logger.info(f"Visualization video written: {visualization_path}")


def run_visualization(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    video_path: Optional[str] = None,
) -> Path:
    """Render visualization.mp4 from an artifact directory."""
    if output_dir is None:
        output_dir = input_dir
    if video_path is None:
        raise ValueError("video_path is required for visualization rendering")

    birdseye_output_path = input_dir / const.BIRDSEYE_PROJECTION_JSON
    if not birdseye_output_path.exists():
        raise FileNotFoundError(f"Required artifact not found: {birdseye_output_path}")

    visualization_path = output_dir / const.VISUALIZATION_VIDEO
    render_visualization_from_artifact(
        video_path=video_path,
        birdseye_output_path=str(birdseye_output_path),
        visualization_path=str(visualization_path),
    )
    return visualization_path