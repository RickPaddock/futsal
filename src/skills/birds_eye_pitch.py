"""
Bird's-eye pitch projection stage.

Projects committed player identities and ball states into court coordinates and
optionally renders a top-right inset debug video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
from tqdm import tqdm

from utils.homography import CourtHomography, create_homography_from_config
from utils.pitch_drawing import create_court_view

from ..core import constants as const
from ..core.data_models import (
    BallInterpolationOutput,
    BallPosition,
    BirdseyeBallFrame,
    BirdseyeFrame,
    BirdseyePlayerPosition,
    BirdseyeProjectionOutput,
    CommittedIdentity,
    Detection,
    Pass1Output,
    Pass2COutput,
    Pass3COutput,
    ScoredFragment,
)
from ..core.schemas import BIRDSEYE_PROJECTION_OUTPUT_SCHEMA, VALIDATION_RESULT_SCHEMA
from ..core.types import BallState, TeamID
from ..utils.file_utils import load_json, save_json
from ..utils.logging_utils import get_logger
from ..utils.video_io import VideoReader
from ..validation.validator import Validator
from .pass3_debug_visualizer import _ghost_bbox_for_frame

logger = get_logger("birds_eye_pitch")


TEAM_COLORS: Dict[TeamID, Tuple[int, int, int]] = {
    TeamID.TEAM_A: (255, 100, 60),
    TeamID.TEAM_B: (60, 80, 255),
}
BALL_REAL_COLOR = (0, 215, 255)
BALL_INTERP_COLOR = (255, 255, 0)


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


def _processed_frame_range(pass1_output: Pass1Output) -> Tuple[int, int]:
    start = pass1_output.processed_start_frame
    end_exclusive = pass1_output.processed_end_frame_exclusive
    if end_exclusive is None:
        end_exclusive = pass1_output.total_frames
    return start, end_exclusive


def _load_calibration_config(calibration_path: Path) -> Dict[str, Any]:
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Pitch calibration not found: {calibration_path}. "
            "Create config/pitch_calibration.json from the click-point workflow."
        )
    calibration = load_json(str(calibration_path))
    if not isinstance(calibration, dict):
        raise ValueError(f"Pitch calibration must be a JSON object: {calibration_path}")
    return calibration


def _bbox_anchor_point(bbox: List[float], homography: CourtHomography) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    center_x = (x1 + x2) / 2.0
    foot_y = min(y2, y1 + float(getattr(homography, "head_to_foot_offset", 150.0)))
    return [center_x, foot_y]


def _project_image_point(
    homography: CourtHomography,
    image_point: List[float],
) -> Tuple[List[float], List[float]]:
    court_x, court_y = homography.pixel_to_court(image_point[0], image_point[1])
    render_x, render_y = homography.court_to_pixel_2d(court_x, court_y)
    render_y = max(0, min(int(homography.output_h - 1 - render_y), homography.output_h - 1))
    return [float(court_x), float(court_y)], [float(render_x), float(render_y)]


def _project_ball_frame(
    ball_position: BallPosition,
    homography: CourtHomography,
) -> BirdseyeBallFrame:
    state = ball_position.state
    if isinstance(state, BallState):
        state_enum = state
    else:
        state_enum = BallState(str(state))

    if ball_position.centroid is None or state_enum == BallState.UNKNOWN:
        return BirdseyeBallFrame(
            frame_idx=ball_position.frame_idx,
            state=state_enum,
            confidence=ball_position.confidence,
            image_position=ball_position.centroid,
            court_position=None,
            render_position=None,
        )

    court_position, render_position = _project_image_point(homography, list(ball_position.centroid))
    return BirdseyeBallFrame(
        frame_idx=ball_position.frame_idx,
        state=state_enum,
        confidence=ball_position.confidence,
        image_position=list(ball_position.centroid),
        court_position=court_position,
        render_position=render_position,
    )


def build_birds_eye_projection_output(
    pass1_output: Pass1Output,
    pass2c_output: Pass2COutput,
    pass3_output: Pass3COutput,
    ball_output: BallInterpolationOutput,
    calibration_config: Dict[str, Any],
    calibration_path: Optional[Path] = None,
) -> BirdseyeProjectionOutput:
    """Build the bird's-eye artifact from existing pass outputs."""
    homography = create_homography_from_config(calibration_config)
    if homography is None:
        raise ValueError("Homography config is missing source_points/dest_points")

    start_frame, end_frame_exclusive = _processed_frame_range(pass1_output)
    detections_by_id: Dict[str, Detection] = {
        detection.detection_id: detection for detection in pass1_output.detections
    }
    identity_by_fragment: Dict[str, CommittedIdentity] = {
        identity.fragment_id: identity for identity in pass3_output.identities
    }
    fragment_by_id: Dict[str, ScoredFragment] = {
        fragment.fragment_id: fragment for fragment in pass2c_output.fragments
    }
    ghost_activity_windows = pass3_output.solver_log.get("ghost_active_windows") if isinstance(pass3_output.solver_log, dict) else None
    if not isinstance(ghost_activity_windows, dict):
        ghost_activity_windows = {}

    frame_players: Dict[int, List[BirdseyePlayerPosition]] = {}
    projected_player_count = 0
    estimated_player_count = 0

    for fragment in pass2c_output.fragments:
        identity = identity_by_fragment.get(fragment.fragment_id)
        if identity is None:
            continue

        is_ghost = bool(getattr(fragment, "is_ghost", False))
        if is_ghost:
            ghost_window = ghost_activity_windows.get(fragment.fragment_id)
            if not isinstance(ghost_window, dict):
                continue
            ghost_start = max(start_frame, int(ghost_window.get("start_frame", fragment.start_frame)))
            ghost_end = min(end_frame_exclusive - 1, int(ghost_window.get("end_frame", fragment.end_frame)))
            for frame_idx in range(ghost_start, ghost_end + 1):
                bbox = _ghost_bbox_for_frame(
                    fragment,
                    frame_idx,
                    ghost_window,
                    fragment_by_id,
                    detections_by_id,
                )
                if bbox is None:
                    continue
                image_anchor = _bbox_anchor_point(cast(List[float], bbox), homography)
                court_position, render_position = _project_image_point(homography, image_anchor)
                frame_players.setdefault(frame_idx, []).append(
                    BirdseyePlayerPosition(
                        frame_idx=frame_idx,
                        fragment_id=fragment.fragment_id,
                        player_id=identity.player_id,
                        team=identity.team,
                        jersey_number=identity.jersey_number,
                        track_id=fragment.original_track_id,
                        is_ghost=True,
                        is_estimated=True,
                        image_bbox=list(bbox),
                        image_anchor=image_anchor,
                        court_position=court_position,
                        render_position=render_position,
                    )
                )
                projected_player_count += 1
                estimated_player_count += 1
            continue

        for detection_id in fragment.detection_ids:
            detection = detections_by_id.get(detection_id)
            if detection is None:
                continue
            if detection.frame_idx < start_frame or detection.frame_idx >= end_frame_exclusive:
                continue
            image_anchor = _bbox_anchor_point(detection.bbox, homography)
            court_position, render_position = _project_image_point(homography, image_anchor)
            frame_players.setdefault(detection.frame_idx, []).append(
                BirdseyePlayerPosition(
                    frame_idx=detection.frame_idx,
                    fragment_id=fragment.fragment_id,
                    player_id=identity.player_id,
                    team=identity.team,
                    jersey_number=identity.jersey_number,
                    track_id=fragment.original_track_id,
                    is_ghost=False,
                    is_estimated=False,
                    image_bbox=list(detection.bbox),
                    image_anchor=image_anchor,
                    court_position=court_position,
                    render_position=render_position,
                )
            )
            projected_player_count += 1

    ball_by_frame: Dict[int, BallPosition] = {
        ball_position.frame_idx: ball_position for ball_position in ball_output.ball_positions
    }
    frames: List[BirdseyeFrame] = []

    for frame_idx in range(start_frame, end_frame_exclusive):
        players = sorted(
            frame_players.get(frame_idx, []),
            key=lambda player: (player.team.value, player.player_id, player.fragment_id),
        )
        source_ball = ball_by_frame.get(frame_idx)
        if source_ball is None:
            source_ball = BallPosition(
                frame_idx=frame_idx,
                state=BallState.UNKNOWN,
                centroid=None,
                bbox=None,
                confidence=0.0,
            )
        projected_ball = _project_ball_frame(source_ball, homography)
        frames.append(BirdseyeFrame(frame_idx=frame_idx, players=players, ball=projected_ball))

    homography_config = calibration_config.get("homography", {}) if isinstance(calibration_config, dict) else {}
    return BirdseyeProjectionOutput(
        video_name=pass1_output.video_name,
        fps=pass1_output.fps,
        total_frames=pass1_output.total_frames,
        processed_start_frame=start_frame,
        processed_end_frame_exclusive=end_frame_exclusive,
        court_length_m=float(homography_config.get("court_length", homography.pitch_width_m)),
        court_width_m=float(homography_config.get("court_width", homography.pitch_height_m)),
        output_pixel_scale=int(homography_config.get("output_pixel_scale", homography.output_pixel_scale)),
        frames=frames,
        diagnostics={
            "calibration_path": str(calibration_path) if calibration_path is not None else None,
            "calibration_point_count": len(homography_config.get("source_points", [])),
            "projected_player_positions": projected_player_count,
            "estimated_player_positions": estimated_player_count,
            "ghost_windows_seen": len(ghost_activity_windows),
        },
    )


def _draw_player_on_pitch(canvas: np.ndarray, player: BirdseyePlayerPosition) -> None:
    color = TEAM_COLORS.get(player.team, (200, 200, 200))
    center = tuple(int(round(v)) for v in player.render_position)
    radius = 10 if player.jersey_number is not None else 9

    if player.is_estimated:
        cv2.circle(canvas, center, radius, color, 2)
    else:
        cv2.circle(canvas, center, radius, color, -1)
        cv2.circle(canvas, center, radius, (255, 255, 255), 1)

    if player.jersey_number is not None:
        text = str(player.jersey_number)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        text_org = (center[0] - text_w // 2, center[1] + text_h // 2)
        cv2.putText(canvas, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_ball_on_pitch(canvas: np.ndarray, ball: BirdseyeBallFrame) -> None:
    if ball.render_position is None:
        return
    center = tuple(int(round(v)) for v in ball.render_position)
    state = ball.state.value if isinstance(ball.state, BallState) else str(ball.state)
    if state == BallState.REAL.value:
        cv2.circle(canvas, center, 5, BALL_REAL_COLOR, -1)
        cv2.circle(canvas, center, 5, (0, 0, 0), 1)
    elif state == BallState.INTERPOLATED.value:
        cv2.circle(canvas, center, 6, BALL_INTERP_COLOR, 2)
        cv2.line(canvas, (center[0] - 4, center[1]), (center[0] + 4, center[1]), BALL_INTERP_COLOR, 2)
        cv2.line(canvas, (center[0], center[1] - 4), (center[0], center[1] + 4), BALL_INTERP_COLOR, 2)


def _render_pitch_inset(frame_projection: BirdseyeFrame, output: BirdseyeProjectionOutput) -> np.ndarray:
    width = int(output.court_length_m * output.output_pixel_scale)
    height = int(output.court_width_m * output.output_pixel_scale)
    pitch = create_court_view(
        width=width,
        height=height,
        court_length=output.court_length_m,
        court_width=output.court_width_m,
    )
    for player in frame_projection.players:
        _draw_player_on_pitch(pitch, player)
    _draw_ball_on_pitch(pitch, frame_projection.ball)
    return pitch


def render_birds_eye_debug_video_from_artifact(
    video_path: str,
    birdseye_output_path: str,
    debug_video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    """Render bird's-eye inset debug video from birdseye_projection.json."""
    birdseye_output = load_json(birdseye_output_path, BirdseyeProjectionOutput)
    frame_map = {frame.frame_idx: frame for frame in birdseye_output.frames}

    reader = VideoReader(video_path)
    writer = None

    try:
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc_fn is None:
            fourcc_fn = cv2.VideoWriter.fourcc

        writer = cv2.VideoWriter(
            debug_video_path,
            fourcc_fn(*"mp4v"),
            float(reader.fps),
            (int(reader.width), int(reader.height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open birdseye debug video writer: {debug_video_path}")

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

        inset_width = max(280, int(reader.width * 0.30))
        inset_height = max(140, int(inset_width * (birdseye_output.court_width_m / birdseye_output.court_length_m)))

        total_render_frames = render_end_exclusive - render_start
        with tqdm(total=total_render_frames, desc="Birdseye debug render", unit="frame") as pbar:
            for frame_idx, frame in reader.iter_frames():
                if frame_idx < render_start:
                    continue
                if frame_idx >= render_end_exclusive:
                    break

                projection = frame_map.get(frame_idx)
                if projection is None:
                    continue

                overlay = frame.copy()
                inset = _render_pitch_inset(projection, birdseye_output)
                inset = cv2.resize(inset, (inset_width, inset_height), interpolation=cv2.INTER_AREA)

                pad = 12
                x1 = overlay.shape[1] - inset_width - pad
                y1 = pad
                x2 = x1 + inset_width
                y2 = y1 + inset_height
                overlay[y1:y2, x1:x2] = inset
                cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)

                _draw_text_with_bg(overlay, "BIRD'S-EYE INSET", (x1, max(20, y1 - 6)), 0.48, (255, 255, 255), 1)
                player_count = len(projection.players)
                estimated_count = sum(1 for player in projection.players if player.is_estimated)
                ball_state = projection.ball.state.value if isinstance(projection.ball.state, BallState) else str(projection.ball.state)
                _draw_text_with_bg(
                    overlay,
                    f"frame={frame_idx} players={player_count} estimated={estimated_count} ball={ball_state}",
                    (12, 24),
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

    logger.info(f"Birdseye debug video written: {debug_video_path}")


class BirdsEyePitchProjector:
    """Pass runner for the bird's-eye projection stage."""

    def run(
        self,
        pass1_path: Path,
        pass2c_path: Path,
        pass3_path: Path,
        ball_path: Path,
        output_path: Path,
        validation_output_path: Path,
        calibration_path: Path,
        video_path: Optional[str] = None,
        debug_video_path: Optional[str] = None,
    ) -> BirdseyeProjectionOutput:
        pass1_output = load_json(pass1_path, Pass1Output)
        pass2c_output = load_json(pass2c_path, Pass2COutput)
        pass3_output = load_json(pass3_path, Pass3COutput)
        ball_output = load_json(ball_path, BallInterpolationOutput)
        calibration_config = _load_calibration_config(calibration_path)

        birdseye_output = build_birds_eye_projection_output(
            pass1_output=pass1_output,
            pass2c_output=pass2c_output,
            pass3_output=pass3_output,
            ball_output=ball_output,
            calibration_config=calibration_config,
            calibration_path=calibration_path,
        )

        validator = Validator()
        validation_result = validator.validate_birdseye(birdseye_output)
        if not validation_result.passed:
            messages = "; ".join(violation.message for violation in validation_result.violations[:5])
            raise ValueError(f"Birdseye projection validation failed: {messages}")

        save_json(birdseye_output.model_dump(), str(output_path), BIRDSEYE_PROJECTION_OUTPUT_SCHEMA)
        save_json(validation_result.model_dump(), str(validation_output_path), VALIDATION_RESULT_SCHEMA)

        if debug_video_path is not None:
            if video_path is None:
                raise ValueError("video_path is required when rendering the birdseye debug video")
            render_birds_eye_debug_video_from_artifact(
                video_path=video_path,
                birdseye_output_path=str(output_path),
                debug_video_path=str(debug_video_path),
                start_frame=pass1_output.processed_start_frame,
                end_frame=pass1_output.processed_end_frame_exclusive,
            )

        return birdseye_output


def run_birds_eye_pitch(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    video_path: Optional[str] = None,
    debug_video_path: Optional[str] = None,
    calibration_path: Optional[Path] = None,
) -> BirdseyeProjectionOutput:
    """Execute the bird's-eye projection stage from an artifact directory."""
    if output_dir is None:
        output_dir = input_dir
    if calibration_path is None:
        calibration_path = const.PITCH_CALIBRATION_JSON

    pass1_path = input_dir / const.PASS1_RAW_JSON
    pass2c_path = input_dir / const.PASS2_GHOSTS_JSON
    pass3_path = input_dir / const.PASS3_IDENTITY_COMMIT_JSON
    ball_path = input_dir / const.BALL_INTERPOLATION_JSON
    output_path = output_dir / const.BIRDSEYE_PROJECTION_JSON
    validation_output_path = output_dir / const.BIRDSEYE_VALIDATION_JSON

    for required_path in (pass1_path, pass2c_path, pass3_path, ball_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required artifact not found: {required_path}")

    projector = BirdsEyePitchProjector()
    return projector.run(
        pass1_path=pass1_path,
        pass2c_path=pass2c_path,
        pass3_path=pass3_path,
        ball_path=ball_path,
        output_path=output_path,
        validation_output_path=validation_output_path,
        calibration_path=Path(calibration_path),
        video_path=video_path,
        debug_video_path=debug_video_path,
    )