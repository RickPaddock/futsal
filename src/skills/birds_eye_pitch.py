"""
Bird's-eye pitch projection stage.

Projects committed player identities and ball states into court coordinates and
optionally renders a top-right inset debug video.
"""

from __future__ import annotations

from collections import deque
from math import hypot
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
from ..utils.position_stabilizer import PositionStabilizer, foot_point_from_bbox
from ..utils.video_io import VideoReader
from ..validation.validator import Validator
from .pass3_debug_visualizer import _ghost_bbox_for_frame

logger = get_logger("birds_eye_pitch")


TEAM_COLORS: Dict[TeamID, Tuple[int, int, int]] = {
    TeamID.TEAM_A: (0, 0, 0),
    TeamID.TEAM_B: (0, 140, 255),
}
BALL_REAL_COLOR = (0, 215, 255)
BALL_INTERP_COLOR = (255, 255, 0)
PLAYER_MAX_SPEED_MPS = 8.0
PLAYER_BASE_STABILIZATION_ALPHA = 0.14
PLAYER_VARIABLE_STABILIZATION_ALPHA = 0.48
PLAYER_MAX_MEASUREMENT_OFFSET_M = 0.45
PLAYER_DIRECTION_HISTORY = 4
PLAYER_DIRECTION_MIN_STEP_M = 0.03
PLAYER_JOURNEY_MIN_DISTANCE_M = 5.0
PLAYER_JOURNEY_MIN_STEP_M = 0.08
PLAYER_JOURNEY_MIN_FRAMES = 3
PLAYER_JOURNEY_MAX_DIRECTION_CHANGE_DEG = 55.0
PLAYER_JOURNEY_MAX_HEADING_DEVIATION_DEG = 42.0
PLAYER_JOURNEY_MIN_STRAIGHTNESS_RATIO = 0.78
PLAYER_JOURNEY_MIN_LOCAL_ALIGNMENT = 0.5
PLAYER_JOURNEY_MAX_DRIFT_M = 0.9
GROUND_CONTACT_WINDOW = 5
PLAYER_HEIGHT_PRIOR_BINS = 16
PLAYER_HEIGHT_PRIOR_QUANTILE = 85.0
PLAYER_HEIGHT_RESTORE_BLEND = 0.72
PLAYER_TRAJECTORY_MIN_FRAMES = 6
PLAYER_TRAJECTORY_MIN_TRUST = 0.45
PLAYER_TRAJECTORY_MIN_STEP_M = 0.06
PLAYER_TRAJECTORY_STATIONARY_STEP_M = 0.04
PLAYER_TRAJECTORY_STATIONARY_MIN_FRAMES = 4
PLAYER_TRAJECTORY_STATIONARY_MAX_DRIFT_M = 0.12
PLAYER_TRAJECTORY_MAX_DIRECTION_CHANGE_DEG = 35.0
PLAYER_TRAJECTORY_COHERENT_MAX_DIRECTION_DEG = 24.0
PLAYER_TRAJECTORY_REACTIVE_DIRECTION_DEG = 70.0
PLAYER_TRAJECTORY_MAX_TRUST_CLIFF = 0.35
PLAYER_TRAJECTORY_MAX_FIT_MEAN_ERROR_M = 0.14
PLAYER_TRAJECTORY_MAX_FIT_PEAK_ERROR_M = 0.28
PLAYER_TRAJECTORY_BLEND_MAX_ALPHA = 0.28
PLAYER_TRAJECTORY_MAX_DRIFT_M = 0.20
POSE_MIN_BBOX_HEIGHT = 90.0
POSE_MIN_KEYPOINT_CONFIDENCE = 0.2
POSE_UPPER_BODY_KEYPOINT_CONFIDENCE = 0.25
POSE_TOP_PADDING_RATIO = 0.05
POSE_SHOULDER_TO_FOOT_RATIO = 0.82
POSE_HIP_TO_FOOT_RATIO = 0.48
POSE_MAX_ANCHOR_SHIFT_RATIO = 0.16
POSE_MODEL_NAME = "yolo11n-pose.pt"
_POSE_MODEL = None


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
    return [center_x, y2]


def _project_image_point(
    homography: CourtHomography,
    image_point: List[float],
) -> Tuple[List[float], List[float]]:
    court_x, court_y = homography.pixel_to_court(image_point[0], image_point[1])
    return [float(court_x), float(court_y)], _court_to_render_point(homography, [float(court_x), float(court_y)])


def _bbox_head_point(bbox: List[float]) -> List[float]:
    x1, y1, x2, _ = [float(v) for v in bbox]
    return [float((x1 + x2) / 2.0), float(y1)]


def _court_to_render_point(
    homography: CourtHomography,
    court_point: List[float],
) -> List[float]:
    render_x, render_y = homography.court_to_pixel_2d(court_point[0], court_point[1])
    render_y = max(0, min(int(homography.output_h - 1 - render_y), homography.output_h - 1))
    return [float(render_x), float(render_y)]


def _bbox_stats(bbox: List[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area = width * height
    center_x = (x1 + x2) / 2.0
    return width, height, area, center_x


def _bbox_center(bbox: List[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]


def _build_player_height_prior(frames: List[BirdseyeFrame]) -> Dict[str, float]:
    samples: List[Tuple[float, float]] = []
    for frame in frames:
        for player in frame.players:
            if player.is_estimated:
                continue
            center_y = _bbox_center(player.image_bbox)[1]
            _, height, _, _ = _bbox_stats(player.image_bbox)
            samples.append((center_y, height))

    if not samples:
        return {
            "slope": 0.0,
            "intercept": 180.0,
            "min_height": 120.0,
            "max_height": 260.0,
        }

    sample_array = np.asarray(samples, dtype=float)
    order = np.argsort(sample_array[:, 0])
    sample_array = sample_array[order]
    bin_count = max(2 if len(sample_array) >= 2 else 1, min(PLAYER_HEIGHT_PRIOR_BINS, len(sample_array) // 10 or 1))

    y_values: List[float] = []
    h_values: List[float] = []
    for chunk in np.array_split(sample_array, bin_count):
        if len(chunk) == 0:
            continue
        y_values.append(float(np.median(chunk[:, 0])))
        h_values.append(float(np.percentile(chunk[:, 1], PLAYER_HEIGHT_PRIOR_QUANTILE)))

    if len(y_values) >= 2 and (max(y_values) - min(y_values)) > 1e-6:
        slope, intercept = np.polyfit(np.asarray(y_values, dtype=float), np.asarray(h_values, dtype=float), 1)
        slope = float(max(0.0, slope))
        intercept = float(intercept)
    else:
        slope = 0.0
        intercept = float(h_values[0])

    heights = sample_array[:, 1]
    return {
        "slope": slope,
        "intercept": intercept,
        "min_height": float(np.percentile(heights, 20)),
        "max_height": float(np.percentile(heights, 98)),
    }


def _expected_player_height(height_prior: Dict[str, float], image_center_y: float) -> float:
    expected_height = height_prior["intercept"] + height_prior["slope"] * float(image_center_y)
    return float(min(max(expected_height, height_prior["min_height"]), height_prior["max_height"]))


def _build_head_height_prior(frames: List[BirdseyeFrame]) -> Dict[str, float]:
    samples: List[Tuple[float, float]] = []
    for frame in frames:
        for player in frame.players:
            if player.is_estimated:
                continue
            head_y = _bbox_head_point(player.image_bbox)[1]
            _, height, _, _ = _bbox_stats(player.image_bbox)
            samples.append((head_y, height))

    if not samples:
        return {
            "slope": 0.0,
            "intercept": 180.0,
            "min_height": 120.0,
            "max_height": 260.0,
        }

    sample_array = np.asarray(samples, dtype=float)
    order = np.argsort(sample_array[:, 0])
    sample_array = sample_array[order]
    bin_count = max(2 if len(sample_array) >= 2 else 1, min(PLAYER_HEIGHT_PRIOR_BINS, len(sample_array) // 10 or 1))

    y_values: List[float] = []
    h_values: List[float] = []
    for chunk in np.array_split(sample_array, bin_count):
        if len(chunk) == 0:
            continue
        y_values.append(float(np.median(chunk[:, 0])))
        h_values.append(float(np.percentile(chunk[:, 1], PLAYER_HEIGHT_PRIOR_QUANTILE)))

    if len(y_values) >= 2 and (max(y_values) - min(y_values)) > 1e-6:
        slope, intercept = np.polyfit(np.asarray(y_values, dtype=float), np.asarray(h_values, dtype=float), 1)
        slope = float(max(0.0, slope))
        intercept = float(intercept)
    else:
        slope = 0.0
        intercept = float(h_values[0])

    heights = sample_array[:, 1]
    return {
        "slope": slope,
        "intercept": intercept,
        "min_height": float(np.percentile(heights, 20)),
        "max_height": float(np.percentile(heights, 98)),
    }


def _apply_head_based_projection(
    frames: List[BirdseyeFrame],
    homography: CourtHomography,
) -> Dict[str, float]:
    height_prior = _build_head_height_prior(frames)
    head_projection_anchors = 0
    total_anchor_adjustment_px = 0.0
    max_anchor_adjustment_px = 0.0

    for frame in frames:
        for player in frame.players:
            head_anchor = _bbox_head_point(player.image_bbox)
            expected_height = _expected_player_height(height_prior, head_anchor[1])
            image_anchor = [float(head_anchor[0]), float(head_anchor[1] + expected_height)]
            court_position, render_position = _project_image_point(homography, image_anchor)

            bbox_anchor = _bbox_anchor_point(player.image_bbox, homography)
            anchor_adjustment_px = hypot(image_anchor[0] - bbox_anchor[0], image_anchor[1] - bbox_anchor[1])

            player.image_anchor = list(image_anchor)
            player.raw_image_anchor = list(image_anchor)
            player.court_position = list(court_position)
            player.raw_court_position = list(court_position)
            player.render_position = list(render_position)
            player.raw_render_position = list(render_position)
            player.stabilization_trust = 1.0

            head_projection_anchors += 1
            total_anchor_adjustment_px += anchor_adjustment_px
            max_anchor_adjustment_px = max(max_anchor_adjustment_px, anchor_adjustment_px)

    mean_anchor_adjustment_px = (
        total_anchor_adjustment_px / head_projection_anchors if head_projection_anchors else 0.0
    )
    return {
        "head_projection_anchors": float(head_projection_anchors),
        "refined_projection_anchors": 0.0,
        "mean_anchor_adjustment_px": float(mean_anchor_adjustment_px),
        "max_anchor_adjustment_px": float(max_anchor_adjustment_px),
    }


def _measurement_trust(
    current: BirdseyePlayerPosition,
    previous: BirdseyePlayerPosition,
    raw_delta_m: float,
    fps: float,
) -> float:
    trust = 1.0
    curr_width, curr_height, curr_area, curr_center_x = _bbox_stats(current.image_bbox)
    prev_width, prev_height, prev_area, prev_center_x = _bbox_stats(previous.image_bbox)

    area_ratio = curr_area / max(prev_area, 1.0)
    width_ratio = curr_width / max(prev_width, 1.0)
    height_ratio = curr_height / max(prev_height, 1.0)
    center_shift_px = abs(curr_center_x - prev_center_x)

    if area_ratio < 0.82:
        trust *= max(0.18, area_ratio / 0.82)
    if width_ratio < 0.86:
        trust *= max(0.2, width_ratio / 0.86)
    if height_ratio < 0.9:
        trust *= max(0.35, height_ratio / 0.9)
    if center_shift_px > prev_width * 0.18 and area_ratio < 0.95:
        trust *= 0.45

    max_step_m = PLAYER_MAX_SPEED_MPS / max(fps, 1.0)
    if raw_delta_m > max_step_m:
        trust *= max(0.15, max_step_m / max(raw_delta_m, 1e-6))

    return float(min(max(trust, 0.0), 1.0))


def _limit_step(
    from_point: List[float],
    to_point: List[float],
    max_distance_m: float,
) -> List[float]:
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    distance = hypot(dx, dy)
    if distance <= max_distance_m or distance == 0.0:
        return [float(to_point[0]), float(to_point[1])]

    scale = max_distance_m / distance
    return [float(from_point[0] + dx * scale), float(from_point[1] + dy * scale)]


def _movement_direction(
    recent_motion: deque[List[float]],
    fallback_motion: List[float],
) -> Optional[List[float]]:
    sum_x = float(sum(vector[0] for vector in recent_motion))
    sum_y = float(sum(vector[1] for vector in recent_motion))
    magnitude = hypot(sum_x, sum_y)
    if magnitude >= PLAYER_DIRECTION_MIN_STEP_M:
        return [sum_x / magnitude, sum_y / magnitude]

    fallback_magnitude = hypot(fallback_motion[0], fallback_motion[1])
    if fallback_magnitude >= PLAYER_DIRECTION_MIN_STEP_M:
        return [fallback_motion[0] / fallback_magnitude, fallback_motion[1] / fallback_magnitude]

    return None


def _blend_directional_motion(
    predicted: List[float],
    measurement: List[float],
    motion_direction: List[float],
    base_alpha: float,
) -> List[float]:
    direction_x, direction_y = motion_direction
    perpendicular_x = -direction_y
    perpendicular_y = direction_x

    delta_x = measurement[0] - predicted[0]
    delta_y = measurement[1] - predicted[1]
    along_delta = delta_x * direction_x + delta_y * direction_y
    perpendicular_delta = delta_x * perpendicular_x + delta_y * perpendicular_y

    along_alpha = min(0.82, base_alpha + 0.22)
    perpendicular_alpha = max(0.06, base_alpha * 0.22)

    return [
        float(predicted[0] + along_delta * along_alpha * direction_x + perpendicular_delta * perpendicular_alpha * perpendicular_x),
        float(predicted[1] + along_delta * along_alpha * direction_y + perpendicular_delta * perpendicular_alpha * perpendicular_y),
    ]


def _vector_angle_degrees(vector_a: List[float], vector_b: List[float]) -> float:
    magnitude_a = hypot(vector_a[0], vector_a[1])
    magnitude_b = hypot(vector_b[0], vector_b[1])
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    dot = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
    cosine = float(np.clip(dot / (magnitude_a * magnitude_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _fit_linear_trajectory_span(
    span: List[BirdseyePlayerPosition],
) -> Tuple[List[List[float]], float, float]:
    frame_values = np.asarray([position.frame_idx for position in span], dtype=float)
    t_values = frame_values - frame_values[0]
    measurement_values = np.asarray([position.court_position for position in span], dtype=float)

    x_coeffs = np.polyfit(t_values, measurement_values[:, 0], 1)
    y_coeffs = np.polyfit(t_values, measurement_values[:, 1], 1)

    fitted_values = np.column_stack(
        (
            x_coeffs[0] * t_values + x_coeffs[1],
            y_coeffs[0] * t_values + y_coeffs[1],
        )
    )
    residuals = np.linalg.norm(fitted_values - measurement_values, axis=1)
    return fitted_values.tolist(), float(np.mean(residuals)), float(np.max(residuals))


def _trajectory_segment_break(
    previous: BirdseyePlayerPosition,
    current: BirdseyePlayerPosition,
    previous_motion: Optional[List[float]],
) -> bool:
    frame_gap = current.frame_idx - previous.frame_idx
    if frame_gap != 1:
        return True
    if previous.is_estimated or current.is_estimated or previous.is_ghost or current.is_ghost:
        return True
    if min(previous.stabilization_trust, current.stabilization_trust) < PLAYER_TRAJECTORY_MIN_TRUST:
        return True
    if abs(current.stabilization_trust - previous.stabilization_trust) > PLAYER_TRAJECTORY_MAX_TRUST_CLIFF:
        return True

    current_motion = [
        current.court_position[0] - previous.court_position[0],
        current.court_position[1] - previous.court_position[1],
    ]
    current_speed = hypot(current_motion[0], current_motion[1])
    if current_speed < PLAYER_TRAJECTORY_STATIONARY_STEP_M:
        return True

    if previous_motion is None:
        return False

    previous_speed = hypot(previous_motion[0], previous_motion[1])
    if previous_speed >= PLAYER_TRAJECTORY_MIN_STEP_M * 1.75 and current_speed < PLAYER_TRAJECTORY_STATIONARY_STEP_M * 1.25:
        return True

    if previous_speed >= PLAYER_TRAJECTORY_MIN_STEP_M and current_speed >= PLAYER_TRAJECTORY_MIN_STEP_M:
        if _vector_angle_degrees(previous_motion, current_motion) >= PLAYER_TRAJECTORY_MAX_DIRECTION_CHANGE_DEG:
            return True

    return False


def _trajectory_span_alpha(span: List[BirdseyePlayerPosition], index: int) -> float:
    span_length = len(span)
    if span_length < 3:
        return 0.0

    average_trust = float(np.mean([position.stabilization_trust for position in span]))
    base_alpha = min(
        PLAYER_TRAJECTORY_BLEND_MAX_ALPHA,
        0.14 + max(0.0, 0.8 - average_trust) * 0.18,
    )
    edge_distance = min(index, span_length - 1 - index)
    edge_scale = min(1.0, edge_distance / 2.0)
    return float(base_alpha * edge_scale)


def _span_motion_vectors(span: List[BirdseyePlayerPosition]) -> List[List[float]]:
    return [
        [
            span[index].court_position[0] - span[index - 1].court_position[0],
            span[index].court_position[1] - span[index - 1].court_position[1],
        ]
        for index in range(1, len(span))
        if span[index].frame_idx - span[index - 1].frame_idx == 1
    ]


def _classify_motion_span(span: List[BirdseyePlayerPosition]) -> str:
    if len(span) < 2:
        return "transition"

    motions = _span_motion_vectors(span)
    if not motions:
        return "transition"

    speeds = [hypot(vector[0], vector[1]) for vector in motions]
    average_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))
    average_trust = float(np.mean([position.stabilization_trust for position in span]))
    max_trust_delta = float(
        max(position.stabilization_trust for position in span)
        - min(position.stabilization_trust for position in span)
    )

    nonzero_vectors = [vector for vector, speed in zip(motions, speeds) if speed >= PLAYER_TRAJECTORY_MIN_STEP_M]
    direction_changes = [
        _vector_angle_degrees(nonzero_vectors[index - 1], nonzero_vectors[index])
        for index in range(1, len(nonzero_vectors))
    ]
    max_direction_change = float(max(direction_changes)) if direction_changes else 0.0

    if len(span) >= PLAYER_TRAJECTORY_STATIONARY_MIN_FRAMES and max_speed <= PLAYER_TRAJECTORY_STATIONARY_STEP_M * 1.1:
        return "stationary"

    if average_trust < PLAYER_TRAJECTORY_MIN_TRUST or max_trust_delta > PLAYER_TRAJECTORY_MAX_TRUST_CLIFF:
        return "transition"

    if max_direction_change >= PLAYER_TRAJECTORY_REACTIVE_DIRECTION_DEG:
        return "reactive"

    if len(span) >= PLAYER_TRAJECTORY_MIN_FRAMES and average_speed >= PLAYER_TRAJECTORY_MIN_STEP_M and max_direction_change <= PLAYER_TRAJECTORY_COHERENT_MAX_DIRECTION_DEG:
        return "coherent"

    return "transition"


def _journey_segment_break(
    span: List[BirdseyePlayerPosition],
    current: BirdseyePlayerPosition,
    previous_motion: Optional[List[float]],
) -> bool:
    previous = span[-1]
    if current.frame_idx - previous.frame_idx != 1:
        return True
    if previous.is_estimated or current.is_estimated or previous.is_ghost or current.is_ghost:
        return True

    current_motion = [
        current.court_position[0] - previous.court_position[0],
        current.court_position[1] - previous.court_position[1],
    ]
    current_step = hypot(current_motion[0], current_motion[1])
    if current_step < PLAYER_JOURNEY_MIN_STEP_M:
        return False

    if previous_motion is not None:
        if _vector_angle_degrees(previous_motion, current_motion) > PLAYER_JOURNEY_MAX_DIRECTION_CHANGE_DEG:
            return True

    overall_motion = [
        previous.court_position[0] - span[0].court_position[0],
        previous.court_position[1] - span[0].court_position[1],
    ]
    overall_distance = hypot(overall_motion[0], overall_motion[1])
    if overall_distance >= PLAYER_JOURNEY_MIN_STEP_M:
        if _vector_angle_degrees(overall_motion, current_motion) > PLAYER_JOURNEY_MAX_HEADING_DEVIATION_DEG:
            return True

    return False


def _journey_summary(
    span: List[BirdseyePlayerPosition],
    fps: float,
) -> Optional[Dict[str, Any]]:
    if len(span) < PLAYER_JOURNEY_MIN_FRAMES:
        return None

    motions = _span_motion_vectors(span)
    if not motions:
        return None

    start_point = list(span[0].court_position)
    end_point = list(span[-1].court_position)
    net_distance = hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
    path_distance = float(sum(hypot(vector[0], vector[1]) for vector in motions))
    if net_distance < PLAYER_JOURNEY_MIN_DISTANCE_M or path_distance < PLAYER_JOURNEY_MIN_DISTANCE_M:
        return None

    straightness_ratio = net_distance / max(path_distance, 1e-6)
    step_x = [vector[0] for vector in motions]
    step_y = [vector[1] for vector in motions]
    net_dx = end_point[0] - start_point[0]
    net_dy = end_point[1] - start_point[1]
    dominant_net = max(abs(net_dx), abs(net_dy))
    dominant_path = max(sum(abs(value) for value in step_x), sum(abs(value) for value in step_y), 1e-6)
    dominant_progress_ratio = dominant_net / dominant_path
    if straightness_ratio < PLAYER_JOURNEY_MIN_STRAIGHTNESS_RATIO:
        return None

    duration_frames = span[-1].frame_idx - span[0].frame_idx
    if duration_frames <= 0:
        return None

    duration_s = duration_frames / max(fps, 1.0)
    section_distance_m = net_distance / duration_frames
    section_speed_mps = section_distance_m * fps

    return {
        "player_id": span[0].player_id,
        "start_frame": int(span[0].frame_idx),
        "end_frame": int(span[-1].frame_idx),
        "point_a": [float(start_point[0]), float(start_point[1])],
        "point_b": [float(end_point[0]), float(end_point[1])],
        "distance_m": float(net_distance),
        "path_distance_m": float(path_distance),
        "duration_s": float(duration_s),
        "section_distance_m": float(section_distance_m),
        "section_speed_mps": float(section_speed_mps),
        "frame_count": int(len(span)),
        "straightness_ratio": float(straightness_ratio),
        "dominant_progress_ratio": float(dominant_progress_ratio),
    }


def _apply_journey_path_smoothing(
    frames: List[BirdseyeFrame],
    homography: CourtHomography,
    fps: float,
    court_length_m: float,
    court_width_m: float,
) -> Dict[str, Any]:
    players_by_id: Dict[str, List[BirdseyePlayerPosition]] = {}
    for frame in frames:
        for player in frame.players:
            players_by_id.setdefault(player.player_id, []).append(player)

    segments_detected = 0
    segments_smoothed = 0
    frames_modified = 0
    total_delta_m = 0.0
    max_delta_m = 0.0
    journeys: List[Dict[str, Any]] = []

    for positions in players_by_id.values():
        positions.sort(key=lambda player: player.frame_idx)
        if len(positions) < PLAYER_JOURNEY_MIN_FRAMES:
            continue

        spans: List[List[BirdseyePlayerPosition]] = []
        current_span: List[BirdseyePlayerPosition] = [positions[0]]
        previous_motion: Optional[List[float]] = None

        for current in positions[1:]:
            if _journey_segment_break(current_span, current, previous_motion):
                spans.append(current_span)
                current_span = [current]
                previous_motion = None
                continue

            previous = current_span[-1]
            current_motion = [
                current.court_position[0] - previous.court_position[0],
                current.court_position[1] - previous.court_position[1],
            ]
            if hypot(current_motion[0], current_motion[1]) >= PLAYER_JOURNEY_MIN_STEP_M:
                previous_motion = current_motion
            current_span.append(current)

        spans.append(current_span)

        for span in spans:
            summary = _journey_summary(span, fps)
            if summary is None:
                continue

            segments_detected += 1
            journeys.append(summary)
            start_frame = span[0].frame_idx
            end_frame = span[-1].frame_idx
            duration_frames = max(1, end_frame - start_frame)
            point_a = summary["point_a"]
            point_b = summary["point_b"]
            direction_x = point_b[0] - point_a[0]
            direction_y = point_b[1] - point_a[1]
            direction_norm = hypot(direction_x, direction_y)
            if direction_norm <= 1e-6:
                continue
            direction_unit = [direction_x / direction_norm, direction_y / direction_norm]
            raw_positions = [list(position.court_position) for position in span]
            span_changed = False

            for index, position in enumerate(span):
                alpha = (position.frame_idx - start_frame) / duration_frames
                target = [
                    point_a[0] + (point_b[0] - point_a[0]) * alpha,
                    point_a[1] + (point_b[1] - point_a[1]) * alpha,
                ]
                target[0] = min(max(float(target[0]), 0.0), court_length_m)
                target[1] = min(max(float(target[1]), 0.0), court_width_m)
                measurement = list(raw_positions[index])

                if len(span) > 1:
                    if index == 0:
                        local_motion = [
                            raw_positions[1][0] - raw_positions[0][0],
                            raw_positions[1][1] - raw_positions[0][1],
                        ]
                    elif index == len(span) - 1:
                        local_motion = [
                            raw_positions[index][0] - raw_positions[index - 1][0],
                            raw_positions[index][1] - raw_positions[index - 1][1],
                        ]
                    else:
                        local_motion = [
                            raw_positions[index + 1][0] - raw_positions[index - 1][0],
                            raw_positions[index + 1][1] - raw_positions[index - 1][1],
                        ]
                    local_norm = hypot(local_motion[0], local_motion[1])
                    if local_norm > 1e-6:
                        local_alignment = (
                            local_motion[0] * direction_unit[0] + local_motion[1] * direction_unit[1]
                        ) / local_norm
                        if local_alignment < PLAYER_JOURNEY_MIN_LOCAL_ALIGNMENT:
                            continue

                target = _limit_step(measurement, target, PLAYER_JOURNEY_MAX_DRIFT_M)
                delta_m = hypot(target[0] - measurement[0], target[1] - measurement[1])
                if delta_m <= 1e-6:
                    continue

                position.court_position = [float(target[0]), float(target[1])]
                position.render_position = _court_to_render_point(homography, position.court_position)
                frames_modified += 1
                total_delta_m += delta_m
                max_delta_m = max(max_delta_m, delta_m)
                span_changed = True

            if span_changed:
                segments_smoothed += 1

    mean_delta_m = total_delta_m / frames_modified if frames_modified else 0.0
    return {
        "journey_segments_detected": float(segments_detected),
        "journey_segments_smoothed": float(segments_smoothed),
        "journey_frames_modified": float(frames_modified),
        "journey_mean_delta_m": float(mean_delta_m),
        "journey_max_delta_m": float(max_delta_m),
        "journeys": journeys,
    }


def _stationary_pair(
    previous: BirdseyePlayerPosition,
    current: BirdseyePlayerPosition,
) -> bool:
    if current.frame_idx - previous.frame_idx != 1:
        return False
    if previous.is_estimated or current.is_estimated or previous.is_ghost or current.is_ghost:
        return False
    if min(previous.stabilization_trust, current.stabilization_trust) < PLAYER_TRAJECTORY_MIN_TRUST:
        return False
    if abs(current.stabilization_trust - previous.stabilization_trust) > PLAYER_TRAJECTORY_MAX_TRUST_CLIFF:
        return False

    delta_x = current.court_position[0] - previous.court_position[0]
    delta_y = current.court_position[1] - previous.court_position[1]
    return hypot(delta_x, delta_y) <= PLAYER_TRAJECTORY_STATIONARY_STEP_M


def _apply_stationary_span_smoothing(
    frames: List[BirdseyeFrame],
    homography: CourtHomography,
    court_length_m: float,
    court_width_m: float,
) -> Dict[str, float]:
    players_by_id: Dict[str, List[BirdseyePlayerPosition]] = {}
    for frame in frames:
        for player in frame.players:
            players_by_id.setdefault(player.player_id, []).append(player)

    segments_detected = 0
    segments_smoothed = 0
    frames_modified = 0
    total_delta_m = 0.0
    max_delta_m = 0.0
    segment_types = {
        "stationary": 0,
        "coherent": 0,
        "transition": 0,
        "reactive": 0,
    }

    for positions in players_by_id.values():
        positions.sort(key=lambda player: player.frame_idx)
        active_span: List[BirdseyePlayerPosition] = []

        def flush_span(span: List[BirdseyePlayerPosition]) -> None:
            nonlocal segments_detected, segments_smoothed, frames_modified, total_delta_m, max_delta_m
            if len(span) < PLAYER_TRAJECTORY_STATIONARY_MIN_FRAMES:
                return

            segment_type = _classify_motion_span(span)
            segment_types[segment_type] += 1
            if segment_type != "stationary":
                return

            segments_detected += 1
            median_x = float(np.median([position.court_position[0] for position in span]))
            median_y = float(np.median([position.court_position[1] for position in span]))
            span_changed = False

            for position in span:
                measurement = list(position.court_position)
                bounded = _limit_step(measurement, [median_x, median_y], PLAYER_TRAJECTORY_STATIONARY_MAX_DRIFT_M)
                bounded[0] = min(max(bounded[0], 0.0), court_length_m)
                bounded[1] = min(max(bounded[1], 0.0), court_width_m)
                delta_m = hypot(bounded[0] - measurement[0], bounded[1] - measurement[1])
                if delta_m <= 1e-6:
                    continue

                position.court_position = [float(bounded[0]), float(bounded[1])]
                position.render_position = _court_to_render_point(homography, position.court_position)
                frames_modified += 1
                total_delta_m += delta_m
                max_delta_m = max(max_delta_m, delta_m)
                span_changed = True

            if span_changed:
                segments_smoothed += 1

        for index, position in enumerate(positions):
            if index == 0:
                active_span = [position]
                continue

            previous = positions[index - 1]
            if _stationary_pair(previous, position):
                if not active_span:
                    active_span = [previous, position]
                elif active_span[-1].frame_idx == previous.frame_idx:
                    active_span.append(position)
                else:
                    flush_span(active_span)
                    active_span = [previous, position]
                continue

            flush_span(active_span)
            active_span = [position]

        flush_span(active_span)

    mean_delta_m = total_delta_m / frames_modified if frames_modified else 0.0
    return {
        "trajectory_stationary_segments_detected": float(segments_detected),
        "trajectory_stationary_segments_smoothed": float(segments_smoothed),
        "trajectory_stationary_frames_modified": float(frames_modified),
        "trajectory_stationary_mean_delta_m": float(mean_delta_m),
        "trajectory_stationary_max_delta_m": float(max_delta_m),
        "trajectory_stationary_segments_classified_stationary": float(segment_types["stationary"]),
        "trajectory_stationary_segments_classified_transition": float(segment_types["transition"]),
        "trajectory_stationary_segments_classified_reactive": float(segment_types["reactive"]),
    }


def _apply_trajectory_segment_smoothing(
    frames: List[BirdseyeFrame],
    homography: CourtHomography,
    court_length_m: float,
    court_width_m: float,
) -> Dict[str, float]:
    players_by_id: Dict[str, List[BirdseyePlayerPosition]] = {}
    for frame in frames:
        for player in frame.players:
            players_by_id.setdefault(player.player_id, []).append(player)

    segments_detected = 0
    segments_smoothed = 0
    frames_modified = 0
    total_delta_m = 0.0
    max_delta_m = 0.0
    segment_types = {
        "stationary": 0,
        "coherent": 0,
        "transition": 0,
        "reactive": 0,
    }

    for positions in players_by_id.values():
        positions.sort(key=lambda player: player.frame_idx)
        if len(positions) < PLAYER_TRAJECTORY_MIN_FRAMES:
            continue

        spans: List[List[BirdseyePlayerPosition]] = []
        current_span: List[BirdseyePlayerPosition] = [positions[0]]
        previous_motion: Optional[List[float]] = None

        for index in range(1, len(positions)):
            previous = positions[index - 1]
            current = positions[index]
            if _trajectory_segment_break(previous, current, previous_motion):
                spans.append(current_span)
                current_span = [current]
                previous_motion = None
                continue

            current_motion = [
                current.court_position[0] - previous.court_position[0],
                current.court_position[1] - previous.court_position[1],
            ]
            current_span.append(current)
            previous_motion = current_motion

        spans.append(current_span)

        for span in spans:
            segments_detected += 1
            segment_type = _classify_motion_span(span)
            segment_types[segment_type] += 1
            if segment_type != "coherent":
                continue

            fitted_values, mean_error, max_error = _fit_linear_trajectory_span(span)
            if mean_error > PLAYER_TRAJECTORY_MAX_FIT_MEAN_ERROR_M or max_error > PLAYER_TRAJECTORY_MAX_FIT_PEAK_ERROR_M:
                continue

            segment_changed = False
            for index, (position, fitted_position) in enumerate(zip(span, fitted_values)):
                alpha = _trajectory_span_alpha(span, index)
                if alpha <= 0.0:
                    continue

                measurement = list(position.court_position)
                blended = [
                    measurement[0] + alpha * (fitted_position[0] - measurement[0]),
                    measurement[1] + alpha * (fitted_position[1] - measurement[1]),
                ]
                bounded = _limit_step(measurement, blended, PLAYER_TRAJECTORY_MAX_DRIFT_M)
                bounded[0] = min(max(bounded[0], 0.0), court_length_m)
                bounded[1] = min(max(bounded[1], 0.0), court_width_m)

                delta_m = hypot(bounded[0] - measurement[0], bounded[1] - measurement[1])
                if delta_m <= 1e-6:
                    continue

                position.court_position = [float(bounded[0]), float(bounded[1])]
                position.render_position = _court_to_render_point(homography, position.court_position)
                frames_modified += 1
                total_delta_m += delta_m
                max_delta_m = max(max_delta_m, delta_m)
                segment_changed = True

            if segment_changed:
                segments_smoothed += 1

    mean_delta_m = total_delta_m / frames_modified if frames_modified else 0.0
    return {
        "trajectory_smoothing_segments_detected": float(segments_detected),
        "trajectory_smoothing_segments_smoothed": float(segments_smoothed),
        "trajectory_smoothing_frames_modified": float(frames_modified),
        "trajectory_smoothing_mean_delta_m": float(mean_delta_m),
        "trajectory_smoothing_max_delta_m": float(max_delta_m),
        "trajectory_smoothing_segments_classified_coherent": float(segment_types["coherent"]),
        "trajectory_smoothing_segments_classified_stationary": float(segment_types["stationary"]),
        "trajectory_smoothing_segments_classified_transition": float(segment_types["transition"]),
        "trajectory_smoothing_segments_classified_reactive": float(segment_types["reactive"]),
    }


def _refine_ground_contact_anchors(
    frames: List[BirdseyeFrame],
    homography: CourtHomography,
) -> Dict[str, float]:
    players_by_id: Dict[str, List[BirdseyePlayerPosition]] = {}
    for frame in frames:
        for player in frame.players:
            players_by_id.setdefault(player.player_id, []).append(player)

    height_prior = _build_player_height_prior(frames)

    adjusted_anchor_count = 0
    total_adjustment_px = 0.0
    max_adjustment_px = 0.0

    for positions in players_by_id.values():
        positions.sort(key=lambda player: player.frame_idx)
        recent_center_x: deque[float] = deque(maxlen=GROUND_CONTACT_WINDOW)
        recent_top_y: deque[float] = deque(maxlen=GROUND_CONTACT_WINDOW)
        recent_reference_heights: deque[float] = deque(maxlen=GROUND_CONTACT_WINDOW)
        recent_widths: deque[float] = deque(maxlen=GROUND_CONTACT_WINDOW)
        previous_anchor: Optional[List[float]] = None

        for position in positions:
            raw_anchor = list(position.image_anchor)
            position.raw_image_anchor = raw_anchor
            x1, y1, _, _ = [float(v) for v in position.image_bbox]
            bbox_center = _bbox_center(position.image_bbox)
            width, height, _, _ = _bbox_stats(position.image_bbox)
            recent_center_x.append(bbox_center[0])
            recent_widths.append(width)
            recent_top_y.append(y1)

            expected_height = _expected_player_height(height_prior, bbox_center[1])
            historical_height = float(np.percentile(list(recent_reference_heights), 80)) if recent_reference_heights else expected_height
            target_height = max(height, expected_height, historical_height)
            restored_height = float(height + max(0.0, target_height - height) * PLAYER_HEIGHT_RESTORE_BLEND)
            recent_reference_heights.append(max(height, restored_height))

            smoothed_center_x = float(np.median(list(recent_center_x)))
            smoothed_top_y = float(np.median(list(recent_top_y)))
            smoothed_height = float(np.percentile(list(recent_reference_heights), 75))
            smoothed_width = float(np.median(list(recent_widths)))
            refined_x = smoothed_center_x
            refined_y = smoothed_top_y + smoothed_height

            if previous_anchor is not None:
                max_sideways_step_px = max(2.0, smoothed_width * 0.15)
                refined_x = previous_anchor[0] + max(
                    -max_sideways_step_px,
                    min(max_sideways_step_px, refined_x - previous_anchor[0]),
                )

                max_upward_step_px = max(1.5, smoothed_height * 0.04)
                if refined_y < previous_anchor[1]:
                    refined_y = max(refined_y, previous_anchor[1] - max_upward_step_px)

                max_downward_step_px = max(2.0, smoothed_height * 0.08)
                if refined_y > previous_anchor[1]:
                    refined_y = min(refined_y, previous_anchor[1] + max_downward_step_px)

            refined_anchor = [float(refined_x), float(refined_y)]
            adjustment_px = hypot(refined_anchor[0] - raw_anchor[0], refined_anchor[1] - raw_anchor[1])
            if adjustment_px > 1e-6:
                adjusted_anchor_count += 1
                total_adjustment_px += adjustment_px
                max_adjustment_px = max(max_adjustment_px, adjustment_px)

            position.image_anchor = refined_anchor
            position.court_position, position.render_position = _project_image_point(homography, refined_anchor)
            previous_anchor = refined_anchor

    mean_adjustment_px = total_adjustment_px / adjusted_anchor_count if adjusted_anchor_count else 0.0
    return {
        "refined_projection_anchors": float(adjusted_anchor_count),
        "mean_anchor_adjustment_px": float(mean_adjustment_px),
        "max_anchor_adjustment_px": float(max_adjustment_px),
    }


def _load_pose_model():
    global _POSE_MODEL
    if _POSE_MODEL is None:
        from ultralytics import YOLO

        _POSE_MODEL = YOLO(POSE_MODEL_NAME)
    return _POSE_MODEL


def _pose_anchor_from_keypoints(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    fallback_anchor: List[float],
    crop_height: float,
) -> Optional[List[float]]:
    visible_ankles = [
        keypoints_xy[index]
        for index in (15, 16)
        if index < len(keypoints_conf) and float(keypoints_conf[index]) >= POSE_MIN_KEYPOINT_CONFIDENCE
    ]
    if visible_ankles:
        ankle_points = np.asarray(visible_ankles, dtype=float)
        return [float(np.mean(ankle_points[:, 0])), float(np.max(ankle_points[:, 1]))]

    visible_knees = [
        keypoints_xy[index]
        for index in (13, 14)
        if index < len(keypoints_conf) and float(keypoints_conf[index]) >= POSE_MIN_KEYPOINT_CONFIDENCE
    ]
    if visible_knees:
        knee_points = np.asarray(visible_knees, dtype=float)
        est_x = float(np.mean(knee_points[:, 0]))
        est_y = float(np.max(knee_points[:, 1]) + crop_height * 0.12)
        return [est_x, est_y]

    visible_hips = [
        keypoints_xy[index]
        for index in (11, 12)
        if index < len(keypoints_conf) and float(keypoints_conf[index]) >= POSE_MIN_KEYPOINT_CONFIDENCE
    ]
    if visible_hips:
        hip_points = np.asarray(visible_hips, dtype=float)
        est_x = float(np.mean(hip_points[:, 0]))
        est_y = float(np.max(hip_points[:, 1]) + crop_height * 0.28)
        return [est_x, est_y]

    return None


def _pose_anchor_from_upper_body_keypoints(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    fallback_anchor: List[float],
    expected_height: float,
) -> Optional[List[float]]:
    visible_head_points = [
        keypoints_xy[index]
        for index in (0, 1, 2, 3, 4)
        if index < len(keypoints_conf) and float(keypoints_conf[index]) >= POSE_UPPER_BODY_KEYPOINT_CONFIDENCE
    ]
    visible_shoulders = [
        keypoints_xy[index]
        for index in (5, 6)
        if index < len(keypoints_conf) and float(keypoints_conf[index]) >= POSE_UPPER_BODY_KEYPOINT_CONFIDENCE
    ]
    visible_hips = [
        keypoints_xy[index]
        for index in (11, 12)
        if index < len(keypoints_conf) and float(keypoints_conf[index]) >= POSE_UPPER_BODY_KEYPOINT_CONFIDENCE
    ]

    if visible_head_points:
        head_points = np.asarray(visible_head_points, dtype=float)
        anchor_x = float(np.mean(head_points[:, 0]))
        top_y = float(np.min(head_points[:, 1]) - expected_height * POSE_TOP_PADDING_RATIO)
        return [anchor_x, top_y + expected_height]

    if visible_shoulders:
        shoulder_points = np.asarray(visible_shoulders, dtype=float)
        anchor_x = float(np.mean(shoulder_points[:, 0]))
        shoulder_y = float(np.mean(shoulder_points[:, 1]))
        return [anchor_x, shoulder_y + expected_height * POSE_SHOULDER_TO_FOOT_RATIO]

    if visible_hips:
        hip_points = np.asarray(visible_hips, dtype=float)
        anchor_x = float(np.mean(hip_points[:, 0]))
        hip_y = float(np.mean(hip_points[:, 1]))
        return [anchor_x, hip_y + expected_height * POSE_HIP_TO_FOOT_RATIO]

    return None


def _refine_pose_anchors(
    frames: List[BirdseyeFrame],
    video_path: str,
    homography: CourtHomography,
    start_frame: int,
    end_frame_exclusive: int,
) -> Dict[str, float]:
    try:
        pose_model = _load_pose_model()
    except Exception as exc:
        logger.warning(f"Pose anchor refinement unavailable, falling back to Level 2 anchors: {exc}")
        return {
            "pose_anchor_refinements": 0.0,
            "pose_anchor_attempts": 0.0,
            "pose_anchor_failures": 0.0,
            "pose_upper_body_refinements": 0.0,
        }

    players_by_frame = {frame.frame_idx: frame.players for frame in frames}
    height_prior = _build_player_height_prior(frames)
    refined_count = 0
    attempt_count = 0
    failure_count = 0
    upper_body_refinement_count = 0

    reader = VideoReader(video_path)
    try:
        for frame_idx, frame in reader.iter_frames():
            if frame_idx < start_frame:
                continue
            if frame_idx >= end_frame_exclusive:
                break

            players = players_by_frame.get(frame_idx)
            if not players:
                continue

            for player in players:
                if player.is_estimated:
                    continue

                _, height, _, _ = _bbox_stats(player.image_bbox)
                if height < POSE_MIN_BBOX_HEIGHT:
                    continue

                attempt_count += 1
                x1, y1, x2, y2 = [int(round(v)) for v in player.image_bbox]
                pad_x = max(6, int((x2 - x1) * 0.15))
                pad_top = max(4, int((y2 - y1) * 0.10))
                pad_bottom = max(6, int((y2 - y1) * 0.08))
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_top)
                crop_x2 = min(frame.shape[1], x2 + pad_x)
                crop_y2 = min(frame.shape[0], y2 + pad_bottom)
                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    failure_count += 1
                    continue

                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                result = pose_model(crop, verbose=False, imgsz=320)[0]
                if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
                    failure_count += 1
                    continue

                best_index = int(np.argmax(result.boxes.conf.cpu().numpy()))
                keypoints_xy = result.keypoints.xy.cpu().numpy()[best_index]
                keypoints_conf = result.keypoints.conf.cpu().numpy()[best_index]
                bbox_center = _bbox_center(player.image_bbox)
                expected_height = max(
                    _expected_player_height(height_prior, bbox_center[1]),
                    height,
                )
                crop_anchor = _pose_anchor_from_upper_body_keypoints(
                    keypoints_xy=keypoints_xy,
                    keypoints_conf=keypoints_conf,
                    fallback_anchor=player.image_anchor,
                    expected_height=float(expected_height),
                )
                if crop_anchor is not None:
                    upper_body_refinement_count += 1
                else:
                    crop_anchor = _pose_anchor_from_keypoints(
                        keypoints_xy=keypoints_xy,
                        keypoints_conf=keypoints_conf,
                        fallback_anchor=player.image_anchor,
                        crop_height=float(crop.shape[0]),
                    )
                if crop_anchor is None:
                    failure_count += 1
                    continue

                refined_anchor = [float(crop_x1 + crop_anchor[0]), float(crop_y1 + crop_anchor[1])]
                max_anchor_shift_px = max(8.0, expected_height * POSE_MAX_ANCHOR_SHIFT_RATIO)
                bounded_anchor = [
                    float(player.image_anchor[0] + max(-max_anchor_shift_px, min(max_anchor_shift_px, refined_anchor[0] - player.image_anchor[0]))),
                    float(player.image_anchor[1] + max(-max_anchor_shift_px, min(max_anchor_shift_px, refined_anchor[1] - player.image_anchor[1]))),
                ]
                player.image_anchor = bounded_anchor
                player.court_position, player.render_position = _project_image_point(homography, bounded_anchor)
                refined_count += 1

    finally:
        reader.close()

    return {
        "pose_anchor_refinements": float(refined_count),
        "pose_anchor_attempts": float(attempt_count),
        "pose_anchor_failures": float(failure_count),
        "pose_upper_body_refinements": float(upper_body_refinement_count),
    }


def _stabilize_player_positions(
    frames: List[BirdseyeFrame],
    homography: CourtHomography,
    fps: float,
    court_length_m: float,
    court_width_m: float,
) -> Dict[str, float]:
    players_by_id: Dict[str, List[BirdseyePlayerPosition]] = {}
    for frame in frames:
        for player in frame.players:
            player.raw_court_position = list(player.court_position)
            player.raw_render_position = list(player.render_position)
            players_by_id.setdefault(player.player_id, []).append(player)

    adjusted_positions = 0
    total_offset_m = 0.0
    max_offset_m = 0.0

    for positions in players_by_id.values():
        positions.sort(key=lambda player: player.frame_idx)
        previous_position: Optional[BirdseyePlayerPosition] = None
        previous_stabilized: Optional[List[float]] = None
        previous_velocity = [0.0, 0.0]

        for position in positions:
            raw_court = list(position.raw_court_position or position.court_position)
            if previous_position is None or previous_stabilized is None:
                position.stabilization_trust = 1.0
                previous_position = position
                previous_stabilized = raw_court
                continue

            frame_gap = max(1, position.frame_idx - previous_position.frame_idx)
            predicted = [
                previous_stabilized[0] + previous_velocity[0] * frame_gap,
                previous_stabilized[1] + previous_velocity[1] * frame_gap,
            ]
            max_step_m = PLAYER_MAX_SPEED_MPS * frame_gap / max(fps, 1.0)
            limited_measurement = _limit_step(predicted, raw_court, max_step_m)
            raw_delta_m = hypot(raw_court[0] - previous_stabilized[0], raw_court[1] - previous_stabilized[1])
            trust = _measurement_trust(position, previous_position, raw_delta_m, fps)
            alpha = PLAYER_BASE_STABILIZATION_ALPHA + PLAYER_VARIABLE_STABILIZATION_ALPHA * trust
            stabilized = [
                predicted[0] + alpha * (limited_measurement[0] - predicted[0]),
                predicted[1] + alpha * (limited_measurement[1] - predicted[1]),
            ]
            stabilized = _limit_step(limited_measurement, stabilized, PLAYER_MAX_MEASUREMENT_OFFSET_M)
            stabilized[0] = min(max(stabilized[0], 0.0), court_length_m)
            stabilized[1] = min(max(stabilized[1], 0.0), court_width_m)

            offset_m = hypot(stabilized[0] - raw_court[0], stabilized[1] - raw_court[1])
            if offset_m > 1e-6:
                adjusted_positions += 1
                total_offset_m += offset_m
                max_offset_m = max(max_offset_m, offset_m)

            position.stabilization_trust = trust
            position.court_position = [float(stabilized[0]), float(stabilized[1])]
            position.render_position = _court_to_render_point(homography, position.court_position)

            instant_velocity = [
                (position.court_position[0] - previous_stabilized[0]) / frame_gap,
                (position.court_position[1] - previous_stabilized[1]) / frame_gap,
            ]
            previous_velocity = [
                previous_velocity[0] * 0.65 + instant_velocity[0] * 0.35,
                previous_velocity[1] * 0.65 + instant_velocity[1] * 0.35,
            ]
            previous_stabilized = list(position.court_position)
            previous_position = position

    mean_offset_m = total_offset_m / adjusted_positions if adjusted_positions else 0.0
    return {
        "stabilized_player_positions": float(adjusted_positions),
        "mean_player_stabilization_offset_m": float(mean_offset_m),
        "max_player_stabilization_offset_m": float(max_offset_m),
    }


def _apply_preprojection_position_stabilization(
    frames: List[BirdseyeFrame],
    homography: CourtHomography,
) -> Dict[str, float]:
    stabilizer = PositionStabilizer(logger=logger)
    anchor_count = 0
    adjusted_positions = 0
    total_anchor_adjustment_px = 0.0
    max_anchor_adjustment_px = 0.0
    total_offset_m = 0.0
    max_offset_m = 0.0

    for frame in frames:
        for player in frame.players:
            raw_anchor = foot_point_from_bbox(player.image_bbox)
            raw_court_position, raw_render_position = _project_image_point(homography, raw_anchor)

            player.raw_image_anchor = list(raw_anchor)
            player.raw_court_position = list(raw_court_position)
            player.raw_render_position = list(raw_render_position)

            result = stabilizer.stabilize(
                track_key=(player.player_id, int(player.track_id)),
                frame_idx=player.frame_idx,
                bbox=player.image_bbox,
                use_measurement=not player.is_estimated,
                debug_label=f"player={player.player_id} track={player.track_id}",
            )
            player.image_anchor = list(result.stabilized_point)
            player.court_position, player.render_position = _project_image_point(homography, player.image_anchor)
            player.stabilization_trust = 0.0 if result.prediction_only else 1.0

            anchor_count += 1
            anchor_adjustment_px = hypot(
                player.image_anchor[0] - player.raw_image_anchor[0],
                player.image_anchor[1] - player.raw_image_anchor[1],
            )
            if anchor_adjustment_px > 1e-6:
                adjusted_positions += 1
                total_anchor_adjustment_px += anchor_adjustment_px
                max_anchor_adjustment_px = max(max_anchor_adjustment_px, anchor_adjustment_px)

            offset_m = hypot(
                player.court_position[0] - player.raw_court_position[0],
                player.court_position[1] - player.raw_court_position[1],
            )
            if offset_m > 1e-6:
                total_offset_m += offset_m
                max_offset_m = max(max_offset_m, offset_m)

    mean_anchor_adjustment_px = total_anchor_adjustment_px / adjusted_positions if adjusted_positions else 0.0
    mean_offset_m = total_offset_m / adjusted_positions if adjusted_positions else 0.0
    return {
        "head_projection_anchors": 0.0,
        "foot_point_projection_anchors": float(anchor_count),
        "refined_projection_anchors": float(adjusted_positions),
        "mean_anchor_adjustment_px": float(mean_anchor_adjustment_px),
        "max_anchor_adjustment_px": float(max_anchor_adjustment_px),
        "stabilized_player_positions": float(adjusted_positions),
        "mean_player_stabilization_offset_m": float(mean_offset_m),
        "max_player_stabilization_offset_m": float(max_offset_m),
        **stabilizer.diagnostics(),
    }


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
            image_bbox=list(ball_position.bbox) if ball_position.bbox is not None else None,
            image_position=ball_position.centroid,
            court_position=None,
            render_position=None,
        )

    court_position, render_position = _project_image_point(homography, list(ball_position.centroid))
    return BirdseyeBallFrame(
        frame_idx=ball_position.frame_idx,
        state=state_enum,
        confidence=ball_position.confidence,
        image_bbox=list(ball_position.bbox) if ball_position.bbox is not None else None,
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
    video_path: Optional[str] = None,
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
                        raw_image_anchor=image_anchor,
                        raw_court_position=court_position,
                        raw_render_position=render_position,
                        stabilization_trust=1.0,
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
                    raw_image_anchor=image_anchor,
                    raw_court_position=court_position,
                    raw_render_position=render_position,
                    stabilization_trust=1.0,
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
    court_length_m = float(homography_config.get("court_length", homography.pitch_width_m))
    court_width_m = float(homography_config.get("court_width", homography.pitch_height_m))
    anchor_metrics = _apply_preprojection_position_stabilization(frames=frames, homography=homography)
    pose_metrics = {
        "pose_anchor_refinements": 0.0,
        "pose_anchor_attempts": 0.0,
        "pose_anchor_failures": 0.0,
        "pose_upper_body_refinements": 0.0,
    }
    stabilization_metrics = {}
    trajectory_metrics = {
        "trajectory_smoothing_segments_detected": 0.0,
        "trajectory_smoothing_segments_smoothed": 0.0,
        "trajectory_smoothing_frames_modified": 0.0,
        "trajectory_smoothing_mean_delta_m": 0.0,
        "trajectory_smoothing_max_delta_m": 0.0,
        "trajectory_smoothing_segments_classified_coherent": 0.0,
        "trajectory_smoothing_segments_classified_stationary": 0.0,
        "trajectory_smoothing_segments_classified_transition": 0.0,
        "trajectory_smoothing_segments_classified_reactive": 0.0,
    }
    stationary_metrics = {
        "trajectory_stationary_segments_detected": 0.0,
        "trajectory_stationary_segments_smoothed": 0.0,
        "trajectory_stationary_frames_modified": 0.0,
        "trajectory_stationary_mean_delta_m": 0.0,
        "trajectory_stationary_max_delta_m": 0.0,
        "trajectory_stationary_segments_classified_stationary": 0.0,
        "trajectory_stationary_segments_classified_transition": 0.0,
        "trajectory_stationary_segments_classified_reactive": 0.0,
    }
    journey_metrics = {
        "journey_segments_detected": 0.0,
        "journey_segments_smoothed": 0.0,
        "journey_frames_modified": 0.0,
        "journey_mean_delta_m": 0.0,
        "journey_max_delta_m": 0.0,
        "journeys": [],
    }

    return BirdseyeProjectionOutput(
        video_name=pass1_output.video_name,
        fps=pass1_output.fps,
        total_frames=pass1_output.total_frames,
        processed_start_frame=start_frame,
        processed_end_frame_exclusive=end_frame_exclusive,
        court_length_m=court_length_m,
        court_width_m=court_width_m,
        output_pixel_scale=int(homography_config.get("output_pixel_scale", homography.output_pixel_scale)),
        frames=frames,
        diagnostics={
            "calibration_path": str(calibration_path) if calibration_path is not None else None,
            "calibration_point_count": len(homography_config.get("source_points", [])),
            "projected_player_positions": projected_player_count,
            "estimated_player_positions": estimated_player_count,
            "ghost_windows_seen": len(ghost_activity_windows),
            **anchor_metrics,
            **pose_metrics,
            **stabilization_metrics,
            **trajectory_metrics,
            **stationary_metrics,
            **journey_metrics,
        },
    )


def _draw_player_on_pitch(canvas: np.ndarray, player: BirdseyePlayerPosition) -> None:
    color = TEAM_COLORS.get(player.team, (200, 200, 200))
    center = tuple(int(round(v)) for v in player.render_position)
    radius = 10 if player.jersey_number is not None else 9

    if player.raw_render_position is not None:
        raw_center = tuple(int(round(v)) for v in player.raw_render_position)
        if raw_center != center:
            cv2.line(canvas, raw_center, center, (220, 220, 220), 1)
            cv2.circle(canvas, raw_center, 4, (245, 245, 245), 1)

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


def _build_journey_activity_lookup(diagnostics: Dict[str, Any]) -> Dict[int, List[str]]:
    journeys = diagnostics.get("journeys") if isinstance(diagnostics, dict) else None
    if not isinstance(journeys, list):
        return {}

    activity_by_frame: Dict[int, List[str]] = {}
    for journey in journeys:
        if not isinstance(journey, dict):
            continue
        player_id = journey.get("player_id")
        start_frame = journey.get("start_frame")
        end_frame = journey.get("end_frame")
        if not isinstance(player_id, str) or not isinstance(start_frame, int) or not isinstance(end_frame, int):
            continue

        for frame_idx in range(start_frame, end_frame + 1):
            active_players = activity_by_frame.setdefault(frame_idx, [])
            if player_id not in active_players:
                active_players.append(player_id)

    return activity_by_frame


def _player_debug_label(player: BirdseyePlayerPosition) -> str:
    label_parts = [f"trk {player.track_id}"]
    if player.jersey_number is not None:
        label_parts.insert(0, f"#{player.jersey_number}")
    if player.is_estimated:
        label_parts.append("est")
    return " ".join(label_parts)


def _draw_outlined_rectangle(
    canvas: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), thickness + 2)
    cv2.rectangle(canvas, top_left, bottom_right, color, thickness)


def _draw_player_overlay_on_frame(
    canvas: np.ndarray,
    player: BirdseyePlayerPosition,
    journey_active: bool = False,
) -> None:
    color = TEAM_COLORS.get(player.team, (200, 200, 200))
    x1, y1, x2, y2 = [int(round(v)) for v in player.image_bbox]
    _draw_outlined_rectangle(canvas, (x1, y1), (x2, y2), color, 1 if player.is_estimated else 2)

    if player.raw_image_anchor is not None:
        raw_anchor_x, raw_anchor_y = [int(round(v)) for v in player.raw_image_anchor]
        cv2.circle(canvas, (raw_anchor_x, raw_anchor_y), 4, (255, 255, 255), 1)

    anchor_x, anchor_y = [int(round(v)) for v in player.image_anchor]
    if player.raw_image_anchor is not None and (anchor_x, anchor_y) != (raw_anchor_x, raw_anchor_y):
        cv2.line(canvas, (raw_anchor_x, raw_anchor_y), (anchor_x, anchor_y), (255, 255, 255), 1)
    cv2.circle(canvas, (anchor_x, anchor_y), 6, (255, 255, 255), -1)
    cv2.circle(canvas, (anchor_x, anchor_y), 4, color, -1)

    label_y = y1 - 8 if y1 > 18 else y1 + 16
    _draw_text_with_bg(canvas, _player_debug_label(player), (x1, label_y), 0.42, (255, 255, 255), 1)
    is_journey_modified = False
    if player.raw_court_position is not None:
        is_journey_modified = hypot(
            player.court_position[0] - player.raw_court_position[0],
            player.court_position[1] - player.raw_court_position[1],
        ) > 1e-6
    if journey_active or is_journey_modified:
        journey_label_y = min(canvas.shape[0] - 8, y2 + 16)
        _draw_text_with_bg(canvas, "journey", (x1, journey_label_y), 0.42, (0, 215, 255), 1)


def _draw_ball_overlay_on_frame(canvas: np.ndarray, ball: BirdseyeBallFrame) -> None:
    state = ball.state.value if isinstance(ball.state, BallState) else str(ball.state)
    if state == BallState.UNKNOWN.value:
        return

    color = BALL_REAL_COLOR if state == BallState.REAL.value else BALL_INTERP_COLOR

    if ball.image_bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in ball.image_bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_y = y1 - 8 if y1 > 18 else y1 + 16
        _draw_text_with_bg(canvas, f"ball {state}", (x1, label_y), 0.42, color, 1)

    if ball.image_position is None:
        return

    center_x, center_y = [int(round(v)) for v in ball.image_position]
    if state == BallState.REAL.value:
        cv2.circle(canvas, (center_x, center_y), 5, color, -1)
        cv2.circle(canvas, (center_x, center_y), 5, (0, 0, 0), 1)
        return

    cv2.circle(canvas, (center_x, center_y), 7, color, 2)
    cv2.line(canvas, (center_x - 5, center_y), (center_x + 5, center_y), color, 2)
    cv2.line(canvas, (center_x, center_y - 5), (center_x, center_y + 5), color, 2)
    _draw_text_with_bg(canvas, f"ball {state}", (center_x + 8, center_y - 8), 0.42, color, 1)


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
                for player in projection.players:
                    _draw_player_overlay_on_frame(overlay, player)
                _draw_ball_overlay_on_frame(overlay, projection.ball)

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
            video_path=video_path,
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