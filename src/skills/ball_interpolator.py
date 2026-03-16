"""
Ball interpolation pass.

Builds a frame-complete ball state timeline from Pass 1 ball detections and can
render a dedicated debug video showing real, interpolated, and unknown ball
states.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from ..core import constants as const
from ..core.data_models import BallDetection, BallInterpolationOutput, BallPosition, Pass1Output
from ..core.schemas import BALL_INTERPOLATION_OUTPUT_SCHEMA
from ..core.types import BallState, InterpolationMethod
from ..utils.file_utils import load_json, save_json
from ..utils.video_io import VideoReader
from ..validation.validator import Validator

logger = logging.getLogger(__name__)

INTERPOLATED_CONFIDENCE = 0.5
FALSE_CAPTURE_MAX_CONFIDENCE = 0.65
FALSE_CAPTURE_MIN_DEVIATION_PX = 80.0
FALSE_CAPTURE_DEVIATION_PER_FRAME = 35.0
FALSE_CAPTURE_STABLE_STREAK_MIN_LENGTH = 3
FALSE_CAPTURE_SHORT_BRANCH_MAX_LENGTH = 2
FALSE_CAPTURE_DETACHED_ISLAND_MAX_LENGTH = 8
FALSE_CAPTURE_LOCAL_SUPPORT_MAX_GAP = 20


def _state_value(state: BallState | str) -> str:
    return state.value if isinstance(state, BallState) else str(state)


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


def _collapse_ball_detections(ball_detections: List[BallDetection]) -> List[BallDetection]:
    """Keep the highest-confidence ball detection for each frame."""
    best_by_frame: Dict[int, BallDetection] = {}

    for detection in sorted(ball_detections, key=lambda det: (det.frame_idx, -det.confidence)):
        current = best_by_frame.get(detection.frame_idx)
        if current is None or detection.confidence > current.confidence:
            best_by_frame[detection.frame_idx] = detection

    return [best_by_frame[frame_idx] for frame_idx in sorted(best_by_frame)]


def _centroid_distance(a: List[float], b: List[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return float((dx * dx + dy * dy) ** 0.5)


def _average_speed(first: BallDetection, second: BallDetection) -> float:
    frame_gap = second.frame_idx - first.frame_idx
    if frame_gap <= 0:
        return float("inf")
    return _centroid_distance(first.centroid, second.centroid) / float(frame_gap)


def _find_support_pair(detections: List[BallDetection], idx: int) -> Optional[Tuple[int, int]]:
    current = detections[idx]
    best_pair: Optional[Tuple[int, int]] = None
    best_span: Optional[int] = None

    for prev_idx in range(idx - 1, -1, -1):
        previous = detections[prev_idx]
        if current.frame_idx - previous.frame_idx > const.MAX_BALL_GAP_FRAMES:
            break

        for next_idx in range(idx + 1, len(detections)):
            following = detections[next_idx]
            if following.frame_idx - current.frame_idx > const.MAX_BALL_GAP_FRAMES:
                break

            if _average_speed(previous, following) > const.BALL_MAX_SPEED_PX_PER_FRAME:
                continue

            span = following.frame_idx - previous.frame_idx
            if best_span is None or span < best_span:
                best_pair = (prev_idx, next_idx)
                best_span = span

    return best_pair


def _stable_streak_bounds(detections: List[BallDetection], idx: int) -> Tuple[int, int]:
    start_idx = idx
    end_idx = idx

    while start_idx > 0:
        previous = detections[start_idx - 1]
        current = detections[start_idx]
        if current.frame_idx - previous.frame_idx != 1:
            break
        if _average_speed(previous, current) > const.BALL_MAX_SPEED_PX_PER_FRAME:
            break
        start_idx -= 1

    while end_idx + 1 < len(detections):
        current = detections[end_idx]
        following = detections[end_idx + 1]
        if following.frame_idx - current.frame_idx != 1:
            break
        if _average_speed(current, following) > const.BALL_MAX_SPEED_PX_PER_FRAME:
            break
        end_idx += 1

    return start_idx, end_idx


def _belongs_to_stable_streak(detections: List[BallDetection], idx: int) -> bool:
    start_idx, end_idx = _stable_streak_bounds(detections, idx)
    return (end_idx - start_idx + 1) >= FALSE_CAPTURE_STABLE_STREAK_MIN_LENGTH


def _is_short_branch_after_stable_streak(detections: List[BallDetection], idx: int) -> bool:
    branch_start_idx, branch_end_idx = _stable_streak_bounds(detections, idx)
    branch_length = branch_end_idx - branch_start_idx + 1

    if branch_length > FALSE_CAPTURE_SHORT_BRANCH_MAX_LENGTH:
        return False
    if branch_start_idx == 0:
        return False

    previous_idx = branch_start_idx - 1
    previous = detections[previous_idx]
    branch_start = detections[branch_start_idx]

    if branch_start.frame_idx - previous.frame_idx != 1:
        return False
    if not _belongs_to_stable_streak(detections, previous_idx):
        return False
    if _average_speed(previous, branch_start) <= const.BALL_MAX_SPEED_PX_PER_FRAME:
        return False

    branch_confidence = max(
        detections[branch_idx].confidence
        for branch_idx in range(branch_start_idx, branch_end_idx + 1)
    )
    return branch_confidence < FALSE_CAPTURE_MAX_CONFIDENCE


def _find_previous_stable_streak_bounds(
    detections: List[BallDetection],
    idx: int,
) -> Optional[Tuple[int, int]]:
    search_idx = idx
    while search_idx >= 0:
        if _belongs_to_stable_streak(detections, search_idx):
            return _stable_streak_bounds(detections, search_idx)
        search_idx -= 1
    return None


def _find_next_stable_streak_bounds(
    detections: List[BallDetection],
    idx: int,
) -> Optional[Tuple[int, int]]:
    search_idx = idx
    while search_idx < len(detections):
        if _belongs_to_stable_streak(detections, search_idx):
            return _stable_streak_bounds(detections, search_idx)
        search_idx += 1
    return None


def _find_adjacent_stable_support_pair(
    detections: List[BallDetection],
    streak_start_idx: int,
    streak_end_idx: int,
) -> Optional[Tuple[int, int]]:
    previous_bounds = _find_previous_stable_streak_bounds(detections, streak_start_idx - 1)
    next_bounds = _find_next_stable_streak_bounds(detections, streak_end_idx + 1)
    if previous_bounds is None or next_bounds is None:
        return None
    return previous_bounds[1], next_bounds[0]


def _has_bidirectional_stable_support(
    detections: List[BallDetection],
    streak_start_idx: int,
    streak_end_idx: int,
) -> bool:
    streak_start = detections[streak_start_idx]
    streak_end = detections[streak_end_idx]

    previous_supported = False
    previous_search_idx = streak_start_idx - 1
    while previous_search_idx >= 0:
        previous_bounds = _find_previous_stable_streak_bounds(detections, previous_search_idx)
        if previous_bounds is None:
            break
        previous_end = detections[previous_bounds[1]]
        previous_gap = streak_start.frame_idx - previous_end.frame_idx
        if previous_gap > FALSE_CAPTURE_LOCAL_SUPPORT_MAX_GAP:
            break
        if previous_gap <= const.MAX_BALL_GAP_FRAMES and _average_speed(
            previous_end,
            streak_start,
        ) <= const.BALL_MAX_SPEED_PX_PER_FRAME:
            previous_supported = True
            break
        previous_search_idx = previous_bounds[0] - 1

    next_supported = False
    next_search_idx = streak_end_idx + 1
    while next_search_idx < len(detections):
        next_bounds = _find_next_stable_streak_bounds(detections, next_search_idx)
        if next_bounds is None:
            break
        next_start = detections[next_bounds[0]]
        next_gap = next_start.frame_idx - streak_end.frame_idx
        if next_gap > FALSE_CAPTURE_LOCAL_SUPPORT_MAX_GAP:
            break
        if next_gap <= const.MAX_BALL_GAP_FRAMES and _average_speed(
            streak_end,
            next_start,
        ) <= const.BALL_MAX_SPEED_PX_PER_FRAME:
            next_supported = True
            break
        next_search_idx = next_bounds[1] + 1

    return previous_supported and next_supported


def _support_pair_metrics(
    detections: List[BallDetection],
    streak_start_idx: int,
    streak_end_idx: int,
    previous_idx: int,
    next_idx: int,
) -> Optional[Tuple[float, float, float, float]]:
    previous = detections[previous_idx]
    streak_start = detections[streak_start_idx]
    streak_end = detections[streak_end_idx]
    following = detections[next_idx]

    support_speed = _average_speed(previous, following)
    if support_speed > const.BALL_MAX_SPEED_PX_PER_FRAME:
        return None

    support_total_gap = following.frame_idx - previous.frame_idx
    if support_total_gap <= 0:
        return None

    deviations: List[float] = []
    for candidate_idx in range(streak_start_idx, streak_end_idx + 1):
        candidate = detections[candidate_idx]
        ratio = (candidate.frame_idx - previous.frame_idx) / float(support_total_gap)
        expected_centroid = [
            previous.centroid[0] + ((following.centroid[0] - previous.centroid[0]) * ratio),
            previous.centroid[1] + ((following.centroid[1] - previous.centroid[1]) * ratio),
        ]
        deviations.append(_centroid_distance(candidate.centroid, expected_centroid))

    if not deviations:
        return None

    nearest_support_gap = min(
        streak_start.frame_idx - previous.frame_idx,
        following.frame_idx - streak_end.frame_idx,
    )
    allowed_deviation = max(
        FALSE_CAPTURE_MIN_DEVIATION_PX,
        FALSE_CAPTURE_DEVIATION_PER_FRAME * float(nearest_support_gap),
    )

    left_edge_speed = _average_speed(previous, streak_start)
    right_edge_speed = _average_speed(streak_end, following)
    return left_edge_speed, right_edge_speed, min(deviations), allowed_deviation


def _find_streak_support_pair(
    detections: List[BallDetection],
    streak_start_idx: int,
    streak_end_idx: int,
) -> Optional[Tuple[int, int]]:
    best_pair: Optional[Tuple[int, int]] = None
    best_speed: Optional[float] = None
    best_span: Optional[int] = None

    streak_start_frame = detections[streak_start_idx].frame_idx
    streak_end_frame = detections[streak_end_idx].frame_idx

    for prev_idx in range(streak_start_idx - 1, -1, -1):
        previous = detections[prev_idx]
        if streak_start_frame - previous.frame_idx > const.MAX_BALL_GAP_FRAMES:
            break
        if not _belongs_to_stable_streak(detections, prev_idx):
            continue

        for next_idx in range(streak_end_idx + 1, len(detections)):
            following = detections[next_idx]
            if following.frame_idx - streak_end_frame > const.MAX_BALL_GAP_FRAMES:
                break
            if not _belongs_to_stable_streak(detections, next_idx):
                continue

            support_speed = _average_speed(previous, following)
            if support_speed > const.BALL_MAX_SPEED_PX_PER_FRAME:
                continue

            support_span = following.frame_idx - previous.frame_idx
            if (
                best_speed is None
                or support_speed < best_speed
                or (abs(support_speed - best_speed) < 1e-9 and (best_span is None or support_span < best_span))
            ):
                best_pair = (prev_idx, next_idx)
                best_speed = support_speed
                best_span = support_span

    return best_pair


def _is_detached_island_between_stable_streaks(detections: List[BallDetection], idx: int) -> bool:
    island_start_idx, island_end_idx = _stable_streak_bounds(detections, idx)
    island_length = island_end_idx - island_start_idx + 1

    if island_length > FALSE_CAPTURE_DETACHED_ISLAND_MAX_LENGTH:
        return False
    support_pair = _find_streak_support_pair(detections, island_start_idx, island_end_idx)
    if support_pair is None:
        return False

    has_bidirectional_support = (
        island_length > FALSE_CAPTURE_SHORT_BRANCH_MAX_LENGTH
        and _has_bidirectional_stable_support(detections, island_start_idx, island_end_idx)
    )

    adjacent_support_pair = _find_adjacent_stable_support_pair(
        detections,
        island_start_idx,
        island_end_idx,
    )
    if adjacent_support_pair is not None:
        adjacent_metrics = _support_pair_metrics(
            detections,
            island_start_idx,
            island_end_idx,
            adjacent_support_pair[0],
            adjacent_support_pair[1],
        )
        if adjacent_metrics is not None:
            _, _, adjacent_min_deviation, adjacent_allowed_deviation = adjacent_metrics
            if adjacent_min_deviation > adjacent_allowed_deviation and not has_bidirectional_support:
                return True

    if has_bidirectional_support:
        return False

    previous_idx, next_idx = support_pair
    island_confidence = max(
        detections[candidate_idx].confidence
        for candidate_idx in range(island_start_idx, island_end_idx + 1)
    )

    metrics = _support_pair_metrics(
        detections,
        island_start_idx,
        island_end_idx,
        previous_idx,
        next_idx,
    )
    if metrics is None:
        return False

    left_edge_speed, right_edge_speed, min_deviation, allowed_deviation = metrics

    if min_deviation <= allowed_deviation:
        return False

    if (
        left_edge_speed <= const.BALL_MAX_SPEED_PX_PER_FRAME
        and right_edge_speed <= const.BALL_MAX_SPEED_PX_PER_FRAME
    ):
        return (
            island_length <= FALSE_CAPTURE_SHORT_BRANCH_MAX_LENGTH
            and island_confidence < FALSE_CAPTURE_MAX_CONFIDENCE
        )

    return True


def _is_false_capture_candidate(detections: List[BallDetection], idx: int) -> bool:
    current = detections[idx]

    if _is_detached_island_between_stable_streaks(detections, idx):
        return True

    if _is_short_branch_after_stable_streak(detections, idx):
        return True

    if _belongs_to_stable_streak(detections, idx):
        return False

    previous = detections[idx - 1] if idx > 0 else None
    following = detections[idx + 1] if idx + 1 < len(detections) else None

    has_implausible_local_edge = False
    if previous is not None and _average_speed(previous, current) > const.BALL_MAX_SPEED_PX_PER_FRAME:
        has_implausible_local_edge = True
    if following is not None and _average_speed(current, following) > const.BALL_MAX_SPEED_PX_PER_FRAME:
        has_implausible_local_edge = True

    if not has_implausible_local_edge and current.confidence >= FALSE_CAPTURE_MAX_CONFIDENCE:
        return False

    support_pair = _find_support_pair(detections, idx)
    if support_pair is None:
        return False

    prev_idx, next_idx = support_pair
    previous = detections[prev_idx]
    following = detections[next_idx]

    total_gap = following.frame_idx - previous.frame_idx
    if total_gap <= 0:
        return False

    ratio = (current.frame_idx - previous.frame_idx) / float(total_gap)
    expected_centroid = [
        previous.centroid[0] + ((following.centroid[0] - previous.centroid[0]) * ratio),
        previous.centroid[1] + ((following.centroid[1] - previous.centroid[1]) * ratio),
    ]
    deviation = _centroid_distance(current.centroid, expected_centroid)
    nearest_support_gap = min(
        current.frame_idx - previous.frame_idx,
        following.frame_idx - current.frame_idx,
    )
    allowed_deviation = max(
        FALSE_CAPTURE_MIN_DEVIATION_PX,
        FALSE_CAPTURE_DEVIATION_PER_FRAME * float(nearest_support_gap),
    )

    if deviation <= allowed_deviation:
        return False

    return bool(has_implausible_local_edge or current.confidence < FALSE_CAPTURE_MAX_CONFIDENCE)


def _filter_false_captures(ball_detections: List[BallDetection]) -> List[BallDetection]:
    """Drop brief false captures that break a plausible ball trajectory."""
    filtered = list(ball_detections)

    while True:
        to_remove: List[int] = []
        for idx in range(len(filtered)):
            if _is_false_capture_candidate(filtered, idx):
                to_remove.append(idx)

        if not to_remove:
            break

        removed_frames = [filtered[idx].frame_idx for idx in to_remove]
        logger.info("Ball interpolation dropped implausible detections at frames: %s", removed_frames)
        filtered = [detection for idx, detection in enumerate(filtered) if idx not in set(to_remove)]

    return filtered


def _interpolate_centroid(
    start_centroid: List[float],
    end_centroid: List[float],
    offset: int,
    gap_size: int,
) -> List[float]:
    ratio = offset / float(gap_size + 1)
    x = start_centroid[0] + ((end_centroid[0] - start_centroid[0]) * ratio)
    y = start_centroid[1] + ((end_centroid[1] - start_centroid[1]) * ratio)
    return [float(x), float(y)]


def _unknown_position(frame_idx: int) -> BallPosition:
    return BallPosition(
        frame_idx=frame_idx,
        state=BallState.UNKNOWN,
        centroid=None,
        bbox=None,
        confidence=0.0,
    )


def _real_position(detection: BallDetection) -> BallPosition:
    return BallPosition(
        frame_idx=detection.frame_idx,
        state=BallState.REAL,
        centroid=list(detection.centroid),
        bbox=list(detection.bbox),
        confidence=float(detection.confidence),
    )


def _gap_entry(
    gap_start: int,
    gap_end: int,
    previous_real_frame: Optional[int],
    next_real_frame: Optional[int],
    fill_mode: str,
) -> Dict[str, Any]:
    return {
        "gap_start": gap_start,
        "gap_end": gap_end,
        "gap_length": max(0, gap_end - gap_start + 1),
        "previous_real_frame": previous_real_frame,
        "next_real_frame": next_real_frame,
        "fill_mode": fill_mode,
    }


def build_ball_interpolation_output(pass1_output: Pass1Output) -> BallInterpolationOutput:
    """Build the frame-complete ball state artifact from Pass 1 output."""
    processed_end = pass1_output.processed_end_frame_exclusive
    if processed_end is None:
        processed_end = pass1_output.total_frames

    if pass1_output.processed_start_frame != 0 or processed_end != pass1_output.total_frames:
        raise ValueError(
            "Ball interpolation requires Pass 1 output covering the full clip "
            f"[0, {pass1_output.total_frames}), got "
            f"[{pass1_output.processed_start_frame}, {processed_end})."
        )

    total_frames = pass1_output.total_frames
    collapsed_detections = _collapse_ball_detections(pass1_output.ball_detections)
    filtered_detections = _filter_false_captures(collapsed_detections)
    detections_by_frame = {detection.frame_idx: detection for detection in filtered_detections}
    ball_positions: Dict[int, BallPosition] = {
        frame_idx: _real_position(detection)
        for frame_idx, detection in detections_by_frame.items()
    }
    interpolated_frames: List[int] = []
    gap_summary: List[Dict[str, Any]] = []

    real_frames = sorted(detections_by_frame)

    if not real_frames:
        for frame_idx in range(total_frames):
            ball_positions[frame_idx] = _unknown_position(frame_idx)
        if total_frames > 0:
            gap_summary.append(
                _gap_entry(
                    gap_start=0,
                    gap_end=total_frames - 1,
                    previous_real_frame=None,
                    next_real_frame=None,
                    fill_mode=BallState.UNKNOWN.value,
                )
            )
        return BallInterpolationOutput(
            ball_positions=[ball_positions[frame_idx] for frame_idx in range(total_frames)],
            interpolation_method=InterpolationMethod.LINEAR,
            total_frames=total_frames,
            interpolated_frames=interpolated_frames,
            gap_summary=gap_summary,
        )

    first_real_frame = real_frames[0]
    if first_real_frame > 0:
        for frame_idx in range(0, first_real_frame):
            ball_positions[frame_idx] = _unknown_position(frame_idx)
        gap_summary.append(
            _gap_entry(
                gap_start=0,
                gap_end=first_real_frame - 1,
                previous_real_frame=None,
                next_real_frame=first_real_frame,
                fill_mode=BallState.UNKNOWN.value,
            )
        )

    for previous_real_frame, next_real_frame in zip(real_frames, real_frames[1:]):
        gap_start = previous_real_frame + 1
        gap_end = next_real_frame - 1
        gap_size = gap_end - gap_start + 1
        if gap_size <= 0:
            continue

        previous_detection = detections_by_frame[previous_real_frame]
        next_detection = detections_by_frame[next_real_frame]

        if gap_size <= const.MAX_BALL_GAP_FRAMES:
            for offset, frame_idx in enumerate(range(gap_start, gap_end + 1), start=1):
                centroid = _interpolate_centroid(
                    previous_detection.centroid,
                    next_detection.centroid,
                    offset,
                    gap_size,
                )
                ball_positions[frame_idx] = BallPosition(
                    frame_idx=frame_idx,
                    state=BallState.INTERPOLATED,
                    centroid=centroid,
                    bbox=None,
                    confidence=INTERPOLATED_CONFIDENCE,
                )
                interpolated_frames.append(frame_idx)

            gap_summary.append(
                _gap_entry(
                    gap_start=gap_start,
                    gap_end=gap_end,
                    previous_real_frame=previous_real_frame,
                    next_real_frame=next_real_frame,
                    fill_mode=BallState.INTERPOLATED.value,
                )
            )
        else:
            for frame_idx in range(gap_start, gap_end + 1):
                ball_positions[frame_idx] = _unknown_position(frame_idx)

            gap_summary.append(
                _gap_entry(
                    gap_start=gap_start,
                    gap_end=gap_end,
                    previous_real_frame=previous_real_frame,
                    next_real_frame=next_real_frame,
                    fill_mode=BallState.UNKNOWN.value,
                )
            )

    last_real_frame = real_frames[-1]
    if last_real_frame < total_frames - 1:
        for frame_idx in range(last_real_frame + 1, total_frames):
            ball_positions[frame_idx] = _unknown_position(frame_idx)
        gap_summary.append(
            _gap_entry(
                gap_start=last_real_frame + 1,
                gap_end=total_frames - 1,
                previous_real_frame=last_real_frame,
                next_real_frame=None,
                fill_mode=BallState.UNKNOWN.value,
            )
        )

    ordered_positions = [ball_positions[frame_idx] for frame_idx in range(total_frames)]

    return BallInterpolationOutput(
        ball_positions=ordered_positions,
        interpolation_method=InterpolationMethod.LINEAR,
        total_frames=total_frames,
        interpolated_frames=interpolated_frames,
        gap_summary=gap_summary,
    )


def _draw_interpolated_marker(
    overlay: np.ndarray,
    centroid: List[float],
    color: Tuple[int, int, int],
) -> None:
    cx, cy = [int(round(v)) for v in centroid]
    cv2.circle(overlay, (cx, cy), 8, color, 2)
    cv2.line(overlay, (cx - 5, cy), (cx + 5, cy), color, 2)
    cv2.line(overlay, (cx, cy - 5), (cx, cy + 5), color, 2)


def _draw_ball_debug_overlay_frame(
    frame: np.ndarray,
    frame_idx: int,
    ball_position: BallPosition,
    gap_context: Optional[Dict[str, Any]],
    rejected_detection: Optional[BallDetection] = None,
) -> np.ndarray:
    overlay = frame.copy()
    state = _state_value(ball_position.state)

    if rejected_detection is not None:
        rx1, ry1, rx2, ry2 = [int(round(v)) for v in rejected_detection.bbox]
        rcx, rcy = [int(round(v)) for v in rejected_detection.centroid]
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        cv2.line(overlay, (rcx - 6, rcy - 6), (rcx + 6, rcy + 6), (0, 0, 255), 2)
        cv2.line(overlay, (rcx - 6, rcy + 6), (rcx + 6, rcy - 6), (0, 0, 255), 2)
        _draw_text_with_bg(
            overlay,
            f"raw_rejected conf={rejected_detection.confidence:.2f}",
            (rx1, min(overlay.shape[0] - 8, ry2 + 18)),
            0.40,
            (0, 0, 255),
            1,
        )

    if state == BallState.REAL.value:
        color = (0, 215, 255)
        if ball_position.bbox is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in ball_position.bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            _draw_text_with_bg(overlay, f"state=real conf={ball_position.confidence:.2f}", (x1, max(18, y1 - 6)), 0.45, color, 1)
        if ball_position.centroid is not None:
            cx, cy = [int(round(v)) for v in ball_position.centroid]
            cv2.circle(overlay, (cx, cy), 5, color, -1)
    elif state == BallState.INTERPOLATED.value and ball_position.centroid is not None:
        color = (255, 255, 0)
        _draw_interpolated_marker(overlay, ball_position.centroid, color)
        cx, cy = [int(round(v)) for v in ball_position.centroid]
        _draw_text_with_bg(overlay, f"state=interpolated conf={ball_position.confidence:.2f}", (max(8, cx + 10), max(18, cy - 10)), 0.42, color, 1)
    elif state == BallState.UNKNOWN.value:
        color = (0, 0, 255)
        frame_h, frame_w = overlay.shape[:2]
        cv2.rectangle(overlay, (4, 4), (frame_w - 5, frame_h - 5), color, 3)
        center_x = frame_w // 2
        center_y = frame_h // 2
        cv2.circle(overlay, (center_x, center_y), 18, color, 3)
        cv2.line(overlay, (center_x - 12, center_y - 12), (center_x + 12, center_y + 12), color, 3)
        cv2.line(overlay, (center_x - 12, center_y + 12), (center_x + 12, center_y - 12), color, 3)
        _draw_text_with_bg(overlay, "state=unknown", (max(12, center_x - 58), max(24, center_y - 28)), 0.55, color, 1)

    lines = [
        "Ball interpolation debug view",
        "real = yellow bbox + solid marker | interpolated = cyan crosshair | unknown = red border + red X",
        f"frame={frame_idx} state={state}",
    ]

    if gap_context is not None:
        lines.append(
            "prev_real="
            f"{gap_context.get('previous_real_frame')} next_real={gap_context.get('next_real_frame')}"
        )
        lines.append(
            f"gap={gap_context.get('gap_length')} fill={gap_context.get('fill_mode')}"
        )
    else:
        lines.append("prev_real=n/a next_real=n/a")
        lines.append("gap=0 fill=real_detection")

    if rejected_detection is not None:
        lines.append(f"rejected_raw_frame={rejected_detection.frame_idx} reason=implausible_jump")

    frame_h, frame_w = overlay.shape[:2]
    top = max(12, int(frame_h * 0.055))
    left = 12
    line_gap = 20
    title_y = top + 14
    first_line_y = title_y + line_gap
    footer_y = first_line_y + (len(lines) * line_gap) + 8
    panel_bottom = min(frame_h - 8, footer_y + 18)

    panel = overlay.copy()
    panel_top = max(6, top - 12)
    cv2.rectangle(panel, (6, panel_top), (frame_w - 6, panel_bottom), (0, 0, 0), -1)
    overlay = cv2.addWeighted(panel, 0.55, overlay, 0.45, 0)

    _draw_text_with_bg(overlay, "BALL INTERPOLATION (DEBUG)", (left, title_y), 0.50, (255, 255, 255), 1)
    for idx, line in enumerate(lines):
        y = first_line_y + idx * line_gap
        _draw_text_with_bg(overlay, f"- {line}", (left + 4, y), 0.46, (255, 255, 255), 1)

    return overlay


def render_ball_interpolation_debug_video_from_artifact(
    video_path: str,
    pass1_output_path: str,
    ball_output_path: str,
    debug_video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    """Render the dedicated debug video for the ball interpolation pass."""
    pass1_output = load_json(Path(pass1_output_path), Pass1Output)
    ball_output = load_json(Path(ball_output_path), BallInterpolationOutput)

    raw_collapsed = _collapse_ball_detections(pass1_output.ball_detections)
    filtered = _filter_false_captures(raw_collapsed)
    filtered_frames = {detection.frame_idx for detection in filtered}
    rejected_by_frame = {
        detection.frame_idx: detection
        for detection in raw_collapsed
        if detection.frame_idx not in filtered_frames
    }
    ball_by_frame = {ball_position.frame_idx: ball_position for ball_position in ball_output.ball_positions}
    gap_by_frame: Dict[int, Dict[str, Any]] = {}
    for gap in ball_output.gap_summary:
        gap_start = int(gap.get("gap_start", -1))
        gap_end = int(gap.get("gap_end", -1))
        if gap_end < gap_start:
            continue
        for frame_idx in range(gap_start, gap_end + 1):
            gap_by_frame[frame_idx] = gap

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
            raise RuntimeError(f"Failed to open ball interpolation debug video writer: {debug_video_path}")

        artifact_start = pass1_output.processed_start_frame
        artifact_end_exclusive = pass1_output.processed_end_frame_exclusive
        if artifact_end_exclusive is None:
            artifact_end_exclusive = pass1_output.total_frames

        render_start = max(start_frame, artifact_start)
        requested_end = artifact_end_exclusive if end_frame is None else end_frame
        render_end_exclusive = min(requested_end, artifact_end_exclusive)

        if render_end_exclusive <= render_start:
            raise ValueError(
                f"Invalid render window: start={render_start}, end={render_end_exclusive}. "
                f"Artifact range is [{artifact_start}, {artifact_end_exclusive})."
            )

        total_render_frames = render_end_exclusive - render_start

        with tqdm(total=total_render_frames, desc="Ball interpolation debug render", unit="frame") as pbar:
            for frame_idx, frame in reader.iter_frames():
                if frame_idx < render_start:
                    continue
                if frame_idx >= render_end_exclusive:
                    break

                ball_position = ball_by_frame.get(frame_idx)
                if ball_position is None:
                    ball_position = _unknown_position(frame_idx)
                debug_frame = _draw_ball_debug_overlay_frame(
                    frame,
                    frame_idx,
                    ball_position,
                    gap_by_frame.get(frame_idx),
                    rejected_detection=rejected_by_frame.get(frame_idx),
                )
                writer.write(debug_frame)
                pbar.update(1)
    finally:
        if writer is not None:
            writer.release()
        reader.close()

    logger.info("Ball interpolation debug video written: %s", debug_video_path)


class BallInterpolator:
    """Pass runner for ball interpolation."""

    def run(
        self,
        pass1_path: Path,
        output_path: Path,
        video_path: Optional[str] = None,
        debug_video_path: Optional[str] = None,
    ) -> BallInterpolationOutput:
        pass1_output = load_json(pass1_path, Pass1Output)
        ball_output = build_ball_interpolation_output(pass1_output)

        validator = Validator()
        validation_result = validator.validate_ball_interpolation(ball_output, pass1_output.total_frames)
        if not validation_result.passed:
            messages = "; ".join(violation.message for violation in validation_result.violations[:5])
            raise ValueError(f"Ball interpolation validation failed: {messages}")

        save_json(ball_output.model_dump(), str(output_path), BALL_INTERPOLATION_OUTPUT_SCHEMA)

        if debug_video_path is not None:
            if video_path is None:
                raise ValueError("video_path is required when rendering the ball interpolation debug video")
            render_ball_interpolation_debug_video_from_artifact(
                video_path=video_path,
                pass1_output_path=str(pass1_path),
                ball_output_path=str(output_path),
                debug_video_path=str(debug_video_path),
            )

        return ball_output


def run_ball_interpolation(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    video_path: Optional[str] = None,
    debug_video_path: Optional[str] = None,
) -> BallInterpolationOutput:
    """Execute the ball interpolation pass from an artifact directory."""
    if output_dir is None:
        output_dir = input_dir

    pass1_path = input_dir / const.PASS1_RAW_JSON
    output_path = output_dir / const.BALL_INTERPOLATION_JSON

    if not pass1_path.exists():
        raise FileNotFoundError(f"pass1_raw.json not found: {pass1_path}")

    if debug_video_path is None and video_path is not None:
        debug_video_path = str(output_dir / const.BALL_INTERPOLATION_DEBUG_VIDEO)

    interpolator = BallInterpolator()
    return interpolator.run(
        pass1_path=pass1_path,
        output_path=output_path,
        video_path=video_path,
        debug_video_path=debug_video_path,
    )