"""Lightweight per-track image-space position stabilization."""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from math import hypot
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np


EMA_ALPHA = 0.2
LATERAL_CLAMP_RATIO = 0.5
MIN_LATERAL_CLAMP_PX = 1.0
INITIAL_POSITION_VARIANCE = 25.0
INITIAL_VELOCITY_VARIANCE = 100.0
PROCESS_NOISE = 4.0
MEASUREMENT_NOISE = 16.0


def foot_point_from_bbox(bbox: List[float]) -> List[float]:
    """Return the bottom-center foot point for a bbox."""
    x1, _, x2, y2 = [float(value) for value in bbox]
    return [float((x1 + x2) / 2.0), float(y2)]


def _bbox_scale_ratios(curr_bbox: List[float], prev_bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in curr_bbox]
    px1, py1, px2, py2 = [float(value) for value in prev_bbox]

    curr_h = y2 - y1
    prev_h = py2 - py1
    curr_w = x2 - x1
    prev_w = px2 - px1

    if prev_h == 0.0 or prev_w == 0.0:
        return 1.0, 1.0

    return float(curr_h / prev_h), float(curr_w / prev_w)


def is_bad_bbox(curr_bbox: List[float], prev_bbox: Optional[List[float]]) -> bool:
    """Reject abrupt bbox shrinkage that usually indicates occlusion or a partial box."""
    if prev_bbox is None:
        return False

    height_ratio, width_ratio = _bbox_scale_ratios(curr_bbox, prev_bbox)
    if height_ratio < 0.6 or width_ratio < 0.6:
        return True

    return False


class _ConstantVelocityKalmanFilter:
    """Small constant-velocity Kalman filter over image-space foot points."""

    def __init__(self, initial_position: List[float]) -> None:
        self._state = np.asarray([initial_position[0], initial_position[1], 0.0, 0.0], dtype=float)
        self._covariance = np.diag(
            [
                INITIAL_POSITION_VARIANCE,
                INITIAL_POSITION_VARIANCE,
                INITIAL_VELOCITY_VARIANCE,
                INITIAL_VELOCITY_VARIANCE,
            ]
        ).astype(float)
        self._measurement_matrix = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self._measurement_noise = np.eye(2, dtype=float) * MEASUREMENT_NOISE

    @property
    def velocity(self) -> List[float]:
        return [float(self._state[2]), float(self._state[3])]

    def _transition_matrix(self, dt: float) -> np.ndarray:
        return np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _process_noise(self, dt: float) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return PROCESS_NOISE * np.asarray(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=float,
        )

    def _predict(self, dt: float) -> None:
        transition = self._transition_matrix(dt)
        self._state = transition @ self._state
        self._covariance = transition @ self._covariance @ transition.T + self._process_noise(dt)

    def predict_only(self, dt: float) -> List[float]:
        self._predict(dt)
        return [float(self._state[0]), float(self._state[1])]

    def update(self, measurement: List[float], dt: float) -> List[float]:
        self._predict(dt)
        measurement_vector = np.asarray(measurement, dtype=float)
        innovation = measurement_vector - self._measurement_matrix @ self._state
        innovation_covariance = (
            self._measurement_matrix @ self._covariance @ self._measurement_matrix.T
            + self._measurement_noise
        )
        kalman_gain = self._covariance @ self._measurement_matrix.T @ np.linalg.inv(innovation_covariance)
        self._state = self._state + kalman_gain @ innovation
        identity = np.eye(4, dtype=float)
        self._covariance = (identity - kalman_gain @ self._measurement_matrix) @ self._covariance
        return [float(self._state[0]), float(self._state[1])]


@dataclass
class StabilizedPositionResult:
    raw_point: List[float]
    stabilized_point: List[float]
    prediction_only: bool
    bbox_rejected: bool
    lateral_clamped: bool


@dataclass
class _TrackState:
    filter: _ConstantVelocityKalmanFilter
    last_frame_idx: int
    last_output: List[float]
    previous_bbox: Optional[List[float]]


class PositionStabilizer:
    """Maintain per-track image-space stabilization state."""

    def __init__(self, logger: Optional[Logger] = None) -> None:
        self._logger = logger
        self._states: Dict[Hashable, _TrackState] = {}
        self._metrics: Dict[str, float] = {
            "position_stabilizer_tracks_initialized": 0.0,
            "position_stabilizer_bbox_rejections": 0.0,
            "position_stabilizer_prediction_only_frames": 0.0,
            "position_stabilizer_ema_smoothed_frames": 0.0,
            "position_stabilizer_lateral_clamps": 0.0,
        }

    def diagnostics(self) -> Dict[str, float]:
        return dict(self._metrics)

    def stabilize(
        self,
        track_key: Hashable,
        frame_idx: int,
        bbox: List[float],
        use_measurement: bool = True,
        debug_label: Optional[str] = None,
    ) -> StabilizedPositionResult:
        raw_point = foot_point_from_bbox(bbox)
        state = self._states.get(track_key)
        if state is None:
            self._states[track_key] = _TrackState(
                filter=_ConstantVelocityKalmanFilter(raw_point),
                last_frame_idx=int(frame_idx),
                last_output=list(raw_point),
                previous_bbox=[float(value) for value in bbox],
            )
            self._metrics["position_stabilizer_tracks_initialized"] += 1.0
            return StabilizedPositionResult(
                raw_point=list(raw_point),
                stabilized_point=list(raw_point),
                prediction_only=False,
                bbox_rejected=False,
                lateral_clamped=False,
            )

        dt = max(1.0, float(frame_idx - state.last_frame_idx))
        bbox_rejected = False
        prediction_only = not use_measurement

        if use_measurement:
            bbox_rejected = is_bad_bbox(bbox, state.previous_bbox)
            if bbox_rejected:
                prediction_only = True
                self._metrics["position_stabilizer_bbox_rejections"] += 1.0
                if self._logger is not None:
                    height_ratio, width_ratio = _bbox_scale_ratios(bbox, state.previous_bbox or bbox)
                    self._logger.debug(
                        f"Position stabilizer rejected bbox for {debug_label or track_key} "
                        f"at frame={frame_idx} (height_ratio={height_ratio:.3f} width_ratio={width_ratio:.3f})"
                    )

        if prediction_only:
            filtered_point = state.filter.predict_only(dt)
            self._metrics["position_stabilizer_prediction_only_frames"] += 1.0
            if self._logger is not None:
                self._logger.debug(
                    f"Position stabilizer using prediction-only for {debug_label or track_key} "
                    f"at frame={frame_idx}"
                )
        else:
            filtered_point = state.filter.update(raw_point, dt)
            state.previous_bbox = [float(value) for value in bbox]

        previous_output = state.last_output
        smoothed_point = [
            float(EMA_ALPHA * filtered_point[0] + (1.0 - EMA_ALPHA) * previous_output[0]),
            float(EMA_ALPHA * filtered_point[1] + (1.0 - EMA_ALPHA) * previous_output[1]),
        ]
        if smoothed_point != filtered_point:
            self._metrics["position_stabilizer_ema_smoothed_frames"] += 1.0

        velocity_x, velocity_y = state.filter.velocity
        speed = max(hypot(velocity_x, velocity_y), abs(smoothed_point[0] - previous_output[0]) / max(dt, 1e-6))
        max_dx = max(MIN_LATERAL_CLAMP_PX, speed * dt * LATERAL_CLAMP_RATIO)
        delta_x = smoothed_point[0] - previous_output[0]
        lateral_clamped = False
        if abs(delta_x) > max_dx:
            smoothed_point[0] = float(previous_output[0] + max_dx * (1.0 if delta_x > 0.0 else -1.0))
            lateral_clamped = True
            self._metrics["position_stabilizer_lateral_clamps"] += 1.0

        state.last_output = list(smoothed_point)
        state.last_frame_idx = int(frame_idx)

        return StabilizedPositionResult(
            raw_point=list(raw_point),
            stabilized_point=list(smoothed_point),
            prediction_only=prediction_only,
            bbox_rejected=bbox_rejected,
            lateral_clamped=lateral_clamped,
        )