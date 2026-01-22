"""fusbal_pipeline.calibration.diagnostics

PROV: FUSBAL.PIPELINE.TASK_CAL_002.SUB_001
REQ: FUSBAL-V1-CAL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
WHY: Provide a stable calibration diagnostics schema and deterministic math helpers for V1.
"""

from __future__ import annotations

import math
from typing import Literal, NotRequired, TypedDict

CalibrationStatus = Literal["success", "fail"]


class CalibrationDiagnosticsV1(TypedDict):
    schema_version: Literal[1]
    status: CalibrationStatus
    rms_reprojection_error_px: float
    inlier_ratio: float
    num_inliers: int
    marking_coverage_score_0_to_1: float
    failure_reason: NotRequired[str]
    notes: NotRequired[str]


def rms(values: list[float]) -> float:
    if not values:
        return float("inf")
    return math.sqrt(sum(v * v for v in values) / float(len(values)))


def euclidean_px(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    dx = p0[0] - p1[0]
    dy = p0[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)

