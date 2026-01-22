# PROV: FUSBAL.PIPELINE.BALL.DETECTIONS.01
# REQ: FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Provide deterministic, trust-first per-frame ball detection records with explicit fields.

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict

from ..contract import TrackRecordV1
from ..diagnostics_keys import (
    FRAME_INDEX,
    GATING_REASON,
    NUM_CANDIDATES,
    NUM_EMITTED,
    UNKNOWN_REASON,
    GATING_NONE,
    GATING_LOW_CONFIDENCE,
    MISSING_REASON,
    MISSING_DETECTOR,
    MISSING_LOW_CONF,
    BALL_UNKNOWN_DETECTOR_ERROR,
    BALL_UNKNOWN_FRAME_UNAVAILABLE,
)


class RawBallDetection(TypedDict):
    bbox_xyxy_px: list[int]
    confidence: float
    diagnostics: NotRequired[dict[str, object]]


class FrameDetectionsDiagnostics(TypedDict):
    frame_index: int
    num_candidates: int
    num_emitted: int
    min_confidence: float
    gating_reason: str


@dataclass(frozen=True)
class DetectionGatingConfig:
    min_confidence: float = 0.6


def _clamp_0_1(value: float) -> float:
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    return float(value)


def _normalize_bbox_xyxy_px(bbox: list[int]) -> list[int] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in bbox):
        return None
    x1, y1, x2, y2 = bbox
    if x2 < x1 or y2 < y1:
        return None
    return [x1, y1, x2, y2]


def emit_ball_detections_v1(
    *,
    frame_index: int,
    t_ms: int,
    source: str,
    candidates: list[RawBallDetection],
    gating: DetectionGatingConfig | None = None,
    unevaluable_reason: str | None = None,
) -> tuple[list[TrackRecordV1], FrameDetectionsDiagnostics]:
    """Convert raw detector candidates into V1 track records (image_px) deterministically.

    Trust-first: candidates below min_confidence are suppressed (not emitted).
    """

    cfg = gating or DetectionGatingConfig()
    if unevaluable_reason is not None and not str(unevaluable_reason).strip():
        unevaluable_reason = BALL_UNKNOWN_DETECTOR_ERROR
    if unevaluable_reason is not None:
        u = str(unevaluable_reason).strip()
        if u not in (BALL_UNKNOWN_FRAME_UNAVAILABLE, BALL_UNKNOWN_DETECTOR_ERROR):
            unevaluable_reason = BALL_UNKNOWN_DETECTOR_ERROR
    normalized: list[tuple[list[int], float, dict[str, object]]] = []
    for cand in candidates:
        bbox = _normalize_bbox_xyxy_px(cand.get("bbox_xyxy_px"))
        conf = cand.get("confidence")
        if bbox is None or not isinstance(conf, (int, float)) or isinstance(conf, bool):
            continue
        diag = cand.get("diagnostics") if isinstance(cand.get("diagnostics"), dict) else {}
        normalized.append((bbox, _clamp_0_1(float(conf)), dict(diag)))

    normalized.sort(key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], -x[1]))

    emitted: list[TrackRecordV1] = []
    for i, (bbox, conf, diag) in enumerate(normalized):
        if conf < cfg.min_confidence:
            continue
        det_id = f"ball_det_{frame_index:06d}_{i:02d}"
        diagnostics = {FRAME_INDEX: int(frame_index), GATING_REASON: GATING_NONE}
        diagnostics.update(diag)
        emitted.append(
            {
                "schema_version": 1,
                "t_ms": int(t_ms),
                "entity_type": "ball",
                "entity_id": det_id,
                "track_id": det_id,
                "source": str(source),
                "frame": "image_px",
                "pos_state": "present",
                "bbox_xyxy_px": bbox,
                "confidence": conf,
                "diagnostics": diagnostics,
            }
        )

    gating_reason: str = GATING_NONE
    missing_reason = MISSING_DETECTOR
    if not normalized:
        gating_reason = "detector_missing"
        missing_reason = MISSING_DETECTOR
    elif normalized and not emitted:
        gating_reason = GATING_LOW_CONFIDENCE
        missing_reason = MISSING_LOW_CONF

    frame_diag: FrameDetectionsDiagnostics = {
        FRAME_INDEX: int(frame_index),
        NUM_CANDIDATES: int(len(normalized)),
        NUM_EMITTED: int(len(emitted)),
        "min_confidence": float(cfg.min_confidence),
        GATING_REASON: str(gating_reason),
    }

    # Trust-first (INT-040): do not represent missing/unknown implicitly by gaps.
    # When nothing is emitted for this frame, emit exactly one explicit state record.
    if emitted:
        return emitted, frame_diag

    state_id = f"ball_det_state_{frame_index:06d}"
    base_diagnostics: dict[str, object] = dict(frame_diag)
    base_diagnostics[MISSING_REASON] = str(missing_reason)
    base_diagnostics.setdefault(GATING_REASON, str(gating_reason))
    if unevaluable_reason is not None:
        base_diagnostics.pop(MISSING_REASON, None)
        base_diagnostics[UNKNOWN_REASON] = str(unevaluable_reason)
        return (
            [
                {
                    "schema_version": 1,
                    "t_ms": int(t_ms),
                    "entity_type": "ball",
                    "entity_id": state_id,
                    "track_id": state_id,
                    "source": str(source),
                    "frame": "image_px",
                    "pos_state": "unknown",
                    "confidence": 0.0,
                    "diagnostics": base_diagnostics,
                }
            ],
            frame_diag,
        )

    return (
        [
            {
                "schema_version": 1,
                "t_ms": int(t_ms),
                "entity_type": "ball",
                "entity_id": state_id,
                "track_id": state_id,
                "segment_id": "ball_det_seg_0001",
                "source": str(source),
                "frame": "image_px",
                "pos_state": "missing",
                "confidence": 0.0,
                "quality": 0.0,
                "break_reason": "detector_missing",
                "diagnostics": base_diagnostics,
            }
        ],
        frame_diag,
    )
