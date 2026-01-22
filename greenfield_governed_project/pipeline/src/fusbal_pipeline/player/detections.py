# PROV: FUSBAL.PIPELINE.PLAYER.DETECTIONS.01
# REQ: FUSBAL-V1-PLAYER-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Provide deterministic, trust-first per-frame player detection records with explicit fields.

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict

from ..contract import TrackRecordV1


class RawPlayerDetection(TypedDict):
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
    min_confidence: float = 0.5


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


def emit_player_detections_v1(
    *,
    frame_index: int,
    t_ms: int,
    source: str,
    candidates: list[RawPlayerDetection],
    gating: DetectionGatingConfig | None = None,
) -> tuple[list[TrackRecordV1], FrameDetectionsDiagnostics]:
    """Convert raw detector candidates into V1 track records (image_px) deterministically.

    Trust-first: candidates below min_confidence are suppressed (not emitted).
    """

    cfg = gating or DetectionGatingConfig()
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
        det_id = f"det_{frame_index:06d}_{i:02d}"
        diagnostics = {"frame_index": frame_index, "gating_reason": "none"}
        diagnostics.update(diag)
        emitted.append(
            {
                "schema_version": 1,
                "t_ms": int(t_ms),
                "entity_type": "player",
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

    frame_diag: FrameDetectionsDiagnostics = {
        "frame_index": int(frame_index),
        "num_candidates": int(len(normalized)),
        "num_emitted": int(len(emitted)),
        "min_confidence": float(cfg.min_confidence),
        "gating_reason": "low_confidence_suppressed" if len(normalized) and not emitted else "none",
    }
    return emitted, frame_diag

