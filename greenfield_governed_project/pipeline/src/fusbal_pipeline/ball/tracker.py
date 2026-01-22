# PROV: FUSBAL.PIPELINE.BALL.TRACKER.01
# REQ: FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Fuse per-frame ball detections into a trust-first track with explicit missing records.

from __future__ import annotations

from dataclasses import dataclass, field

from ..contract import TrackRecordV1
from .track_types import BallTrackConfig, MissingReason


def _bbox_center_xy(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


@dataclass
class BallTracker:
    source: str
    cfg: BallTrackConfig = field(default_factory=BallTrackConfig)
    track_id: str = "ball_trk_0001"
    segment_id: str = "ball_seg_0001"

    _last_center_xy: tuple[float, float] | None = None
    _last_confidence: float = 0.0

    def update(
        self,
        *,
        t_ms: int,
        frame_index: int,
        detections: list[TrackRecordV1],
    ) -> TrackRecordV1:
        """Update the ball track from per-frame detection-style records (frame=image_px).

        Output is always exactly one V1 track record per frame (present or missing).
        """

        present: list[TrackRecordV1] = [
            d
            for d in detections
            if d.get("entity_type") == "ball"
            and d.get("frame") == "image_px"
            and d.get("pos_state") == "present"
            and isinstance(d.get("bbox_xyxy_px"), list)
        ]

        best: TrackRecordV1 | None = None
        for d in present:
            conf = d.get("confidence")
            if not isinstance(conf, (int, float)) or isinstance(conf, bool):
                continue
            if best is None or float(conf) > float(best.get("confidence") or 0.0):
                best = d

        if not best:
            return self._emit_missing(t_ms=t_ms, frame_index=frame_index, reason="detector_missing")

        bbox = best.get("bbox_xyxy_px")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return self._emit_missing(t_ms=t_ms, frame_index=frame_index, reason="detector_missing")

        conf = float(best.get("confidence") or 0.0)
        if conf < self.cfg.min_detection_confidence:
            return self._emit_missing(t_ms=t_ms, frame_index=frame_index, reason="low_confidence")

        center = _bbox_center_xy([int(x) for x in bbox])
        if self._last_center_xy is not None:
            jump_px = _dist(self._last_center_xy, center)
            if jump_px > self.cfg.max_center_jump_px:
                return self._emit_missing(
                    t_ms=t_ms, frame_index=frame_index, reason="jump_rejected", jump_px=jump_px
                )

        quality = conf
        self._last_center_xy = center
        self._last_confidence = conf

        diagnostics: dict[str, object] = {
            "frame_index": int(frame_index),
            "missing_reason": None,
            "jump_px": 0.0,
        }
        if isinstance(best.get("diagnostics"), dict):
            # Merge but keep our stable keys present.
            diagnostics.update(dict(best["diagnostics"]))
            diagnostics.setdefault("frame_index", int(frame_index))

        return {
            "schema_version": 1,
            "t_ms": int(t_ms),
            "entity_type": "ball",
            "entity_id": self.track_id,
            "track_id": self.track_id,
            "segment_id": self.segment_id,
            "source": str(self.source),
            "frame": "image_px",
            "pos_state": "present",
            "bbox_xyxy_px": [int(x) for x in bbox],
            "confidence": conf,
            "quality": float(quality),
            "diagnostics": diagnostics,
        }

    def _emit_missing(
        self,
        *,
        t_ms: int,
        frame_index: int,
        reason: MissingReason,
        jump_px: float | None = None,
    ) -> TrackRecordV1:
        diagnostics: dict[str, object] = {
            "frame_index": int(frame_index),
            "missing_reason": str(reason),
            "jump_px": float(jump_px) if jump_px is not None else 0.0,
        }
        return {
            "schema_version": 1,
            "t_ms": int(t_ms),
            "entity_type": "ball",
            "entity_id": self.track_id,
            "track_id": self.track_id,
            "segment_id": self.segment_id,
            "source": str(self.source),
            "frame": "image_px",
            "pos_state": "missing",
            "confidence": float(self._last_confidence),
            "quality": 0.0,
            "break_reason": "detector_missing",
            "diagnostics": diagnostics,
        }

