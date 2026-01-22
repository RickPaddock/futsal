# PROV: FUSBAL.PIPELINE.BALL.TRACKER.01
# REQ: FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Fuse per-frame ball detections into a trust-first track with explicit missing records.

from __future__ import annotations

from dataclasses import dataclass, field

from ..contract import TrackRecordV1
from ..diagnostics_keys import (
    FRAME_INDEX,
    MISSING_REASON,
    JUMP_PX,
    UNKNOWN_REASON,
    MISSING_DETECTOR,
    MISSING_LOW_CONF,
    MISSING_JUMP_REJECTED,
    BREAK_DETECTOR_MISSING,
    BALL_UNKNOWN_DETECTOR_ERROR,
    BALL_UNKNOWN_FRAME_UNAVAILABLE,
)
from .track_types import BallTrackConfig, MissingReason

_BALL_UNKNOWN_REASON_VOCAB: tuple[str, ...] = (
    BALL_UNKNOWN_FRAME_UNAVAILABLE,
    BALL_UNKNOWN_DETECTOR_ERROR,
)

_BALL_MISSING_TO_BREAK_REASON: dict[MissingReason, str] = {
    "detector_missing": BREAK_DETECTOR_MISSING,
    "low_confidence": BREAK_DETECTOR_MISSING,
    "jump_rejected": BREAK_DETECTOR_MISSING,
}


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
    _has_seen_present: bool = False
    _segment_counter: int = 1
    _last_pos_state: str | None = None

    def update(
        self,
        *,
        t_ms: int,
        frame_index: int,
        detections: list[TrackRecordV1],
        unevaluable_reason: str | None = None,
    ) -> TrackRecordV1:
        """Update the ball track from per-frame detection-style records (frame=image_px).

        Output is always exactly one V1 track record per frame (present/missing/unknown).
        """

        if unevaluable_reason is not None:
            reason = str(unevaluable_reason).strip() or BALL_UNKNOWN_DETECTOR_ERROR
            if reason not in _BALL_UNKNOWN_REASON_VOCAB:
                reason = BALL_UNKNOWN_DETECTOR_ERROR
            rec = self._emit_unknown(t_ms=t_ms, frame_index=frame_index, reason=reason)
            self._last_pos_state = "unknown"
            return rec

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
            rec = self._emit_missing(t_ms=t_ms, frame_index=frame_index, reason="detector_missing")
            self._last_pos_state = "missing"
            return rec

        bbox = best.get("bbox_xyxy_px")
        if not isinstance(bbox, list) or len(bbox) != 4:
            rec = self._emit_missing(t_ms=t_ms, frame_index=frame_index, reason="detector_missing")
            self._last_pos_state = "missing"
            return rec

        conf = float(best.get("confidence") or 0.0)
        if conf < self.cfg.min_detection_confidence:
            rec = self._emit_missing(t_ms=t_ms, frame_index=frame_index, reason="low_confidence")
            self._last_pos_state = "missing"
            return rec

        center = _bbox_center_xy([int(x) for x in bbox])
        if self._last_center_xy is not None:
            jump_px = _dist(self._last_center_xy, center)
            if jump_px > self.cfg.max_center_jump_px:
                rec = self._emit_missing(
                    t_ms=t_ms, frame_index=frame_index, reason="jump_rejected", jump_px=jump_px
                )
                self._last_pos_state = "missing"
                return rec

        if self._has_seen_present and self._last_pos_state in ("missing", "unknown"):
            self._segment_counter += 1
            self.segment_id = f"ball_seg_{self._segment_counter:04d}"

        quality = conf
        self._last_center_xy = center
        self._last_confidence = conf
        self._has_seen_present = True

        diagnostics: dict[str, object] = {
            FRAME_INDEX: int(frame_index),
            MISSING_REASON: None,
            JUMP_PX: 0.0,
        }
        if isinstance(best.get("diagnostics"), dict):
            # Merge but keep our stable keys present.
            diagnostics.update(dict(best["diagnostics"]))
            diagnostics.setdefault(FRAME_INDEX, int(frame_index))

        rec: TrackRecordV1 = {
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
        self._last_pos_state = "present"
        return rec

    def _emit_missing(
        self,
        *,
        t_ms: int,
        frame_index: int,
        reason: MissingReason,
        jump_px: float | None = None,
    ) -> TrackRecordV1:
        diagnostics: dict[str, object] = {
            FRAME_INDEX: int(frame_index),
            MISSING_REASON: str(reason),
            JUMP_PX: float(jump_px) if jump_px is not None else 0.0,
        }
        break_reason = _BALL_MISSING_TO_BREAK_REASON.get(reason, BREAK_DETECTOR_MISSING)
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
            "break_reason": break_reason,
            "diagnostics": diagnostics,
        }

    def _emit_unknown(
        self,
        *,
        t_ms: int,
        frame_index: int,
        reason: str,
    ) -> TrackRecordV1:
        diagnostics: dict[str, object] = {
            FRAME_INDEX: int(frame_index),
            UNKNOWN_REASON: str(reason),
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
            "pos_state": "unknown",
            "confidence": 0.0,
            "quality": 0.0,
            "diagnostics": diagnostics,
        }


def compute_ball_quality_metrics_v1(
    *,
    tracks: list[TrackRecordV1],
    cfg: BallTrackConfig,
) -> dict[str, object]:
    """Compute deterministic, aggregated ball tracking quality metrics for diagnostics/quality_summary.json."""

    # Filter to tracker-style ball records (not detection-style records).
    ball = [
        r
        for r in tracks
        if r.get("entity_type") == "ball"
        and r.get("frame") == "image_px"
        and isinstance(r.get("t_ms"), int)
        and "quality" in r
    ]
    ball.sort(key=lambda r: (int(r.get("t_ms", 0)), int(r.get("diagnostics", {}).get(FRAME_INDEX, 0)) if isinstance(r.get("diagnostics"), dict) else 0))

    total = len(ball)
    counts = {"present": 0, "missing": 0, "unknown": 0}
    missing_reason_counts: dict[str, int] = {
        MISSING_DETECTOR: 0,
        MISSING_LOW_CONF: 0,
        MISSING_JUMP_REJECTED: 0,
    }
    present_conf: list[float] = []

    missing_runs = 0
    unknown_runs = 0
    current_missing = 0
    current_unknown = 0
    max_missing = 0
    max_unknown = 0

    for rec in ball:
        ps = rec.get("pos_state")
        if ps in counts:
            counts[str(ps)] += 1
        diag = rec.get("diagnostics") if isinstance(rec.get("diagnostics"), dict) else {}
        if ps == "present":
            current_missing = 0
            current_unknown = 0
            conf = rec.get("confidence")
            if isinstance(conf, (int, float)) and not isinstance(conf, bool):
                v = float(conf)
                if v < 0:
                    v = 0.0
                if v > 1:
                    v = 1.0
                present_conf.append(v)
        elif ps == "missing":
            if current_missing == 0:
                missing_runs += 1
            current_missing += 1
            max_missing = max(max_missing, current_missing)
            current_unknown = 0
            mr = diag.get(MISSING_REASON)
            if isinstance(mr, str) and mr in missing_reason_counts:
                missing_reason_counts[mr] += 1
        elif ps == "unknown":
            if current_unknown == 0:
                unknown_runs += 1
            current_unknown += 1
            max_unknown = max(max_unknown, current_unknown)
            current_missing = 0

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0}
        return {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(sum(values) / len(values)),
        }

    return {
        "schema_version": 1,
        "total_records": int(total),
        "pos_state_counts": {k: int(v) for k, v in counts.items()},
        "missing_reason_counts": {k: int(v) for k, v in missing_reason_counts.items()},
        "missing_runs": int(missing_runs),
        "unknown_runs": int(unknown_runs),
        "max_missing_run_records": int(max_missing),
        "max_unknown_run_records": int(max_unknown),
        "present_confidence": stats(present_conf),
        "thresholds": {
            "min_detection_confidence": float(cfg.min_detection_confidence),
            "max_center_jump_px": float(cfg.max_center_jump_px),
        },
    }
