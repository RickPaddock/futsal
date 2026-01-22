# PROV: FUSBAL.PIPELINE.TESTS.BALL_V1.01
# REQ: SYS-ARCH-15, FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001
# WHY: Guard trust-first ball detection/tracking behavior (no hallucination, explicit missing).

from __future__ import annotations

from fusbal_pipeline.ball.detections import DetectionGatingConfig, emit_ball_detections_v1
from fusbal_pipeline.ball.tracker import BallTracker
from fusbal_pipeline.contract import validate_track_record_v1


def test_emit_ball_detections_v1_suppresses_low_confidence() -> None:
    dets, diag = emit_ball_detections_v1(
        frame_index=0,
        t_ms=0,
        source="test",
        candidates=[
            {"bbox_xyxy_px": [0, 0, 10, 10], "confidence": 0.2},
            {"bbox_xyxy_px": [10, 10, 12, 12], "confidence": 0.59},
        ],
        gating=DetectionGatingConfig(min_confidence=0.6),
    )
    assert dets == []
    assert diag["gating_reason"] == "low_confidence_suppressed"


def test_ball_tracker_emits_explicit_missing_records() -> None:
    tracker = BallTracker(source="test")

    rec0 = tracker.update(t_ms=0, frame_index=0, detections=[])
    assert rec0["pos_state"] == "missing"
    assert rec0["entity_type"] == "ball"
    assert rec0["track_id"] == "ball_trk_0001"
    assert validate_track_record_v1(rec0) == []

    dets, _diag = emit_ball_detections_v1(
        frame_index=1,
        t_ms=33,
        source="test_det",
        candidates=[{"bbox_xyxy_px": [0, 0, 10, 10], "confidence": 0.9}],
    )
    rec1 = tracker.update(t_ms=33, frame_index=1, detections=dets)
    assert rec1["pos_state"] == "present"
    assert validate_track_record_v1(rec1) == []

    # Large jump is rejected conservatively (prefer missing over wrong).
    dets2, _diag2 = emit_ball_detections_v1(
        frame_index=2,
        t_ms=66,
        source="test_det",
        candidates=[{"bbox_xyxy_px": [1000, 0, 1010, 10], "confidence": 0.95}],
    )
    rec2 = tracker.update(t_ms=66, frame_index=2, detections=dets2)
    assert rec2["pos_state"] == "missing"
    assert rec2.get("diagnostics", {}).get("missing_reason") == "jump_rejected"
    assert validate_track_record_v1(rec2) == []

