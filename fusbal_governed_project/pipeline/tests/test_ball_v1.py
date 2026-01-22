# PROV: FUSBAL.PIPELINE.TESTS.BALL_V1.01
# REQ: SYS-ARCH-15, FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001
# WHY: Guard trust-first ball detection/tracking behavior (no hallucination, explicit missing).

from __future__ import annotations

from fusbal_pipeline.ball.detections import DetectionGatingConfig, emit_ball_detections_v1
from fusbal_pipeline.ball.tracker import BallTracker, compute_ball_quality_metrics_v1
from fusbal_pipeline.contract import validate_track_record_v1
from fusbal_pipeline.diagnostics_keys import (
    BALL_UNKNOWN_FRAME_UNAVAILABLE,
    MISSING_REASON,
    UNKNOWN_REASON,
)


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
    assert len(dets) == 1
    assert dets[0]["pos_state"] == "missing"
    assert dets[0].get("diagnostics", {}).get(MISSING_REASON) == "low_confidence"
    assert diag["gating_reason"] == "low_confidence_suppressed"


def test_emit_ball_detections_v1_emits_unknown_when_unevaluable() -> None:
    dets, diag = emit_ball_detections_v1(
        frame_index=7,
        t_ms=70,
        source="test",
        candidates=[],
        unevaluable_reason=BALL_UNKNOWN_FRAME_UNAVAILABLE,
    )
    assert len(dets) == 1
    assert dets[0]["pos_state"] == "unknown"
    assert dets[0].get("diagnostics", {}).get(UNKNOWN_REASON) == BALL_UNKNOWN_FRAME_UNAVAILABLE
    assert validate_track_record_v1(dets[0]) == []
    assert diag["frame_index"] == 7


def test_emit_ball_detections_v1_emits_missing_when_no_candidates() -> None:
    dets, diag = emit_ball_detections_v1(
        frame_index=0,
        t_ms=0,
        source="test",
        candidates=[],
    )
    assert len(dets) == 1
    assert dets[0]["pos_state"] == "missing"
    assert dets[0].get("diagnostics", {}).get(MISSING_REASON) == "detector_missing"
    assert validate_track_record_v1(dets[0]) == []
    assert diag["gating_reason"] == "detector_missing"


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


def test_ball_tracker_emits_unknown_when_unevaluable() -> None:
    tracker = BallTracker(source="test")
    rec = tracker.update(
        t_ms=0,
        frame_index=0,
        detections=[],
        unevaluable_reason=BALL_UNKNOWN_FRAME_UNAVAILABLE,
    )
    assert rec["pos_state"] == "unknown"
    assert rec.get("diagnostics", {}).get(UNKNOWN_REASON) == BALL_UNKNOWN_FRAME_UNAVAILABLE
    assert validate_track_record_v1(rec) == []


def test_ball_tracker_missing_reason_mapping_and_segment_increments() -> None:
    tracker = BallTracker(source="test")
    # First present starts segment_0001.
    rec0 = tracker.update(
        t_ms=0,
        frame_index=0,
        detections=[
            {
                "schema_version": 1,
                "t_ms": 0,
                "entity_type": "ball",
                "entity_id": "ball_det_000000_00",
                "track_id": "ball_det_000000_00",
                "source": "test",
                "frame": "image_px",
                "pos_state": "present",
                "bbox_xyxy_px": [0, 0, 10, 10],
                "confidence": 0.9,
            }
        ],
    )
    assert rec0["pos_state"] == "present"
    assert rec0["segment_id"] == "ball_seg_0001"

    # Low-confidence present candidate (below tracker threshold) yields missing=low_confidence.
    rec1 = tracker.update(
        t_ms=33,
        frame_index=1,
        detections=[
            {
                "schema_version": 1,
                "t_ms": 33,
                "entity_type": "ball",
                "entity_id": "ball_det_000001_00",
                "track_id": "ball_det_000001_00",
                "source": "test",
                "frame": "image_px",
                "pos_state": "present",
                "bbox_xyxy_px": [2, 0, 12, 10],
                "confidence": 0.59,
            }
        ],
    )
    assert rec1["pos_state"] == "missing"
    assert rec1.get("diagnostics", {}).get(MISSING_REASON) == "low_confidence"
    assert rec1.get("break_reason") == "detector_missing"
    assert validate_track_record_v1(rec1) == []

    # When the ball returns, start a new segment deterministically.
    rec2 = tracker.update(
        t_ms=66,
        frame_index=2,
        detections=[
            {
                "schema_version": 1,
                "t_ms": 66,
                "entity_type": "ball",
                "entity_id": "ball_det_000002_00",
                "track_id": "ball_det_000002_00",
                "source": "test",
                "frame": "image_px",
                "pos_state": "present",
                "bbox_xyxy_px": [4, 0, 14, 10],
                "confidence": 0.9,
            }
        ],
    )
    assert rec2["pos_state"] == "present"
    assert rec2["segment_id"] == "ball_seg_0002"
    assert validate_track_record_v1(rec2) == []


def test_compute_ball_quality_metrics_v1_is_deterministic() -> None:
    tracker = BallTracker(source="test")
    tracks = [
        tracker.update(
            t_ms=0,
            frame_index=0,
            detections=[
                {
                    "schema_version": 1,
                    "t_ms": 0,
                    "entity_type": "ball",
                    "entity_id": "ball_det_000000_00",
                    "track_id": "ball_det_000000_00",
                    "source": "test",
                    "frame": "image_px",
                    "pos_state": "present",
                    "bbox_xyxy_px": [0, 0, 10, 10],
                    "confidence": 0.9,
                }
            ],
        ),
        tracker.update(t_ms=33, frame_index=1, detections=[]),
        tracker.update(
            t_ms=66,
            frame_index=2,
            detections=[],
            unevaluable_reason=BALL_UNKNOWN_FRAME_UNAVAILABLE,
        ),
    ]

    metrics = compute_ball_quality_metrics_v1(tracks=tracks, cfg=tracker.cfg)
    assert metrics["schema_version"] == 1
    assert metrics["total_records"] == 3
    assert metrics["pos_state_counts"]["present"] == 1
    assert metrics["pos_state_counts"]["missing"] == 1
    assert metrics["pos_state_counts"]["unknown"] == 1
    assert metrics["missing_runs"] == 1
    assert metrics["unknown_runs"] == 1
