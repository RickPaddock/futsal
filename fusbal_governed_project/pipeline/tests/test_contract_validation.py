# PROV: FUSBAL.PIPELINE.TESTS.CONTRACT_VALIDATION.01
# REQ: SYS-ARCH-15, FUSBAL-V1-DATA-001
# WHY: Guard core contract validation behavior (bounds, required fields, trust-first errors).

from __future__ import annotations

from fusbal_pipeline.contract import validate_event_record_v1, validate_track_record_v1


def test_track_confidence_out_of_range_reports_value() -> None:
    obj = {
        "schema_version": 1,
        "t_ms": 0,
        "entity_type": "player",
        "entity_id": "p1",
        "track_id": "t1",
        "source": "demo",
        "frame": "pitch",
        "pos_state": "missing",
        "confidence": 1.5,
    }
    errs = validate_track_record_v1(obj)
    assert any("track.confidence" in e and "1.5" in e for e in errs)


def test_track_wgs84_bounds_enforced() -> None:
    obj = {
        "schema_version": 1,
        "t_ms": 0,
        "entity_type": "player",
        "entity_id": "p1",
        "track_id": "t1",
        "source": "demo",
        "frame": "wgs84",
        "pos_state": "present",
        "lat": 91.0,
        "lon": -181.0,
    }
    errs = validate_track_record_v1(obj)
    assert any("track.lat" in e and "91.0" in e for e in errs)
    assert any("track.lon" in e and "-181.0" in e for e in errs)


def test_event_confidence_bounds_enforced() -> None:
    obj = {
        "schema_version": 1,
        "t_ms": 0,
        "event_type": "goal",
        "event_state": "candidate",
        "confidence": -0.1,
        "evidence": [
            {
                "artifact_id": "tracks_jsonl",
                "time_range_ms": {"start_ms": 0, "end_ms": 0},
            }
        ],
    }
    errs = validate_event_record_v1(obj)
    assert any("event.confidence" in e and "-0.1" in e for e in errs)


def test_event_evidence_requires_exactly_one_range() -> None:
    obj = {
        "schema_version": 1,
        "t_ms": 0,
        "event_type": "shot",
        "event_state": "candidate",
        "confidence": 0.5,
        "evidence": [
            {
                "artifact_id": "tracks_jsonl",
                "time_range_ms": {"start_ms": 0, "end_ms": 1},
                "frame_range": {"start_frame": 0, "end_frame": 1},
            }
        ],
    }
    errs = validate_event_record_v1(obj)
    assert any("exactly one of time_range_ms or frame_range" in e for e in errs)


def test_track_missing_requires_break_reason() -> None:
    obj = {
        "schema_version": 1,
        "t_ms": 0,
        "entity_type": "player",
        "entity_id": "p1",
        "track_id": "t1",
        "source": "demo",
        "frame": "pitch",
        "pos_state": "missing",
        "segment_id": "seg_0001",
    }
    errs = validate_track_record_v1(obj)
    assert any("break_reason is required" in e for e in errs)


def test_track_break_reason_requires_segment_id() -> None:
    obj = {
        "schema_version": 1,
        "t_ms": 0,
        "entity_type": "player",
        "entity_id": "p1",
        "track_id": "t1",
        "source": "demo",
        "frame": "pitch",
        "pos_state": "missing",
        "break_reason": "occlusion",
    }
    errs = validate_track_record_v1(obj)
    assert any("segment_id is required" in e for e in errs)


def test_track_break_reason_only_allowed_when_missing() -> None:
    obj = {
        "schema_version": 1,
        "t_ms": 0,
        "entity_type": "player",
        "entity_id": "p1",
        "track_id": "t1",
        "source": "demo",
        "frame": "image_px",
        "pos_state": "present",
        "bbox_xyxy_px": [0, 0, 10, 10],
        "break_reason": "manual_reset",
        "segment_id": "seg_0001",
    }
    errs = validate_track_record_v1(obj)
    assert any("only allowed when track.pos_state=missing" in e for e in errs)


def test_ball_missing_requires_missing_reason_and_detector_missing_break_reason() -> None:
    obj = {
        "schema_version": 1,
        "t_ms": 0,
        "entity_type": "ball",
        "entity_id": "ball_trk_0001",
        "track_id": "ball_trk_0001",
        "segment_id": "ball_seg_0001",
        "source": "demo",
        "frame": "image_px",
        "pos_state": "missing",
        "break_reason": "detector_missing",
        "confidence": 0.0,
        "quality": 0.0,
        "diagnostics": {"frame_index": 0, "jump_px": 0.0},
    }
    errs = validate_track_record_v1(obj)
    assert any("diagnostics.missing_reason" in e for e in errs)


def test_ball_unknown_requires_allowed_unknown_reason() -> None:
    obj = {
        "schema_version": 1,
        "t_ms": 0,
        "entity_type": "ball",
        "entity_id": "ball_trk_0001",
        "track_id": "ball_trk_0001",
        "segment_id": "ball_seg_0001",
        "source": "demo",
        "frame": "image_px",
        "pos_state": "unknown",
        "confidence": 0.0,
        "quality": 0.0,
        "diagnostics": {"frame_index": 0, "unknown_reason": "not_allowed"},
    }
    errs = validate_track_record_v1(obj)
    assert any("diagnostics.unknown_reason" in e for e in errs)


def test_ball_present_requires_frame_index_in_diagnostics() -> None:
    obj = {
        "schema_version": 1,
        "t_ms": 0,
        "entity_type": "ball",
        "entity_id": "ball_trk_0001",
        "track_id": "ball_trk_0001",
        "segment_id": "ball_seg_0001",
        "source": "demo",
        "frame": "image_px",
        "pos_state": "present",
        "bbox_xyxy_px": [0, 0, 10, 10],
        "confidence": 0.9,
        "quality": 0.9,
        "diagnostics": {},
    }
    errs = validate_track_record_v1(obj)
    assert any("track.diagnostics.frame_index" in e for e in errs)
