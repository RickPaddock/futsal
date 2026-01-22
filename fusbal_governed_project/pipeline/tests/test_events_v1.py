# PROV: FUSBAL.PIPELINE.TESTS.EVENTS_V1.01
# REQ: SYS-ARCH-15, FUSBAL-V1-EVENT-001, FUSBAL-V1-TRUST-001
# WHY: Guard conservative shots/goals inference and evidence pointer well-formedness.

from __future__ import annotations

from fusbal_pipeline.contract import validate_event_record_v1
from fusbal_pipeline.events.shots_goals import ShotsGoalsConfig, infer_shots_goals_v1


def test_infer_shots_goals_v1_emits_candidate_not_confirmed() -> None:
    tracks = [
        {
            "schema_version": 1,
            "t_ms": 0,
            "entity_type": "ball",
            "entity_id": "ball_trk_0001",
            "track_id": "ball_trk_0001",
            "source": "test",
            "frame": "image_px",
            "pos_state": "present",
            "bbox_xyxy_px": [0, 0, 10, 10],
            "confidence": 0.95,
        },
        {
            "schema_version": 1,
            "t_ms": 100,
            "entity_type": "ball",
            "entity_id": "ball_trk_0001",
            "track_id": "ball_trk_0001",
            "source": "test",
            "frame": "image_px",
            "pos_state": "present",
            "bbox_xyxy_px": [200, 0, 210, 10],
            "confidence": 0.95,
        },
        {
            "schema_version": 1,
            "t_ms": 200,
            "entity_type": "ball",
            "entity_id": "ball_trk_0001",
            "track_id": "ball_trk_0001",
            "source": "test",
            "frame": "image_px",
            "pos_state": "missing",
            "confidence": 0.95,
        },
        {
            "schema_version": 1,
            "t_ms": 1800,
            "entity_type": "ball",
            "entity_id": "ball_trk_0001",
            "track_id": "ball_trk_0001",
            "source": "test",
            "frame": "image_px",
            "pos_state": "missing",
            "confidence": 0.95,
        },
    ]
    events = infer_shots_goals_v1(
        tracks=tracks, cfg=ShotsGoalsConfig(min_shot_speed_px_per_s=500.0, goal_missing_ms=1500)
    )
    assert events
    assert all(e["event_state"] != "confirmed" for e in events)
    for e in events:
        assert validate_event_record_v1(e) == []


def test_infer_shots_goals_v1_emits_no_events_for_stationary_ball() -> None:
    tracks = [
        {
            "schema_version": 1,
            "t_ms": 0,
            "entity_type": "ball",
            "entity_id": "ball_trk_0001",
            "track_id": "ball_trk_0001",
            "segment_id": "ball_seg_0001",
            "source": "test",
            "frame": "image_px",
            "pos_state": "present",
            "bbox_xyxy_px": [0, 0, 10, 10],
            "confidence": 0.95,
            "quality": 0.95,
            "diagnostics": {"frame_index": 0, "missing_reason": None, "jump_px": 0.0},
        },
        {
            "schema_version": 1,
            "t_ms": 200,
            "entity_type": "ball",
            "entity_id": "ball_trk_0001",
            "track_id": "ball_trk_0001",
            "segment_id": "ball_seg_0001",
            "source": "test",
            "frame": "image_px",
            "pos_state": "present",
            "bbox_xyxy_px": [0, 0, 10, 10],
            "confidence": 0.95,
            "quality": 0.95,
            "diagnostics": {"frame_index": 2, "missing_reason": None, "jump_px": 0.0},
        },
    ]
    events = infer_shots_goals_v1(tracks=tracks, cfg=ShotsGoalsConfig(min_shot_speed_px_per_s=10.0))
    assert events == []


def test_infer_shots_goals_v1_emits_no_events_when_dt_is_too_small() -> None:
    tracks = [
        {
            "schema_version": 1,
            "t_ms": 0,
            "entity_type": "ball",
            "entity_id": "ball_trk_0001",
            "track_id": "ball_trk_0001",
            "segment_id": "ball_seg_0001",
            "source": "test",
            "frame": "image_px",
            "pos_state": "present",
            "bbox_xyxy_px": [0, 0, 10, 10],
            "confidence": 0.95,
            "quality": 0.95,
            "diagnostics": {"frame_index": 0, "missing_reason": None, "jump_px": 0.0},
        },
        {
            "schema_version": 1,
            "t_ms": 5,
            "entity_type": "ball",
            "entity_id": "ball_trk_0001",
            "track_id": "ball_trk_0001",
            "segment_id": "ball_seg_0001",
            "source": "test",
            "frame": "image_px",
            "pos_state": "present",
            "bbox_xyxy_px": [500, 0, 510, 10],
            "confidence": 0.95,
            "quality": 0.95,
            "diagnostics": {"frame_index": 1, "missing_reason": None, "jump_px": 0.0},
        },
    ]
    events = infer_shots_goals_v1(
        tracks=tracks,
        cfg=ShotsGoalsConfig(min_dt_ms=10, min_shot_speed_px_per_s=1.0),
    )
    assert events == []


def test_infer_shots_goals_v1_emits_no_events_for_missing_only_sequences() -> None:
    tracks = [
        {
            "schema_version": 1,
            "t_ms": 0,
            "entity_type": "ball",
            "entity_id": "ball_trk_0001",
            "track_id": "ball_trk_0001",
            "segment_id": "ball_seg_0001",
            "source": "test",
            "frame": "image_px",
            "pos_state": "missing",
            "confidence": 0.0,
            "quality": 0.0,
            "break_reason": "detector_missing",
            "diagnostics": {"frame_index": 0, "missing_reason": "detector_missing", "jump_px": 0.0},
        },
        {
            "schema_version": 1,
            "t_ms": 1000,
            "entity_type": "ball",
            "entity_id": "ball_trk_0001",
            "track_id": "ball_trk_0001",
            "segment_id": "ball_seg_0001",
            "source": "test",
            "frame": "image_px",
            "pos_state": "missing",
            "confidence": 0.0,
            "quality": 0.0,
            "break_reason": "detector_missing",
            "diagnostics": {"frame_index": 10, "missing_reason": "detector_missing", "jump_px": 0.0},
        },
    ]
    events = infer_shots_goals_v1(tracks=tracks, cfg=ShotsGoalsConfig(goal_missing_ms=200))
    assert events == []
