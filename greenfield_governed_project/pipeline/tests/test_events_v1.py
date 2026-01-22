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

