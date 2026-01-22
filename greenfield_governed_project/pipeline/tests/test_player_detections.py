# PROV: FUSBAL.PIPELINE.TESTS.PLAYER_DETECTIONS.01
# REQ: SYS-ARCH-15, FUSBAL-V1-PLAYER-001, FUSBAL-V1-TRUST-001
# WHY: Lock in deterministic, trust-first detection record emission.

from __future__ import annotations

from fusbal_pipeline.player.detections import DetectionGatingConfig, emit_player_detections_v1


def test_emit_player_detections_v1_gates_low_confidence_and_is_deterministic() -> None:
    candidates = [
        {"bbox_xyxy_px": [50, 10, 90, 110], "confidence": 0.49, "diagnostics": {"x": 1}},
        {"bbox_xyxy_px": [10, 10, 40, 110], "confidence": 0.90, "diagnostics": {"x": 2}},
        {"bbox_xyxy_px": [10, 10, 40, 110], "confidence": 0.80, "diagnostics": {"x": 3}},
    ]

    gating = DetectionGatingConfig(min_confidence=0.5)

    out1, diag1 = emit_player_detections_v1(
        frame_index=7,
        t_ms=700,
        source="fixture",
        candidates=list(candidates),
        gating=gating,
    )
    out2, diag2 = emit_player_detections_v1(
        frame_index=7,
        t_ms=700,
        source="fixture",
        candidates=list(candidates),
        gating=gating,
    )

    assert diag1 == diag2
    assert diag1["num_emitted"] == 2

    # Deterministic ordering: bbox sort then higher confidence first.
    assert [r["bbox_xyxy_px"] for r in out1] == [r["bbox_xyxy_px"] for r in out2]
    assert [r["confidence"] for r in out1] == [r["confidence"] for r in out2]

    # Stable minimal fields + stable diagnostics key.
    for rec in out1:
        assert rec["schema_version"] == 1
        assert rec["entity_type"] == "player"
        assert rec["frame"] == "image_px"
        assert rec["pos_state"] == "present"
        assert isinstance(rec.get("confidence"), float)
        assert isinstance(rec.get("diagnostics"), dict)
        assert "gating_reason" in rec["diagnostics"]


def test_emit_player_detections_v1_rejects_invalid_bbox() -> None:
    out, diag = emit_player_detections_v1(
        frame_index=0,
        t_ms=0,
        source="fixture",
        candidates=[{"bbox_xyxy_px": [10, 10, 5, 20], "confidence": 0.9}],
        gating=DetectionGatingConfig(min_confidence=0.0),
    )
    assert out == []
    assert diag["num_candidates"] == 0
