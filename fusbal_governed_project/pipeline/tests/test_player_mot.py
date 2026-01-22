# PROV: FUSBAL.PIPELINE.TESTS.PLAYER_MOT.01
# REQ: SYS-ARCH-15, FUSBAL-V1-PLAYER-001, FUSBAL-V1-TRUST-001
# WHY: Lock in swap-avoidant behavior (breaks over swaps) and deterministic outputs.

from __future__ import annotations

from fusbal_pipeline.player.detections import DetectionGatingConfig, emit_player_detections_v1
from fusbal_pipeline.player.mot import SwapAvoidantMOT


def test_mot_breaks_on_ambiguous_association_instead_of_swapping() -> None:
    mot = SwapAvoidantMOT(source="fixture")
    gating = DetectionGatingConfig(min_confidence=0.0)

    # Frame 0: create two tracks close to each other.
    dets0, _ = emit_player_detections_v1(
        frame_index=0,
        t_ms=0,
        source="fixture",
        candidates=[
            {"bbox_xyxy_px": [100, 100, 120, 200], "confidence": 0.9},
            {"bbox_xyxy_px": [130, 100, 150, 200], "confidence": 0.9},
        ],
        gating=gating,
    )
    out0 = mot.update(t_ms=0, detections=dets0)
    present0 = [r for r in out0 if r.get("pos_state") == "present"]
    assert len(present0) == 2

    # Frame 1: one detection roughly between both track centers.
    dets1, _ = emit_player_detections_v1(
        frame_index=1,
        t_ms=100,
        source="fixture",
        candidates=[{"bbox_xyxy_px": [115, 100, 135, 200], "confidence": 0.9}],
        gating=gating,
    )
    out1 = mot.update(t_ms=100, detections=dets1)

    # Ambiguity should break both existing tracks (no swap), emitting missing records.
    breaks = [
        r
        for r in out1
        if r.get("pos_state") == "missing" and r.get("break_reason") == "ambiguous_association"
    ]
    assert len(breaks) == 2

    # Deterministic ordering by track_id.
    track_ids = [r.get("track_id") for r in breaks]
    assert track_ids == sorted(track_ids)


def test_mot_emits_explicit_unknown_records_for_ambiguous_detections() -> None:
    mot = SwapAvoidantMOT(source="fixture")
    gating = DetectionGatingConfig(min_confidence=0.0)

    # Create a detection that will be ambiguous (doesn't match existing tracks cleanly)
    dets, _ = emit_player_detections_v1(
        frame_index=0,
        t_ms=0,
        source="fixture", 
        candidates=[
            {"bbox_xyxy_px": [100, 100, 120, 200], "confidence": 0.9},
            {"bbox_xyxy_px": [130, 100, 150, 200], "confidence": 0.9},
        ],
        gating=gating,
    )
    out0 = mot.update(t_ms=0, detections=dets)
    
    # Ambiguous detection in next frame - one detection between two existing tracks
    dets1, _ = emit_player_detections_v1(
        frame_index=1,
        t_ms=100,
        source="fixture",
        candidates=[{"bbox_xyxy_px": [115, 100, 135, 200], "confidence": 0.9}],
        gating=gating,
    )
    out1 = mot.update(t_ms=100, detections=dets1)
    
    # Should emit explicit unknown record for the ambiguous detection
    unknown_records = [
        r for r in out1 
        if r.get("pos_state") == "unknown" and "amb_" in r.get("entity_id", "")
    ]
    assert len(unknown_records) == 1
    assert unknown_records[0]["diagnostics"]["unknown_reason"] == "ambiguous_association"


def test_mot_out_of_view_heuristic() -> None:
    from fusbal_pipeline.player.track_types import MotConfig
    
    # Configure with small out_of_view threshold for testing
    cfg = MotConfig(out_of_view_distance_px=50.0)
    mot = SwapAvoidantMOT(source="fixture", cfg=cfg)
    gating = DetectionGatingConfig(min_confidence=0.0)
    
    # Create initial track
    dets0, _ = emit_player_detections_v1(
        frame_index=0,
        t_ms=0,
        source="fixture",
        candidates=[{"bbox_xyxy_px": [100, 100, 120, 200], "confidence": 0.9}],
        gating=gating,
    )
    out0 = mot.update(t_ms=0, detections=dets0)
    present0 = [r for r in out0 if r.get("pos_state") == "present"]
    assert len(present0) == 1
    
    # Next frame with detection far away (out of view)
    dets1, _ = emit_player_detections_v1(
        frame_index=1,
        t_ms=100,
        source="fixture", 
        candidates=[{"bbox_xyxy_px": [300, 300, 320, 400], "confidence": 0.9}],  # Far from original
        gating=gating,
    )
    out1 = mot.update(t_ms=100, detections=dets1)
    
    # Should break with out_of_view reason
    breaks = [
        r for r in out1 
        if r.get("pos_state") == "missing" and r.get("break_reason") == "out_of_view"
    ]
    assert len(breaks) == 1
