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
