# PROV: FUSBAL.PIPELINE.TESTS.VALIDATE_STRICT_V1.01
# REQ: FUSBAL-V1-BALL-001, SYS-ARCH-15
# WHY: Ensure optional strict validation catches unsafe invariants.

from __future__ import annotations

import json
from pathlib import Path

from fusbal_pipeline.contract import validate_tracks_jsonl


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n", encoding="utf8")


def test_strict_ball_frame_index_monotonicity_violation(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.jsonl"

    # Same track_id with a repeated frame_index should be rejected in strict mode.
    rec0 = {
        "schema_version": 1,
        "frame": "image_px",
        "entity_type": "ball",
        "entity_id": "B1",
        "track_id": "B1",
        "segment_id": "S1",
        "source": "video",
        "t_ms": 0,
        "pos_state": "present",
        "confidence": 0.9,
        "quality": 1.0,
        "bbox_xyxy_px": [0, 0, 1, 1],
        "diagnostics": {"frame_index": 10},
    }
    rec1 = {
        **rec0,
        "t_ms": 100,
        "diagnostics": {"frame_index": 10},
    }

    _write_jsonl(tracks, [rec0, rec1])

    errors_default = validate_tracks_jsonl(tracks)
    assert errors_default == []

    errors_strict = validate_tracks_jsonl(tracks, strict_frame_index_monotonicity=True)
    assert any("strict" in e.lower() or "monotonic" in e.lower() for e in errors_strict)
