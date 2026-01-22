# PROV: FUSBAL.PIPELINE.TESTS.MANIFEST_DETERMINISM.01
# REQ: SYS-ARCH-15, FUSBAL-V1-OUT-001
# WHY: Ensure manifest generation remains deterministic and artifact lists are stable.

from __future__ import annotations

import json
from pathlib import Path

from fusbal_pipeline.cli import _build_manifest_v1


def test_manifest_json_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    m1 = _build_manifest_v1(
        match_id="MATCH_001",
        video_paths=["a.mp4", "b.mp4"],
        sensor_paths=[],
        notes=None,
    )
    m2 = _build_manifest_v1(
        match_id="MATCH_001",
        video_paths=["a.mp4", "b.mp4"],
        sensor_paths=[],
        notes=None,
    )
    s1 = json.dumps(m1, indent=2, sort_keys=True)
    s2 = json.dumps(m2, indent=2, sort_keys=True)
    assert s1 == s2


def test_manifest_artifacts_are_stably_sorted() -> None:
    m = _build_manifest_v1(match_id="MATCH_001", video_paths=["a.mp4"], sensor_paths=[], notes=None)
    ids = [a["artifact_id"] for a in m["artifacts"]]
    assert ids == sorted(ids)
