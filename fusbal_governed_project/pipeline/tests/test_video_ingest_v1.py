# PROV: FUSBAL.PIPELINE.TESTS.VIDEO_INGEST_V1.01
# REQ: REQ-V1-VIDEO-INGEST-001, SYS-ARCH-15
# WHY: Guard deterministic V1 timebase computation and explicit error handling for missing inputs.

from __future__ import annotations

from pathlib import Path

import pytest

from fusbal_pipeline.video.ingest import VideoIngestError, compute_t_ms_from_frame_index, probe_video_metadata


def test_compute_t_ms_from_frame_index_is_deterministic() -> None:
    assert compute_t_ms_from_frame_index(frame_index=0, fps=10.0) == 0
    assert compute_t_ms_from_frame_index(frame_index=1, fps=10.0) == 100
    assert compute_t_ms_from_frame_index(frame_index=2, fps=10.0) == 200


def test_probe_video_metadata_missing_file_is_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"
    with pytest.raises(VideoIngestError) as e:
        probe_video_metadata(video_path=missing, source_rel_path="nope.mp4")
    assert "video file not found" in str(e.value)
    assert "nope.mp4" in str(e.value)
    assert getattr(e.value, "exit_code", None) == 3


def test_probe_video_metadata_missing_ffprobe_override_is_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"")
    monkeypatch.setenv("FUSBAL_FFPROBE_PATH", str(tmp_path / "nope_ffprobe"))
    with pytest.raises(VideoIngestError) as e:
        probe_video_metadata(video_path=vid, source_rel_path="clip.mp4")
    assert "configured tool path does not exist" in str(e.value)
    assert getattr(e.value, "exit_code", None) == 4
