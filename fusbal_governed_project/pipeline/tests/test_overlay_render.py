# PROV: FUSBAL.PIPELINE.TESTS.OVERLAY_RENDER.01
# REQ: SYS-ARCH-15, FUSBAL-V1-OUT-001, FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001
# WHY: Guard deterministic overlay command construction and trust-first overlay behavior.

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from fusbal_pipeline.overlay.render import OverlayError, build_ball_overlay_filter, build_ffmpeg_overlay_args


def _write_tracks(tmp: Path, lines: list[str]) -> Path:
    p = tmp / "tracks.jsonl"
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf8")
    return p


def test_build_ball_overlay_filter_null_when_no_present_ball_records(tmp_path: Path) -> None:
    tracks = _write_tracks(
        tmp_path,
        [
            '{"schema_version":1,"t_ms":0,"entity_type":"ball","entity_id":"b","track_id":"b","source":"s","frame":"image_px","pos_state":"missing","confidence":0.9}',
        ],
    )
    vf = build_ball_overlay_filter(tracks_jsonl=tracks, fps=30.0)
    assert vf == "null"


def test_build_ball_overlay_filter_deterministic(tmp_path: Path) -> None:
    tracks = _write_tracks(
        tmp_path,
        [
            '{"schema_version":1,"t_ms":33,"entity_type":"ball","entity_id":"b","track_id":"b","source":"s","frame":"image_px","pos_state":"present","bbox_xyxy_px":[10,20,30,60],"confidence":0.75}',
            '{"schema_version":1,"t_ms":66,"entity_type":"ball","entity_id":"b","track_id":"b","source":"s","frame":"image_px","pos_state":"present","bbox_xyxy_px":[12,22,32,62],"confidence":0.80}',
        ],
    )
    vf1 = build_ball_overlay_filter(tracks_jsonl=tracks, fps=30.0)
    vf2 = build_ball_overlay_filter(tracks_jsonl=tracks, fps=30.0)
    assert vf1 == vf2
    assert "drawbox=" in vf1
    assert "drawtext=" in vf1


def test_build_ffmpeg_overlay_args_requires_ffmpeg(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"not_a_real_video")
    tracks = _write_tracks(tmp_path, [])
    out = tmp_path / "overlay.mp4"
    with patch("shutil.which", return_value=None):
        with pytest.raises(OverlayError, match="ffmpeg is required"):
            build_ffmpeg_overlay_args(
                video_path=video,
                tracks_jsonl=tracks,
                out_mp4=out,
                fps=30.0,
            )

