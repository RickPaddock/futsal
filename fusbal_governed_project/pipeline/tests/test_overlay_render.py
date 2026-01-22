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


def test_build_ball_overlay_filter_uses_n_based_enable(tmp_path: Path) -> None:
    tracks = _write_tracks(
        tmp_path,
        [
            '{"schema_version":1,"t_ms":33,"entity_type":"ball","entity_id":"b","track_id":"b","source":"s","frame":"image_px","pos_state":"present","bbox_xyxy_px":[10,20,30,60],"confidence":0.75}',
        ],
    )
    vf = build_ball_overlay_filter(tracks_jsonl=tracks, fps=30.0)
    # 33ms @ 30fps ~= frame 1
    assert "enable='eq(n\\,1)'" in vf


def test_build_ball_overlay_filter_dedupes_per_frame(tmp_path: Path) -> None:
    tracks = _write_tracks(
        tmp_path,
        [
            # Both map to the same rounded frame index at 30fps.
            '{"schema_version":1,"t_ms":33,"entity_type":"ball","entity_id":"b","track_id":"b","source":"s","frame":"image_px","pos_state":"present","bbox_xyxy_px":[10,20,30,60],"confidence":0.10}',
            '{"schema_version":1,"t_ms":34,"entity_type":"ball","entity_id":"b","track_id":"b","source":"s","frame":"image_px","pos_state":"present","bbox_xyxy_px":[11,21,31,61],"confidence":0.90}',
        ],
    )
    vf = build_ball_overlay_filter(tracks_jsonl=tracks, fps=30.0)
    assert vf.count("drawbox=") == 1
    assert vf.count("drawtext=") == 1
    # Highest confidence wins.
    assert "x=11" in vf


def test_build_ball_overlay_filter_clamps_or_skips_out_of_range_bbox(tmp_path: Path) -> None:
    tracks = _write_tracks(
        tmp_path,
        [
            # Clamp to frame bounds: x1/y1 should become 0.
            '{"schema_version":1,"t_ms":33,"entity_type":"ball","entity_id":"b","track_id":"b","source":"s","frame":"image_px","pos_state":"present","bbox_xyxy_px":[-10,-10,5,5],"confidence":0.75}',
            # Fully out of bounds after clamp -> skipped.
            '{"schema_version":1,"t_ms":66,"entity_type":"ball","entity_id":"b","track_id":"b","source":"s","frame":"image_px","pos_state":"present","bbox_xyxy_px":[200,200,210,210],"confidence":0.80}',
        ],
    )
    vf = build_ball_overlay_filter(tracks_jsonl=tracks, fps=30.0, width_px=100, height_px=100)
    assert "x=0:y=0" in vf
    # Second bbox is skipped, so only one overlay.
    assert vf.count("drawbox=") == 1


def test_build_ball_overlay_filter_truncates_deterministically(tmp_path: Path) -> None:
    lines = []
    # 10 distinct frames (t_ms values) at 30fps; all valid.
    for i in range(10):
        t_ms = (i + 1) * 33
        lines.append(
            '{"schema_version":1,"t_ms":%d,"entity_type":"ball","entity_id":"b","track_id":"b","source":"s","frame":"image_px","pos_state":"present","bbox_xyxy_px":[%d,20,%d,60],"confidence":0.50}'
            % (t_ms, 10 + i, 30 + i)
        )
    tracks = _write_tracks(tmp_path, lines)
    # max_ops=4 => max_boxes=2
    vf1 = build_ball_overlay_filter(tracks_jsonl=tracks, fps=30.0, max_ops=4)
    vf2 = build_ball_overlay_filter(tracks_jsonl=tracks, fps=30.0, max_ops=4)
    assert vf1 == vf2
    assert vf1.count("drawbox=") == 2
    assert vf1.count("drawtext=") == 2


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


def test_build_ffmpeg_overlay_args_respects_ffmpeg_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"not_a_real_video")
    tracks = _write_tracks(tmp_path, [])
    out = tmp_path / "overlay.mp4"
    monkeypatch.setenv("FUSBAL_FFMPEG_PATH", "ffmpeg_custom")
    with patch("shutil.which", return_value="/opt/bin/ffmpeg_custom"):
        args = build_ffmpeg_overlay_args(
            video_path=video,
            tracks_jsonl=tracks,
            out_mp4=out,
            fps=30.0,
        )
    assert args[0] == "/opt/bin/ffmpeg_custom"

