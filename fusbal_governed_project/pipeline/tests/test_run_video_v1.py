# PROV: FUSBAL.PIPELINE.TESTS.RUN_VIDEO_V1.01
# REQ: REQ-V1-VIDEO-RUNNER-001, REQ-V1-BALL-DETECT-BASELINE-001, SYS-ARCH-15
# WHY: Guard run-video determinism and baseline detector trust-first behavior on synthetic frames.

from __future__ import annotations

import json
from pathlib import Path

from fusbal_pipeline.ball.detector import BaselineBallDetector, BaselineBallDetectorConfig
from fusbal_pipeline.cli import _run_video_write_bundle
from fusbal_pipeline.video.ingest import VideoFrame, VideoMetadata


def test_run_video_is_deterministic_with_null_detector(tmp_path: Path) -> None:
    meta = VideoMetadata(
        fps=10.0,
        width_px=8,
        height_px=8,
        nb_frames=2,
        duration_s=0.2,
        source_rel_path="clip.mp4",
        diagnostics={},
    )
    # Two trivial frames (content irrelevant when detector disabled).
    frame_bytes = bytes([0] * (4 * 4 * 3))
    frames = [
        VideoFrame(frame_index=0, t_ms=0, width_px=4, height_px=4, pix_fmt="rgb24", rgb24=frame_bytes),
        VideoFrame(frame_index=1, t_ms=100, width_px=4, height_px=4, pix_fmt="rgb24", rgb24=frame_bytes),
    ]

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _run_video_write_bundle(
        out_dir=out_a,
        match_id="M1",
        meta=meta,
        frames=frames,
        ball_detector_enabled=False,
        ball_detector_cfg=BaselineBallDetectorConfig(),
        min_confidence=0.6,
    )
    _run_video_write_bundle(
        out_dir=out_b,
        match_id="M1",
        meta=meta,
        frames=frames,
        ball_detector_enabled=False,
        ball_detector_cfg=BaselineBallDetectorConfig(),
        min_confidence=0.6,
    )

    tracks_a = (out_a / "tracks.jsonl").read_bytes()
    tracks_b = (out_b / "tracks.jsonl").read_bytes()
    events_a = (out_a / "events.json").read_bytes()
    events_b = (out_b / "events.json").read_bytes()
    assert tracks_a == tracks_b
    assert events_a == events_b

    # Trust-first: when detector is disabled, there must be zero present ball records.
    lines = tracks_a.decode("utf8").splitlines()
    present = 0
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("entity_type") == "ball" and rec.get("pos_state") == "present":
            present += 1
    assert present == 0


def test_baseline_ball_detector_detects_bright_low_saturation_pixel() -> None:
    # 4x4 decode frame with one white pixel at (1,2).
    w, h = 4, 4
    buf = bytearray([0] * (w * h * 3))
    x, y = 1, 2
    i = (y * w + x) * 3
    buf[i] = 255
    buf[i + 1] = 255
    buf[i + 2] = 255

    det = BaselineBallDetector(
        decode_width_px=w,
        decode_height_px=h,
        src_width_px=8,
        src_height_px=8,
        cfg=BaselineBallDetectorConfig(
            sample_step_px=1,
            min_luma_0_to_255=200,
            max_saturation_0_to_255=20,
            bbox_radius_px=1,
            blob_window_radius_px=1,
            blob_min_ratio_0_to_1=0.0,
            blob_high_ratio_threshold_0_to_1=1.0,
            blob_high_ratio_conf_scale=1.0,
            min_abs_delta_luma_0_to_255=0,
        ),
    )
    # First call seeds motion reference; second call produces the detection.
    det.detect(frame_index=0, t_ms=0, image_bytes=bytes(buf))
    out = det.detect(frame_index=1, t_ms=100, image_bytes=bytes(buf))
    assert len(out) == 1
    assert out[0]["confidence"] >= 0.0
    bbox = out[0]["bbox_xyxy_px"]
    assert isinstance(bbox, list) and len(bbox) == 4
