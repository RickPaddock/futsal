# PROV: FUSBAL.PIPELINE.TESTS.BALL_DET_REGRESSION_V1.01
# REQ: REQ-V1-BALL-DETECT-BASELINE-001, FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Prevent regressions in baseline ball detection/gating/tracking on a deterministic synthetic sequence.

from __future__ import annotations

from fusbal_pipeline.ball.detector import BaselineBallDetector, BaselineBallDetectorConfig
from fusbal_pipeline.ball.detections import DetectionGatingConfig, emit_ball_detections_v1
from fusbal_pipeline.ball.tracker import BallTracker
from fusbal_pipeline.contract import validate_track_record_v1


def test_baseline_detector_regression_emits_at_least_10_present_frames() -> None:
    w, h = 8, 8
    black = bytes([0] * (w * h * 3))

    def frame_with_white_pixel(*, x: int, y: int) -> bytes:
        buf = bytearray(black)
        i = (y * w + x) * 3
        buf[i] = 255
        buf[i + 1] = 255
        buf[i + 2] = 255
        return bytes(buf)

    det = BaselineBallDetector(
        decode_width_px=w,
        decode_height_px=h,
        src_width_px=w,
        src_height_px=h,
        cfg=BaselineBallDetectorConfig(
            sample_step_px=1,
            min_luma_0_to_255=200,
            max_saturation_0_to_255=30,
            bbox_radius_px=1,
            edge_margin_ratio_0_to_1=0.0,
            blob_window_radius_px=1,
            blob_min_ratio_0_to_1=0.02,
            blob_high_ratio_threshold_0_to_1=1.0,
            blob_high_ratio_conf_scale=1.0,
            min_abs_delta_luma_0_to_255=3,
        ),
    )
    gating = DetectionGatingConfig(min_confidence=0.6)
    tracker = BallTracker(source="test")

    present = 0
    for frame_index in range(0, 32):
        t_ms = frame_index * 100
        img = frame_with_white_pixel(x=3, y=3) if (frame_index % 2 == 1) else black
        candidates = det.detect(frame_index=frame_index, t_ms=t_ms, image_bytes=img)
        det_records, _diag = emit_ball_detections_v1(
            frame_index=frame_index,
            t_ms=t_ms,
            source="test",
            candidates=candidates,
            gating=gating,
        )
        rec = tracker.update(t_ms=t_ms, frame_index=frame_index, detections=det_records)
        assert validate_track_record_v1(rec) == []
        if rec.get("pos_state") == "present":
            present += 1

    assert present >= 10

