# PROV: FUSBAL.PIPELINE.BALL.DETECTOR.01
# REQ: FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Define a minimal ball detector interface with trust-first defaults (missing over wrong).

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .detections import DetectionGatingConfig, FrameDetectionsDiagnostics, RawBallDetection
from ..diagnostics_keys import FRAME_INDEX, GATING_REASON, NUM_CANDIDATES, NUM_EMITTED


class BallDetector(Protocol):
    def detect(self, *, frame_index: int, t_ms: int, image_bytes: bytes | None = None) -> list[RawBallDetection]:
        raise NotImplementedError


@dataclass(frozen=True)
class NullBallDetector:
    """Trust-first detector that never emits detections (explicitly missing)."""

    def detect(
        self, *, frame_index: int, t_ms: int, image_bytes: bytes | None = None
    ) -> list[RawBallDetection]:
        return []


@dataclass(frozen=True)
class StaticBallDetector:
    """Deterministic detector backed by a pre-defined per-frame candidate list (for fixtures/tests)."""

    per_frame: dict[int, list[RawBallDetection]]

    def detect(
        self, *, frame_index: int, t_ms: int, image_bytes: bytes | None = None
    ) -> list[RawBallDetection]:
        return list(self.per_frame.get(int(frame_index), []))


@dataclass(frozen=True)
class BaselineBallDetectorConfig:
    """Conservative, dependency-free baseline heuristic for local UAT."""

    sample_step_px: int = 3
    min_luma_0_to_255: int = 220
    max_saturation_0_to_255: int = 40
    bbox_radius_px: int = 6
    edge_margin_ratio_0_to_1: float = 0.08
    blob_window_radius_px: int = 6
    blob_min_ratio_0_to_1: float = 0.02
    blob_high_ratio_threshold_0_to_1: float = 1.0
    blob_high_ratio_conf_scale: float = 1.0
    min_abs_delta_luma_0_to_255: int = 3


@dataclass
class BaselineBallDetector:
    """Heuristic ball detector on rgb24 frames (decoded via ffmpeg).

    This is a plumbing/UAT baseline only. It is intentionally conservative and low-recall.
    """

    decode_width_px: int
    decode_height_px: int
    src_width_px: int
    src_height_px: int
    cfg: BaselineBallDetectorConfig = BaselineBallDetectorConfig()
    _prev_rgb24: bytes | None = None

    def detect(
        self, *, frame_index: int, t_ms: int, image_bytes: bytes | None = None
    ) -> list[RawBallDetection]:
        if image_bytes is None:
            return []
        if not isinstance(image_bytes, (bytes, bytearray)):
            return []
        expected = int(self.decode_width_px * self.decode_height_px * 3)
        if len(image_bytes) != expected:
            return []

        if self._prev_rgb24 is None:
            # First frame: no motion reference, be conservative.
            self._prev_rgb24 = bytes(image_bytes)
            return []

        prev = self._prev_rgb24
        self._prev_rgb24 = bytes(image_bytes)

        step = int(self.cfg.sample_step_px)
        if step <= 0:
            step = 1
        best: tuple[int, int, float, int, int, int] | None = None  # x,y,score,luma,sat,delta_luma

        # Scan a subsampled grid for bright, low-saturation pixels.
        w = int(self.decode_width_px)
        h = int(self.decode_height_px)
        buf = image_bytes
        margin = float(self.cfg.edge_margin_ratio_0_to_1)
        if margin < 0:
            margin = 0.0
        if margin > 0.45:
            margin = 0.45
        margin_x = int(round(w * margin))
        margin_y = int(round(h * margin))
        for y in range(margin_y, max(margin_y, h - margin_y), step):
            row = y * w * 3
            for x in range(margin_x, max(margin_x, w - margin_x), step):
                i = row + x * 3
                r = buf[i]
                g = buf[i + 1]
                b = buf[i + 2]
                mx = r if r >= g and r >= b else (g if g >= b else b)
                mn = r if r <= g and r <= b else (g if g <= b else b)
                luma = (int(r) + int(g) + int(b)) // 3
                sat = int(mx) - int(mn)
                if luma < int(self.cfg.min_luma_0_to_255):
                    continue
                if sat > int(self.cfg.max_saturation_0_to_255):
                    continue
                pr = prev[i]
                pg = prev[i + 1]
                pb = prev[i + 2]
                prev_luma = (int(pr) + int(pg) + int(pb)) // 3
                delta = abs(int(luma) - int(prev_luma))
                if delta < int(self.cfg.min_abs_delta_luma_0_to_255):
                    continue
                score = float(luma) * (1.0 - float(sat) / 255.0)
                if best is None or score > best[2]:
                    best = (x, y, score, luma, sat, int(delta))

        if best is None:
            return []

        x, y, score, luma, sat, delta_luma = best

        # Second-stage conservatism: estimate how "blob-like" the local neighborhood is.
        # Very large bright regions (e.g., court lines/highlights) are treated as missing.
        win_r = int(self.cfg.blob_window_radius_px)
        if win_r < 1:
            win_r = 1
        q = 0
        area = 0
        for yy in range(max(0, y - win_r), min(h, y + win_r + 1)):
            for xx in range(max(0, x - win_r), min(w, x + win_r + 1)):
                area += 1
                j = (yy * w + xx) * 3
                rr = buf[j]
                gg = buf[j + 1]
                bb = buf[j + 2]
                mx2 = rr if rr >= gg and rr >= bb else (gg if gg >= bb else bb)
                mn2 = rr if rr <= gg and rr <= bb else (gg if gg <= bb else bb)
                l2 = (int(rr) + int(gg) + int(bb)) // 3
                sat2 = int(mx2) - int(mn2)
                if l2 >= int(self.cfg.min_luma_0_to_255) and sat2 <= int(self.cfg.max_saturation_0_to_255):
                    q += 1
        ratio = float(q / area) if area else 0.0
        if ratio < float(self.cfg.blob_min_ratio_0_to_1):
            return []

        # Scale bbox back into source pixel coordinates (frame=image_px refers to source video pixels).
        scale_x = float(self.src_width_px) / float(max(1, self.decode_width_px))
        scale_y = float(self.src_height_px) / float(max(1, self.decode_height_px))
        r_px = int(self.cfg.bbox_radius_px)
        cx = int(round(x * scale_x))
        cy = int(round(y * scale_y))
        rr_x = int(round(r_px * scale_x))
        rr_y = int(round(r_px * scale_y))
        x1 = max(0, cx - rr_x)
        y1 = max(0, cy - rr_y)
        x2 = min(int(self.src_width_px) - 1, cx + rr_x)
        y2 = min(int(self.src_height_px) - 1, cy + rr_y)
        if x2 <= x1 or y2 <= y1:
            return []

        conf = (float(luma) / 255.0) * (1.0 - float(sat) / 255.0)
        if ratio > float(self.cfg.blob_high_ratio_threshold_0_to_1):
            conf *= float(self.cfg.blob_high_ratio_conf_scale)
        if conf < 0:
            conf = 0.0
        if conf > 1:
            conf = 1.0

        diagnostics: dict[str, object] = {
            FRAME_INDEX: int(frame_index),
            "method": "bright_low_saturation",
            "sample_step_px": int(step),
            "luma_0_to_255": int(luma),
            "sat_0_to_255": int(sat),
            "abs_delta_luma_0_to_255": int(delta_luma),
            "bbox_radius_px": int(r_px),
            "edge_margin_ratio_0_to_1": float(margin),
            "blob_window_radius_px": int(win_r),
            "blob_qualifying_pixels": int(q),
            "blob_window_area": int(area),
            "blob_ratio_0_to_1": float(ratio),
            "blob_high_ratio_threshold_0_to_1": float(self.cfg.blob_high_ratio_threshold_0_to_1),
            "blob_high_ratio_conf_scale": float(self.cfg.blob_high_ratio_conf_scale),
            "decode_size_px": [int(self.decode_width_px), int(self.decode_height_px)],
            "source_size_px": [int(self.src_width_px), int(self.src_height_px)],
        }
        return [
            {
                "bbox_xyxy_px": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "diagnostics": diagnostics,
            }
        ]


def default_detection_diagnostics(
    *, frame_index: int, candidates_len: int, gating: DetectionGatingConfig | None
) -> FrameDetectionsDiagnostics:
    cfg = gating or DetectionGatingConfig()
    return {
        FRAME_INDEX: int(frame_index),
        NUM_CANDIDATES: int(candidates_len),
        NUM_EMITTED: 0,
        "min_confidence": float(cfg.min_confidence),
        GATING_REASON: "detector_missing" if candidates_len == 0 else "none",
    }
