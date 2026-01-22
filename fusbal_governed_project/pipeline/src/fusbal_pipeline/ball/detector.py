# PROV: FUSBAL.PIPELINE.BALL.DETECTOR.01
# REQ: FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Define a minimal ball detector interface with trust-first defaults (missing over wrong).

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .detections import DetectionGatingConfig, FrameDetectionsDiagnostics, RawBallDetection


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


def default_detection_diagnostics(
    *, frame_index: int, candidates_len: int, gating: DetectionGatingConfig | None
) -> FrameDetectionsDiagnostics:
    cfg = gating or DetectionGatingConfig()
    return {
        "frame_index": int(frame_index),
        "num_candidates": int(candidates_len),
        "num_emitted": 0,
        "min_confidence": float(cfg.min_confidence),
        "gating_reason": "detector_missing" if candidates_len == 0 else "none",
    }

