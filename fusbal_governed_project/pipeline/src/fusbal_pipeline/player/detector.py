# PROV: FUSBAL.PIPELINE.PLAYER.DETECTOR.01
# REQ: FUSBAL-V1-PLAYER-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Define a minimal detector interface with trust-first defaults (missing over wrong).

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .detections import DetectionGatingConfig, FrameDetectionsDiagnostics, RawPlayerDetection


class PlayerDetector(Protocol):
    def detect(self, *, frame_index: int, t_ms: int, image_bytes: bytes | None = None) -> list[RawPlayerDetection]:
        raise NotImplementedError


@dataclass(frozen=True)
class NullPlayerDetector:
    """Trust-first detector that never emits detections (explicitly missing)."""

    def detect(
        self, *, frame_index: int, t_ms: int, image_bytes: bytes | None = None
    ) -> list[RawPlayerDetection]:
        return []


@dataclass(frozen=True)
class StaticPlayerDetector:
    """Deterministic detector backed by a pre-defined per-frame candidate list (for fixtures/tests)."""

    per_frame: dict[int, list[RawPlayerDetection]]

    def detect(
        self, *, frame_index: int, t_ms: int, image_bytes: bytes | None = None
    ) -> list[RawPlayerDetection]:
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

