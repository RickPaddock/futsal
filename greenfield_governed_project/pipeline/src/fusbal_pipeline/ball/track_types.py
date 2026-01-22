# PROV: FUSBAL.PIPELINE.BALL.TRACK_TYPES.01
# REQ: FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Provide stable, explicit ball tracking vocabulary and configuration.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MissingReason = Literal[
    "detector_missing",
    "low_confidence",
    "jump_rejected",
]

MISSING_REASON_VOCAB: tuple[MissingReason, ...] = (
    "detector_missing",
    "low_confidence",
    "jump_rejected",
)


@dataclass(frozen=True)
class BallTrackConfig:
    min_detection_confidence: float = 0.6
    max_center_jump_px: float = 120.0

