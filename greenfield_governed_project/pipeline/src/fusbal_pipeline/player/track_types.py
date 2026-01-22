# PROV: FUSBAL.PIPELINE.PLAYER.TRACK_TYPES.01
# REQ: FUSBAL-V1-PLAYER-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Provide stable, explicit tracking vocabulary and record helpers for swap-avoidant MOT.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BreakReason = Literal[
    "occlusion",
    "ambiguous_association",
    "out_of_view",
    "detector_missing",
    "manual_reset",
]

BREAK_REASON_VOCAB: tuple[BreakReason, ...] = (
    "occlusion",
    "ambiguous_association",
    "out_of_view",
    "detector_missing",
    "manual_reset",
)


@dataclass(frozen=True)
class MotConfig:
    max_center_distance_px: float = 80.0
    min_association_score: float = 0.5
    max_active_tracks: int = 22  # Futsal has max 10 players per team + some buffer
    out_of_view_distance_px: float = 200.0  # Heuristic threshold for out_of_view vs occlusion
    track_timeout_frames: int = 30  # Max frames a track can be missing before forced cleanup

