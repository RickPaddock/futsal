# PROV: FUSBAL.PIPELINE.CONTRACT.01
# REQ: FUSBAL-V1-DATA-001
# WHY: Capture the canonical track/event data contract in code for validation and tooling.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict


Frame = Literal["pitch", "enu", "wgs84"]
EntityType = Literal["player", "ball"]


class TrackSample(TypedDict):
    t_ms: int
    entity_type: EntityType
    entity_id: str
    source: str
    frame: Frame
    x_m: NotRequired[float]
    y_m: NotRequired[float]
    lat: NotRequired[float]
    lon: NotRequired[float]
    sigma_m: NotRequired[float]
    quality: NotRequired[float]


@dataclass(frozen=True)
class Event:
    t_ms: int
    event_type: Literal["shot", "goal"]
    confidence: float
    notes: str | None = None
