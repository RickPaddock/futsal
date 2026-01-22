# PROV: FUSBAL.PIPELINE.EVENTS.TYPES.01
# REQ: FUSBAL-V1-EVENT-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Define stable event vocabulary and evidence pointer types used in the V1 contract.

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

EventState = Literal["confirmed", "candidate", "unknown"]


class TimeRangeMs(TypedDict):
    start_ms: int
    end_ms: int


class FrameRange(TypedDict):
    start_frame: int
    end_frame: int


class EvidencePointerV1(TypedDict):
    artifact_id: str
    time_range_ms: NotRequired[TimeRangeMs]
    frame_range: NotRequired[FrameRange]

