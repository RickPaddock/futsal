# PROV: FUSBAL.PIPELINE.CONTRACT.01
# REQ: FUSBAL-V1-DATA-001, SYS-ARCH-15
# WHY: Capture the canonical track/event data contract in code for validation and tooling.

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict


Frame = Literal["pitch", "enu", "wgs84"]
EntityType = Literal["player", "ball"]
TeamLabel = Literal["A", "B", "unknown"]
PosState = Literal["present", "missing", "unknown"]
EventType = Literal["shot", "goal"]

TRACK_RECORD_SCHEMA_VERSION = 1
EVENT_RECORD_SCHEMA_VERSION = 1


class TrackRecordV1(TypedDict):
    schema_version: Literal[1]
    t_ms: int
    entity_type: EntityType
    entity_id: str  # stable within the bundle (no implicit meaning)
    track_id: str  # stable within the bundle; may differ from entity_id
    source: str
    frame: Frame
    pos_state: PosState
    x_m: NotRequired[float]
    y_m: NotRequired[float]
    lat: NotRequired[float]
    lon: NotRequired[float]
    sigma_m: NotRequired[float]
    confidence: NotRequired[float]
    quality: NotRequired[float]
    team: NotRequired[TeamLabel]
    diagnostics: NotRequired[dict[str, object]]


class EventRecordV1(TypedDict):
    schema_version: Literal[1]
    t_ms: int
    event_type: EventType
    confidence: float
    source: NotRequired[str]
    notes: NotRequired[str]
    diagnostics: NotRequired[dict[str, object]]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_track_record_v1(obj: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["track record must be a JSON object"]

    schema_version = obj.get("schema_version")
    if schema_version != TRACK_RECORD_SCHEMA_VERSION:
        errors.append(f"track.schema_version must be {TRACK_RECORD_SCHEMA_VERSION}")

    t_ms = obj.get("t_ms")
    if not isinstance(t_ms, int) or isinstance(t_ms, bool):
        errors.append("track.t_ms must be an integer (milliseconds)")

    entity_type = obj.get("entity_type")
    if entity_type not in ("player", "ball"):
        errors.append("track.entity_type must be one of: player, ball")

    for key in ("entity_id", "track_id", "source"):
        val = obj.get(key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"track.{key} must be a non-empty string")

    frame = obj.get("frame")
    if frame not in ("pitch", "enu", "wgs84"):
        errors.append("track.frame must be one of: pitch, enu, wgs84")

    pos_state = obj.get("pos_state")
    if pos_state not in ("present", "missing", "unknown"):
        errors.append("track.pos_state must be one of: present, missing, unknown")

    if isinstance(pos_state, str) and pos_state == "present":
        if frame in ("pitch", "enu"):
            if not _is_number(obj.get("x_m")) or not _is_number(obj.get("y_m")):
                errors.append("track.x_m and track.y_m are required numbers when pos_state=present and frame=pitch/enu")
        elif frame == "wgs84":
            if not _is_number(obj.get("lat")) or not _is_number(obj.get("lon")):
                errors.append("track.lat and track.lon are required numbers when pos_state=present and frame=wgs84")

    if "sigma_m" in obj and not _is_number(obj.get("sigma_m")):
        errors.append("track.sigma_m must be a number when present")

    for key in ("confidence", "quality"):
        if key in obj and not _is_number(obj.get(key)):
            errors.append(f"track.{key} must be a number when present")

    if "team" in obj and obj.get("team") not in ("A", "B", "unknown"):
        errors.append("track.team must be one of: A, B, unknown")

    if "diagnostics" in obj and not isinstance(obj.get("diagnostics"), dict):
        errors.append("track.diagnostics must be an object when present")

    return errors


def validate_event_record_v1(obj: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["event record must be a JSON object"]

    schema_version = obj.get("schema_version")
    if schema_version != EVENT_RECORD_SCHEMA_VERSION:
        errors.append(f"event.schema_version must be {EVENT_RECORD_SCHEMA_VERSION}")

    t_ms = obj.get("t_ms")
    if not isinstance(t_ms, int) or isinstance(t_ms, bool):
        errors.append("event.t_ms must be an integer (milliseconds)")

    event_type = obj.get("event_type")
    if event_type not in ("shot", "goal"):
        errors.append("event.event_type must be one of: shot, goal")

    confidence = obj.get("confidence")
    if not _is_number(confidence):
        errors.append("event.confidence must be a number")

    if "source" in obj and (not isinstance(obj.get("source"), str) or not obj.get("source")):
        errors.append("event.source must be a non-empty string when present")

    if "notes" in obj and (not isinstance(obj.get("notes"), str) or not obj.get("notes")):
        errors.append("event.notes must be a non-empty string when present")

    if "diagnostics" in obj and not isinstance(obj.get("diagnostics"), dict):
        errors.append("event.diagnostics must be an object when present")

    return errors


def validate_tracks_jsonl(path: Path) -> list[str]:
    if not path.is_file():
        return [f"tracks file missing: {path}"]
    errors: list[str] = []
    for idx, raw in enumerate(path.read_text(encoding="utf8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{path}: line {idx}: invalid JSON: {e}")
            continue
        for err in validate_track_record_v1(obj):
            errors.append(f"{path}: line {idx}: {err}")
    return errors


def validate_events_json(path: Path) -> list[str]:
    if not path.is_file():
        return [f"events file missing: {path}"]
    try:
        obj = json.loads(path.read_text(encoding="utf8"))
    except json.JSONDecodeError as e:
        return [f"{path}: invalid JSON: {e}"]
    if not isinstance(obj, list):
        return [f"{path}: must be a JSON array of event records"]
    errors: list[str] = []
    for i, item in enumerate(obj):
        for err in validate_event_record_v1(item):
            errors.append(f"{path}: index {i}: {err}")
    return errors
