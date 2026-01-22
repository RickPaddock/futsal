# PROV: FUSBAL.PIPELINE.CONTRACT.01
# REQ: FUSBAL-V1-DATA-001, FUSBAL-V1-PLAYER-001, FUSBAL-V1-TEAM-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Capture the canonical track/event data contract in code for validation and tooling.

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from .ball.track_types import MISSING_REASON_VOCAB
from .bundle import BUNDLE_ARTIFACT_SPECS_V1
from .diagnostics_keys import (
    BALL_UNKNOWN_DETECTOR_ERROR,
    BALL_UNKNOWN_FRAME_UNAVAILABLE,
    FRAME_INDEX,
    MISSING_REASON,
    UNKNOWN_REASON,
)

Frame = Literal["pitch", "enu", "wgs84", "image_px"]
EntityType = Literal["player", "ball"]
TeamLabel = Literal["A", "B", "unknown"]
PosState = Literal["present", "missing", "unknown"]
EventType = Literal["shot", "goal"]
EventState = Literal["confirmed", "candidate", "unknown"]
BreakReason = Literal[
    "occlusion",
    "ambiguous_association",
    "out_of_view",
    "detector_missing",
    "manual_reset",
]

TRACK_RECORD_SCHEMA_VERSION = 1
EVENT_RECORD_SCHEMA_VERSION = 1

# Calibration contract (V1) - see INT-020.
PITCH_TEMPLATE_SCHEMA_VERSION = 1
CALIBRATION_INPUT_SCHEMA_VERSION = 1
MARKINGS_OBSERVATIONS_SCHEMA_VERSION = 1
MANUAL_CORRESPONDENCES_SCHEMA_VERSION = 1
CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION = 1


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
    bbox_xyxy_px: NotRequired[list[int]]
    sigma_m: NotRequired[float]
    confidence: NotRequired[float]
    quality: NotRequired[float]
    team: NotRequired[TeamLabel]
    team_confidence: NotRequired[float]
    segment_id: NotRequired[str]
    break_reason: NotRequired[BreakReason]
    diagnostics: NotRequired[dict[str, object]]


class EvidenceTimeRangeMsV1(TypedDict):
    start_ms: int
    end_ms: int


class EvidenceFrameRangeV1(TypedDict):
    start_frame: int
    end_frame: int


class EvidencePointerV1(TypedDict):
    artifact_id: str
    time_range_ms: NotRequired[EvidenceTimeRangeMsV1]
    frame_range: NotRequired[EvidenceFrameRangeV1]


class EventRecordV1(TypedDict):
    schema_version: Literal[1]
    t_ms: int
    event_type: EventType
    event_state: EventState
    confidence: float
    evidence: list[EvidencePointerV1]
    source: NotRequired[str]
    notes: NotRequired[str]
    diagnostics: NotRequired[dict[str, object]]


class PitchTemplateLineV1(TypedDict):
    label: str
    p0_xy_m: list[float]
    p1_xy_m: list[float]


class PitchTemplateV1(TypedDict):
    schema_version: Literal[1]
    pitch_template_id: str
    dimensions_m: dict[str, float]
    frame: Literal["pitch_v1"]
    frame_origin: Literal["lower_left_corner"]
    markings_v1: dict[str, object]


class CalibrationInputV1(TypedDict):
    schema_version: Literal[1]
    venue_id: str
    pitch_id: str
    pitch_template_ref: dict[str, str]
    camera_id: str
    image_pre_undistorted: bool
    image_size_px: NotRequired[list[int]]
    source_video_path: NotRequired[str]


class MarkingsSegmentV1(TypedDict):
    label: str
    p0_xy_px: list[float]
    p1_xy_px: list[float]


class MarkingsObservationsV1(TypedDict):
    schema_version: Literal[1]
    camera_id: str
    frame_ref: NotRequired[dict[str, object]]
    segments: list[MarkingsSegmentV1]


class ManualCorrespondenceV1(TypedDict):
    image_xy_px: list[float]
    pitch_xy_m: list[float]
    label: NotRequired[str]


class ManualCorrespondencesV1(TypedDict):
    schema_version: Literal[1]
    camera_id: str
    pitch_template_ref: dict[str, str]
    correspondences: list[ManualCorrespondenceV1]


class CalibrationDiagnosticsV1(TypedDict, total=False):
    schema_version: Literal[1]
    status: Literal["success", "fail"]
    rms_reprojection_error_px: float
    inlier_ratio: float
    num_inliers: int
    marking_coverage_score_0_to_1: float
    failure_reason: str
    notes: str


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    if not _is_number(value):
        return False
    return math.isfinite(float(value))


def _fmt_value(value: object) -> str:
    # Keep error messages deterministic and compact.
    return repr(value)


def validate_track_record_v1(obj: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["track record must be a JSON object"]

    schema_version = obj.get("schema_version")
    if schema_version != TRACK_RECORD_SCHEMA_VERSION:
        errors.append(f"track.schema_version must be {TRACK_RECORD_SCHEMA_VERSION}")

    t_ms = obj.get("t_ms")
    if not isinstance(t_ms, int) or isinstance(t_ms, bool) or t_ms < 0:
        errors.append(f"track.t_ms must be an integer >= 0 (got {_fmt_value(t_ms)})")

    entity_type = obj.get("entity_type")
    if entity_type not in ("player", "ball"):
        errors.append("track.entity_type must be one of: player, ball")

    for key in ("entity_id", "track_id", "source"):
        val = obj.get(key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"track.{key} must be a non-empty string")

    frame = obj.get("frame")
    if frame not in ("pitch", "enu", "wgs84", "image_px"):
        errors.append("track.frame must be one of: pitch, enu, wgs84, image_px")

    pos_state = obj.get("pos_state")
    if pos_state not in ("present", "missing", "unknown"):
        errors.append("track.pos_state must be one of: present, missing, unknown")

    # Semantic rules for breaks/segments:
    # - break_reason is only meaningful for missing records
    # - missing records must include break_reason
    # - break_reason requires segment_id (segment context is explicit)
    has_break_reason = "break_reason" in obj
    has_segment_id = "segment_id" in obj
    if pos_state == "missing" and not has_break_reason:
        errors.append("track.break_reason is required when track.pos_state=missing")
    if pos_state in ("present", "unknown") and has_break_reason:
        errors.append("track.break_reason is only allowed when track.pos_state=missing")
    if has_break_reason and not has_segment_id:
        errors.append("track.segment_id is required when track.break_reason is present")

    if isinstance(pos_state, str) and pos_state == "present":
        if frame in ("pitch", "enu"):
            if not _is_number(obj.get("x_m")) or not _is_number(obj.get("y_m")):
                errors.append(
                    "track.x_m and track.y_m are required numbers when pos_state=present and "
                    "frame=pitch/enu"
                )
        elif frame == "wgs84":
            if not _is_number(obj.get("lat")) or not _is_number(obj.get("lon")):
                errors.append(
                    "track.lat and track.lon are required numbers when pos_state=present and "
                    "frame=wgs84"
                )
            else:
                lat = obj.get("lat")
                lon = obj.get("lon")
                if _is_number(lat) and (float(lat) < -90 or float(lat) > 90):
                    errors.append(
                        f"track.lat must be in [-90, 90] when frame=wgs84 (got {_fmt_value(lat)})"
                    )
                if _is_number(lon) and (float(lon) < -180 or float(lon) > 180):
                    errors.append(
                        f"track.lon must be in [-180, 180] when frame=wgs84 (got {_fmt_value(lon)})"
                    )
        elif frame == "image_px":
            bbox = obj.get("bbox_xyxy_px")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in bbox)
            ):
                errors.append(
                    "track.bbox_xyxy_px is required as [x1,y1,x2,y2] integers when "
                    "pos_state=present and frame=image_px"
                )
            else:
                x1, y1, x2, y2 = bbox
                if x2 < x1 or y2 < y1:
                    errors.append("track.bbox_xyxy_px must satisfy x2>=x1 and y2>=y1")

    if "sigma_m" in obj:
        sigma_m = obj.get("sigma_m")
        if not _is_finite_number(sigma_m) or float(sigma_m) < 0:
            errors.append(
                f"track.sigma_m must be a finite number >= 0 when present "
                f"(got {_fmt_value(sigma_m)})"
            )

    for key in ("confidence", "quality"):
        if key in obj:
            val = obj.get(key)
            if not _is_finite_number(val) or float(val) < 0 or float(val) > 1:
                errors.append(
                    f"track.{key} must be a finite number in [0, 1] when present "
                    f"(got {_fmt_value(val)})"
                )

    if "team" in obj and obj.get("team") not in ("A", "B", "unknown"):
        errors.append("track.team must be one of: A, B, unknown")

    if "team_confidence" in obj:
        tc = obj.get("team_confidence")
        if "team" not in obj:
            errors.append("track.team_confidence requires track.team to be present")
        if not _is_finite_number(tc) or float(tc) < 0 or float(tc) > 1:
            errors.append(
                f"track.team_confidence must be a finite number in [0, 1] when present "
                f"(got {_fmt_value(tc)})"
            )

    if "segment_id" in obj:
        seg = obj.get("segment_id")
        if not isinstance(seg, str) or not seg.strip():
            errors.append("track.segment_id must be a non-empty string when present")

    if "break_reason" in obj:
        br = obj.get("break_reason")
        allowed = (
            "occlusion",
            "ambiguous_association",
            "out_of_view",
            "detector_missing",
            "manual_reset",
        )
        if br not in allowed:
            errors.append(
                "track.break_reason must be one of: occlusion, ambiguous_association, "
                "out_of_view, detector_missing, manual_reset"
            )

    if "diagnostics" in obj and not isinstance(obj.get("diagnostics"), dict):
        errors.append("track.diagnostics must be an object when present")

    # Ball-specific semantic checks (INT-040 trust-first).
    if entity_type == "ball":
        diag = obj.get("diagnostics")
        if not isinstance(diag, dict):
            errors.append("track.diagnostics is required when entity_type=ball")
            diag = {}
        fi = diag.get(FRAME_INDEX)
        if not isinstance(fi, int) or isinstance(fi, bool) or fi < 0:
            errors.append(f"track.diagnostics.{FRAME_INDEX} must be an integer >= 0 when entity_type=ball")

        if pos_state == "missing":
            mr = diag.get(MISSING_REASON)
            if mr not in MISSING_REASON_VOCAB:
                allowed = ", ".join(MISSING_REASON_VOCAB)
                errors.append(
                    f"track.diagnostics.{MISSING_REASON} must be one of: {allowed} when entity_type=ball and pos_state=missing"
                )
            # For V1, ball missing is always a detector-derived break (coarse category).
            if obj.get("break_reason") != "detector_missing":
                errors.append(
                    "track.break_reason must be detector_missing when entity_type=ball and pos_state=missing"
                )

        if pos_state == "unknown":
            ur = diag.get(UNKNOWN_REASON)
            allowed = (BALL_UNKNOWN_FRAME_UNAVAILABLE, BALL_UNKNOWN_DETECTOR_ERROR)
            if ur not in allowed:
                errors.append(
                    f"track.diagnostics.{UNKNOWN_REASON} must be one of: {allowed[0]}, {allowed[1]} when entity_type=ball and pos_state=unknown"
                )

    return errors


def validate_event_record_v1(obj: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["event record must be a JSON object"]

    schema_version = obj.get("schema_version")
    if schema_version != EVENT_RECORD_SCHEMA_VERSION:
        errors.append(f"event.schema_version must be {EVENT_RECORD_SCHEMA_VERSION}")

    t_ms = obj.get("t_ms")
    if not isinstance(t_ms, int) or isinstance(t_ms, bool) or t_ms < 0:
        errors.append(f"event.t_ms must be an integer >= 0 (got {_fmt_value(t_ms)})")

    event_type = obj.get("event_type")
    if event_type not in ("shot", "goal"):
        errors.append("event.event_type must be one of: shot, goal")

    event_state = obj.get("event_state")
    if event_state not in ("confirmed", "candidate", "unknown"):
        errors.append("event.event_state must be one of: confirmed, candidate, unknown")

    confidence = obj.get("confidence")
    if not _is_finite_number(confidence) or float(confidence) < 0 or float(confidence) > 1:
        errors.append(
            f"event.confidence must be a finite number in [0, 1] (got {_fmt_value(confidence)})"
        )

    evidence = obj.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("event.evidence must be a non-empty list of evidence pointers")
        evidence = []
    allowed_artifact_ids = {s.artifact_id for s in BUNDLE_ARTIFACT_SPECS_V1}
    for idx, ptr in enumerate(evidence):
        if not isinstance(ptr, dict):
            errors.append(f"event.evidence[{idx}] must be an object")
            continue
        artifact_id = ptr.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            errors.append(f"event.evidence[{idx}].artifact_id must be a non-empty string")
        elif artifact_id not in allowed_artifact_ids:
            errors.append(
                f"event.evidence[{idx}].artifact_id must be a known bundle artifact id "
                f"(got {_fmt_value(artifact_id)})"
            )

        has_time = "time_range_ms" in ptr
        has_frame = "frame_range" in ptr
        if has_time == has_frame:
            errors.append(
                f"event.evidence[{idx}] must include exactly one of time_range_ms or frame_range"
            )
            continue

        if has_time:
            tr = ptr.get("time_range_ms")
            if not isinstance(tr, dict):
                errors.append(f"event.evidence[{idx}].time_range_ms must be an object")
            else:
                start_ms = tr.get("start_ms")
                end_ms = tr.get("end_ms")
                if not isinstance(start_ms, int) or isinstance(start_ms, bool) or start_ms < 0:
                    errors.append(f"event.evidence[{idx}].time_range_ms.start_ms must be int >= 0")
                if not isinstance(end_ms, int) or isinstance(end_ms, bool) or end_ms < 0:
                    errors.append(f"event.evidence[{idx}].time_range_ms.end_ms must be int >= 0")
                if (
                    isinstance(start_ms, int)
                    and not isinstance(start_ms, bool)
                    and isinstance(end_ms, int)
                    and not isinstance(end_ms, bool)
                    and end_ms < start_ms
                ):
                    errors.append(
                        f"event.evidence[{idx}].time_range_ms must satisfy end_ms>=start_ms"
                    )

        if has_frame:
            fr = ptr.get("frame_range")
            if not isinstance(fr, dict):
                errors.append(f"event.evidence[{idx}].frame_range must be an object")
            else:
                start_frame = fr.get("start_frame")
                end_frame = fr.get("end_frame")
                if (
                    not isinstance(start_frame, int)
                    or isinstance(start_frame, bool)
                    or start_frame < 0
                ):
                    errors.append(
                        f"event.evidence[{idx}].frame_range.start_frame must be int >= 0"
                    )
                if not isinstance(end_frame, int) or isinstance(end_frame, bool) or end_frame < 0:
                    errors.append(f"event.evidence[{idx}].frame_range.end_frame must be int >= 0")
                if (
                    isinstance(start_frame, int)
                    and not isinstance(start_frame, bool)
                    and isinstance(end_frame, int)
                    and not isinstance(end_frame, bool)
                    and end_frame < start_frame
                ):
                    errors.append(
                        f"event.evidence[{idx}].frame_range must satisfy end_frame>=start_frame"
                    )

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
    with path.open("r", encoding="utf8") as f:
        for idx, raw in enumerate(f, start=1):
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


# Alignment hint for downstream consumers (non-normative).
CONTRACT_V1_SCHEMA_PATH = "pipeline/schemas/contract.v1.schema.json"
CONTRACT_V1_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"


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


def _validate_xy_pair(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_is_finite_number(v) for v in value)
    )


def validate_pitch_template_v1(obj: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["pitch_template must be a JSON object"]
    if obj.get("schema_version") != PITCH_TEMPLATE_SCHEMA_VERSION:
        errors.append(f"pitch_template.schema_version must be {PITCH_TEMPLATE_SCHEMA_VERSION}")
    if not isinstance(obj.get("pitch_template_id"), str) or not obj.get("pitch_template_id"):
        errors.append("pitch_template.pitch_template_id must be a non-empty string")

    dims = obj.get("dimensions_m")
    if not isinstance(dims, dict):
        errors.append("pitch_template.dimensions_m must be an object")
        dims = {}
    for key in ("length", "width"):
        val = dims.get(key)
        if not _is_finite_number(val) or float(val) <= 0:
            errors.append(f"pitch_template.dimensions_m.{key} must be a finite number > 0")

    if obj.get("frame") != "pitch_v1":
        errors.append('pitch_template.frame must be "pitch_v1"')
    if obj.get("frame_origin") != "lower_left_corner":
        errors.append('pitch_template.frame_origin must be "lower_left_corner"')

    markings = obj.get("markings_v1")
    if not isinstance(markings, dict):
        errors.append("pitch_template.markings_v1 must be an object")
        markings = {}
    if markings.get("schema_version") != 1:
        errors.append("pitch_template.markings_v1.schema_version must be 1")
    lines = markings.get("lines")
    if not isinstance(lines, list) or not all(isinstance(x, dict) for x in lines):
        errors.append("pitch_template.markings_v1.lines must be a list of objects")
        lines = []
    for line in lines:
        if not isinstance(line.get("label"), str) or not line.get("label"):
            errors.append("markings_v1.lines[].label must be a non-empty string")
        for pkey in ("p0_xy_m", "p1_xy_m"):
            if not _validate_xy_pair(line.get(pkey)):
                errors.append(f"markings_v1.lines[].{pkey} must be [x_m, y_m] finite numbers")

    return errors


def validate_calibration_input_v1(obj: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["calibration_input must be a JSON object"]
    if obj.get("schema_version") != CALIBRATION_INPUT_SCHEMA_VERSION:
        errors.append(f"calibration_input.schema_version must be {CALIBRATION_INPUT_SCHEMA_VERSION}")
    for key in ("venue_id", "pitch_id", "camera_id"):
        if not isinstance(obj.get(key), str) or not obj.get(key):
            errors.append(f"calibration_input.{key} must be a non-empty string")
    if not isinstance(obj.get("image_pre_undistorted"), bool):
        errors.append("calibration_input.image_pre_undistorted must be a boolean")
    if "image_size_px" in obj:
        val = obj.get("image_size_px")
        if (
            not isinstance(val, list)
            or len(val) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in val)
        ):
            errors.append("calibration_input.image_size_px must be [width, height] positive integers")
    if "source_video_path" in obj and (not isinstance(obj.get("source_video_path"), str) or not obj.get("source_video_path")):
        errors.append("calibration_input.source_video_path must be a non-empty string when present")

    ref = obj.get("pitch_template_ref")
    if not isinstance(ref, dict):
        errors.append("calibration_input.pitch_template_ref must be an object")
        ref = {}
    if not isinstance(ref.get("pitch_template_id"), str) or not ref.get("pitch_template_id"):
        errors.append("calibration_input.pitch_template_ref.pitch_template_id must be a non-empty string")
    return errors


def validate_markings_observations_v1(obj: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["markings_observations must be a JSON object"]
    if obj.get("schema_version") != MARKINGS_OBSERVATIONS_SCHEMA_VERSION:
        errors.append(f"markings_observations.schema_version must be {MARKINGS_OBSERVATIONS_SCHEMA_VERSION}")
    if not isinstance(obj.get("camera_id"), str) or not obj.get("camera_id"):
        errors.append("markings_observations.camera_id must be a non-empty string")
    segments = obj.get("segments")
    if not isinstance(segments, list) or not all(isinstance(x, dict) for x in segments):
        errors.append("markings_observations.segments must be a list of objects")
        segments = []
    for seg in segments:
        if not isinstance(seg.get("label"), str) or not seg.get("label"):
            errors.append("segments[].label must be a non-empty string")
        for key in ("p0_xy_px", "p1_xy_px"):
            if not _validate_xy_pair(seg.get(key)):
                errors.append(f"segments[].{key} must be [x, y] finite numbers")
    return errors


def validate_manual_correspondences_v1(obj: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["manual_correspondences must be a JSON object"]
    if obj.get("schema_version") != MANUAL_CORRESPONDENCES_SCHEMA_VERSION:
        errors.append(f"manual_correspondences.schema_version must be {MANUAL_CORRESPONDENCES_SCHEMA_VERSION}")
    if not isinstance(obj.get("camera_id"), str) or not obj.get("camera_id"):
        errors.append("manual_correspondences.camera_id must be a non-empty string")
    ref = obj.get("pitch_template_ref")
    if not isinstance(ref, dict):
        errors.append("manual_correspondences.pitch_template_ref must be an object")
        ref = {}
    if not isinstance(ref.get("pitch_template_id"), str) or not ref.get("pitch_template_id"):
        errors.append("manual_correspondences.pitch_template_ref.pitch_template_id must be a non-empty string")
    corrs = obj.get("correspondences")
    if not isinstance(corrs, list) or not all(isinstance(x, dict) for x in corrs):
        errors.append("manual_correspondences.correspondences must be a list of objects")
        corrs = []
    if len(corrs) < 12:
        errors.append("manual_correspondences.correspondences must have length >= 12 (V1)")
    for c in corrs:
        if not _validate_xy_pair(c.get("image_xy_px")):
            errors.append("correspondences[].image_xy_px must be [x, y] finite numbers")
        if not _validate_xy_pair(c.get("pitch_xy_m")):
            errors.append("correspondences[].pitch_xy_m must be [x_m, y_m] finite numbers")
    return errors


def validate_calibration_diagnostics_v1(obj: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["calibration diagnostics must be a JSON object"]
    if obj.get("schema_version") != CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION:
        errors.append(f"calibration.schema_version must be {CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION}")
    if obj.get("status") not in ("success", "fail"):
        errors.append('calibration.status must be "success" or "fail"')
    for key in (
        "rms_reprojection_error_px",
        "inlier_ratio",
        "marking_coverage_score_0_to_1",
    ):
        if not _is_finite_number(obj.get(key)):
            errors.append(f"calibration.{key} must be a finite number")
    num_inliers = obj.get("num_inliers")
    if not isinstance(num_inliers, int) or isinstance(num_inliers, bool) or num_inliers < 0:
        errors.append("calibration.num_inliers must be an integer >= 0")
    if obj.get("status") == "fail":
        if not isinstance(obj.get("failure_reason"), str) or not obj.get("failure_reason"):
            errors.append("calibration.failure_reason is required when status=fail")
    return errors
