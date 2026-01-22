"""fusbal_pipeline.calibration.pitch_templates

PROV: FUSBAL.PIPELINE.TASK_CAL_001.SUB_001
REQ: FUSBAL-V1-CAL-001, SYS-ARCH-15
WHY: Define a minimal, versioned pitch template schema (dimensions + labeled markings) for calibration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, TypedDict

from fusbal_pipeline.errors import ERROR, ValidationError, make_error

PITCH_TEMPLATE_SCHEMA_VERSION = 1
PitchFrame = Literal["pitch_v1"]
PitchFrameOrigin = Literal["lower_left_corner"]

# Required markings set for V1 marking coverage scoring (see INT-020/TASK-CAL-002).
REQUIRED_MARKING_LABELS_V1: tuple[str, ...] = (
    "touchline_left",
    "touchline_right",
    "center_line",
    "goal_box_left",
    "goal_box_right",
)


class PitchDimensionsM(TypedDict):
    length: float
    width: float


class MarkingLineV1(TypedDict):
    label: str
    p0_xy_m: list[float]  # [x_m, y_m]
    p1_xy_m: list[float]  # [x_m, y_m]


class MarkingsModelV1(TypedDict):
    schema_version: Literal[1]
    lines: list[MarkingLineV1]


class PitchTemplateV1(TypedDict):
    schema_version: Literal[1]
    pitch_template_id: str
    dimensions_m: PitchDimensionsM
    frame: PitchFrame
    frame_origin: PitchFrameOrigin
    markings_v1: MarkingsModelV1


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _validate_point_xy_m(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    return all(_is_finite_number(v) for v in value)


def validate_pitch_template_v1(obj: object, *, path: str | None = None) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if not isinstance(obj, dict):
        return [make_error(ERROR.CAL_PITCH_TEMPLATE_INVALID, "pitch_template must be a JSON object", path=path)]

    if obj.get("schema_version") != PITCH_TEMPLATE_SCHEMA_VERSION:
        errors.append(
            make_error(
                ERROR.CAL_PITCH_TEMPLATE_INVALID,
                f"pitch_template.schema_version must be {PITCH_TEMPLATE_SCHEMA_VERSION}",
                path=path,
                field="schema_version",
                value=obj.get("schema_version"),
            )
        )

    pitch_template_id = obj.get("pitch_template_id")
    if not isinstance(pitch_template_id, str) or not pitch_template_id.strip():
        errors.append(
            make_error(
                ERROR.CAL_PITCH_TEMPLATE_INVALID,
                "pitch_template.pitch_template_id must be a non-empty string",
                path=path,
                field="pitch_template_id",
                value=pitch_template_id,
            )
        )

    dims = obj.get("dimensions_m")
    if not isinstance(dims, dict):
        errors.append(
            make_error(
                ERROR.CAL_PITCH_TEMPLATE_INVALID,
                "pitch_template.dimensions_m must be an object",
                path=path,
                field="dimensions_m",
                value=dims,
            )
        )
        dims = {}

    for key in ("length", "width"):
        val = dims.get(key)
        if not _is_finite_number(val) or float(val) <= 0:
            errors.append(
                make_error(
                    ERROR.CAL_PITCH_TEMPLATE_INVALID,
                    f"pitch_template.dimensions_m.{key} must be a finite number > 0",
                    path=path,
                    field=f"dimensions_m.{key}",
                    value=val,
                )
            )

    if obj.get("frame") != "pitch_v1":
        errors.append(
            make_error(
                ERROR.CAL_PITCH_TEMPLATE_INVALID,
                'pitch_template.frame must be "pitch_v1"',
                path=path,
                field="frame",
                value=obj.get("frame"),
            )
        )

    if obj.get("frame_origin") != "lower_left_corner":
        errors.append(
            make_error(
                ERROR.CAL_PITCH_TEMPLATE_INVALID,
                'pitch_template.frame_origin must be "lower_left_corner"',
                path=path,
                field="frame_origin",
                value=obj.get("frame_origin"),
            )
        )

    markings = obj.get("markings_v1")
    if not isinstance(markings, dict):
        errors.append(
            make_error(
                ERROR.CAL_PITCH_TEMPLATE_INVALID,
                "pitch_template.markings_v1 must be an object",
                path=path,
                field="markings_v1",
                value=markings,
            )
        )
        markings = {}

    if markings.get("schema_version") != 1:
        errors.append(
            make_error(
                ERROR.CAL_PITCH_TEMPLATE_INVALID,
                "pitch_template.markings_v1.schema_version must be 1",
                path=path,
                field="markings_v1.schema_version",
                value=markings.get("schema_version"),
            )
        )

    lines = markings.get("lines")
    if not isinstance(lines, list) or not all(isinstance(x, dict) for x in lines):
        errors.append(
            make_error(
                ERROR.CAL_PITCH_TEMPLATE_INVALID,
                "pitch_template.markings_v1.lines must be a list of objects",
                path=path,
                field="markings_v1.lines",
                value=lines,
            )
        )
        lines = []

    seen: set[str] = set()
    for idx, line in enumerate(lines):
        label = line.get("label")
        if not isinstance(label, str) or not label.strip():
            errors.append(
                make_error(
                    ERROR.CAL_PITCH_TEMPLATE_INVALID,
                    "markings_v1.lines[].label must be a non-empty string",
                    path=path,
                    index=idx,
                    field="markings_v1.lines[].label",
                    value=label,
                )
            )
            continue
        if label in seen:
            errors.append(
                make_error(
                    ERROR.CAL_PITCH_TEMPLATE_INVALID,
                    "markings_v1.lines[].label must be unique",
                    path=path,
                    index=idx,
                    field="markings_v1.lines[].label",
                    value=label,
                )
            )
        seen.add(label)

        for pkey in ("p0_xy_m", "p1_xy_m"):
            pval = line.get(pkey)
            if not _validate_point_xy_m(pval):
                errors.append(
                    make_error(
                        ERROR.CAL_PITCH_TEMPLATE_INVALID,
                        f"markings_v1.lines[].{pkey} must be [x_m, y_m] finite numbers",
                        path=path,
                        index=idx,
                        field=f"markings_v1.lines[].{pkey}",
                        value=pval,
                    )
                )

    # The template can include more markings, but V1 auto-fit expects the required set.
    missing_required = [lbl for lbl in REQUIRED_MARKING_LABELS_V1 if lbl not in seen]
    if missing_required:
        errors.append(
            make_error(
                ERROR.CAL_PITCH_TEMPLATE_INVALID,
                "pitch_template missing required markings labels for V1: "
                + ", ".join(missing_required),
                path=path,
                field="markings_v1.lines",
                value=missing_required,
            )
        )

    return errors


def load_pitch_template(path: Path) -> tuple[PitchTemplateV1 | None, list[ValidationError]]:
    try:
        obj = json.loads(path.read_text(encoding="utf8"))
    except FileNotFoundError:
        return None, [make_error(ERROR.CAL_PITCH_TEMPLATE_MISSING, f"missing pitch template: {path}", path=str(path))]
    except json.JSONDecodeError as e:
        return None, [
            make_error(
                ERROR.CAL_PITCH_TEMPLATE_INVALID_JSON, f"invalid JSON: {e}", path=str(path)
            )
        ]

    errors = validate_pitch_template_v1(obj, path=str(path))
    if errors:
        return None, errors
    return obj, []


def lines_by_label(template: PitchTemplateV1) -> dict[str, MarkingLineV1]:
    out: dict[str, MarkingLineV1] = {}
    for line in template["markings_v1"]["lines"]:
        out[line["label"]] = line
    return out

