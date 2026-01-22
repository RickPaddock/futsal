"""fusbal_pipeline.calibration.manual

PROV: FUSBAL.PIPELINE.TASK_CAL_003.SUB_001
REQ: FUSBAL-V1-CAL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
WHY: Provide a deterministic manual correspondence fallback that produces a calibration result + diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from fusbal_pipeline.calibration.auto_fit import (
    _least_squares_homography_pitch_to_image,
    _project_pitch_to_image,
)
from fusbal_pipeline.calibration.diagnostics import CalibrationDiagnosticsV1, euclidean_px, rms
from fusbal_pipeline.errors import ERROR, ValidationError, make_error

MANUAL_CORRESPONDENCES_SCHEMA_VERSION = 1


class PitchTemplateRefV1(TypedDict):
    pitch_template_id: str


class ManualCorrespondenceV1(TypedDict):
    image_xy_px: list[float]
    pitch_xy_m: list[float]
    label: NotRequired[str]


class ManualCorrespondencesV1(TypedDict):
    schema_version: Literal[1]
    camera_id: str
    pitch_template_ref: PitchTemplateRefV1
    correspondences: list[ManualCorrespondenceV1]


class ManualCalibrationResultV1(TypedDict):
    schema_version: Literal[1]
    calibration_kind: Literal["manual"]
    pitch_template_id: str
    camera_id: str
    homography_pitch_to_image_3x3: list[list[float]]
    diagnostics: CalibrationDiagnosticsV1


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _validate_xy(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    return all(_is_finite_number(v) for v in value)


def validate_manual_correspondences_v1(
    obj: object, *, path: str | None = None
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if not isinstance(obj, dict):
        return [
            make_error(
                ERROR.CAL_MANUAL_INVALID,
                "manual_correspondences must be a JSON object",
                path=path,
            )
        ]

    if obj.get("schema_version") != MANUAL_CORRESPONDENCES_SCHEMA_VERSION:
        errors.append(
            make_error(
                ERROR.CAL_MANUAL_INVALID,
                f"manual_correspondences.schema_version must be {MANUAL_CORRESPONDENCES_SCHEMA_VERSION}",
                path=path,
                field="schema_version",
                value=obj.get("schema_version"),
            )
        )

    if not isinstance(obj.get("camera_id"), str) or not obj.get("camera_id"):
        errors.append(
            make_error(
                ERROR.CAL_MANUAL_INVALID,
                "manual_correspondences.camera_id must be a non-empty string",
                path=path,
                field="camera_id",
                value=obj.get("camera_id"),
            )
        )

    ref = obj.get("pitch_template_ref")
    if not isinstance(ref, dict):
        errors.append(
            make_error(
                ERROR.CAL_MANUAL_INVALID,
                "manual_correspondences.pitch_template_ref must be an object",
                path=path,
                field="pitch_template_ref",
                value=ref,
            )
        )
        ref = {}
    if not isinstance(ref.get("pitch_template_id"), str) or not ref.get("pitch_template_id"):
        errors.append(
            make_error(
                ERROR.CAL_MANUAL_INVALID,
                "manual_correspondences.pitch_template_ref.pitch_template_id must be a non-empty string",
                path=path,
                field="pitch_template_ref.pitch_template_id",
                value=ref.get("pitch_template_id"),
            )
        )

    corrs = obj.get("correspondences")
    if not isinstance(corrs, list) or not all(isinstance(x, dict) for x in corrs):
        errors.append(
            make_error(
                ERROR.CAL_MANUAL_INVALID,
                "manual_correspondences.correspondences must be a list of objects",
                path=path,
                field="correspondences",
                value=corrs,
            )
        )
        corrs = []

    if isinstance(corrs, list) and len(corrs) < 12:
        errors.append(
            make_error(
                ERROR.CAL_MANUAL_INVALID,
                "manual_correspondences.correspondences must have length >= 12 (V1)",
                path=path,
                field="correspondences",
                value=len(corrs),
            )
        )

    for idx, c in enumerate(corrs):
        if not _validate_xy(c.get("image_xy_px")):
            errors.append(
                make_error(
                    ERROR.CAL_MANUAL_INVALID,
                    "correspondences[].image_xy_px must be [x, y] finite numbers",
                    path=path,
                    index=idx,
                    field="correspondences[].image_xy_px",
                    value=c.get("image_xy_px"),
                )
            )
        if not _validate_xy(c.get("pitch_xy_m")):
            errors.append(
                make_error(
                    ERROR.CAL_MANUAL_INVALID,
                    "correspondences[].pitch_xy_m must be [x_m, y_m] finite numbers",
                    path=path,
                    index=idx,
                    field="correspondences[].pitch_xy_m",
                    value=c.get("pitch_xy_m"),
                )
            )
        if "label" in c and not (isinstance(c.get("label"), str) and c.get("label") != ""):
            errors.append(
                make_error(
                    ERROR.CAL_MANUAL_INVALID,
                    "correspondences[].label must be a string when present",
                    path=path,
                    index=idx,
                    field="correspondences[].label",
                    value=c.get("label"),
                )
            )

    return errors


def load_manual_correspondences(
    path: Path,
) -> tuple[ManualCorrespondencesV1 | None, list[ValidationError]]:
    try:
        obj = json.loads(path.read_text(encoding="utf8"))
    except FileNotFoundError:
        return None, [make_error(ERROR.CAL_MANUAL_MISSING, f"missing manual correspondences: {path}", path=str(path))]
    except json.JSONDecodeError as e:
        return None, [make_error(ERROR.CAL_MANUAL_INVALID_JSON, f"invalid JSON: {e}", path=str(path))]

    errors = validate_manual_correspondences_v1(obj, path=str(path))
    if errors:
        return None, errors
    return obj, []


def _manual_fit_result(manual: ManualCorrespondencesV1) -> ManualCalibrationResultV1:
    pitch_xy_m: list[tuple[float, float]] = []
    image_xy_px: list[tuple[float, float]] = []
    for c in manual["correspondences"]:
        pitch_xy_m.append((float(c["pitch_xy_m"][0]), float(c["pitch_xy_m"][1])))
        image_xy_px.append((float(c["image_xy_px"][0]), float(c["image_xy_px"][1])))

    h = _least_squares_homography_pitch_to_image(pitch_xy_m, image_xy_px)
    if h is None:
        diagnostics: CalibrationDiagnosticsV1 = {
            "schema_version": 1,
            "status": "fail",
            "rms_reprojection_error_px": float("inf"),
            "inlier_ratio": 0.0,
            "num_inliers": 0,
            "marking_coverage_score_0_to_1": 1.0,
            "failure_reason": "fit_singular",
        }
        return {
            "schema_version": 1,
            "calibration_kind": "manual",
            "pitch_template_id": manual["pitch_template_ref"]["pitch_template_id"],
            "camera_id": manual["camera_id"],
            "homography_pitch_to_image_3x3": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "diagnostics": diagnostics,
        }

    per_point: list[float] = []
    for (x, y), (u_gt, v_gt) in zip(pitch_xy_m, image_xy_px, strict=True):
        uv = _project_pitch_to_image(h, x, y)
        if uv is None:
            per_point.append(float("inf"))
            continue
        per_point.append(euclidean_px(uv, (u_gt, v_gt)))
    rms_err = rms(per_point)

    diagnostics = {
        "schema_version": 1,
        "status": "success",
        "rms_reprojection_error_px": float(rms_err),
        "inlier_ratio": 1.0,
        "num_inliers": int(len(manual["correspondences"])),
        "marking_coverage_score_0_to_1": 1.0,
    }
    return {
        "schema_version": 1,
        "calibration_kind": "manual",
        "pitch_template_id": manual["pitch_template_ref"]["pitch_template_id"],
        "camera_id": manual["camera_id"],
        "homography_pitch_to_image_3x3": [[float(x) for x in row] for row in h],
        "diagnostics": diagnostics,
    }


def cmd_manual_fit(args: argparse.Namespace) -> int:
    manual, errors = load_manual_correspondences(Path(args.manual))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, sort_keys=True))
        return 2
    assert manual is not None
    result = _manual_fit_result(manual)
    payload = {"ok": result["diagnostics"]["status"] == "success", "result": result}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fusbal_pipeline.calibration.manual")
    parser.add_argument("--manual", required=True, help="Path to manual_correspondences.json")
    args = parser.parse_args(argv)
    return cmd_manual_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())

