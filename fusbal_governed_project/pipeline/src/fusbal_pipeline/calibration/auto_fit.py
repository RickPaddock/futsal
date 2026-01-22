"""fusbal_pipeline.calibration.auto_fit

PROV: FUSBAL.PIPELINE.TASK_CAL_002.SUB_002
REQ: FUSBAL-V1-CAL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
WHY: Implement a deterministic V1 auto-fit from labeled field markings to a planar homography with diagnostics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from fusbal_pipeline.calibration.diagnostics import CalibrationDiagnosticsV1, euclidean_px, rms
from fusbal_pipeline.calibration.inputs import CalibrationInputV1, load_calibration_input
from fusbal_pipeline.calibration.pitch_templates import (
    REQUIRED_MARKING_LABELS_V1,
    PitchTemplateV1,
    lines_by_label,
    load_pitch_template,
)
from fusbal_pipeline.errors import ERROR, ValidationError, make_error

MARKINGS_OBSERVATIONS_SCHEMA_VERSION = 1
CALIBRATION_RESULT_SCHEMA_VERSION = 1


class MarkingSegmentV1(TypedDict):
    label: str
    p0_xy_px: list[float]
    p1_xy_px: list[float]


class MarkingsObservationsV1(TypedDict):
    schema_version: Literal[1]
    camera_id: str
    frame_ref: NotRequired[dict[str, object]]
    segments: list[MarkingSegmentV1]


class CalibrationResultV1(TypedDict):
    schema_version: Literal[1]
    calibration_kind: Literal["auto_fit"]
    pitch_template_id: str
    camera_id: str
    homography_pitch_to_image_3x3: list[list[float]]
    diagnostics: CalibrationDiagnosticsV1


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _validate_xy_px(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    return all(_is_finite_number(v) for v in value)


def validate_markings_observations_v1(
    obj: object, *, path: str | None = None
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if not isinstance(obj, dict):
        return [make_error(ERROR.CAL_MARKINGS_INVALID, "markings_observations must be a JSON object", path=path)]

    if obj.get("schema_version") != MARKINGS_OBSERVATIONS_SCHEMA_VERSION:
        errors.append(
            make_error(
                ERROR.CAL_MARKINGS_INVALID,
                f"markings_observations.schema_version must be {MARKINGS_OBSERVATIONS_SCHEMA_VERSION}",
                path=path,
                field="schema_version",
                value=obj.get("schema_version"),
            )
        )

    if not isinstance(obj.get("camera_id"), str) or not obj.get("camera_id"):
        errors.append(
            make_error(
                ERROR.CAL_MARKINGS_INVALID,
                "markings_observations.camera_id must be a non-empty string",
                path=path,
                field="camera_id",
                value=obj.get("camera_id"),
            )
        )

    segments = obj.get("segments")
    if not isinstance(segments, list) or not all(isinstance(x, dict) for x in segments):
        errors.append(
            make_error(
                ERROR.CAL_MARKINGS_INVALID,
                "markings_observations.segments must be a list of objects",
                path=path,
                field="segments",
                value=segments,
            )
        )
        segments = []

    for idx, seg in enumerate(segments):
        label = seg.get("label")
        if not isinstance(label, str) or not label.strip():
            errors.append(
                make_error(
                    ERROR.CAL_MARKINGS_INVALID,
                    "segments[].label must be a non-empty string",
                    path=path,
                    index=idx,
                    field="segments[].label",
                    value=label,
                )
            )
        for pkey in ("p0_xy_px", "p1_xy_px"):
            pval = seg.get(pkey)
            if not _validate_xy_px(pval):
                errors.append(
                    make_error(
                        ERROR.CAL_MARKINGS_INVALID,
                        f"segments[].{pkey} must be [x, y] finite numbers",
                        path=path,
                        index=idx,
                        field=f"segments[].{pkey}",
                        value=pval,
                    )
                )

    return errors


def load_markings_observations(path: Path) -> tuple[MarkingsObservationsV1 | None, list[ValidationError]]:
    try:
        obj = json.loads(path.read_text(encoding="utf8"))
    except FileNotFoundError:
        return None, [make_error(ERROR.CAL_MARKINGS_MISSING, f"missing markings observations: {path}", path=str(path))]
    except json.JSONDecodeError as e:
        return None, [make_error(ERROR.CAL_MARKINGS_INVALID_JSON, f"invalid JSON: {e}", path=str(path))]

    errors = validate_markings_observations_v1(obj, path=str(path))
    if errors:
        return None, errors
    return obj, []


def marking_coverage_score_v1(segments: list[MarkingSegmentV1]) -> float:
    if not segments:
        return 0.0
    present = {s.get("label") for s in segments if isinstance(s.get("label"), str)}
    required = set(REQUIRED_MARKING_LABELS_V1)
    return float(len(required.intersection(present))) / 5.0


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(a)
    if n == 0 or any(len(row) != n for row in a) or len(b) != n:
        return None
    # Gaussian elimination with partial pivoting (deterministic).
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        inv = 1.0 / m[col][col]
        for j in range(col, n + 1):
            m[col][j] *= inv
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def _least_squares_homography_pitch_to_image(
    pitch_xy_m: list[tuple[float, float]],
    image_xy_px: list[tuple[float, float]],
) -> list[list[float]] | None:
    if len(pitch_xy_m) != len(image_xy_px) or len(pitch_xy_m) < 4:
        return None

    a_rows: list[list[float]] = []
    b_vals: list[float] = []
    for (x, y), (u, v) in zip(pitch_xy_m, image_xy_px, strict=True):
        a_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        b_vals.append(u)
        a_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        b_vals.append(v)

    # Normal equations: (A^T A) h = A^T b
    ata = [[0.0 for _ in range(8)] for _ in range(8)]
    atb = [0.0 for _ in range(8)]
    for row, bv in zip(a_rows, b_vals, strict=True):
        for i in range(8):
            atb[i] += row[i] * bv
            ri = row[i]
            for j in range(8):
                ata[i][j] += ri * row[j]

    h = _solve_linear_system(ata, atb)
    if h is None:
        return None
    h11, h12, h13, h21, h22, h23, h31, h32 = h
    return [
        [h11, h12, h13],
        [h21, h22, h23],
        [h31, h32, 1.0],
    ]


def _project_pitch_to_image(h: list[list[float]], x: float, y: float) -> tuple[float, float] | None:
    denom = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(denom) < 1e-12:
        return None
    u = (h[0][0] * x + h[0][1] * y + h[0][2]) / denom
    v = (h[1][0] * x + h[1][1] * y + h[1][2]) / denom
    return (u, v)


def _compute_fit_metrics(
    h: list[list[float]],
    pitch_xy_m: list[tuple[float, float]],
    image_xy_px: list[tuple[float, float]],
    *,
    inlier_threshold_px: float,
) -> tuple[float, int, float, list[float]]:
    per_point: list[float] = []
    for (x, y), (u_gt, v_gt) in zip(pitch_xy_m, image_xy_px, strict=True):
        uv = _project_pitch_to_image(h, x, y)
        if uv is None:
            per_point.append(float("inf"))
            continue
        per_point.append(euclidean_px(uv, (u_gt, v_gt)))
    num_inliers = sum(1 for e in per_point if e <= inlier_threshold_px)
    inlier_ratio = float(num_inliers) / float(len(per_point)) if per_point else 0.0
    return rms(per_point), num_inliers, inlier_ratio, per_point


def _pick_first_segment_by_label(segments: list[MarkingSegmentV1]) -> dict[str, MarkingSegmentV1]:
    out: dict[str, MarkingSegmentV1] = {}
    for seg in segments:
        label = seg.get("label")
        if not isinstance(label, str):
            continue
        if label not in out:
            out[label] = seg
    return out


def _auto_fit_from_required_markings(
    *,
    inputs: CalibrationInputV1,
    pitch_template: PitchTemplateV1,
    markings: MarkingsObservationsV1,
) -> CalibrationResultV1:
    coverage = marking_coverage_score_v1(markings["segments"])

    if inputs["image_pre_undistorted"] is not True:
        diagnostics: CalibrationDiagnosticsV1 = {
            "schema_version": 1,
            "status": "fail",
            "rms_reprojection_error_px": float("inf"),
            "inlier_ratio": 0.0,
            "num_inliers": 0,
            "marking_coverage_score_0_to_1": coverage,
            "failure_reason": "image_not_pre_undistorted_v1",
        }
        return {
            "schema_version": 1,
            "calibration_kind": "auto_fit",
            "pitch_template_id": pitch_template["pitch_template_id"],
            "camera_id": inputs["camera_id"],
            "homography_pitch_to_image_3x3": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "diagnostics": diagnostics,
        }

    seg_by_label = _pick_first_segment_by_label(markings["segments"])
    missing = [lbl for lbl in REQUIRED_MARKING_LABELS_V1 if lbl not in seg_by_label]
    if missing:
        diagnostics = {
            "schema_version": 1,
            "status": "fail",
            "rms_reprojection_error_px": float("inf"),
            "inlier_ratio": 0.0,
            "num_inliers": 0,
            "marking_coverage_score_0_to_1": coverage,
            "failure_reason": "insufficient_marking_coverage",
            "notes": "missing required labels: " + ", ".join(missing),
        }
        return {
            "schema_version": 1,
            "calibration_kind": "auto_fit",
            "pitch_template_id": pitch_template["pitch_template_id"],
            "camera_id": inputs["camera_id"],
            "homography_pitch_to_image_3x3": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "diagnostics": diagnostics,
        }

    line_map = lines_by_label(pitch_template)
    pitch_lines = [line_map[lbl] for lbl in REQUIRED_MARKING_LABELS_V1]
    image_segs = [seg_by_label[lbl] for lbl in REQUIRED_MARKING_LABELS_V1]

    pitch_points: list[tuple[float, float]] = []
    image_points: list[tuple[float, float]] = []
    for pl, seg in zip(pitch_lines, image_segs, strict=True):
        p0 = (float(pl["p0_xy_m"][0]), float(pl["p0_xy_m"][1]))
        p1 = (float(pl["p1_xy_m"][0]), float(pl["p1_xy_m"][1]))
        pm = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
        u0 = (float(seg["p0_xy_px"][0]), float(seg["p0_xy_px"][1]))
        u1 = (float(seg["p1_xy_px"][0]), float(seg["p1_xy_px"][1]))
        um = ((u0[0] + u1[0]) / 2.0, (u0[1] + u1[1]) / 2.0)
        # Add endpoints + midpoint for each marking to reach >= 12 correspondences deterministically.
        pitch_points.extend([p0, p1, pm])
        image_points.extend([u0, u1, um])

    # Orientation ambiguity: each labeled segment may be flipped.
    best: tuple[float, str, list[list[float]], int, float] | None = None
    inlier_threshold_px = 4.0
    for flips in itertools.product([False, True], repeat=len(REQUIRED_MARKING_LABELS_V1)):
        pitch_xy_m: list[tuple[float, float]] = []
        image_xy_px: list[tuple[float, float]] = []
        for i, flip in enumerate(flips):
            p0 = pitch_points[3 * i]
            p1 = pitch_points[3 * i + 1]
            pm = pitch_points[3 * i + 2]
            u0 = image_points[3 * i]
            u1 = image_points[3 * i + 1]
            um = image_points[3 * i + 2]
            if not flip:
                pitch_xy_m.extend([p0, p1, pm])
                image_xy_px.extend([u0, u1, um])
            else:
                pitch_xy_m.extend([p0, p1, pm])
                image_xy_px.extend([u1, u0, um])

        h = _least_squares_homography_pitch_to_image(pitch_xy_m, image_xy_px)
        if h is None:
            continue
        rms_err, num_inliers, inlier_ratio, _ = _compute_fit_metrics(
            h, pitch_xy_m, image_xy_px, inlier_threshold_px=inlier_threshold_px
        )
        key = "".join("1" if f else "0" for f in flips)
        if best is None or (rms_err, key) < (best[0], best[1]):
            best = (rms_err, key, h, num_inliers, inlier_ratio)

    if best is None:
        diagnostics = {
            "schema_version": 1,
            "status": "fail",
            "rms_reprojection_error_px": float("inf"),
            "inlier_ratio": 0.0,
            "num_inliers": 0,
            "marking_coverage_score_0_to_1": coverage,
            "failure_reason": "fit_singular",
        }
        return {
            "schema_version": 1,
            "calibration_kind": "auto_fit",
            "pitch_template_id": pitch_template["pitch_template_id"],
            "camera_id": inputs["camera_id"],
            "homography_pitch_to_image_3x3": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "diagnostics": diagnostics,
        }

    rms_err, _key, h, num_inliers, inlier_ratio = best

    # Trust-first: only succeed if V1 BEV thresholds are met (see TASK-CAL-003).
    reasons: list[str] = []
    if rms_err > 4.0:
        reasons.append("rms_reprojection_error_px_above_threshold")
    if inlier_ratio < 0.60:
        reasons.append("inlier_ratio_below_threshold")
    if num_inliers < 12:
        reasons.append("num_inliers_below_threshold")
    if coverage < 0.50:
        reasons.append("marking_coverage_score_below_threshold")

    if reasons:
        diagnostics = {
            "schema_version": 1,
            "status": "fail",
            "rms_reprojection_error_px": float(rms_err),
            "inlier_ratio": float(inlier_ratio),
            "num_inliers": int(num_inliers),
            "marking_coverage_score_0_to_1": float(coverage),
            "failure_reason": "quality_thresholds_not_met",
            "notes": "reasons: " + ", ".join(reasons),
        }
    else:
        diagnostics = {
            "schema_version": 1,
            "status": "success",
            "rms_reprojection_error_px": float(rms_err),
            "inlier_ratio": float(inlier_ratio),
            "num_inliers": int(num_inliers),
            "marking_coverage_score_0_to_1": float(coverage),
        }

    return {
        "schema_version": 1,
        "calibration_kind": "auto_fit",
        "pitch_template_id": pitch_template["pitch_template_id"],
        "camera_id": inputs["camera_id"],
        "homography_pitch_to_image_3x3": [[float(x) for x in row] for row in h],
        "diagnostics": diagnostics,
    }


def _resolve_pitch_template_path(inputs_path: Path, pitch_template_arg: str | None) -> Path:
    if pitch_template_arg:
        return Path(pitch_template_arg)
    return inputs_path.parent / "pitch_template.json"


def cmd_auto_fit(args: argparse.Namespace) -> int:
    inputs_path = Path(args.inputs)
    markings_path = Path(args.markings)
    pitch_template_path = _resolve_pitch_template_path(inputs_path, getattr(args, "pitch_template", None))

    inputs, in_errs = load_calibration_input(inputs_path)
    pitch, p_errs = load_pitch_template(pitch_template_path)
    markings, m_errs = load_markings_observations(markings_path)
    errors = in_errs + p_errs + m_errs
    if errors:
        payload = {"ok": False, "errors": errors}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    assert inputs is not None and pitch is not None and markings is not None
    result = _auto_fit_from_required_markings(inputs=inputs, pitch_template=pitch, markings=markings)
    payload = {"ok": result["diagnostics"]["status"] == "success", "result": result}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fusbal_pipeline.calibration.auto_fit")
    parser.add_argument("--inputs", required=True, help="Path to calibration_input.json")
    parser.add_argument("--markings", required=True, help="Path to markings_observations.json")
    parser.add_argument(
        "--pitch-template",
        help="Optional path to pitch_template.json (defaults to sibling of --inputs).",
    )
    args = parser.parse_args(argv)
    return cmd_auto_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
