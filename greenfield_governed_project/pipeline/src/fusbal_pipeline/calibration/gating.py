"""fusbal_pipeline.calibration.gating

PROV: FUSBAL.PIPELINE.TASK_CAL_003.SUB_002
REQ: FUSBAL-V1-TRUST-001, FUSBAL-V1-BEV-001, SYS-ARCH-15
WHY: Enforce deterministic V1 calibration quality gates for BEV output suppression with explicit reasons.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Literal, TypedDict

from fusbal_pipeline.calibration.inputs import load_calibration_input
from fusbal_pipeline.errors import ERROR, ValidationError, make_error

GATING_SCHEMA_VERSION = 1

THRESHOLDS_V1 = {
    "rms_reprojection_error_px_max": 4.0,
    "inlier_ratio_min": 0.60,
    "num_inliers_min": 12,
    "marking_coverage_score_min": 0.50,
}


class BevGateV1(TypedDict):
    status: Literal["pass", "fail"]
    reasons: list[str]
    metrics: dict[str, Any]
    thresholds: dict[str, Any]


class CalibrationGatingSummaryV1(TypedDict):
    schema_version: Literal[1]
    bev_gate: BevGateV1


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _load_json(path: Path) -> tuple[object | None, list[ValidationError]]:
    try:
        return json.loads(path.read_text(encoding="utf8")), []
    except FileNotFoundError:
        return None, [make_error(ERROR.CAL_RESULT_MISSING, f"missing calibration result: {path}", path=str(path))]
    except json.JSONDecodeError as e:
        return None, [make_error(ERROR.CAL_RESULT_INVALID_JSON, f"invalid JSON: {e}", path=str(path))]


def _extract_metrics(result_obj: object, *, path: str) -> tuple[dict[str, Any] | None, list[ValidationError]]:
    if not isinstance(result_obj, dict):
        return None, [make_error(ERROR.CAL_RESULT_INVALID, "calibration result must be a JSON object", path=path)]

    diagnostics = result_obj.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None, [
            make_error(
                ERROR.CAL_RESULT_INVALID,
                "calibration result diagnostics must be an object",
                path=path,
                field="diagnostics",
                value=diagnostics,
            )
        ]

    metrics: dict[str, Any] = {}
    for key in (
        "rms_reprojection_error_px",
        "inlier_ratio",
        "num_inliers",
        "marking_coverage_score_0_to_1",
        "status",
        "failure_reason",
    ):
        metrics[key] = diagnostics.get(key)
    return metrics, []


def gate_bev_v1(metrics: dict[str, Any]) -> BevGateV1:
    reasons: list[str] = []
    status = metrics.get("status")
    if status != "success":
        reasons.append("calibration_failed")

    rms_err = metrics.get("rms_reprojection_error_px")
    inlier_ratio = metrics.get("inlier_ratio")
    num_inliers = metrics.get("num_inliers")
    coverage = metrics.get("marking_coverage_score_0_to_1")

    if not _is_finite_number(rms_err):
        reasons.append("missing_rms_reprojection_error_px")
    elif float(rms_err) > THRESHOLDS_V1["rms_reprojection_error_px_max"]:
        reasons.append("rms_reprojection_error_px_above_threshold")

    if not _is_finite_number(inlier_ratio):
        reasons.append("missing_inlier_ratio")
    elif float(inlier_ratio) < THRESHOLDS_V1["inlier_ratio_min"]:
        reasons.append("inlier_ratio_below_threshold")

    if not isinstance(num_inliers, int) or isinstance(num_inliers, bool):
        reasons.append("missing_num_inliers")
    elif int(num_inliers) < int(THRESHOLDS_V1["num_inliers_min"]):
        reasons.append("num_inliers_below_threshold")

    if not _is_finite_number(coverage):
        reasons.append("missing_marking_coverage_score")
    elif float(coverage) < THRESHOLDS_V1["marking_coverage_score_min"]:
        reasons.append("marking_coverage_score_below_threshold")

    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "metrics": {
            "status": status,
            "failure_reason": metrics.get("failure_reason"),
            "rms_reprojection_error_px": rms_err,
            "inlier_ratio": inlier_ratio,
            "num_inliers": num_inliers,
            "marking_coverage_score_0_to_1": coverage,
        },
        "thresholds": dict(THRESHOLDS_V1),
    }


def cmd_gate(args: argparse.Namespace) -> int:
    _, input_errors = load_calibration_input(Path(args.inputs))
    result_obj, result_errors = _load_json(Path(args.result))
    errors = input_errors + result_errors
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, sort_keys=True))
        return 2
    assert result_obj is not None
    metrics, metric_errors = _extract_metrics(result_obj, path=str(args.result))
    if metric_errors:
        print(json.dumps({"ok": False, "errors": metric_errors}, indent=2, sort_keys=True))
        return 2
    assert metrics is not None
    gate = gate_bev_v1(metrics)
    out: CalibrationGatingSummaryV1 = {"schema_version": 1, "bev_gate": gate}
    payload = {"ok": gate["status"] == "pass", "gating": out}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fusbal_pipeline.calibration.gating")
    parser.add_argument("--inputs", required=True, help="Path to calibration_input.json (context)")
    parser.add_argument("--result", required=True, help="Path to auto-fit or manual calibration result JSON")
    args = parser.parse_args(argv)
    return cmd_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
