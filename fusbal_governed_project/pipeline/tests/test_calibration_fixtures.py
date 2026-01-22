# PROV: FUSBAL.PIPELINE.TESTS.CALIBRATION_FIXTURES.01
# REQ: SYS-ARCH-15, FUSBAL-V1-CAL-001, FUSBAL-V1-TRUST-001
# WHY: Ensure calibration fixtures validate and produce deterministic success/failure semantics for guardrails.

from __future__ import annotations

from pathlib import Path

from fusbal_pipeline.calibration.auto_fit import (
    _auto_fit_from_required_markings,
    load_markings_observations,
)
from fusbal_pipeline.calibration.gating import THRESHOLDS_V1, gate_bev_v1
from fusbal_pipeline.calibration.inputs import load_calibration_input
from fusbal_pipeline.calibration.manual import _manual_fit_result, load_manual_correspondences
from fusbal_pipeline.calibration.pitch_templates import load_pitch_template


def _fixture_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "calibration"
        / "venues"
        / "example_venue"
        / "pitch_a"
    )


def test_autofit_fixture_produces_success_and_gate_passes() -> None:
    d = _fixture_dir()
    inputs, in_errs = load_calibration_input(d / "calibration_input.json")
    pitch, p_errs = load_pitch_template(d / "pitch_template.json")
    markings, m_errs = load_markings_observations(d / "markings_observations.json")
    assert not (in_errs or p_errs or m_errs)
    assert inputs is not None and pitch is not None and markings is not None

    result = _auto_fit_from_required_markings(inputs=inputs, pitch_template=pitch, markings=markings)
    diag = result["diagnostics"]
    assert diag["status"] == "success"
    assert diag["rms_reprojection_error_px"] <= THRESHOLDS_V1["rms_reprojection_error_px_max"]
    assert diag["inlier_ratio"] >= THRESHOLDS_V1["inlier_ratio_min"]
    assert diag["num_inliers"] >= THRESHOLDS_V1["num_inliers_min"]
    assert diag["marking_coverage_score_0_to_1"] >= THRESHOLDS_V1["marking_coverage_score_min"]

    gate = gate_bev_v1(diag)
    assert gate["status"] == "pass"


def test_manual_fixture_produces_success_and_gate_passes() -> None:
    d = _fixture_dir()
    manual, errs = load_manual_correspondences(d / "manual_correspondences.json")
    assert not errs
    assert manual is not None

    result = _manual_fit_result(manual)
    diag = result["diagnostics"]
    assert diag["status"] == "success"
    gate = gate_bev_v1(diag)
    assert gate["status"] == "pass"

