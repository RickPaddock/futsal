---
generated: true
source: spec/md/docs/runbooks/calibration-and-bev.mdt
source_sha256: sha256:142a84e5232f331bb7449461350d217a121adc3c783f1faf5dc2a459516d266e
---

# Calibration + BEV mapping (V1)

This runbook is a single-source navigation entry for camera→pitch calibration artifacts and BEV gating semantics.

## What this covers

- How calibration inputs and pitch templates are represented on disk (versioned JSON).
- How auto-fit consumes markings observations and emits diagnostics.
- How manual correspondences can be used when auto-fit fails.
- How calibration quality deterministically gates BEV outputs (trust-first).

Out of scope:
- Building an interactive UI for calibration.
- Guaranteed calibration success for every venue (V1 prefers explicit failure + diagnostics).

## Stable on-disk layout (V1)

All calibration artefacts live under:

`pipeline/fixtures/calibration/venues/<venue_id>/<pitch_id>/`

Expected files:
- `pitch_template.json`
- `calibration_input.json`
- `markings_observations.json` (for auto-fit)
- `manual_correspondences.json` (for manual fallback)

## Coordinate frames (V1)

Pitch frame is `pitch_v1`:
- Units: meters
- Origin: lower-left corner
- Axes: `x_m` along pitch length, `y_m` along width

Image frame is pixel coordinates:
- `xy_px`: `[x, y]` in pixels
- V1 assumption: frames are pre-undistorted (`image_pre_undistorted: true`)

## Auto-fit workflow

Inputs:
- `pitch_template.json`
- `calibration_input.json`
- `markings_observations.json` (labeled line segments in pixel coordinates)

Outputs (generated bundle diagnostics):
- `diagnostics/calibration.json` with explicit `status: success|fail` and required metrics

Trust-first rules:
- If markings are missing/insufficient or fit is unreliable, return explicit failure with diagnostics (do not guess).
- Distortion handling is explicit: V1 assumes pre-undistorted inputs; if not, fail explicitly.

## Manual fallback workflow

When auto-fit fails, provide `manual_correspondences.json` with explicit versioning and coordinate frames.

Trust-first rules:
- Validate correspondences strictly (minimum count, bounds, schema_version).
- Emit explicit failures with actionable errors when invalid.

## BEV gating (deterministic)

BEV is allowed only when all thresholds pass:
- `rms_reprojection_error_px <= 4.0`
- `inlier_ratio >= 0.60`
- `num_inliers >= 12`
- `marking_coverage_score_0_to_1 >= 0.50`

On fail:
- `bev.mp4` MUST NOT be produced.
- `manifest.json` MUST still include an `artifacts[]` entry for `bev.mp4` with `status="missing"` and an explicit `missing_reason` (e.g. `calibration_gated`).
- `diagnostics/quality_summary.json` MUST include `bev_gate.status="fail"` and reasons, plus metrics+thresholds.

## Evidence-first checks (recommended)

When implementing INT-020 tasks, ensure there are deterministic commands that validate fixtures and emit the expected diagnostics/gating records.
