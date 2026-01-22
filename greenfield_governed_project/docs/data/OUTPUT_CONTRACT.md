---
generated: true
source: spec/md/docs/data/OUTPUT_CONTRACT.mdt
source_sha256: sha256:b1c2fbac1a64663227eb520a20a7c74adc83e3ab9cd2b6afee623c46077e2803
---

# Output contract (V1)

V1 produces a deterministic per-match bundle under `output/<match_id>/` (bundle layout version `1`).

## Bundle manifest (`manifest.json`)

`manifest.json` is required and is the canonical index for the bundle:

- `schema_version`: `1`
- `bundle_layout_version`: `1`
- `match_id`: string
- `inputs.video_paths`: list of strings (sorted + unique; provenance only)
- `inputs.sensor_paths`: list of strings (sorted + unique; provenance only)
- `artifacts[]`: expected artifact list with stable ids
- `notes` (optional): string

Determinism rules:
- JSON must be written with stable key ordering (e.g. `sort_keys=true`).
- `inputs.video_paths` and `inputs.sensor_paths` must be sorted + unique.
- `artifacts[]` must be sorted by `artifact_id` and must match the V1 expected list exactly.
- No timestamps, random IDs, or machine-specific data beyond the explicit `inputs.*` paths.

### Artifact entries (`artifacts[]`)

Each `artifacts[]` entry is an object:

- `artifact_id`: string (stable id; used for deterministic ordering)
- `path`: string (relative to bundle root)
- `required`: boolean
- `status`: `"present"` | `"missing"`
- `missing_reason` (optional): string (required when `status="missing"`)

For optional artifacts (e.g. `bev.mp4`), include the entry even when missing, with `status="missing"` and an explicit `missing_reason` such as `calibration_gated` or `calibration_failed`.

## Required artifacts

- `overlay.mp4`
  - Player boxes + track IDs (ID breaks are acceptable; swaps are not).
  - Team label (A/B/Unknown) + confidence.
  - Ball marker when tracked + confidence; explicit “ball missing” state.
- `bev.mp4` (best-effort)
  - Bird’s-eye view animation in pitch meters; only when calibration quality passes.
- `manifest.json`
  - Bundle index + provenance + deterministic artifact list.
- `tracks.jsonl`
  - Canonical time series; one JSON object per entity per time sample.
- `events.json`
  - Shots + goals only in V1; conservative (high precision).
- `report.json` + `report.html`
  - Summary stats + confidence + diagnostics and failure reasons.
- `diagnostics/`
  - `calibration.json`
  - `sync.json` (required if multi-camera or sensors are used)
  - `quality_summary.json`

## Calibration artifacts (V1)

V1 supports explicit, versioned calibration inputs and diagnostics. BEV outputs MUST be gated on calibration quality.

### Pitch template (`pitch_template.json`)

Stored per venue/pitch under a stable layout:

`pipeline/fixtures/calibration/venues/<venue_id>/<pitch_id>/pitch_template.json` (example fixtures live under `pipeline/fixtures/`).

Minimum fields:
- `schema_version`: `1`
- `pitch_template_id`: string (stable id, e.g. `<venue_id>/<pitch_id>`)
- `dimensions_m.length`: number
- `dimensions_m.width`: number
- `frame`: `"pitch_v1"`
- `frame_origin`: `"lower_left_corner"`

Pitch frame semantics (`pitch_v1`):
- Units: meters.
- Origin: lower-left corner of the pitch (touchline/goal-line intersection when facing along +x).
- Axes: `x_m` increases along pitch length; `y_m` increases along pitch width.

### Calibration input (`calibration_input.json`)

Stored alongside pitch template:

`pipeline/fixtures/calibration/venues/<venue_id>/<pitch_id>/calibration_input.json`

Minimum fields:
- `schema_version`: `1`
- `venue_id`: string
- `pitch_id`: string
- `pitch_template_ref.pitch_template_id`: string
- `camera_id`: string
- `image_pre_undistorted`: boolean (V1 assumes `true`; if `false`, calibration MUST fail explicitly in V1)
- `source_video_path` (provenance only): string

### Markings observations (`markings_observations.json`)

Auto-fit consumes an explicit markings observations record:

`pipeline/fixtures/calibration/venues/<venue_id>/<pitch_id>/markings_observations.json`

Minimum fields:
- `schema_version`: `1`
- `camera_id`: string
- `frame_ref`: object (e.g. `{ "t_ms": 123456 }`)
- `segments[]`: list of labeled line segments:
  - `label`: string (e.g. `"touchline_left"`, `"touchline_right"`, `"center_line"`, `"goal_box_left"`, `"goal_box_right"`)
  - `p0_xy_px`: `[x, y]`
  - `p1_xy_px`: `[x, y]`

Deterministic V1 coverage score:
- Define required labels set = {`touchline_left`, `touchline_right`, `center_line`, `goal_box_left`, `goal_box_right`}.
- `marking_coverage_score_0_to_1 = (# of required labels present at least once in segments[].label) / 5`.
- If `segments[]` is empty, score is `0.0`.

### Manual correspondences (`manual_correspondences.json`)

When auto-fit fails, V1 supports a minimal manual fallback input record:

`pipeline/fixtures/calibration/venues/<venue_id>/<pitch_id>/manual_correspondences.json`

Minimum fields:
- `schema_version`: `1`
- `camera_id`: string
- `pitch_template_ref.pitch_template_id`: string
- `correspondences[]`: list of point correspondences:
  - `image_xy_px`: `[x, y]`
  - `pitch_xy_m`: `[x_m, y_m]`
  - `label` (optional): string

### Calibration diagnostics (`diagnostics/calibration.json`)

Required minimum fields:
- `schema_version`: `1`
- `status`: `"success"` | `"fail"`
- `rms_reprojection_error_px`: number
- `inlier_ratio`: number (0..1)
- `num_inliers`: integer
- `marking_coverage_score_0_to_1`: number (0..1)
- `failure_reason` (required when fail): string
- `notes` (optional): string

### Calibration gating summary (`diagnostics/quality_summary.json`)

Required minimum fields:
- `schema_version`: `1`
- `bev_gate.status`: `"pass"` | `"fail"`
- `bev_gate.reasons[]`: list of strings (non-empty on fail)
- `bev_gate.metrics`: object including the measured diagnostics metrics
- `bev_gate.thresholds`: object including the threshold numbers used

V1 BEV gating thresholds (no overrides):
- Allow BEV only when all are true:
  - `rms_reprojection_error_px <= 4.0`
  - `inlier_ratio >= 0.60`
  - `num_inliers >= 12`
  - `marking_coverage_score_0_to_1 >= 0.50`

## Canonical track record (JSONL)

Each line is a single `TrackRecordV1` sample (schema version `1`).

Required fields:
- `schema_version`: `1`
- `t_ms`: integer milliseconds
- `entity_type`: `"player"` or `"ball"`
- `entity_id`: string (stable within the bundle; no implicit meaning)
- `track_id`: string (stable within the bundle; may differ from `entity_id`)
- `source`: string (e.g. `"vision_cam_A"`)
- `frame`: `"pitch"`, `"enu"`, `"wgs84"`, or `"image_px"`
- `pos_state`: `"present"`, `"missing"`, or `"unknown"`

Geometry rules (explicit missing/unknown semantics):
- If `pos_state="present"` and `frame` is `"pitch"` or `"enu"`, then `x_m` and `y_m` are required numbers.
- If `pos_state="present"` and `frame` is `"wgs84"`, then `lat` and `lon` are required numbers.
- If `pos_state="present"` and `frame` is `"image_px"`, then `bbox_xyxy_px` is required.

Optional fields:
- `sigma_m`: number (1σ positional uncertainty in meters)
- `confidence`: number (0..1)
- `quality`: number (0..1)
- `team`: `"A"`, `"B"`, or `"unknown"` (typically for players)
- `team_confidence`: number (0..1) (only when `team` is present)
- `segment_id`: string (required when emitting explicit track segments for players)
- `break_reason`: string (required when a player track segment ends due to ambiguity/gating)
- `bbox_xyxy_px`: `[x1, y1, x2, y2]` integers (required when `frame="image_px"`; optional otherwise)
- `diagnostics`: object (structured; stable keys listed below)

### Player detections (trust-first)

Player detections MAY be emitted as `entity_type="player"` records in `frame="image_px"` with `track_id==entity_id` (no identity continuity), or as part of tracking output.

Minimum recommended fields for detection-style records:
- `entity_type="player"`
- `frame="image_px"`
- `pos_state="present"` or `"unknown"`
- `bbox_xyxy_px`
- `confidence` (0..1)
- `diagnostics.gating_reason` when detections are suppressed or downgraded

### Player tracking (swap-avoidant segmentation)

When tracking is enabled, player records SHOULD use stable `track_id` and emit `segment_id` for contiguous spans.

Trust-first rule:
- If association is ambiguous, end the current segment and emit `break_reason` rather than swapping the identity of an existing `track_id`.

`break_reason` vocabulary (finite, minimum set):
- `occlusion`
- `ambiguous_association`
- `out_of_view`
- `detector_missing`
- `manual_reset`

Example segment end (explicit break instead of swap):

```json
{"schema_version":1,"t_ms":1235000,"entity_type":"player","entity_id":"camA_entity_12","track_id":"camA_track_12","segment_id":"seg_03","source":"vision_cam_A","frame":"image_px","pos_state":"missing","confidence":0.62,"break_reason":"ambiguous_association","diagnostics":{"association_score":0.12}}
```

### Team assignment (A/B/Unknown)

Team assignment is expressed via `team`:
- Team A: `team="A"`
- Team B: `team="B"`
- Unknown: `team="unknown"`

When `team` is present, emit `team_confidence` and include smoothing parameters in diagnostics.

### Diagnostics keys (stable)

Diagnostics MUST be an object and SHOULD use stable keys so audits do not depend on ad-hoc strings.

Recommended stable keys (extendable):
- `gating_reason`: string (e.g., `"low_confidence"`, `"outside_pitch_roi"`)
- `association_score`: number (0..1)
- `color_evidence`: object (summary stats; not raw images)
- `smoothing`: object (e.g., `{ "window_frames": 15, "hysteresis": 0.1 }`)
- `unknown_reason`: string (why `pos_state` or `team` is unknown)

```json
{"schema_version":1,"t_ms":1234567,"entity_type":"player","entity_id":"camA_entity_12","track_id":"camA_track_12","segment_id":"seg_03","source":"vision_cam_A","frame":"pitch","pos_state":"present","x_m":8.3,"y_m":12.1,"sigma_m":0.25,"confidence":0.91,"quality":0.86,"team":"unknown","team_confidence":0.42,"diagnostics":{"smoothing":{"window_frames":15,"hysteresis":0.1}}}
```

```json
{"schema_version":1,"t_ms":1234567,"entity_type":"player","entity_id":"det_000123_05","track_id":"det_000123_05","source":"vision_cam_A","frame":"image_px","pos_state":"present","bbox_xyxy_px":[120,80,200,300],"confidence":0.77,"diagnostics":{"gating_reason":"none"}}
```

## Canonical events (`events.json`)

`events.json` is a JSON array of `EventRecordV1` objects (schema version `1`).

Required fields:
- `schema_version`: `1`
- `t_ms`: integer milliseconds
- `event_type`: `"shot"` or `"goal"`
- `event_state`: `"confirmed"`, `"candidate"`, or `"unknown"`
- `confidence`: number (0..1)
- `evidence[]`: list of evidence pointers (must be non-empty)

Event state semantics (trust-first):
- `confirmed`: high-confidence; MUST NOT be emitted unless evidence is sufficient and unambiguous.
- `candidate`: plausible but uncertain; safe to show in UI but MUST NOT be counted as truth by default.
- `unknown`: ambiguous/low-quality; primarily for audit visibility.

### Evidence pointers

Each `evidence[]` entry MUST be an object:
- `artifact_id`: string (MUST match a bundle artifact id from the V1 manifest, e.g. `tracks_jsonl`, `overlay_mp4`, `events_json`)
- Exactly one of:
  - `time_range_ms`: `{ "start_ms": int >= 0, "end_ms": int >= 0 }` with `end_ms >= start_ms`
  - `frame_range`: `{ "start_frame": int >= 0, "end_frame": int >= 0 }` with `end_frame >= start_frame`

Optional fields:
- `source`: string
- `notes`: string
- `diagnostics`: object (free-form)

Example (conservative shot candidate with time-bounded evidence on `tracks.jsonl`):

```json
{
  "schema_version": 1,
  "t_ms": 1234500,
  "event_type": "shot",
  "event_state": "candidate",
  "confidence": 0.62,
  "source": "shots_goals_v1",
  "evidence": [
    {
      "artifact_id": "tracks_jsonl",
      "time_range_ms": { "start_ms": 1234200, "end_ms": 1234800 }
    }
  ],
  "diagnostics": { "speed_px_per_s": 942.1 }
}
```

```json
[
  {"schema_version":1,"t_ms":1234000,"event_type":"shot","confidence":0.92,"source":"vision_cam_A"}
]
```
