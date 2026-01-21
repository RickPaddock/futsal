---
generated: true
source: spec/md/docs/data/OUTPUT_CONTRACT.mdt
source_sha256: sha256:2b58d0a779df5d2aaebd912a1631412271afda01800b1fe38b51426826e57728
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

## Canonical track record (JSONL)

Each line is a single `TrackRecordV1` sample (schema version `1`).

Required fields:
- `schema_version`: `1`
- `t_ms`: integer milliseconds
- `entity_type`: `"player"` or `"ball"`
- `entity_id`: string (stable within the bundle; no implicit meaning)
- `track_id`: string (stable within the bundle; may differ from `entity_id`)
- `source`: string (e.g. `"vision_cam_A"`)
- `frame`: `"pitch"`, `"enu"`, or `"wgs84"`
- `pos_state`: `"present"`, `"missing"`, or `"unknown"`

Geometry rules (explicit missing/unknown semantics):
- If `pos_state="present"` and `frame` is `"pitch"` or `"enu"`, then `x_m` and `y_m` are required numbers.
- If `pos_state="present"` and `frame` is `"wgs84"`, then `lat` and `lon` are required numbers.

Optional fields:
- `sigma_m`: number (1σ positional uncertainty in meters)
- `confidence`: number (0..1)
- `quality`: number (0..1)
- `team`: `"A"`, `"B"`, or `"unknown"` (typically for players)
- `diagnostics`: object (free-form, versioned by the schema version)

```json
{"schema_version":1,"t_ms":1234567,"entity_type":"player","entity_id":"camA_entity_12","track_id":"camA_track_12","source":"vision_cam_A","frame":"pitch","pos_state":"present","x_m":8.3,"y_m":12.1,"sigma_m":0.25,"confidence":0.91,"quality":0.86,"team":"unknown"}
```

## Canonical events (`events.json`)

`events.json` is a JSON array of `EventRecordV1` objects (schema version `1`).

Required fields:
- `schema_version`: `1`
- `t_ms`: integer milliseconds
- `event_type`: `"shot"` or `"goal"`
- `confidence`: number (0..1)

Optional fields:
- `source`: string
- `notes`: string
- `diagnostics`: object (free-form)

```json
[
  {"schema_version":1,"t_ms":1234000,"event_type":"shot","confidence":0.92,"source":"vision_cam_A"}
]
```
