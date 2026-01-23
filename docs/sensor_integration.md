# Sensor Integration (Future-Proofing)

This project’s MVP should be **sellable with venue cameras + bib colors only**. Sensors (GNSS balls, UWB tags, player GPS wearables) are **optional inputs** that must *only improve* outputs when present and healthy, and must never be required for a baseline result.

## Design principle
Treat every input as a `TrackSource` that can emit time-stamped positions in a known coordinate frame with an uncertainty estimate. The core pipeline consumes a **canonical track format** and performs fusion/selection based on quality gates.

If a sensor is missing or low quality, the system falls back to camera-only tracking without changing the output contract.

## Canonical data model (internal)
Represent any source (vision, UWB, GNSS, etc.) as samples:

- `t_ms`: integer milliseconds (monotonic or UTC epoch; specify `timebase` in metadata)
- `entity_type`: `"player"` or `"ball"`
- `entity_id`: stable id within the source (e.g., `"track_12"`, `"tag_03"`, `"ball_gnss"`)
- `frame`: coordinate frame of `x_m,y_m`:
  - `"pitch"`: pitch coordinates in meters (preferred for fusion)
  - `"enu"`: local tangent plane meters (east/north), anchored to a venue origin
  - `"wgs84"`: latitude/longitude (ingest only; must be converted before fusion)
- `x_m`, `y_m`: floats (meters) when `frame != "wgs84"`
- `lat`, `lon`: floats when `frame == "wgs84"`
- `sigma_m` or `cov_2x2`: uncertainty (required for gating / weighting)
- `quality`: 0–1 optional score (derived from fix type / DOP / vendor quality)
- `source`: `"vision_cam_A"`, `"uwb"`, `"gnss_ball"`, `"player_gps_phone"`, etc.

**Suggested storage:** JSON Lines (`.jsonl`) for large time series; JSON for metadata.

## Ingest formats (external)
Support adapters that parse common exports into the canonical model:
- CSV/JSON (vendor SDK exports)
- GPX (common consumer GPS)
- FIT (sports devices; optional, if needed)

Adapters should be isolated per vendor/format and output the canonical model.

## Venue georeference (outdoor)
To fuse GNSS with video, the system needs a mapping from WGS84 → pitch meters.

Recommended approach (one-time per pitch):
- Capture 2–4 pitch anchor points in WGS84 (corners are best) via a phone GPS survey or vendor-provided pitch geo data.
- Store anchors as `pitch_geo.json` (per pitch): corner `lat/lon` + the corresponding pitch coordinates `(0,0), (L,0), (0,W), (L,W)`.
- Convert sensor WGS84 samples → ENU → pitch meters using that mapping.

If pitch geo is missing, GNSS can still be ingested, but must be treated as **non-fusable** (no BEV fusion; optionally show as “coarse location” only).

## Time sync (video ↔ sensor)
Prefer sensors that provide UTC timestamps. Otherwise:
- Estimate time offset via correlation between sensor speed spikes and vision ball-speed spikes (or audio kick peaks).
- Provide a simple manual fallback: user supplies “kick at hh:mm:ss” marker.

Always persist estimated offset and a sync quality score.

## Fusion policy (recommended defaults)
- **Ball:** if GNSS/ball sensor quality is high, use it as a prior to constrain vision search + fill gaps; never “snap” to low-quality GNSS.
- **Players:** if UWB-in-bibs is present, use it to stabilize identity and positions; use vision to render overlays and recover from tag dropouts.
- **User GPS wearables:** treat primarily as *identity hints and stat validation* (distance/speed), not as authoritative positions on a small pitch unless accuracy is proven.

## Product tiers (conceptual)
- **Base:** venue cameras + bib colors only (always works).
- **Pro:** adds ball sensor / GNSS ball (improves shots/goals reliability).
- **Premium:** adds UWB-in-bibs for player identity/positions (+ optional ball).

## Non-goals for sensor integration
- Do not require a specific vendor in the core design.
- Do not block camera-only processing if sensor ingestion fails.
- Do not assume indoor GNSS reliability (this repo targets outdoor futsal, but deployments may vary).
