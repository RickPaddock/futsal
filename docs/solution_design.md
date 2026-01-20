# Futsal Tracking — High-Level Design (V1 → Premium)

This document defines a complete, implementation-ready high-level design for an offline (post-processed) futsal analysis system that produces **trustworthy** player + ball tracking, conservative events, and a bird’s-eye tactical view. It is written to be specific enough that another LLM (or engineer) can implement the system end-to-end.

## 0) Product framing (trust-first)

### Target user
- Venue owners/operators and regular players of **outdoor futsal** who want automated match stats and replays with minimal setup.

### Core promise
- Upload (or ingest) match video from **venue-installed cameras** (1–4 supported; **2 recommended**) and receive:
  - An **overlay video** with player tracks/IDs and ball tracking when visible.
  - A **bird’s-eye view (BEV)** reconstruction with movement and conservative events (shots/goals v1).
  - A **report** (stats + confidence + “what we’re unsure about”).

### Non-negotiables (V1)
- **Accuracy and trust > features.** The system must avoid hallucinating the ball/events and avoid identity swaps.
- **Low friction onboarding.**
  - Venue install is a **one-time calibration**, with periodic health checks and quick re-calibration if bumped.
  - Bib colors are allowed/required to improve team separation.
- **No special hardware required for baseline.**
  - Sensors (UWB/GNSS/wearables) are *optional upsells* and must never be required for a sellable output.
  - Future sensor contract is defined in `docs/sensor_integration.md`.

## 1) V1 definition (concrete)

### V1 environment assumptions
- Outdoor futsal.
- Venue camera mounts up to ~3m height, **not overhead**.
- Cameras are fixed during a match (minor shake acceptable; no handheld).
- Venue provides two distinct **bib colors**; all players wear one of the two bib colors.
- Cloud batch processing is allowed; compute cost should be minimized by design.

### V1 required outputs
Deliver a bundle per match:
1) `overlay.mp4`
   - Player boxes + **track IDs** (IDs may break; swaps are unacceptable).
   - Team color label (A/B/Unknown) + confidence.
   - Ball marker when tracked + confidence; explicit “ball missing” state.
2) `bev.mp4` (bird’s-eye animation)
   - Player positions projected into pitch coordinates.
   - Ball position when available.
3) `tracks.jsonl` (canonical time series; one record per entity per time sample)
4) `events.json` (shots + goals only in v1, conservative)
5) `report.json` + `report.html` (stats + confidence + diagnostics)

### V1 acceptability / error modes (trust-preserving)
- **Ball tracking:** may be missing for long periods. Must not hallucinate. Prefer “unknown” over wrong.
- **Player identities:** prefer **ID breaks** over swaps. Track continuity is secondary to correctness.
- **Events:** high precision, low recall is OK. If uncertain, omit or mark as “candidate”.
- **Calibration:** if pitch mapping is weak, still deliver overlay and non-BEV stats; BEV outputs may be disabled with clear reason.

### Explicitly out-of-scope for V1
- Passes/possession attribution (requires robust ball track + possession model).
- Jersey number OCR as a requirement.
- Tactical formations / press metrics / networks.
- Real-time processing.

## 2) Tiering / ROI ladder (cameras + sensors)

### Base (sellable)
- Venue cameras + bib colors only.
- 1–4 cameras supported; **2 cameras recommended default install**.
- Outputs: tracking + BEV + conservative shots/goals best-effort.

### Pro (upsell)
- Adds ball sensor stream (GNSS/UWB/etc.) if available in future:
  - Improves shot/goal reliability and reduces compute by narrowing vision search.

### Premium (upsell)
- Adds UWB-in-bibs for players:
  - Stabilizes identity and positions; reduces camera-only failure modes and GPU load.

Sensors and their integration contracts are future work; the architecture must treat them as optional `TrackSource`s (see `docs/sensor_integration.md`).

## 3) System architecture (offline batch)

### High-level components
1) **Ingest**
   - Read N video files, metadata, and optional sensor logs.
   - Produce normalized clips (optional trimming/compression for iteration; see `utils/trim_compress_video.py`).
2) **Calibration**
   - Per camera: lens handling (optional), pitch model fit, homography to pitch plane.
   - Cross-camera: time sync and camera-to-pitch alignment.
3) **Per-camera vision**
   - Player detection → per-camera tracking → appearance embeddings.
   - Ball detection → ball tracking (often intermittent).
4) **Multi-camera fusion**
   - Merge per-camera tracks into pitch-space tracks with confidence.
5) **Team assignment**
   - Use bib color models + temporal smoothing; output A/B/Unknown.
6) **Event inference**
   - Shots/goals in v1, conservative confidence thresholds.
7) **Rendering + reporting**
   - Overlay + BEV videos, JSON outputs, HTML report with diagnostics.

### Operating principle: “best available sources”
- The pipeline must always produce a baseline output from whatever sources pass quality checks.
- Optional sources (extra cameras, user cameras, sensors) are only used if they can be aligned + synced with sufficient confidence.

### Suggested repository structure (implementation target)
This repo is currently minimal; a typical build-out:
- `futsal/`
  - `cli.py` (entrypoints)
  - `config/` (schemas + defaults)
  - `io/` (video + sensor ingest)
  - `calibration/` (pitch model, homography, quality checks)
  - `vision/`
    - `detectors/` (players, ball)
    - `tracking/` (per-camera MOT)
    - `reid/` (embeddings + association)
  - `fusion/` (multi-camera fusion, track stitching)
  - `events/` (shots/goals)
  - `render/` (overlay + BEV)
  - `report/` (html/json reports)
- `docs/` (this spec + calibration guide)
- `utils/` (small scripts like point clicking, trimming)

## 4) Data contracts (what modules exchange)

### Canonical coordinate frames
- **Image frame:** pixels `(u_px, v_px)` per camera.
- **Pitch frame:** meters `(x_m, y_m)` in a standardized pitch coordinate system:
  - `x_m` increases left→right along length.
  - `y_m` increases top→bottom along width.
  - Origin at top-left corner when viewed in BEV.

### Canonical track record (internal)
Use the schema described in `docs/sensor_integration.md` for all sources. In practice, vision modules should emit pitch-frame tracks too (after calibration).

Example JSONL record:
```json
{"t_ms":1234567,"entity_type":"player","entity_id":"camA_track_12","source":"vision_cam_A","frame":"pitch","x_m":8.3,"y_m":12.1,"sigma_m":0.25,"quality":0.86}
```

### Match bundle layout (output)
- `output/<match_id>/`
  - `overlay.mp4`
  - `bev.mp4`
  - `tracks.jsonl`
  - `events.json`
  - `report.json`
  - `report.html`
  - `diagnostics/`
    - `calibration.json`
    - `sync.json`
    - `quality_summary.json`

## 5) Camera strategy

### Recommended “2-camera default” placements (non-overhead)
Define a small set of enforceable presets to reduce calibration variance:
- **Preset S+G (recommended):**
  - Cam A: high sideline near halfway, slightly angled down.
  - Cam B: behind-goal or goal-side corner, capturing goal mouth + near corner.
- **Preset S+S (simpler):**
  - Two high sideline cameras on opposite sides for occlusion reduction.
- **Preset S+C (ok):**
  - Sideline + corner.

Each preset comes with:
- Expected visible markings for calibration (touchlines, penalty box lines, center circle).
- A “quality checklist” to accept/reject a setup.

### Supporting 1–4 venue cameras
Implementation note:
- Treat each camera independently up to pitch mapping and per-camera tracking.
- Multi-camera fusion consumes only pitch-frame tracks, so adding cameras is additive.

### User cameras (“additive only”)
User cameras are accepted only if:
- Time sync confidence is high enough, and
- Pitch alignment quality passes thresholds.
Otherwise, user camera is ignored and the system still produces the venue-camera result.

## 6) Calibration & pitch mapping

### Inputs
- Video frames from each camera.
- Known pitch dimensions (or a small set of supported pitch templates):
  - Standard futsal sizes vary; store per venue/pitch.

### Steps
1) **Lens handling**
   - If lens intrinsics are known (camera model), undistort fisheye/wide-angle.
   - Otherwise, do not overcomplicate v1: rely on robust line fitting + homography; treat residual distortion as uncertainty.
2) **Auto pitch feature detection**
   - Detect: touchlines, goal lines, penalty box lines, center circle arc if visible.
   - Use edge + line detection + semantic segmentation for field markings as needed.
3) **Model fit**
   - Fit a parametric pitch model to detected lines.
   - Solve for homography (or piecewise mapping if required later).
4) **Quality gating**
   - Compute reprojection error on detected features.
   - Compute expected metric sanity checks (e.g., distances between known lines).
   - Output `calibration_quality` (0–1) and failure reasons.
5) **Manual fallback**
   - If auto fit fails, use a minimal click flow to collect correspondences (see `utils/click_points.py`).
   - Store `pitch_calibration.json` per pitch and reuse across matches.

### Calibration outputs
- Per camera:
  - Homography `H_img_to_pitch`
  - Inverse `H_pitch_to_img` (for overlay)
  - Quality score + diagnostics

## 7) Per-camera player detection

### Model requirements
- Must handle:
  - Small/medium players at distance
  - Partial occlusions
  - Outdoor lighting (shadows/glare)
  - Bib colors and non-uniform kits

### Output
- Per frame: list of detections with:
  - `bbox_xyxy`, `score`, `class="player"`, optional `pose_keypoints` (future)

### Compute controls (cost)
- Run detector at reduced FPS (e.g., 10–15fps) then interpolate/track between frames.
- Adaptive scheduling:
  - Increase detection FPS during crowded play or sudden motion.

## 8) Per-camera multi-object tracking (MOT)

### Goals
- Produce stable tracklets with low ID swaps.
- It is acceptable to break tracks rather than swap identities.

### Recommended approach
- Tracking-by-detection with a strong association method:
  - Motion model (Kalman)
  - IoU matching
  - Appearance embeddings (ReID)
  - Optional camera-specific constraints (field boundaries)

### Track output
- `track_id` per camera, per frame (or per timestamp).
- Track confidence derived from detection scores + association stability.

### Identity policy (trust-first)
- Penalize associations that would swap identities in close interactions.
- Prefer ending a track and starting a new one when ambiguous.

## 9) Appearance embeddings (ReID) and stitching

### Purpose
- Re-acquire players after occlusions or leaving/entering frame.
- Support cross-camera association later (optional; pitch-space fusion can often avoid cross-camera ReID for v1).

### Usage policy
- Compute embeddings:
  - On keyframes per tracklet, not every frame.
- Use embeddings only when motion/IoU are insufficient and confidence is high.

## 10) Team assignment (bib colors)

### Requirements
- Bib colors are required (two distinct colors).
- The system outputs team A/B/Unknown with confidence per player track.

### Algorithm (robust, simple)
1) For each track, sample pixels inside torso region across time (avoid shorts/skin/background).
2) Convert to a stable color space (e.g., Lab).
3) Fit a 2-cluster model (k-means/GMM) across all tracks, weighted by sample quality.
4) Assign each track to cluster with temporal smoothing.
5) Output “Unknown” when confidence is low (e.g., goalkeeper wearing different kit, or severe lighting issues).

### Manual override
- Provide a lightweight override (e.g., swap cluster labels; mark a track as team A/B).

## 11) Ball detection & tracking (best-effort in v1)

### Problem reality
- Ball is small, fast, frequently occluded, and visually ambiguous.
- V1 must avoid hallucinating the ball.

### Strategy
- Dedicated small-object ball detector.
- Candidate filtering:
  - Ignore impossible locations (outside field mask).
  - Ignore detections inconsistent with plausible speed.
- Temporal linking:
  - Kalman filter with strong gating.
  - Explicit missing state.

### Confidence output
- Ball track points include `quality` and `sigma_m`.
- Downstream events only consume ball states above thresholds.

## 12) Multi-camera fusion (pitch-space)

### Inputs
- Per-camera player tracks mapped to pitch coordinates with uncertainties.
- Per-camera ball tracks in pitch coordinates.

### Time sync
V1 requirements:
- Cameras are “close enough” synced, or we can estimate offsets.
Strategies:
- Use audio correlation (crowd/kicks/whistle) to estimate offset.
- Use visual correlation (global motion patterns) when audio is absent.
- Persist `sync.json` per match with `offset_ms` per camera and a confidence score.

### Fusion algorithm (simple, robust)
- For each timestamp bucket:
  - Cluster observations in pitch space (players) using distance threshold scaled by uncertainty.
  - For each fused entity:
    - Choose a representative position by weighted average (inverse variance).
    - Keep provenance (which cameras contributed).
  - Maintain fused tracks over time with a separate pitch-space tracker.

### Cross-camera identity (v1 policy)
- Default: treat fusion as “multiple noisy measurements of the same physical players” without trying to preserve a single global ID across cameras unless confidence is very high.
- If you do global IDs in v1, enforce the “no swaps” policy:
  - Prefer splitting IDs rather than incorrectly merging.

## 13) Event inference (v1: shots + goals only)

### Inputs
- Ball track with confidence.
- Player tracks.
- Goal geometry in pitch coordinates (from pitch template).

### Definitions (conservative)
- **Shot candidate**
  - Ball is tracked continuously for a minimum window (e.g., 0.5–1.0s).
  - Ball velocity magnitude exceeds threshold.
  - Ball trajectory heads toward the goal region.
  - Confidence requires good ball quality for most of the window.
- **Goal**
  - Ball crosses goal line region in pitch coordinates with high confidence, OR
  - Ball disappears near goal line following a high-confidence shot and reappears outside play in a consistent way (lower confidence).

### Output
- Each event includes:
  - `type`, `t_start_ms`, `t_end_ms`
  - `confidence`
  - `supporting_evidence` (ball quality stats, contributing cameras)

### V1 non-goals
- Do not infer passes/possession unless ball tracking is robust enough to meet a precision target.

## 14) Rendering

### Overlay video
- For each camera:
  - Draw player boxes, track IDs, team label, and confidence.
  - Draw ball marker if tracked, plus “missing” indicator if not.
  - Draw calibration mask/lines optionally as a debug toggle.

### Bird’s-eye video
- Render pitch template with:
  - Player dots (colored by team), smoothed trajectories.
  - Ball dot when available.
  - Event markers for shots/goals.

## 15) Reporting (trust UI)

The report is as important as accuracy: it communicates uncertainty.

### Required report sections
- Calibration quality per camera (pass/warn/fail) and why.
- Tracking quality metrics (e.g., track fragmentation rate; time with “unknown” ball).
- Stats:
  - Per team: possession is v2; in v1 only ball-visible stats.
  - Per player: distance, speed percentiles, heatmaps.
- Events list with confidence and replay timestamps.

### Trust-preserving UX rules
- Never present low-confidence outputs as definitive.
- Prefer “unknown” to wrong, especially for events and identity.

## 16) Compute & cost controls (cloud batch)

Given 40 matches/pitch/week at 60 minutes each:
- Video minutes/week/pitch = 2400 per camera.
- Compute scales roughly linearly with camera count unless sensors reduce vision workload.

Primary levers:
- Detector FPS reduction + CPU interpolation.
- Only run expensive ReID on track breaks.
- Ball detection scheduling (candidate windows).
- Early exits on low calibration quality (skip BEV and event inference to save cost).

## 17) Quality metrics and acceptance gates

### Offline metrics (internal validation)
- Tracking:
  - IDF1 / HOTA on annotated clips.
  - Identity swaps per minute (must be very low).
  - Fragmentation rate (acceptable if swaps are avoided).
- Ball:
  - “Hallucination rate” (false ball presence) should be near zero.
  - Visible recall can be moderate; precision must be high.
- Events:
  - Precision target high (e.g., ≥90%) even if recall is low.
- Calibration:
  - Reprojection error thresholds; pitch-scale sanity checks.

### Runtime gates (per match)
Produce a `quality_summary.json` and decide what to ship:
- If calibration fails: ship overlay only + non-BEV stats.
- If ball quality low: ship tracking + BEV, but events omitted or marked candidate-only.

## 18) Implementation roadmap (LLM-friendly)

### Milestone 0 — scaffolding
- Create CLI skeleton and output bundle layout.
- Implement `MatchConfig` (pitch dims, camera list, calibration paths).

### Milestone 1 — single-camera baseline
- Implement:
  - Video ingest
  - Manual calibration fallback (reuse `utils/click_points.py` output)
  - Player detection + tracking
  - BEV projection
  - Overlay + BEV render

### Milestone 2 — team assignment (bibs)
- Add robust bib clustering + confidence + overrides.

### Milestone 3 — best-effort ball + shots/goals
- Add ball detection/tracking + conservative event inference.

### Milestone 4 — two-camera support
- Add time sync estimation, per-camera calibration, pitch-space fusion.
- Define and enforce installation presets + quality checks.

### Milestone 5 — premium hooks (no implementation yet)
- Implement adapters/interfaces for sensors per `docs/sensor_integration.md` (parsers + canonical model).
- Do not fuse until quality gates and pitch geo exist.

## 19) Interfaces / CLIs (suggested)

### Core batch command
Example:
```bash
python -m futsal.cli process \
  --match-id 2026-01-20_pitch1_19h \
  --pitch-config input/pitches/pitch1.json \
  --video camA=input/videos/pitch1_camA.mp4 \
  --video camB=input/videos/pitch1_camB.mp4 \
  --out output/2026-01-20_pitch1_19h
```

### Calibration command
```bash
python -m futsal.cli calibrate \
  --pitch-config input/pitches/pitch1.json \
  --video camA=input/videos/pitch1_camA.mp4
```

### Sensor ingest (future)
```bash
python -m futsal.cli ingest-sensor \
  --format gpx \
  --entity ball \
  --input input/sensors/ball.gpx \
  --out output/<match_id>/diagnostics/ball_sensor.jsonl
```

## 20) Notes on GNSS / user GPS (future; not in current cycle)

The system must be able to integrate user-owned GNSS-enabled balls and GPS data later, but this is not part of the current development cycle.

Design requirement:
- Implement sensor ingestion as isolated adapters that output canonical records (`tracks.jsonl`-compatible).
- Treat GNSS as a low-rate, noisy prior unless proven otherwise; never let it degrade the baseline camera-only output.

See `docs/sensor_integration.md` for the integration contract.

