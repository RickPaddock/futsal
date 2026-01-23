# MVP Plan (Single-Camera, Trust-First, GCP Batch)

This document defines the **MVP** plan for a futsal tracking product optimized for:
- **Single sideline wide-angle camera** (static).
- **Two bib colors required** (Team A / Team B).
- **Trust-first outputs**: prefer **missing/unknown/breaks** over wrong positions/identities.
- **Sellable outputs**: overlay + BEV (when calibrated) + conservative highlights + basic running stats.
- **GCP batch processing** with cost controls.

This MVP is intentionally not “perfect tracking everywhere”. It is “high-trust tracking where we have evidence” plus explicit uncertainty everywhere else.

---

## 0) MVP definition of done (phased delivery)

A user can upload a match video (single camera) and receive a deterministic bundle:
- `overlay.mp4` (playable, useful)
- `tracks.jsonl` (players; and ball once the learned ball model is integrated)
- `events.json` (shots/goals; conservative; candidate/unknown allowed once ball is available)
- `report.html` + `report.json` (confidence + known gaps)
- `bev.mp4` (only when calibration gate passes)
- `diagnostics/` (calibration + quality + run metadata)

And:
- **Identity trust bar:** no silent identity swaps (breaks allowed).
- **BEV trust bar:** no “wrong meters”: BEV is gated; stats derived from BEV are only computed when gated-on.
- **Cost bar:** the system has a “low cost” mode with bounded heavy-model usage (SAM2 only when needed).

This MVP is delivered in milestones. **MVP-0 ships player-only tracking** while the learned ball model and SAM2 checkpoint are being prepared.

---

## 1) Core product decisions (locked)

1) **Single camera MVP.** Plan for multi-camera later (venue cameras and/or user cameras at arbitrary heights/angles).
2) **Bibs required in MVP.** Always Team A/B (internally allow low-confidence; UI can carry-forward).
3) **Passes/possession are V2.**
4) **Identity policy:** “no swaps, breaks allowed” for truth outputs; optional “estimated continuity” may exist only as a clearly-marked visualization.
5) **BEV is gated:** only output BEV + meters-based stats when calibration quality passes thresholds.

---

## 2) Architecture (single camera, offline)

### 2.1 Data model (two layers)

**Truth layer (drives stats/events):**
- Only uses evidence-backed detections/tracks.
- Emits explicit `pos_state` for players and ball:
  - `present` (evidence-backed)
  - `missing` (evaluable but no acceptable position)
  - `unknown` (unevaluable; must include reason)
- Identity policy: break rather than swap when association is ambiguous.

**Visualization layer (optional, UX continuity):**
- May render “ghost/estimated occluded” markers for short spans.
- Must be visually distinct (dashed/ghosted) and excluded from stats/events.

### 2.2 Tracking stack (players)

Baseline stack for single-camera:
- **T1:** player detector (your finetuned YOLO) + MOT (ByteTrack/BoT-SORT family).
- **Offline association:** stitch tracklets across occlusion using appearance + motion, with a swap-avoidant gate.
- **Team A/B:** bib-based color classification per track (smoothed), with confidence.

### 2.3 Occlusion recovery (players)

Use segmentation only as a **budgeted fallback**:
- Trigger conditions (examples):
  - detected players < expected (e.g., < 12)
  - strong overlap clusters
  - high occlusion score (many close bboxes)
- Apply SAM2 only to:
  - **ROI crops** around overlap clusters, if feasible; otherwise a bounded “all players” fallback.
- Emission:
  - If SAM2 produces a confident mask/box: emit `present` with diagnostics `recovered_by=sam2`.
  - If not: emit `missing` (truth layer), optionally show a ghost marker (visual layer).

### 2.4 Ball tracking (MVP; learned model pending)

Ball is the hardest single-camera element; treat it explicitly:
- MVP-0: ball is optional (may be omitted or heuristic-only).
- MVP-1: integrate a **learned ball detector** checkpoint + lightweight tracker.
- Always: explicit missing/unknown per frame (no hallucination).
- Always: “ball visible quality” metrics drive event gating (shots/goals/highlights).

### 2.5 Calibration + BEV

One-time per venue/camera calibration is recommended:
- **Hybrid**: auto attempt → if fails quality gate → manual click flow (1–2 minutes).
- Store calibration artifact; reuse until camera moves.
- BEV is enabled only when calibration passes gate thresholds (see requirements).

### 2.6 Events + highlights (MVP)

MVP events:
- Shots + goals only, conservative.
- Candidate/unknown allowed; confirmed can be added later with a strict bar.

Highlights:
- Generate highlight clips primarily from:
  - conservative shot/goal candidates
  - ball proximity to goal region (when BEV available)
  - optional audio spike cue (future)

---

## 3) MVP milestones (ordered)

### MVP-0 — “Ship a real overlay, player-only (start now)”
Deliver:
- Player detector + tracker producing player `tracks.jsonl`.
- `overlay.mp4` showing player IDs + team colors (bibs).
- Report that includes identity-break counts and confidence.

Acceptance:
- No swaps on the evaluation clip(s) (breaks allowed).
- Overlay is playable and clearly indicates low-confidence segments.

### MVP-1 — “Add learned ball + conservative shots/goals + highlights (when ball model lands)”
Deliver:
- Learned ball detector + tracker producing ball records in `tracks.jsonl`.
- Conservative `events.json` (shots/goals).
- Highlight clip export for candidate shots/goals.

Acceptance:
- Ball missing is explicit; no ball hallucination.
- Events are conservative (low false positives).

### MVP-2 — “Calibration + BEV + running stats”
Deliver:
- Calibration artifact + gating + diagnostics.
- `bev.mp4` + meters-based stats (distance, speed, heatmap) only when BEV gate passes.

Acceptance:
- When BEV is off, report must explain why and stats are withheld (no “wrong meters”).

### MVP-3 — “Cost/scale hardening + multi-camera prep”
Deliver:
- Budgeted SAM2 usage + telemetry in reports.
- GCP pipeline hardened for batch throughput and predictable cost.
- Interfaces ready for multi-camera and UWB (future inputs).

---

## 4) Evaluation (how we know it works)

### 4.1 Minimal dataset strategy
- Start with your GoPro clip(s) + label:
  - player bboxes on sampled frames
  - identity continuity across a few hard occlusion sequences
  - ball presence/position on sampled windows
  - goal timestamps for highlight evaluation

### 4.2 Metrics to track every build
- **Identity swaps (severity-1):** must be zero on evaluation set.
- **Break rate:** how often tracks break (optimize down over time).
- **Coverage:** fraction of frames with `present` for players/ball.
- **BEV gate pass rate:** how often calibration is good enough.
- **Cost/time:** runtime and GPU seconds per minute of input video.

---

## 5) Deployment modes (three options; crystal clear)

These are **deployment modes** (where the code runs and where inputs/outputs live), not just “settings”.

### A) GCP Speed (cloud, GPU-first)
Goal: fastest turnaround.
- **Runs on:** GCP GPU worker (batch job).
- **Input:** video in **GCS**.
- **Output:** bundle written to **GCS**.
- **Knobs:** higher fps (10–15), larger decode width (e.g., 960–1280), SAM2 allowed but tightly bounded.

### B) GCP Low Cost (cloud, budget-first)
Goal: lowest $/match.
- **Runs on:** GCP worker (GPU or CPU, depending on economics).
- **Input:** video in **GCS**.
- **Output:** bundle written to **GCS**.
- **Knobs:** lower fps (8–10), smaller decode width (e.g., 640–960), strict SAM2 caps, skip expensive work when confidence is low.

### C) Local Laptop (actual laptop / venue PC)
Goal: run fully offline from local files.
- **Runs on:** an actual local machine (your laptop or a venue PC).
- **Input:** local video file path.
- **Output:** bundle written to local disk.
- **Knobs:** smaller decode width + low fps; SAM2 off by default; intended for demos/dev and early venue trials.

### GCP implementation outline (for A/B)
- Storage: input videos in **GCS**, outputs written back to **GCS**.
- Compute: a GPU-enabled batch worker (e.g., GCE GPU VM, GKE job, or Vertex AI custom job).
- Orchestration: queue jobs (Pub/Sub or Cloud Tasks) → worker pulls job → writes bundle + logs.
- Caching: keep model weights on disk on the worker image; avoid redownloading.

### Explicit non-goal (for MVP)
- We do **not** “outsource” GCP jobs to a user laptop. If we ever add a hybrid agent model, it will be a separate, explicit V2 deployment option.

---

## 6) Cost control principles (non-negotiable)

- Decode down (cap width) + run at reduced fps; interpolate for rendering.
- Crop to pitch ROI when possible.
- Run SAM2 only on failure windows; hard cap its total work per match.
- Keep strict separation between truth vs visualization layers to avoid “cheap continuity” corrupting stats.

---

## 7) How Rick’s approach fits MVP

Rick’s T1/T2/T3 concept is directionally right for single-camera occlusions, but we will:
- Keep T1 (trained YOLO + MOT).
- Keep T2 (SAM2), but budget it and prefer ROI usage.
- Replace T3 “invent positions” with:
  - truth layer: `missing/unknown`
  - visualization layer: optional ghost markers (excluded from stats).

---

## 8) V2 preview (not MVP)

- Passes/possession (requires robust ball + identity + BEV).
- Multi-camera fusion and time sync.
- UWB bibs / ankle tags as optional TrackSources for identity stability.
- Jersey OCR and player re-identification across matches.

---

## 9) Model assets (current state)

As of now (local repo state):
- Player detection checkpoints exist under `fusbal/models/` (e.g., `best.pt`, `yolo11m.pt`, `yolo11x.pt`).
- Learned ball model checkpoint: **in progress** (to be integrated in MVP-1).
- SAM2 checkpoint: **pending** (to be integrated as a budgeted fallback in MVP-0/1 once available).
