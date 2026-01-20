---
generated: true
source: spec/requirements/index.json + spec/requirements/areas/core.json + spec/requirements/areas/v1.json
source_sha256: sha256:e9cb8ff6bbea60d676b4f74b1878caecc649e202922041cff8a1d345892c994c
---

# Requirements (generated)

Source: `spec/requirements/index.json`

## SYS-ARCH-15 — All code units MUST be traceable to requirements; shared utilities map to SYS-ARCH-15.

- Status: `canonical`
- Implementation: `todo`
- Owner: `platform`
- Tags: `governance`, `traceability`

Acceptance:
- No code unit can be merged without REQ mapping.
- Shared/plumbing code uses SYS-ARCH-15 with a human WHY.

## AUD-REQ-10 — Generate + enforce Requirement ↔ Code provenance (100% policy with guardrails).

- Status: `canonical`
- Implementation: `todo`
- Owner: `platform`
- Tags: `governance`, `guardrails`

Acceptance:
- Deterministic provenance scanner exists and is enforced by guardrails.
- Derived reports exist and are never hand-edited.

## FUSBAL-V1-TRUST-001 — Trust-first behavior: avoid identity swaps and ball/event hallucinations; prefer Unknown/missing over wrong.

- Status: `canonical`
- Implementation: `todo`
- Owner: `product`
- Tags: `v1`, `trust`

Acceptance:
- Player tracking prefers track breaks over ID swaps when ambiguous.
- Ball tracking explicitly supports missing/unknown state and does not fabricate positions.
- Event inference is high-precision (low-recall acceptable) and emits confidence with evidence links.

## FUSBAL-V1-OUT-001 — Produce a deterministic per-match output bundle (overlay, BEV, tracks, events, report) with diagnostics.

- Status: `canonical`
- Implementation: `todo`
- Owner: `platform`
- Tags: `v1`, `outputs`

Acceptance:
- A match run outputs a stable directory layout under `output/<match_id>/`.
- At minimum, an overlay video and a report describing confidence/known gaps are produced.
- BEV outputs are produced only when calibration quality passes thresholds; otherwise disabled with reasons.

## FUSBAL-V1-DATA-001 — All sources emit a canonical track record format suitable for fusion and reporting.

- Status: `canonical`
- Implementation: `todo`
- Owner: `platform`
- Tags: `v1`, `data`

Acceptance:
- Tracks are stored as `.jsonl` (one record per entity per sample) plus JSON metadata.
- Each sample includes timestamp, coordinate frame, uncertainty, and optional quality (0–1).

## FUSBAL-V1-CAL-001 — Calibration maps each camera to pitch coordinates with explicit quality gating and diagnostics.

- Status: `canonical`
- Implementation: `todo`
- Owner: `vision`
- Tags: `v1`, `calibration`

Acceptance:
- Calibration outputs homographies (or equivalent) and a quality score with failure reasons.
- If auto-calibration fails, a minimal manual fallback can produce a reusable calibration artifact.

## FUSBAL-V1-BEV-001 — When calibration quality is sufficient, generate bird’s-eye-view (BEV) tracks and video in pitch meters.

- Status: `canonical`
- Implementation: `todo`
- Owner: `vision`
- Tags: `v1`, `bev`

Acceptance:
- Players (and ball when available) are projected into a consistent pitch coordinate system in meters.
- BEV video renders movement with clear confidence cues and missing-state semantics.

## FUSBAL-V1-PLAYER-001 — Detect and track all players per match with conservative identity continuity.

- Status: `canonical`
- Implementation: `todo`
- Owner: `vision`
- Tags: `v1`, `players`

Acceptance:
- Per-frame detections carry confidence and are filtered to the pitch area when possible.
- Tracking produces track IDs and per-track confidence; ambiguous associations cause breaks, not swaps.

## FUSBAL-V1-TEAM-001 — Assign players to Team A/B via bib colors (plus smoothing) with an Unknown state.

- Status: `canonical`
- Implementation: `todo`
- Owner: `vision`
- Tags: `v1`, `teams`

Acceptance:
- Team label is output per player sample (A/B/Unknown) with confidence.
- Assignments are temporally smoothed and include diagnostics for ambiguous segments.

## FUSBAL-V1-BALL-001 — Track the ball when visible and explicitly represent missing/unknown spans without hallucination.

- Status: `canonical`
- Implementation: `todo`
- Owner: `vision`
- Tags: `v1`, `ball`

Acceptance:
- Ball detector/track output includes confidence and missing-state semantics.
- Long ball-missing spans are allowed; output must prefer unknown over wrong.

## FUSBAL-V1-EVENT-001 — Infer shots and goals conservatively in V1 (high precision) and emit confidence + evidence pointers.

- Status: `canonical`
- Implementation: `todo`
- Owner: `vision`
- Tags: `v1`, `events`

Acceptance:
- Events are emitted with timestamps and confidence; uncertain events are omitted or marked candidate.
- Goal events include a best-effort evidence window reference (time range / frame ids).

## FUSBAL-V1-SENSOR-001 — Sensors are optional inputs that only improve outputs when healthy; camera-only output remains sellable.

- Status: `canonical`
- Implementation: `todo`
- Owner: `platform`
- Tags: `v1`, `sensors`

Acceptance:
- The pipeline accepts sensor logs as additive TrackSources using the canonical track format.
- If sensor ingestion fails or is low quality, outputs fall back to camera-only without contract changes.

