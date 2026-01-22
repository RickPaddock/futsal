---
generated: true
source: spec/intents/INT-060.json
source_sha256: sha256:df178076de9b730010002de776a7b692c606a949be940c95c4621b57bde045c7
intent_id: INT-060
title: Useful overlay.mp4 rendering for UAT (ball + confidence)
status: closed
created_date: 2026-01-22
closed_date: 2026-01-22
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-060"
---

# Intent: INT-060

- Render a playable, human-useful `overlay.mp4` from a local input video plus contract-valid `tracks.jsonl` (starting with ball in image_px).
- Keep behavior deterministic and trust-first: no overlays are drawn without corresponding track evidence.

## Work packages

### INT-060-001 — Overlay rendering

- TASK-OVERLAY-001 Implement `render-overlay` CLI to produce a useful overlay.mp4 from video + tracks.jsonl.
- TASK-OVERLAY-002 Determinism tests + actionable backend checks for overlay rendering.

### INT-060-002 — Hardening + portal UX

- TASK-OVERLAY-TIMEOUT-STDERR-001 Add ffmpeg timeout and bounded stderr excerpts to reduce hang risk.
- TASK-OVERLAY-FPS-PROBE-001 Probe fps from video metadata when --fps is omitted.
- TASK-OVERLAY-ENABLE-N-EXPR-001 Use frame-index based enable expressions to reduce float drift.
- TASK-OVERLAY-DEDUP-PER-FRAME-001 Deduplicate overlays per frame window (readability + smaller filters).
- TASK-OVERLAY-MAXOPS-TOPK-001 Apply deterministic top-K truncation for max_ops.
- TASK-OVERLAY-FFMPEG-OVERRIDE-VERSION-001 Support FUSBAL_FFMPEG_PATH override and capture ffmpeg version.
- TASK-OVERLAY-BBOX-CLAMP-001 Clamp/skip out-of-frame bboxes using probed video dimensions.
- TASK-OVERLAY-RENDER-REPORT-001 Write overlay_render_report.json for automation.
- TASK-OVERLAY-TRUSTFIRST-MESSAGE-001 Emit clear trust-first note when no markers are rendered.
- TASK-PORTAL-OVERLAY-LINK-001 Surface overlay.mp4 link in portal when present.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `spec/md/docs/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

