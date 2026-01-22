---
generated: true
source: spec/intents/INT-060.json
source_sha256: sha256:0eaef0002eda866fa787ab82d377fb9e1978827e41259fc29de52dec5f28816b
intent_id: INT-060
title: Useful overlay.mp4 rendering for UAT (ball + confidence)
status: todo
created_date: 2026-01-22
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

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `spec/md/docs/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

