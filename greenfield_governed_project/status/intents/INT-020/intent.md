---
generated: true
source: spec/intents/INT-020.json
source_sha256: sha256:4feb7072a7f3ef0d1453ee5af3ed82dfb0237474f95756dd7e7636816ed942f3
intent_id: INT-020
title: Calibration + BEV mapping MVP
status: todo
created_date: 2026-01-20
close_gate:
  - "npm run guardrails"
  - "npm run generate:check"
  - "npm run audit:intent -- --intent-id INT-020"
---

# Intent: INT-020

- Implement pitch mapping that is robust to wide-angle distortion and imperfect markings.
- Gate BEV outputs on calibration quality; always still deliver overlay/video diagnostics.

## Work packages

### INT-020-001 — Pitch model + mapping

- TASK-CAL-001 Define pitch templates and camera calibration inputs per venue/pitch.
- TASK-CAL-002 Implement auto-fit from field markings where possible; record reprojection error diagnostics.

### INT-020-002 — Manual fallback + quality gates

- TASK-CAL-003 Add a minimal correspondence-click fallback and explicit calibration quality gating.

