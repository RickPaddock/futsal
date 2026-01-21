---
generated: true
source: spec/intents/INT-020.json
source_sha256: sha256:096a7cb4a058df958559047d1b2adf0f000a166695eac94477549c609abda7ac
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

- TASK-CAL-001 Pitch templates + calibration input schema (multi-venue/pitch) + contract update.
- TASK-CAL-002 Auto-fit from field markings + reprojection diagnostics + contract update.

### INT-020-002 — Manual fallback + quality gates

- TASK-CAL-003 Manual correspondence fallback input + explicit calibration quality gating + contract update.

