---
generated: true
source: spec/intents/INT-020.json
source_sha256: sha256:b606cdcb828c82dd3a483a3a1d612a5f2361fb877bac492f4bbf331d612bf7e0
intent_id: INT-020
title: Calibration + BEV mapping MVP
status: closed
created_date: 2026-01-20
closed_date: 2026-01-21
close_gate:
  - "npm run guardrails"
  - "npm run generate"
  - "npm run audit:intent -- --intent-id INT-020"
---

# Intent: INT-020

- Implement pitch mapping robust to imperfect/partial markings; V1 assumes frames are pre-undistorted (`image_pre_undistorted=true`) and MUST fail explicitly with diagnostics when that assumption is violated.
- Gate BEV outputs on calibration quality; always still deliver overlay/video diagnostics.

## Work packages

### INT-020-001 — Pitch model + mapping

- TASK-CAL-001 Pitch templates + calibration input schema (multi-venue/pitch) + contract update.
- TASK-CAL-002 Auto-fit from field markings + reprojection diagnostics + contract update.

### INT-020-002 — Manual fallback + quality gates

- TASK-CAL-003 Manual correspondence fallback input + explicit calibration quality gating + contract update.

## Runbooks (LLM navigation)

- Decision: `update`
- Templates: `spec/md/docs/runbooks/calibration-and-bev.mdt`
- Notes: This intent introduces calibration input records, pitch templates, auto-fit diagnostics, and trust-first BEV gating. A runbook is required so future LLMs/humans can navigate and reproduce calibration workflows safely.

## Scope (paths)

- Allowed: `spec/`, `spec/md/docs/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

