---
generated: true
source: spec/intents/INT-074.json
source_sha256: sha256:9e911d22e91320d19017bc0975dd6e1538e9667ad89ad38b70704b4fe2bc2b67
intent_id: INT-074
title: MVP-1: learned ball model integration (checkpoint-driven)
status: todo
created_date: 2026-01-23
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-074"
---

# Intent: INT-074

- Replace/augment the baseline ball detector with a learned checkpoint-driven detector once it is available.
- Preserve trust-first semantics: explicit missing/unknown; no hallucination; quality metrics drive downstream gating.

## Work packages

### INT-074-001 — Detector + tracker

- TASK-BALL-DET-LEARNED-MVP-001 Integrate a learned ball detector backend with checkpoint selection.
- TASK-BALL-TRACK-MVP-001 Track the ball conservatively with explicit missing/unknown per frame.

### INT-074-002 — Contract integration

- TASK-BALL-CONTRACT-INTEGRATION-MVP-001 Ensure learned ball outputs conform to the contract and validators.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

