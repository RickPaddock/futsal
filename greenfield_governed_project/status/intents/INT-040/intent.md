---
generated: true
source: spec/intents/INT-040.json
source_sha256: sha256:2825762f85407ae4bbe6d0f591f5983d4f2bd1699987d085f20e3d011b2701b8
intent_id: INT-040
title: Ball tracking + conservative shots/goals (V1)
status: todo
created_date: 2026-01-20
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-040"
---

# Intent: INT-040

- Track the ball with explicit missing/unknown state; never hallucinate position when not visible.
- Infer shots/goals with high precision; mark candidates when uncertain.

## Work packages

### INT-040-001 — Ball detection + tracking

- TASK-BALL-DET-001 Per-frame ball detections with confidence + conservative gating + contract update.
- TASK-BALL-TRACK-001 Ball tracks with explicit missing spans + quality metrics/diagnostics + contract update.

### INT-040-002 — Shots/goals inference

- TASK-EVENT-001 Conservative shots/goals inference with confidence + evidence pointers + contract update.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `spec/md/docs/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

