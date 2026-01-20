---
generated: true
source: spec/intents/INT-040.json
source_sha256: sha256:8c97ac9ad3a5c5df49d4e4003bfd8e5eb2b46a755ce485b9a2f1380133813e31
intent_id: INT-040
title: Ball tracking + conservative shots/goals (V1)
status: todo
created_date: 2026-01-20
close_gate:
  - "npm run guardrails"
  - "npm run generate:check"
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

