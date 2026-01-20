---
generated: true
source: spec/intents/INT-040.json
source_sha256: sha256:eb7b8f78a3eb883092771e67947abd3a0f7c3d1023dc520eabef42661d3ccd7d
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

- TASK-BALL-DET-001 Implement ball detector(s) and gating to avoid false positives.
- TASK-BALL-TRACK-001 Implement intermittent ball tracking with explicit missing state and quality metrics.

### INT-040-002 — Shots/goals inference

- TASK-EVENT-001 Implement conservative shots/goals inference and emit events with confidence and evidence links.

