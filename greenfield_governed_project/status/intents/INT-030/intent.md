---
generated: true
source: spec/intents/INT-030.json
source_sha256: sha256:053a47f78cb8a989109723a7ab13358d899b86deef3245044edc03ce6a9c2c51
intent_id: INT-030
title: Player tracking MVP (trust-first)
status: todo
created_date: 2026-01-20
close_gate:
  - "npm run guardrails"
  - "npm run generate:check"
  - "npm run audit:intent -- --intent-id INT-030"
---

# Intent: INT-030

- Detect and track all players with conservative identity continuity (breaks preferred over swaps).
- Assign players to teams using bib colors with temporal smoothing and an Unknown state.

## Work packages

### INT-030-001 — Detection + tracking

- TASK-PLAYER-DET-001 Select/implement player detector(s) and output per-frame detections with confidence.
- TASK-PLAYER-MOT-001 Implement MOT with strong swap avoidance; expose track confidence and break reasons.

### INT-030-002 — Team assignment

- TASK-TEAM-001 Implement bib-color team assignment with smoothing and diagnostics (A/B/Unknown).

