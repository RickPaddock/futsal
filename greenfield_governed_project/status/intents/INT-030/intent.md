---
generated: true
source: spec/intents/INT-030.json
source_sha256: sha256:da43467b03ccd2c85571a94553310f7a04f1448e6a891c1fb601b71a4e8f78c2
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

- TASK-PLAYER-DET-001 Per-frame player detections (bbox/pos, confidence, diagnostics) + contract update.
- TASK-PLAYER-MOT-001 Swap-avoidant MOT with breaks, confidence, and break reasons + contract update.

### INT-030-002 — Team assignment

- TASK-TEAM-001 Bib-color team assignment A/B/Unknown with smoothing + diagnostics + contract update.

