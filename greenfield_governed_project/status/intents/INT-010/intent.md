---
generated: true
source: spec/intents/INT-010.json
source_sha256: sha256:610a1c1da7d40d7f22af3d67370f9126b1022094f50effb3ce368b3c58b3bdc9
intent_id: INT-010
title: Define V1 output contract + pipeline scaffolding
status: todo
created_date: 2026-01-20
close_gate:
  - "npm run guardrails"
  - "npm run generate:check"
  - "npm run audit:intent -- --intent-id INT-010"
---

# Intent: INT-010

- Codify the match output bundle and canonical track/event schemas.
- Create a minimal pipeline CLI and directory layout to make subsequent vision/calibration work incremental.

## Work packages

### INT-010-001 — Output bundle + schemas

- TASK-V1-OUT-001 Define match bundle layout and required artifacts (overlay, BEV, tracks, events, report).
- TASK-V1-DATA-001 Define canonical track record and metadata needed for fusion and reporting.

### INT-010-002 — Pipeline scaffolding

- TASK-V1-CLI-001 Create a minimal CLI that initializes/validates bundle structure and manifests.

