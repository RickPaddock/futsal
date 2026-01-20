---
generated: true
source: spec/intents/INT-010.json
source_sha256: sha256:5b61761789c8d8cf21ded433558d4d9e92e73a0bed64e96640e73a7a0d79642a
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

- TASK-V1-OUT-001 Define bundle layout + required artifacts + deterministic manifests (code + contract).
- TASK-V1-DATA-001 Define canonical record schemas + metadata (code + contract).

### INT-010-002 — Pipeline scaffolding

- TASK-V1-CLI-001 Minimal CLI for init/validate of bundle layout with clear errors.

