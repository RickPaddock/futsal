---
generated: true
source: spec/intents/INT-001.json
source_sha256: sha256:94ebcdd9ed98beeb3f6cd1d25c9115aab99cdfa7270c4f090a28ff4f64a2acf0
intent_id: INT-001
title: Bootstrap Fusbal governed project: requirements, provenance, evidence, portal
status: todo
created_date: 2026-01-20
close_gate:
  - "npm run guardrails"
  - "npm run generate:check"
  - "npm run audit:intent -- --intent-id INT-001"
---

# Intent: INT-001

- This intent boots Fusbal into a governed state from day 1 (no backfills).
- All human-readable .md files are generated; humans edit spec/templates only.
- Delivery plans are represented as intents and work packages, then surfaced via portal/status outputs.

## Work packages

### INT-001-001 — Repo scaffolding + generation

- TASK-BOOT-001 Create spec/templates and generators for all .md outputs
- TASK-BOOT-002 Add guardrails to enforce generation (and document provenance tagging)

### INT-001-002 — Evidence + portal

- TASK-BOOT-003 Add evidence run records + a minimal internal portal
- TASK-BOOT-004 Add Fusbal product docs + high-level delivery plan surfaces

