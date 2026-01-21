---
generated: true
source: spec/intents/INT-001.json
source_sha256: sha256:0c20c7d99d9183e6b7b6a971e00d66e2a850fefefdae6852fe5b859be14134a5
intent_id: INT-001
title: Bootstrap Fusbal governed project: requirements, provenance, evidence, portal
status: closed
created_date: 2026-01-20
closed_date: 2026-01-20
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

- TASK-BOOT-001 Deterministic generators + canonical templates for governed surfaces (no .md hand-edits).
- TASK-BOOT-002 Guardrails enforcing governance invariants + provenance/requirements discipline.

### INT-001-002 — Evidence + portal

- TASK-BOOT-003 Evidence run recorder + internal portal pages (intents/tasks + refresh).
- TASK-BOOT-004 Generated docs templates for navigation, runbooks, and delivery plan surfaces.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

