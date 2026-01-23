---
generated: true
source: spec/intents/INT-075.json
source_sha256: sha256:ef498df04e13f9e57cb6170dfb9741d5f18798e636e23e9ae32b2148e37be9da
intent_id: INT-075
title: MVP-1: events + highlights (shots/goals only, conservative)
status: todo
created_date: 2026-01-23
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-075"
---

# Intent: INT-075

- Infer shots/goals conservatively and export highlights suitable for selling value to tracked players.
- Keep event semantics high precision; candidate/unknown allowed; never emit confirmed without a strict bar.

## Work packages

### INT-075-001 — Events

- TASK-EVENTS-SHOTS-GOALS-MVP-001 Implement conservative shot/goal inference suitable for MVP.

### INT-075-002 — Highlights + reporting

- TASK-HIGHLIGHTS-CLIPS-MVP-001 Export highlight clips for candidate shots/goals.
- TASK-EVENTS-REPORTING-MVP-001 Surface events/highlights and confidence in report.html/report.json.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

