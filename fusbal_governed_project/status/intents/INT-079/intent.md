---
generated: true
source: spec/intents/INT-079.json
source_sha256: sha256:6fe6e6d619de8f4b9848f42fe3a9b4e85975aa9dbffd1584a142dfb219196f65
intent_id: INT-079
title: Portal: MVP job runs and artifact surfacing
status: todo
created_date: 2026-01-23
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-079"
---

# Intent: INT-079

- Make the internal portal a practical operations surface for MVP runs across deployment modes.
- Surface the latest bundles/artifacts (overlay, report, bev when present) and show clear run status.

## Work packages

### INT-079-001 — Run list + status

- TASK-PORTAL-JOB-LIST-MVP-001 Add portal surfaces for listing MVP runs across intents/modes.
- TASK-PORTAL-RUN-STATUS-MVP-001 Show run status, mode, timestamps, and failure reasons.

### INT-079-002 — Artifacts

- TASK-PORTAL-ARTIFACTS-MVP-001 Link to overlay/report/bev artifacts when present and show missing reasons.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `apps/portal/`, `status/audit/`, `tools/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

