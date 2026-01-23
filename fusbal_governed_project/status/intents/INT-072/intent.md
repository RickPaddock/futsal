---
generated: true
source: spec/intents/INT-072.json
source_sha256: sha256:e4ebd6ede40cf23402f2a15b203ae6b51fa5f087fc147723d55614fe400da6f3
intent_id: INT-072
title: MVP-0: bundle outputs (overlay + non-placeholder report)
status: todo
created_date: 2026-01-23
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-072"
---

# Intent: INT-072

- Produce a real, human-useful overlay.mp4 for MVP-0 (players + teams + confidence) inside the output bundle.
- Replace placeholder report.html with a minimal but truthful report explaining confidence and known gaps.

## Work packages

### INT-072-001 — Overlay

- TASK-OVERLAY-PLAYERS-MVP-001 Extend overlay rendering to include players + team + confidence cues.
- TASK-OVERLAY-BACKEND-ROBUSTNESS-MVP-001 Ensure overlay can render text reliably across backends.

### INT-072-002 — Reports and placeholders

- TASK-REPORT-HTML-MVP-001 Generate a minimal non-placeholder report.html explaining confidence and gaps.
- TASK-BUNDLE-NO-PLACEHOLDER-MVP-001 Stop shipping placeholder overlay/report for real runs.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

