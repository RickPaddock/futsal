---
generated: true
source: spec/intents/INT-077.json
source_sha256: sha256:cf7e53ea4745ebfdd4470b0db21f2c75bfb666d575ed29736a9cf5c0995713ec
intent_id: INT-077
title: MVP: SAM2 occlusion recovery (budgeted, evidence-backed)
status: todo
created_date: 2026-01-23
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-077"
---

# Intent: INT-077

- Integrate SAM2 as an occlusion-recovery fallback for player tracking once a checkpoint is available.
- Keep it budgeted (windowed, capped) and trust-first (no fabricated truth positions).

## Work packages

### INT-077-001 — Integration

- TASK-SAM2-CHECKPOINT-LOADING-MVP-001 Add portable SAM2 checkpoint/config loading and backend checks.

### INT-077-002 — Budgeting + ROI

- TASK-SAM2-WINDOWING-MVP-001 Trigger SAM2 only on failure windows (count drop/overlap).
- TASK-SAM2-ROI-CROPS-MVP-001 Prefer ROI crops and cap work per match.
- TASK-SAM2-BUDGET-TELEMETRY-MVP-001 Record SAM2 usage in diagnostics/report for cost control.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

