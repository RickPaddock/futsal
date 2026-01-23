---
generated: true
source: spec/intents/INT-076.json
source_sha256: sha256:9a8653b230781ecf9c08674ff0a6ae47965f3f8723cb2aab1221656a45e6fd7f
intent_id: INT-076
title: MVP: evaluation harness + regression gates
status: todo
created_date: 2026-01-23
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-076"
---

# Intent: INT-076

- Define how we measure MVP quality (identity swaps, break rate, coverage, ball visible recall, BEV gating).
- Add repeatable evaluation runs and regression gates so improvements do not silently break trust.

## Work packages

### INT-076-001 — Dataset + metrics

- TASK-EVAL-DATASET-SPEC-MVP-001 Define a compact evaluation dataset format and fixtures.
- TASK-EVAL-METRICS-MVP-001 Implement evaluation metrics for swaps/breaks/coverage/BEV gate.

### INT-076-002 — Regression + evidence

- TASK-EVAL-GOLDEN-RUNS-MVP-001 Create golden evidence runs and store summaries under status/audit.
- TASK-EVAL-REGRESSION-GATES-MVP-001 Add regression thresholds/gates that fail builds on trust regressions.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `scripts/`, `tools/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

