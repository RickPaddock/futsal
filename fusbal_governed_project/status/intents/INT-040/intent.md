---
generated: true
source: spec/intents/INT-040.json
source_sha256: sha256:2c356038594a514503e30a1cabed473cc5e963c1baa94e8a38310e12a939b549
intent_id: INT-040
title: Ball tracking + conservative shots/goals (V1)
status: closed
created_date: 2026-01-20
closed_date: 2026-01-22
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-040"
---

# Intent: INT-040

- Track the ball with explicit missing/unknown state; never hallucinate position when not visible.
- Infer shots/goals with high precision; mark candidates when uncertain.

## Work packages

### INT-040-001 — Ball detection + tracking

- TASK-BALL-DET-001 Per-frame ball detections with confidence + conservative gating + contract update.
- TASK-BALL-TRACK-001 Ball tracks with explicit missing spans + quality metrics/diagnostics + contract update.

### INT-040-002 — Shots/goals inference

- TASK-EVENT-001 Conservative shots/goals inference with confidence + evidence pointers + contract update.

### INT-040-003 — Quality + auditability improvements

- TASK-INT-040-IMP-001 Ball detections emit explicit missing/unknown records.
- TASK-INT-040-IMP-002 Contract template documents ball detection/track schemas and examples.
- TASK-INT-040-IMP-003 Ball detections use centralized diagnostics key constants.
- TASK-INT-040-IMP-004 Ball tracker break_reason/segment semantics aligned with missing reasons.
- TASK-INT-040-IMP-005 Add deterministic negative-scenario tests for shots/goals inference.
- TASK-INT-040-IMP-006 Define and test ball pos_state=unknown semantics.
- TASK-INT-040-IMP-007 Implement explicit unknown emission path in ball tracker.
- TASK-INT-040-IMP-008 Strengthen contract validation for ball-specific semantic checks.
- TASK-INT-040-IMP-009 Emit ball tracking quality metrics into diagnostics/quality_summary.json.
- TASK-INT-040-IMP-010 Portal visualizations for ball missing/unknown and evidence pointers.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `spec/md/docs/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

