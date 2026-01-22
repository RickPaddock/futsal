---
generated: true
source: spec/intents/INT-030.json
source_sha256: sha256:c07cb983393b50b4052036a3fc7c5a1c37f2d6e708ba74870a04d6793f41e76f
intent_id: INT-030
title: Player tracking MVP (trust-first)
status: todo
created_date: 2026-01-20
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-030"
---

# Intent: INT-030

- Detect and track all players with swap avoidance: ambiguous associations must end a segment with an explicit break_reason (breaks preferred over swaps).
- Assign players to teams using bib colors with temporal smoothing; team values are A/B/unknown (unknown is valid and preferred when evidence is weak).

## Work packages

### INT-030-001 — Detection + tracking

- TASK-PLAYER-DET-001 Per-frame player detections (bbox/pos, confidence, diagnostics) + contract update.
- TASK-PLAYER-MOT-001 Swap-avoidant MOT with breaks, confidence, and break reasons + contract update.

### INT-030-002 — Team assignment

- TASK-TEAM-001 Bib-color team assignment A/B/Unknown with smoothing + diagnostics + contract update.

### INT-030-003 — Implementation quality improvements

- TASK-INT-030-IMP-001 Add executable fixture run path that emits tracks.jsonl (detections→MOT→team).
- TASK-INT-030-IMP-002 Add deterministic unit tests for detections, MOT, and team smoothing.
- TASK-INT-030-IMP-003 Tighten contract validation for break_reason/segment semantics.
- TASK-INT-030-IMP-004 Add fixture-based smoke run that generates a valid, non-placeholder bundle.
- TASK-INT-030-IMP-005 Reduce contract template drift with schema snippet injection during generation.
- TASK-INT-030-IMP-006 Emit explicit unknown records/diagnostics for ambiguous detections.
- TASK-INT-030-IMP-007 Bound MOT state and add out_of_view heuristic.
- TASK-INT-030-IMP-008 Centralize diagnostics key constants.
- TASK-INT-030-IMP-009 Improve portal surfacing for per-task quality blockers.
- TASK-INT-030-IMP-010 Add git/environment metadata to evidence run.json.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `spec/md/docs/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

