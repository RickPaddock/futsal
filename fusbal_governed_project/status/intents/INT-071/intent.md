---
generated: true
source: spec/intents/INT-071.json
source_sha256: sha256:88bfde98b1aa998f56d09251a78d49d19f843e0a876ef194e57771e5a18d4a40
intent_id: INT-071
title: MVP-0: player detection + tracking (YOLO + MOT, trust-first)
status: todo
created_date: 2026-01-23
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-071"
---

# Intent: INT-071

- Integrate a real player detector (YOLO checkpoint) and MOT to produce player tracks on real video.
- Enforce trust-first identity: break over swap when association confidence is low or ambiguous; emit Unknown rather than guessing.

## Work packages

### INT-071-001 — Detection + tracking core

- TASK-PLAYER-DET-YOLO-001 Integrate YOLO player detector backend (checkpoint-driven).
- TASK-PLAYER-MOT-MVP-001 Implement MOT and swap-avoidant association policy.

### INT-071-002 — Contract outputs + team assignment

- TASK-PLAYER-TRACKS-JSONL-MVP-001 Emit contract-valid player records (segments, breaks, diagnostics).
- TASK-TEAM-BIBS-MVP-001 Assign Team A/B from bib colors with confidence and smoothing.

### INT-071-003 — Runnable path

- TASK-PLAYER-E2E-RUNNER-MVP-001 Wire the player pipeline into a runnable CLI path that writes a real bundle.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `models/`, `status/intents/`, `status/portal/`, `status/wizard/`

