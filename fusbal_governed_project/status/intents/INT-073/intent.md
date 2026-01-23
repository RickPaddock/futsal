---
generated: true
source: spec/intents/INT-073.json
source_sha256: sha256:535eaa802e2eed3190b49acac755d4055cfd55afd873daf660b38af2addcaf8a
intent_id: INT-073
title: MVP-2: calibration + BEV (quality-gated) + running stats
status: todo
created_date: 2026-01-23
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-073"
---

# Intent: INT-073

- Add one-time calibration (auto-first with manual fallback) for a static single camera.
- Enable BEV outputs and meters-based stats only when calibration quality passes deterministic gates.

## Work packages

### INT-073-001 — Calibration

- TASK-CAL-AUTO-MVP-001 Implement auto-first calibration flow and emit metrics.
- TASK-CAL-MANUAL-CLICK-MVP-001 Implement a minimal manual click fallback and persisted artifact.

### INT-073-002 — BEV + stats

- TASK-BEV-GATE-DIAGNOSTICS-MVP-001 Implement deterministic gating and diagnostic reasons.
- TASK-BEV-RENDER-MVP-001 Render bev.mp4 when gated-on and ensure outputs are deterministic.
- TASK-STATS-METERS-MVP-001 Compute meters-based stats only when gated-on.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

