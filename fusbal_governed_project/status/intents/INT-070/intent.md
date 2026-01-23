---
generated: true
source: spec/intents/INT-070.json
source_sha256: sha256:3617f91fdad8b07c1994add3dff42ef6dad505a6d8f3530fb4a2dd8cf7564d3b
intent_id: INT-070
title: MVP deployment: GCP batch runner (Speed + Low Cost)
status: todo
created_date: 2026-01-23
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-070"
---

# Intent: INT-070

- Run the MVP pipeline as a deterministic batch job on GCP and write a contract-valid bundle to GCS.
- Support two cloud deployment modes: GCP Speed (GPU-first) and GCP Low Cost (budget-first).

## Work packages

### INT-070-001 — Job spec + storage I/O

- TASK-GCP-JOBSPEC-001 Define a batch job spec for MVP runs (inputs, outputs, mode, config).
- TASK-GCP-GCS-IO-001 Implement GCS input download and bundle upload (atomic layout).

### INT-070-002 — Container + orchestration

- TASK-GCP-CONTAINER-001 Containerize the runner with pinned deps and repeatable builds.
- TASK-GCP-ORCHESTRATION-001 Execute jobs via a GCP batch primitive with retries and timeouts.

### INT-070-003 — Telemetry + reporting

- TASK-GCP-RUN-TELEMETRY-001 Emit runtime/cost-oriented telemetry to diagnostics/report and logs.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `scripts/`, `tools/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

