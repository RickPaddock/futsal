---
generated: true
source: spec/intents/INT-070.json
source_sha256: sha256:1328065837a259482b2ad9cc82048a2c9b86a84ded8ef8a7a70364ee8f29671b
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

- Run the MVP pipeline as a batch job on GCP using a deterministically-normalized job spec and write a contract-valid bundle to GCS (deterministic layout + manifest/metadata; encoder bytes may vary).
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

