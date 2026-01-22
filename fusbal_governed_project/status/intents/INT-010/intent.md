---
generated: true
source: spec/intents/INT-010.json
source_sha256: sha256:93c33cb89c55b53d1a7283583b823f2183dc9281c72a2095187da35351990e92
intent_id: INT-010
title: Define V1 output contract + pipeline scaffolding
status: closed
created_date: 2026-01-20
closed_date: 2026-01-21
close_gate:
  - "npm run guardrails"
  - "npm run generate:check"
  - "npm run audit:intent -- --intent-id INT-010"
---

# Intent: INT-010

- Codify the match output bundle and canonical track/event schemas.
- Create a minimal pipeline CLI and directory layout to make subsequent vision/calibration work incremental.

## Work packages

### INT-010-001 — Output bundle + schemas

- TASK-V1-OUT-001 Define bundle layout + required artifacts + deterministic manifests (code + contract).
- TASK-V1-DATA-001 Define canonical record schemas + metadata (code + contract).

### INT-010-002 — Pipeline scaffolding

- TASK-V1-CLI-001 Minimal CLI for init/validate of bundle layout with clear errors.

### INT-010-003 — Follow-up quality improvements

- TASK-INT-010-IMP-001 Stream `tracks.jsonl` validation to avoid loading large files into memory.
- TASK-INT-010-IMP-002 Add structured error codes and optional JSON output for validation.
- TASK-INT-010-IMP-003 Add minimal unit tests for determinism and schema validation.
- TASK-INT-010-IMP-004 Extend `init` to optionally scaffold placeholder outputs.
- TASK-INT-010-IMP-005 Tighten numeric/range validation for track/event fields.
- TASK-INT-010-IMP-006 Improve manifest portability (avoid forced absolute paths).
- TASK-INT-010-IMP-007 Export the V1 contract as JSON Schema for downstream validators.
- TASK-INT-010-IMP-008 Show per-task quality audit results in the internal portal UI.
- TASK-INT-010-IMP-009 Capture evidence for `npm run generate` runs and enforce at close.
- TASK-INT-010-IMP-010 Add lightweight formatting/linting for the pipeline Python package.

## Runbooks (LLM navigation)

- Decision: `update`
- Templates: `spec/md/docs/runbooks/evidence.mdt`, `spec/md/docs/runbooks/intent-quality.mdt`, `spec/md/docs/runbooks/intent-and-task-workflow.mdt`
- Notes: This intent updates audit/runbook guidance so future LLMs understand evidence capture, generation writes, and portal navigation expectations.

## Scope (paths)

- Allowed: `spec/`, `spec/md/docs/`, `pipeline/`, `scripts/`, `tools/`, `apps/portal/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

