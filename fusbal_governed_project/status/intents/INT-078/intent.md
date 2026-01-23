---
generated: true
source: spec/intents/INT-078.json
source_sha256: sha256:889fbba344729f39966298c7ea36f25f333b50ebea29185951ea6c24d4a960f1
intent_id: INT-078
title: MVP deployment: Local Laptop runner (offline bundle)
status: todo
created_date: 2026-01-23
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-078"
---

# Intent: INT-078

- Provide a true local-laptop execution mode: read a local video path and write a bundle to local disk.
- Keep outputs contract-valid and deterministic; provide clear guidance on prerequisites (ffmpeg, CUDA, etc.).

## Work packages

### INT-078-001 — Runner

- TASK-LOCAL-CLI-RUNNER-MVP-001 Add/extend a local CLI command that produces a real bundle.
- TASK-LOCAL-MODES-MVP-001 Add explicit mode selection (local laptop vs cloud profiles) in config.

### INT-078-002 — Docs

- TASK-LOCAL-DOCS-MVP-001 Update quick reference docs templates for local-laptop execution.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `spec/md/docs/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

