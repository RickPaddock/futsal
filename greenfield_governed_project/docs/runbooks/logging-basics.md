---
generated: true
source: spec/md/docs/runbooks/logging-basics.mdt
source_sha256: sha256:a257012fb11e358ad2aea27ee3d0d27bf34c8a0b5845ad05ec990f5b825624ef
---

# Logging basics

## What to log

Every runnable should emit enough context to debug and audit:

- `intent_id` (when applicable)
- `task_id` / `subtask_id` (when applicable)
- `run_id` (when producing artefacts under `status/audit/...`)
- Errors in a stable prefix form: `[<area>:error] <message>`

## Format

- Prefer single-line, grep-friendly messages.
- For machine-readable artefacts, write JSON files and include them as evidence artefacts via `tools/evidence/record_run.mjs --artefact <path>`.

## Evidence integration

Record validation/audit commands as evidence:

`node tools/evidence/record_run.mjs --intent-id INT-001 --out status/audit/INT-001/runs/<run_id>/run.json -- npm run guardrails`

