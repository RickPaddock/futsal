---
generated: true
source: spec/md/AGENTS.mdt
source_sha256: sha256:8652b63d80d1428d7a29ad0ae0f8a1b34577a205ffa9f7c1c9371a31ec4a9d71
---

# Agent instructions (FUSBAL)

This repository is designed to be **movable and standalone**. Avoid machine-specific paths and do not introduce coupling to other repositories.

## Governance hard rules

- **No hand-edits to `.md`**: all human-readable Markdown is machine-generated.
- Edit sources only:
  - Requirements: `spec/requirements/index.json`
  - Intents: `spec/intents/*.json`
  - Tasks: `spec/tasks/*.json`
  - Prompt templates: `spec/prompts/*.prompt.txt`
  - Markdown templates: `spec/md/**/*.mdt`
- After editing sources, regenerate: `npm run generate`
- Guardrails must pass before declaring done: `npm run guardrails`

## Derived artefacts (do not hand-edit)

- `README.md`
- `AGENTS.md`
- `docs/**/*.md`
- `docs/requirements/requirements.md`
- `status/intents/**/intent.md`
- `status/intents/**/scope.json`
- `status/intents/**/work_packages.json`
- `status/portal/internal_intents.json`

## Evidence (recommended)

Record validation runs as a `run.json`:

`node tools/evidence/record_run.mjs --intent-id INT-001 --out status/audit/INT-001/runs/<run_id>/run.json -- npm run guardrails`
