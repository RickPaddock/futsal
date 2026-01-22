---
generated: true
source: spec/md/docs/ops/governance/sources_and_flow.mdt
source_sha256: sha256:e01698fa6238a4dc3557e1bd6d469c5ce1459f5c837e7215fe27a0d413edc643
---

# Governance: sources and flow

This repository enforces a strict separation between **canonical sources** and **generated surfaces**.

## Sources (edited by humans / LLMs)

- Requirements: `spec/requirements/index.json`
- Intents: `spec/intents/*.json`
- Markdown templates: `spec/md/**/*.mdt`
- Runtime code: `apps/**`, `packages/**`, `scripts/**`, `tools/**`

## Generated surfaces (never hand-edit)

- All `.md` outside `spec/`
- `docs/requirements/requirements.md`
- `status/intents/**/{intent.md,scope.json,work_packages.json}`
- `status/portal/internal_intents.json`

## Enforcement

- Generation is deterministic: `npm run generate` and `npm run generate:check`
- Guardrails fail fast if any `.md` is not generated: `npm run guardrails`

