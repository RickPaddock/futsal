---
generated: true
source: spec/md/docs/LLM_NAVIGATION.mdt
source_sha256: sha256:0e45385ce92a1def4d7d2716b21c4c404a2725f30d3e6c676e6f97d1a90503b8
---

# LLM navigation

This repo is governed: edit sources, then regenerate derived surfaces.

## Start here

1) `README.md` (generated overview)
2) `AGENTS.md` (generated agent rules)
3) `docs/LLM_QUICK_REFERENCE.md` (generated commands)
4) `docs/delivery/HL_DELIVERY_PLAN.md` (generated roadmap)
5) `docs/data/OUTPUT_CONTRACT.md` (generated data/outputs contract)
6) `docs/runbooks/README.md` (generated runbook index)
7) `docs/ops/governance/sources_and_flow.md` (generated governance model)

## What is source vs generated?

Sources (edit these):
- Requirements: `spec/requirements/index.json`
- Intents: `spec/intents/*.json`
- Markdown templates: `spec/md/**/*.mdt`
- Code: `apps/**`, `packages/**`, `scripts/**`, `tools/**`

Generated (do not hand-edit):
- All `.md` outside `spec/`
- `docs/requirements/requirements.md`
- `status/intents/**/{intent.md,scope.json,work_packages.json}`
- `status/portal/internal_intents.json`
