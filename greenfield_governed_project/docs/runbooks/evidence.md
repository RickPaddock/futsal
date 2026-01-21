---
generated: true
source: spec/md/docs/runbooks/evidence.mdt
source_sha256: sha256:a56e9985b24f884f3933622e7deb548e61e3836e5eeea316c6df6dfc8f95841c
---

# Evidence

Evidence is captured as machine-readable JSON under `status/audit/<INTENT_ID>/runs/<run_id>/run.json`.

## Record a guardrails run

`node tools/evidence/record_run.mjs --intent-id INT-001 --out status/audit/INT-001/runs/<run_id>/run.json -- npm run guardrails`

## Record a generation drift check

`node tools/evidence/record_run.mjs --intent-id INT-001 --out status/audit/INT-001/runs/<run_id>/run.json -- npm run generate:check`

## Record a preflight review report

Preflight reviews should write a machine-readable JSON report that future humans/LLMs can reference:

`status/audit/<INTENT_ID>/runs/<run_id>/preflight/preflight_report.json`
