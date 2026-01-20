---
generated: true
source: spec/md/docs/runbooks/evidence.mdt
source_sha256: sha256:f973091a753b2660d3dbce6fd4671c3b89c4dea2106838af403d8328e6a41b91
---

# Evidence

Evidence is captured as machine-readable JSON under `status/audit/<INTENT_ID>/runs/<run_id>/run.json`.

## Record a guardrails run

`node tools/evidence/record_run.mjs --intent-id INT-001 --out status/audit/INT-001/runs/<run_id>/run.json -- npm run guardrails`

## Record a generation drift check

`node tools/evidence/record_run.mjs --intent-id INT-001 --out status/audit/INT-001/runs/<run_id>/run.json -- npm run generate:check`

